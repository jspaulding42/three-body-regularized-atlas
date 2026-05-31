"""Independent checker for serialized three-body proof certificates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Iterable

import numpy as np

from .certificate_language import (
    BranchUnionCertificate,
    ChartChainCertificate,
    EventIsolationCertificate,
    FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    GeneralizedFuchsianRemainderMajorantCertificate,
    FuchsianLogTermCertificate,
    OrdinaryChartTransitionCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    PlanarLeviCivitaTransitionCertificate,
    SpatialKSBinaryChartCertificate,
    SpatialKSTransitionCertificate,
    TotalCollisionFuchsianStopChartCertificate,
    TotalCollisionGeneralizedFuchsianStopChartCertificate,
)
from .binary_chart import (
    RegularizedBinaryCollisionChartState,
    planar_accelerations_from_regularized_chart_rhs,
    regularized_binary_collision_chart_rhs,
    regularized_binary_collision_chart_to_planar,
)
from .binary_series import (
    RegularizedBinaryTaylorSolution,
    pair_energy_constraint_coefficients as lc_pair_energy_constraint_coefficients,
    regularized_rhs_coefficients as lc_regularized_rhs_coefficients,
)
from .dynamics import accelerations
from .event_recurrence import PrimitiveCauchyTailInput
from .fuchsian import (
    FiniteFuchsianLogBranch,
    FiniteFuchsianLogTotalCollisionIsolationCertificate,
    FuchsianShapeBranch,
    FuchsianLogTerm,
    _finite_fuchsian_log_branch_derivative_envelope,
    certify_finite_fuchsian_log_total_collision_isolation,
    construct_fuchsian_shape_branch,
)
from .intervals import (
    FloatInterval,
    RationalInterval,
    interval_array_series_eval,
    interval_polyder,
    interval_polynomial_eval,
    interval_sign,
    rational_interval_polyder,
    rational_interval_polynomial_eval,
    rational_interval_sign,
)
from .ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    regularized_ks_binary_chart_rhs,
    spatial_accelerations_from_ks_binary_rhs,
)
from .ks_binary_series import (
    SpatialKSBinaryTaylorSolution,
    ks_horizontal_constraint_coefficients,
    ks_pair_energy_constraint_coefficients,
    regularized_rhs_coefficients as ks_regularized_rhs_coefficients,
)
from .series import acceleration_coefficients
from .stratified_branch_tree import SUPPORTED_STRATIFIED_LEAF_KINDS


@dataclass(frozen=True)
class CertificateCheckObligation:
    """One checker obligation for a serialized certificate."""

    obligation: str
    certified: bool
    detail: str


@dataclass(frozen=True)
class CertificateCheckResult:
    """Result of checking one serialized certificate."""

    certificate_id: str
    certificate_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    max_coefficient_residual: float
    max_sampled_newton_residual: float

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(obligation.certified for obligation in self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if not obligation.certified
        )


@dataclass(frozen=True)
class TransitionCheckResult:
    """Result of checking one serialized chart transition."""

    transition_id: str
    transition_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    max_position_gap: float
    max_velocity_gap: float

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(obligation.certified for obligation in self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if not obligation.certified
        )


@dataclass(frozen=True)
class EventIsolationCheckResult:
    """Result of checking one serialized event-isolation certificate."""

    event_id: str
    event_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    root_interval_width: float

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(obligation.certified for obligation in self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if not obligation.certified
        )


@dataclass(frozen=True)
class BranchUnionCheckResult:
    """Result of checking one serialized finite branch-union certificate."""

    union_id: str
    union_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    leaf_count: int

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(obligation.certified for obligation in self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if not obligation.certified
        )


@dataclass(frozen=True)
class ChartChainCheckResult:
    """Result of checking one serialized finite chart-chain certificate."""

    chain_id: str
    chain_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    chart_count: int

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(obligation.certified for obligation in self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if not obligation.certified
        )


@dataclass(frozen=True)
class ProofGradeArithmeticBackendCertificate:
    """Small certificate for exact rational interval arithmetic support."""

    backend_id: str
    exact_fraction_endpoints: bool
    rational_interval_operations_checked: bool
    rational_interval_polynomial_eval_checked: bool
    rational_interval_sign_trichotomy_checked: bool
    finite_float_to_fraction_embedding_checked: bool
    statement: str
    proof_sketch: str
    witness_source: str = "rational_interval_arithmetic_backend"

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.backend_id == "python_fraction_rational_interval_backend"
            and self.exact_fraction_endpoints
            and self.rational_interval_operations_checked
            and self.rational_interval_polynomial_eval_checked
            and self.rational_interval_sign_trichotomy_checked
            and self.finite_float_to_fraction_embedding_checked
            and self.statement
            and self.proof_sketch
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("exact_fraction_endpoints", self.exact_fraction_endpoints),
            (
                "rational_interval_operations_checked",
                self.rational_interval_operations_checked,
            ),
            (
                "rational_interval_polynomial_eval_checked",
                self.rational_interval_polynomial_eval_checked,
            ),
            (
                "rational_interval_sign_trichotomy_checked",
                self.rational_interval_sign_trichotomy_checked,
            ),
            (
                "finite_float_to_fraction_embedding_checked",
                self.finite_float_to_fraction_embedding_checked,
            ),
        )
        missing = [name for name, certified in fields if not certified]
        if self.backend_id != "python_fraction_rational_interval_backend":
            missing.append("proof_grade_arithmetic_backend_id")
        return tuple(missing)


@dataclass(frozen=True)
class CertificateCheckerKernelSupportCertificate:
    """Theorem-facing support manifest for the independent checker kernel."""

    checker_id: str
    supported_chart_types: tuple[str, ...]
    event_obligation_ids: tuple[str, ...]
    branch_union_obligation_ids: tuple[str, ...]
    chart_chain_obligation_ids: tuple[str, ...]
    transition_obligation_ids: tuple[str, ...]
    proof_grade_arithmetic_backend_certificate: (
        ProofGradeArithmeticBackendCertificate | None
    ) = None
    witness_source: str = "certificate_checker_kernel_support"

    @property
    def proof_grade_arithmetic_backend_sound(self) -> bool:
        return bool(
            self.proof_grade_arithmetic_backend_certificate is not None
            and self.proof_grade_arithmetic_backend_certificate.proof_certified
        )

    @property
    def ordinary_taylor_sound(self) -> bool:
        return "ordinary_taylor" in self.supported_chart_types

    @property
    def levi_civita_sound(self) -> bool:
        return "planar_levi_civita_binary" in self.supported_chart_types

    @property
    def spatial_ks_sound(self) -> bool:
        return "spatial_ks_binary" in self.supported_chart_types

    @property
    def fuchsian_stop_sound(self) -> bool:
        return "total_collision_fuchsian_stop" in self.supported_chart_types

    @property
    def generalized_fuchsian_stop_sound(self) -> bool:
        return (
            "total_collision_generalized_fuchsian_stop"
            in self.supported_chart_types
        )

    @property
    def transition_sound(self) -> bool:
        return bool(self.transition_obligation_ids)

    @property
    def branch_union_sound(self) -> bool:
        return bool(self.branch_union_obligation_ids)

    @property
    def chart_chain_sound(self) -> bool:
        return bool(self.chart_chain_obligation_ids)

    @property
    def verifier_kernel_sound(self) -> bool:
        return bool(
            self.checker_id == "independent_chart_verifier_v1"
            and self.ordinary_taylor_sound
            and self.levi_civita_sound
            and self.spatial_ks_sound
            and self.fuchsian_stop_sound
            and self.generalized_fuchsian_stop_sound
            and self.transition_sound
            and self.branch_union_sound
            and self.chart_chain_sound
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.verifier_kernel_sound
            and self.event_obligation_ids
            and self.proof_grade_arithmetic_backend_sound
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("ordinary_taylor_sound", self.ordinary_taylor_sound),
            ("levi_civita_sound", self.levi_civita_sound),
            ("spatial_ks_sound", self.spatial_ks_sound),
            ("fuchsian_stop_sound", self.fuchsian_stop_sound),
            (
                "generalized_fuchsian_stop_sound",
                self.generalized_fuchsian_stop_sound,
            ),
            ("transition_sound", self.transition_sound),
            ("branch_union_sound", self.branch_union_sound),
            ("chart_chain_sound", self.chart_chain_sound),
            ("event_isolation_sound", bool(self.event_obligation_ids)),
            ("verifier_kernel_sound", self.verifier_kernel_sound),
            (
                "proof_grade_arithmetic_backend_sound",
                self.proof_grade_arithmetic_backend_sound,
            ),
        )
        return tuple(name for name, certified in fields if not certified)


_PROOF_GRADE_CHART_OBLIGATIONS: dict[str, tuple[str, ...]] = {
    "ordinary_taylor": (
        "ordinary_taylor_exact_rational_residual_polynomials",
        "interval_taylor_model_newton_residual",
    ),
    "planar_levi_civita_binary": (
        "interval_physical_time_containment",
        "planar_lc_pair_energy_constraint",
        "planar_lc_exact_rational_regularized_residual_polynomials",
        "interval_planar_lc_regularized_rhs_residual",
        "interval_projected_newton_residual_away_from_binary_collision",
    ),
    "spatial_ks_binary": (
        "interval_physical_time_containment",
        "spatial_ks_pair_energy_and_horizontal_constraints",
        "spatial_ks_exact_rational_regularized_residual_polynomials",
        "interval_spatial_ks_regularized_rhs_residual",
        "interval_spatial_ks_projected_newton_residual_away_from_binary_collision",
    ),
    "total_collision_fuchsian_stop": (
        "interval_fuchsian_lifted_residual_on_punctured_shells",
        "interval_fuchsian_projected_residual_on_punctured_shells",
        "interval_zero_angular_momentum_on_punctured_shells",
        "interval_center_of_mass_and_linear_momentum_on_punctured_shells",
        "interval_fuchsian_supported_scale_finite_energy_matching",
        "interval_total_collision_endpoint_collapse_envelope",
        "fuchsian_primitive_cauchy_inputs_certify",
        "fuchsian_primitive_cauchy_shell_inside_isolation",
        "fuchsian_tail_bound_covers_primitive_cauchy_tail",
        "fuchsian_primitive_cauchy_residual_tail_within_tolerance",
    ),
    "total_collision_generalized_fuchsian_stop": (
        "generalized_fuchsian_remainder_majorant_certifies",
        "generalized_fuchsian_remainder_component_inputs_certify",
        "generalized_fuchsian_remainder_shell_inside_isolation",
        "generalized_fuchsian_tail_bound_covers_remainder_tail",
        "interval_generalized_fuchsian_lifted_residual_on_punctured_shells",
        "generalized_fuchsian_remainder_residual_tail_within_tolerance",
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells",
        "interval_generalized_zero_angular_momentum_on_punctured_shells",
        "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells",
        "generalized_fuchsian_endpoint_collapse_envelope",
    ),
}

_PROOF_GRADE_EVENT_OBLIGATIONS = (
    "event_exact_rational_interval_arithmetic",
)

_PROOF_GRADE_BRANCH_UNION_OBLIGATIONS = (
    "branch_union_exact_rational_interval_aggregation",
)

_PROOF_GRADE_CHART_CHAIN_OBLIGATIONS = (
    "chart_chain_exact_rational_time_coverage",
)

_PROOF_GRADE_TRANSITION_OBLIGATIONS = (
    "transition_exact_rational_state_continuity",
)


def certify_rational_interval_arithmetic_backend_soundness(
    *,
    witness_source: str = "rational_interval_arithmetic_backend",
) -> ProofGradeArithmeticBackendCertificate:
    """Check the exact rational interval operations used by the verifier."""

    third = Fraction(1, 3)
    half = Fraction(1, 2)
    two_fifths = Fraction(2, 5)
    three_fifths = Fraction(3, 5)
    first = RationalInterval(third, half)
    second = RationalInterval(two_fifths, three_fifths)
    exact_fraction_endpoints = bool(
        isinstance(first.lower, Fraction)
        and isinstance(first.upper, Fraction)
        and first.as_tuple() == (third, half)
    )
    rational_interval_operations_checked = bool(
        (first + second).as_tuple() == (Fraction(11, 15), Fraction(11, 10))
        and (first - second).as_tuple() == (Fraction(-4, 15), Fraction(1, 10))
        and (first * second).as_tuple() == (Fraction(2, 15), Fraction(3, 10))
        and second.reciprocal().as_tuple() == (Fraction(5, 3), Fraction(5, 2))
    )
    variable = RationalInterval(third, half)
    polynomial = (
        RationalInterval.point(1),
        RationalInterval.point(2),
        RationalInterval.point(1),
    )
    value = rational_interval_polynomial_eval(polynomial, variable)
    rational_interval_polynomial_eval_checked = bool(
        value.as_tuple() == (Fraction(16, 9), Fraction(9, 4))
        and rational_interval_polyder(polynomial)[0].as_tuple()
        == (Fraction(2, 1), Fraction(2, 1))
    )
    rational_interval_sign_trichotomy_checked = bool(
        rational_interval_sign(RationalInterval(Fraction(1, 5), Fraction(2, 5)))
        == 1
        and rational_interval_sign(
            RationalInterval(Fraction(-2, 5), Fraction(-1, 5))
        )
        == -1
        and rational_interval_sign(RationalInterval(Fraction(-1, 5), Fraction(1, 5)))
        == 0
    )
    embedded = RationalInterval.from_float_interval(0.5, 0.75)
    finite_float_to_fraction_embedding_checked = bool(
        embedded.as_tuple() == (Fraction(1, 2), Fraction(3, 4))
    )
    return ProofGradeArithmeticBackendCertificate(
        backend_id="python_fraction_rational_interval_backend",
        exact_fraction_endpoints=exact_fraction_endpoints,
        rational_interval_operations_checked=rational_interval_operations_checked,
        rational_interval_polynomial_eval_checked=(
            rational_interval_polynomial_eval_checked
        ),
        rational_interval_sign_trichotomy_checked=(
            rational_interval_sign_trichotomy_checked
        ),
        finite_float_to_fraction_embedding_checked=(
            finite_float_to_fraction_embedding_checked
        ),
        statement=(
            "The checker backend represents interval endpoints as exact "
            "Fractions, performs interval arithmetic by exact endpoint "
            "operations, evaluates power-basis polynomials by Horner interval "
            "arithmetic, and uses exact rational sign trichotomy for checker "
            "obligations that claim proof-grade rational arithmetic."
        ),
        proof_sketch=(
            "RationalInterval stores Fraction endpoints and all arithmetic "
            "constructors combine only endpoint Fractions by exact field "
            "operations.  Addition, subtraction, multiplication, reciprocal, "
            "polynomial Horner evaluation, derivative coefficient scaling, and "
            "sign trichotomy are checked here on rational fixtures whose exact "
            "results are known.  Finite binary floats are embedded by "
            "Fraction.from_float, preserving their exact binary-rational value "
            "before the checker performs interval operations."
        ),
        witness_source=str(witness_source),
    )


def certify_certificate_checker_kernel_support(
    *,
    proof_grade_arithmetic_backend_certificate: (
        ProofGradeArithmeticBackendCertificate | None
    ) = None,
    proof_grade_arithmetic_backend_sound: bool = False,
    witness_source: str = "certificate_checker_kernel_support",
) -> CertificateCheckerKernelSupportCertificate:
    """Derive the supported checker-kernel language from the checker tables."""

    if proof_grade_arithmetic_backend_sound and (
        proof_grade_arithmetic_backend_certificate is None
    ):
        raise TypeError(
            "proof_grade_arithmetic_backend_sound=True requires a "
            "ProofGradeArithmeticBackendCertificate"
        )

    return CertificateCheckerKernelSupportCertificate(
        checker_id="independent_chart_verifier_v1",
        supported_chart_types=tuple(sorted(_PROOF_GRADE_CHART_OBLIGATIONS)),
        event_obligation_ids=_PROOF_GRADE_EVENT_OBLIGATIONS,
        branch_union_obligation_ids=_PROOF_GRADE_BRANCH_UNION_OBLIGATIONS,
        chart_chain_obligation_ids=_PROOF_GRADE_CHART_CHAIN_OBLIGATIONS,
        transition_obligation_ids=_PROOF_GRADE_TRANSITION_OBLIGATIONS,
        proof_grade_arithmetic_backend_certificate=(
            proof_grade_arithmetic_backend_certificate
        ),
        witness_source=str(witness_source),
    )


@dataclass(frozen=True)
class IndependentChartVerifierCertificate:
    """Aggregate verifier result for a collection of serialized charts."""

    checker_id: str
    chart_results: tuple[CertificateCheckResult, ...]
    transition_results: tuple[TransitionCheckResult, ...] = ()
    event_results: tuple[EventIsolationCheckResult, ...] = ()
    branch_union_results: tuple[BranchUnionCheckResult, ...] = ()
    chart_chain_results: tuple[ChartChainCheckResult, ...] = ()

    @property
    def checked_certificate_count(self) -> int:
        return len(self.chart_results)

    @property
    def ordinary_taylor_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "ordinary_taylor"
        )

    @property
    def planar_levi_civita_binary_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "planar_levi_civita_binary"
        )

    @property
    def spatial_ks_binary_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "spatial_ks_binary"
        )

    @property
    def total_collision_fuchsian_stop_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "total_collision_fuchsian_stop"
        )

    @property
    def total_collision_generalized_fuchsian_stop_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "total_collision_generalized_fuchsian_stop"
        )

    @property
    def checked_event_count(self) -> int:
        return len(self.event_results)

    @property
    def checked_branch_union_count(self) -> int:
        return len(self.branch_union_results)

    @property
    def checked_chart_chain_count(self) -> int:
        return len(self.chart_chain_results)

    @property
    def certified(self) -> bool:
        return bool(
            self.chart_results
            and all(result.certified for result in self.chart_results)
            and all(result.certified for result in self.transition_results)
            and all(result.certified for result in self.event_results)
            and all(result.certified for result in self.branch_union_results)
            and all(result.certified for result in self.chart_chain_results)
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def proof_grade_arithmetic_obligation_ids(self) -> tuple[str, ...]:
        obligation_ids: list[str] = []
        for result in self.chart_results:
            required = _PROOF_GRADE_CHART_OBLIGATIONS.get(result.certificate_type, ())
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation for obligation in required if obligation in certified
            )
        for result in self.event_results:
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_EVENT_OBLIGATIONS
                if obligation in certified
            )
        for result in self.branch_union_results:
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_BRANCH_UNION_OBLIGATIONS
                if obligation in certified
            )
        for result in self.chart_chain_results:
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_CHART_CHAIN_OBLIGATIONS
                if obligation in certified
            )
        for result in self.transition_results:
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_TRANSITION_OBLIGATIONS
                if obligation in certified
            )
        return tuple(dict.fromkeys(obligation_ids))

    @property
    def proof_grade_arithmetic_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.chart_results and not self.event_results:
            blockers.append("proof_grade_arithmetic_no_checked_chart_or_event")
        for result in self.chart_results:
            required = _PROOF_GRADE_CHART_OBLIGATIONS.get(result.certificate_type)
            if required is None:
                blockers.append(
                    f"{result.certificate_id}:proof_grade_arithmetic_unknown_chart_type"
                )
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in required:
                if obligation not in certified:
                    blockers.append(f"{result.certificate_id}:{obligation}")
        for result in self.event_results:
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_EVENT_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.event_id}:{obligation}")
        for result in self.transition_results:
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_TRANSITION_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.transition_id}:{obligation}")
        for result in self.branch_union_results:
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_BRANCH_UNION_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.union_id}:{obligation}")
        for result in self.chart_chain_results:
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_CHART_CHAIN_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.chain_id}:{obligation}")
        return tuple(dict.fromkeys(blockers))

    @property
    def proof_grade_arithmetic_checked_bundle_certified(self) -> bool:
        return bool(self.certified and not self.proof_grade_arithmetic_blockers)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        for result in self.chart_results:
            missing.extend(result.missing_obligations)
        for result in self.transition_results:
            missing.extend(result.missing_obligations)
        for result in self.event_results:
            missing.extend(result.missing_obligations)
        for result in self.branch_union_results:
            missing.extend(result.missing_obligations)
        for result in self.chart_chain_results:
            missing.extend(result.missing_obligations)
        return tuple(dict.fromkeys(missing))


def _certified_obligation_ids(
    obligations: Iterable[CertificateCheckObligation],
) -> set[str]:
    return {
        obligation.obligation
        for obligation in obligations
        if obligation.certified
    }


def check_ordinary_taylor_chart(
    certificate: OrdinaryTaylorChartCertificate,
) -> CertificateCheckResult:
    """Verify an ordinary Taylor chart from explicit serialized coefficients."""

    q = _coefficient_array(certificate.position_coefficients)
    v = _coefficient_array(certificate.velocity_coefficients)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    residual_tolerance = float(certificate.residual_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    unit_speed_tolerance = (
        coefficient_tolerance
        if np.isfinite(coefficient_tolerance) and coefficient_tolerance >= 0.0
        else 1.0e-14
    )

    shape_ok = bool(
        q.ndim == 3
        and v.shape == q.shape
        and q.shape[0] >= 2
        and q.shape[1] == 3
        and q.shape[2] in (2, 3)
        and masses.shape == (3,)
    )
    finite_arrays = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(v)))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    unit_physical_parameter_speed = bool(
        finite_intervals
        and _ordinary_unit_physical_parameter_speed(
            parameter_interval,
            physical_time_interval,
            tolerance=max(unit_speed_tolerance, 1.0e-14),
        )
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    initial_noncollision = bool(shape_ok and _initial_noncollision(q[0]))

    max_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and initial_noncollision
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_coefficient_recurrence_residual(
                q,
                v,
                masses,
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf

    max_sampled_newton_residual = np.inf
    sampled_residual_certified = False
    max_interval_newton_residual = np.inf
    interval_residual_certified = False
    exact_rational_interval_residual_certified = False
    if (
        recurrence_certified
        and finite_intervals
        and unit_physical_parameter_speed
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            max_sampled_newton_residual = _max_sampled_newton_residual(
                q,
                v,
                masses,
                parameter_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                max_sampled_newton_residual <= residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_sampled_newton_residual = np.inf
    if (
        recurrence_certified
        and finite_intervals
        and unit_physical_parameter_speed
        and finite_tolerances
    ):
        try:
            max_interval_newton_residual = _max_interval_ordinary_taylor_model_residual(
                q,
                v,
                masses,
                parameter_interval,
                tail_bound=tail_bound,
            )
            exact_rational_interval_residual_certified = True
            interval_residual_certified = bool(
                max_interval_newton_residual <= residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_newton_residual = np.inf
            exact_rational_interval_residual_certified = False

    obligations = (
        CertificateCheckObligation(
            "ordinary_chart_type",
            certificate.chart_type == "ordinary_taylor",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "coefficient_array_shape",
            shape_ok,
            f"position_shape={q.shape}; velocity_shape={v.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all position and velocity coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "ordinary_physical_parameter_unit_speed",
            unit_physical_parameter_speed,
            (
                f"parameter_width={parameter_interval[1] - parameter_interval[0]}; "
                f"physical_width={physical_time_interval[1] - physical_time_interval[0]}"
            ),
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            f"coefficient_tolerance={coefficient_tolerance}; residual_tolerance={residual_tolerance}",
        ),
        CertificateCheckObligation(
            "initial_noncollision",
            initial_noncollision,
            "all initial pair distances are positive",
        ),
        CertificateCheckObligation(
            "ordinary_taylor_coefficient_recurrence",
            recurrence_certified,
            f"max_coefficient_residual={max_coefficient_residual}",
        ),
        CertificateCheckObligation(
            "ordinary_taylor_exact_rational_residual_polynomials",
            exact_rational_interval_residual_certified,
            (
                "q'-v and v'-a(q) residual polynomials are evaluated with "
                "exact rational interval arithmetic over serialized "
                "binary-float coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_taylor_model_newton_residual",
            interval_residual_certified,
            (
                "max_interval_taylor_model_newton_residual="
                f"{max_interval_newton_residual}; "
                f"diagnostic_max_sampled_newton_residual="
                f"{max_sampled_newton_residual}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="ordinary_taylor",
        checker_id="independent_ordinary_taylor_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_sampled_newton_residual),
    )


def check_planar_levi_civita_binary_chart(
    certificate: PlanarLeviCivitaBinaryChartCertificate,
) -> CertificateCheckResult:
    """Verify a planar LC binary chart from explicit serialized coefficients."""

    solution = _regularized_binary_solution_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    regularized_residual_tolerance = float(certificate.regularized_residual_tolerance)
    projected_residual_tolerance = float(certificate.projected_residual_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    rho_lower_bound = float(certificate.projection_rho_lower_bound)

    arrays = (
        solution.z,
        solution.z_velocity,
        solution.pair_energy,
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time,
    )
    vector_shape = solution.z.shape
    scalar_shape = solution.pair_energy.shape
    shape_ok = bool(
        solution.z.ndim == 2
        and vector_shape[1:] == (2,)
        and solution.z_velocity.shape == vector_shape
        and solution.binary_center.shape == vector_shape
        and solution.binary_center_velocity.shape == vector_shape
        and solution.third_offset.shape == vector_shape
        and solution.third_offset_velocity.shape == vector_shape
        and vector_shape[0] >= 2
        and solution.pair_energy.shape == (vector_shape[0],)
        and solution.physical_time.shape == (vector_shape[0],)
        and scalar_shape == (vector_shape[0],)
    )
    finite_arrays = bool(all(np.all(np.isfinite(array)) for array in arrays))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    pair_valid = bool(
        len(solution.pair) == 2
        and solution.pair[0] != solution.pair[1]
        and set(solution.pair).issubset({0, 1, 2})
    )
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(regularized_residual_tolerance)
        and regularized_residual_tolerance >= 0.0
        and np.isfinite(projected_residual_tolerance)
        and projected_residual_tolerance >= 0.0
        and np.isfinite(rho_lower_bound)
        and rho_lower_bound >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    (
        physical_time_contained,
        physical_time_enclosure,
    ) = _interval_physical_time_contained(
        solution.physical_time,
        parameter_interval,
        physical_time_interval,
        tolerance=tail_bound,
    )
    sampled_physical_time_contained = bool(
        finite_intervals
        and shape_ok
        and _sampled_physical_time_contained(
            solution.physical_time,
            parameter_interval,
            physical_time_interval,
            sample_count=max(sample_count, 2),
        )
    )

    max_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and pair_valid
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_regularized_binary_coefficient_residual(
                solution,
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf

    max_constraint_residual = np.inf
    pair_energy_constraint_certified = False
    if recurrence_certified:
        try:
            max_constraint_residual = float(
                np.max(
                    np.abs(
                        lc_pair_energy_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            pair_energy_constraint_certified = bool(
                max_constraint_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_constraint_residual = np.inf

    max_regularized_residual = np.inf
    regularized_residual_certified = False
    max_interval_regularized_residual = np.inf
    interval_regularized_residual_certified = False
    exact_rational_regularized_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            max_regularized_residual = _max_sampled_regularized_binary_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
            )
            regularized_residual_certified = bool(
                max_regularized_residual <= regularized_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_regularized_residual = np.inf
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_regularized_residual = (
                _max_interval_planar_lc_taylor_model_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                )
            )
            interval_regularized_residual_certified = bool(
                max_interval_regularized_residual <= regularized_residual_tolerance
            )
            exact_rational_regularized_residual_certified = True
        except (FloatingPointError, ValueError):
            max_interval_regularized_residual = np.inf
            exact_rational_regularized_residual_certified = False

    max_projected_residual = np.inf
    projected_sample_count = 0
    projected_residual_certified = False
    max_interval_projected_residual = np.inf
    interval_projected_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            (
                max_projected_residual,
                projected_sample_count,
            ) = _max_sampled_projected_binary_newton_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
                rho_lower_bound=rho_lower_bound,
            )
            projected_residual_certified = bool(
                projected_sample_count > 0
                and max_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_projected_residual = np.inf
            projected_sample_count = 0
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_projected_residual = (
                _max_interval_planar_lc_projected_newton_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                    rho_lower_bound=rho_lower_bound,
                )
            )
            interval_projected_residual_certified = bool(
                max_interval_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_projected_residual = np.inf

    obligations = (
        CertificateCheckObligation(
            "planar_levi_civita_binary_chart_type",
            certificate.chart_type == "planar_levi_civita_binary",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_coefficient_array_shape",
            shape_ok,
            (
                f"z_shape={solution.z.shape}; z_velocity_shape={solution.z_velocity.shape}; "
                f"pair_energy_shape={solution.pair_energy.shape}; physical_time_shape={solution.physical_time.shape}"
            ),
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all regularized binary coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "binary_pair_valid",
            pair_valid,
            f"pair={solution.pair!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"coefficient_tolerance={coefficient_tolerance}; "
                f"regularized_residual_tolerance={regularized_residual_tolerance}; "
                f"projected_residual_tolerance={projected_residual_tolerance}; "
                f"projection_rho_lower_bound={rho_lower_bound}"
            ),
        ),
        CertificateCheckObligation(
            "interval_physical_time_containment",
            physical_time_contained,
            (
                f"physical_time_interval={physical_time_interval}; "
                f"interval_physical_time_enclosure={physical_time_enclosure}; "
                "diagnostic_sampled_physical_time_contained="
                f"{sampled_physical_time_contained}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_regularized_coefficient_recurrence",
            recurrence_certified,
            f"max_coefficient_residual={max_coefficient_residual}",
        ),
        CertificateCheckObligation(
            "planar_lc_pair_energy_constraint",
            pair_energy_constraint_certified,
            f"max_constraint_residual={max_constraint_residual}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_rational_regularized_residual_polynomials",
            exact_rational_regularized_residual_certified,
            (
                "regularized LC residual polynomials are evaluated with exact "
                "rational interval arithmetic over serialized binary-float "
                "coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_planar_lc_regularized_rhs_residual",
            interval_regularized_residual_certified,
            (
                "max_interval_regularized_residual="
                f"{max_interval_regularized_residual}; "
                f"diagnostic_max_sampled_regularized_residual="
                f"{max_regularized_residual}"
            ),
        ),
        CertificateCheckObligation(
            "interval_projected_newton_residual_away_from_binary_collision",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_projected_sample_count={projected_sample_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="planar_levi_civita_binary",
        checker_id="independent_planar_lc_binary_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_projected_residual),
    )


def check_spatial_ks_binary_chart(
    certificate: SpatialKSBinaryChartCertificate,
) -> CertificateCheckResult:
    """Verify a spatial KS binary chart from explicit serialized coefficients."""

    solution = _spatial_ks_solution_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    regularized_residual_tolerance = float(certificate.regularized_residual_tolerance)
    projected_residual_tolerance = float(certificate.projected_residual_tolerance)
    constraint_tolerance = float(certificate.constraint_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    rho_lower_bound = float(certificate.projection_rho_lower_bound)

    arrays = (
        solution.u,
        solution.u_velocity,
        solution.pair_energy,
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time,
    )
    shape_ok = bool(
        solution.u.ndim == 2
        and solution.u.shape[1:] == (4,)
        and solution.u.shape[0] >= 2
        and solution.u_velocity.shape == solution.u.shape
        and solution.binary_center.shape == (solution.u.shape[0], 3)
        and solution.binary_center_velocity.shape == solution.binary_center.shape
        and solution.third_offset.shape == solution.binary_center.shape
        and solution.third_offset_velocity.shape == solution.binary_center.shape
        and solution.pair_energy.shape == (solution.u.shape[0],)
        and solution.physical_time.shape == (solution.u.shape[0],)
    )
    finite_arrays = bool(all(np.all(np.isfinite(array)) for array in arrays))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    pair_valid = bool(
        len(solution.pair) == 2
        and solution.pair[0] != solution.pair[1]
        and set(solution.pair).issubset({0, 1, 2})
    )
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(regularized_residual_tolerance)
        and regularized_residual_tolerance >= 0.0
        and np.isfinite(projected_residual_tolerance)
        and projected_residual_tolerance >= 0.0
        and np.isfinite(constraint_tolerance)
        and constraint_tolerance >= 0.0
        and np.isfinite(rho_lower_bound)
        and rho_lower_bound >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    (
        physical_time_contained,
        physical_time_enclosure,
    ) = _interval_physical_time_contained(
        solution.physical_time,
        parameter_interval,
        physical_time_interval,
        tolerance=tail_bound,
    )
    sampled_physical_time_contained = bool(
        finite_intervals
        and shape_ok
        and _sampled_physical_time_contained(
            solution.physical_time,
            parameter_interval,
            physical_time_interval,
            sample_count=max(sample_count, 2),
        )
    )

    max_coefficient_residual = np.inf
    max_relative_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and pair_valid
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_spatial_ks_coefficient_residual(
                solution,
            )
            max_relative_coefficient_residual = (
                _max_spatial_ks_relative_coefficient_residual(solution)
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
                or max_relative_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf
            max_relative_coefficient_residual = np.inf

    max_pair_energy_constraint = np.inf
    max_horizontal_constraint = np.inf
    max_interval_pair_energy_constraint = np.inf
    max_interval_horizontal_constraint = np.inf
    constraints_certified = False
    if recurrence_certified:
        try:
            max_pair_energy_constraint = float(
                np.max(
                    np.abs(
                        ks_pair_energy_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            max_horizontal_constraint = float(
                np.max(
                    np.abs(
                        ks_horizontal_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            if finite_intervals:
                (
                    max_interval_pair_energy_constraint,
                    max_interval_horizontal_constraint,
                ) = _max_interval_spatial_ks_constraint_residuals(
                    solution,
                    parameter_interval,
                )
            constraints_certified = bool(
                (
                    max_pair_energy_constraint <= constraint_tolerance
                    and max_horizontal_constraint <= constraint_tolerance
                )
                or (
                    max_interval_pair_energy_constraint <= constraint_tolerance
                    and max_interval_horizontal_constraint <= constraint_tolerance
                )
            )
        except (FloatingPointError, ValueError):
            max_pair_energy_constraint = np.inf
            max_horizontal_constraint = np.inf
            max_interval_pair_energy_constraint = np.inf
            max_interval_horizontal_constraint = np.inf

    max_regularized_residual = np.inf
    regularized_residual_certified = False
    max_interval_regularized_residual = np.inf
    interval_regularized_residual_certified = False
    exact_rational_regularized_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            max_regularized_residual = _max_sampled_spatial_ks_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
            )
            regularized_residual_certified = bool(
                max_regularized_residual <= regularized_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_regularized_residual = np.inf
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_regularized_residual = (
                _max_interval_spatial_ks_taylor_model_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                )
            )
            interval_regularized_residual_certified = bool(
                max_interval_regularized_residual <= regularized_residual_tolerance
            )
            exact_rational_regularized_residual_certified = True
        except (FloatingPointError, ValueError):
            max_interval_regularized_residual = np.inf
            exact_rational_regularized_residual_certified = False

    max_projected_residual = np.inf
    projected_sample_count = 0
    projected_residual_certified = False
    max_interval_projected_residual = np.inf
    interval_projected_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            (
                max_projected_residual,
                projected_sample_count,
            ) = _max_sampled_spatial_ks_projected_newton_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
                rho_lower_bound=rho_lower_bound,
            )
            projected_residual_certified = bool(
                projected_sample_count > 0
                and max_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_projected_residual = np.inf
            projected_sample_count = 0
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_projected_residual = (
                _max_interval_spatial_ks_projected_newton_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                    rho_lower_bound=rho_lower_bound,
                )
            )
            interval_projected_residual_certified = bool(
                max_interval_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_projected_residual = np.inf

    obligations = (
        CertificateCheckObligation(
            "spatial_ks_binary_chart_type",
            certificate.chart_type == "spatial_ks_binary",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "spatial_ks_coefficient_array_shape",
            shape_ok,
            (
                f"u_shape={solution.u.shape}; u_velocity_shape={solution.u_velocity.shape}; "
                f"binary_center_shape={solution.binary_center.shape}; physical_time_shape={solution.physical_time.shape}"
            ),
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all spatial KS binary coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "binary_pair_valid",
            pair_valid,
            f"pair={solution.pair!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"coefficient_tolerance={coefficient_tolerance}; "
                f"regularized_residual_tolerance={regularized_residual_tolerance}; "
                f"projected_residual_tolerance={projected_residual_tolerance}; "
                f"constraint_tolerance={constraint_tolerance}; "
                f"projection_rho_lower_bound={rho_lower_bound}"
            ),
        ),
        CertificateCheckObligation(
            "interval_physical_time_containment",
            physical_time_contained,
            (
                f"physical_time_interval={physical_time_interval}; "
                f"interval_physical_time_enclosure={physical_time_enclosure}; "
                "diagnostic_sampled_physical_time_contained="
                f"{sampled_physical_time_contained}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_regularized_coefficient_recurrence",
            recurrence_certified,
            (
                f"max_coefficient_residual={max_coefficient_residual}; "
                "max_relative_coefficient_residual="
                f"{max_relative_coefficient_residual}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_pair_energy_and_horizontal_constraints",
            constraints_certified,
            (
                f"max_pair_energy_constraint={max_pair_energy_constraint}; "
                f"max_horizontal_constraint={max_horizontal_constraint}; "
                "max_interval_pair_energy_constraint="
                f"{max_interval_pair_energy_constraint}; "
                "max_interval_horizontal_constraint="
                f"{max_interval_horizontal_constraint}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_exact_rational_regularized_residual_polynomials",
            exact_rational_regularized_residual_certified,
            (
                "regularized KS residual polynomials are evaluated with exact "
                "rational interval arithmetic over serialized binary-float "
                "coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_spatial_ks_regularized_rhs_residual",
            interval_regularized_residual_certified,
            (
                "max_interval_regularized_residual="
                f"{max_interval_regularized_residual}"
            ),
        ),
        CertificateCheckObligation(
            "interval_spatial_ks_projected_newton_residual_away_from_binary_collision",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_projected_sample_count={projected_sample_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="spatial_ks_binary",
        checker_id="independent_spatial_ks_binary_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_projected_residual),
    )


def check_total_collision_fuchsian_stop_chart(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    """Verify a serialized finite Fuchsian-log total-collision stop chart."""

    branch = _fuchsian_log_branch_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    central_shape = np.asarray(certificate.central_shape, dtype=float)
    tau_interval = tuple(float(value) for value in certificate.tau_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    event_time = float(certificate.event_physical_time)
    total_collision_tau = float(certificate.total_collision_tau)
    isolation_radius = float(certificate.isolation_radius)
    residual_tolerance = float(certificate.residual_tolerance)
    angular_tolerance = float(certificate.angular_momentum_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)

    shape_ok = bool(
        central_shape.ndim == 2
        and central_shape.shape[0] == 3
        and central_shape.shape[1] in (2, 3)
        and masses.shape == (3,)
    )
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    finite_terms = _fuchsian_log_terms_finite(certificate.terms, central_shape.shape)
    finite_coefficients = bool(
        shape_ok
        and np.all(np.isfinite(central_shape))
        and np.isfinite(certificate.scale_coefficient)
        and finite_terms
    )
    finite_intervals = bool(
        _finite_nonempty_interval(tau_interval)
        and _finite_nonempty_interval(physical_time_interval)
        and np.isfinite(event_time)
        and np.isfinite(total_collision_tau)
    )
    tau_interval_straddles = bool(
        finite_intervals
        and tau_interval[0] < total_collision_tau < tau_interval[1]
        and abs(total_collision_tau) <= 1.0e-15
    )
    physical_time_matches = bool(
        finite_intervals
        and np.isclose(
            physical_time_interval[0],
            event_time + tau_interval[0] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and np.isclose(
            physical_time_interval[1],
            event_time + tau_interval[1] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and physical_time_interval[0] < event_time < physical_time_interval[1]
    )
    finite_tolerances = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(angular_tolerance)
        and angular_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    primitive_cauchy_inputs_present = certificate.primitive_cauchy_inputs is not None
    primitive_cauchy_inputs_certified = False
    primitive_cauchy_tail_bound = np.inf
    primitive_cauchy_detail = "primitive Cauchy inputs not supplied"
    primitive_cauchy_shell_inside_isolation = False
    primitive_residual_tail_certified = False
    primitive_residual_tail_bound = np.inf
    primitive_residual_tail_detail = "primitive Cauchy inputs not supplied"
    endpoint_collapse_certified = False
    endpoint_collapse_detail = "primitive Cauchy inputs not supplied"
    endpoint_first_shell_position_bound = np.inf
    finite_energy_certified = False
    finite_energy_limit = np.inf
    finite_energy_detail = "finite coefficients not certified"
    if finite_coefficients and positive_masses:
        (
            finite_energy_certified,
            finite_energy_limit,
            finite_energy_detail,
        ) = _check_fuchsian_supported_scale_finite_energy_matching(branch)
    if primitive_cauchy_inputs_present:
        (
            primitive_cauchy_inputs_certified,
            primitive_cauchy_tail_bound,
            primitive_cauchy_detail,
        ) = _check_fuchsian_primitive_cauchy_inputs(
            certificate.primitive_cauchy_inputs,
            branch=branch,
        )
        (
            primitive_residual_tail_certified,
            primitive_residual_tail_bound,
            primitive_residual_tail_detail,
        ) = _check_fuchsian_primitive_cauchy_residual_tail(
            certificate.primitive_cauchy_inputs,
            residual_tolerance=residual_tolerance,
        )
        (
            endpoint_collapse_certified,
            endpoint_first_shell_position_bound,
            endpoint_collapse_detail,
        ) = _check_fuchsian_endpoint_collapse_envelope(
            certificate.primitive_cauchy_inputs,
        )
        primitive_cauchy_shell_inside_isolation = bool(
            np.isfinite(certificate.primitive_cauchy_inputs.initial_radius)
            and np.isfinite(isolation_radius)
            and 0.0 < certificate.primitive_cauchy_inputs.initial_radius
            and certificate.primitive_cauchy_inputs.initial_radius <= isolation_radius
        )
    primitive_cauchy_tail_covered = bool(
        primitive_cauchy_inputs_present
        and primitive_cauchy_inputs_certified
        and tail_admissible
        and np.isfinite(primitive_cauchy_tail_bound)
        and tail_bound + 1.0e-15 >= primitive_cauchy_tail_bound
    )

    central_shape_collision_free = bool(
        shape_ok and _minimum_pair_distance(central_shape) > 0.0
    )
    central_floor_gap = np.inf
    shape_floor_gap = np.inf
    deviation_gap = np.inf
    isolation_certified = False
    if (
        shape_ok
        and finite_coefficients
        and positive_masses
        and central_shape_collision_free
        and np.isfinite(isolation_radius)
        and 0.0 < isolation_radius < 1.0
    ):
        try:
            recomputed_isolation = _fuchsian_log_total_collision_isolation_from_certificate(
                certificate,
            )
            central_floor_gap = abs(
                recomputed_isolation.central_shape_pair_distance_floor
                - float(certificate.central_shape_pair_distance_floor)
            )
            shape_floor_gap = abs(
                recomputed_isolation.shape_pair_distance_floor
                - float(certificate.shape_pair_distance_floor)
            )
            deviation_gap = abs(
                recomputed_isolation.shape_deviation_bound
                - float(certificate.shape_deviation_bound)
            )
            isolation_certified = bool(
                recomputed_isolation.certified
                and central_floor_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.central_shape_pair_distance_floor))
                and shape_floor_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.shape_pair_distance_floor))
                and deviation_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.shape_deviation_bound))
            )
        except (FloatingPointError, ValueError):
            central_floor_gap = np.inf
            shape_floor_gap = np.inf
            deviation_gap = np.inf

    max_lifted_residual = np.inf
    max_projected_residual = np.inf
    max_interval_lifted_residual = np.inf
    max_interval_projected_residual = np.inf
    max_interval_angular_momentum = np.inf
    max_interval_center_of_mass = np.inf
    max_interval_linear_momentum = np.inf
    max_angular_momentum = np.inf
    max_position_scale = np.inf
    sampled_count = 0
    interval_lifted_residual_certified = False
    interval_lifted_residual_detail = "primitive Cauchy inputs not supplied"
    interval_projected_residual_certified = False
    interval_projected_residual_detail = "primitive Cauchy inputs not supplied"
    interval_angular_certified = False
    interval_angular_detail = "primitive Cauchy inputs not supplied"
    interval_com_momentum_certified = False
    interval_com_momentum_detail = "primitive Cauchy inputs not supplied"
    sampled_residual_certified = False
    angular_certified = False
    scaling_certified = False
    if (
        isolation_certified
        and tau_interval_straddles
        and physical_time_matches
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            (
                max_lifted_residual,
                max_projected_residual,
                max_angular_momentum,
                max_position_scale,
                sampled_count,
            ) = _max_sampled_fuchsian_stop_chart_quantities(
                branch,
                tau_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                sampled_count > 0
                and max_lifted_residual <= residual_tolerance
                and max_projected_residual <= residual_tolerance
            )
            angular_certified = bool(
                sampled_count > 0 and max_angular_momentum <= angular_tolerance
            )
            scaling_certified = bool(
                sampled_count > 0
                and np.isfinite(max_position_scale)
                and max_position_scale > 0.0
            )
        except (FloatingPointError, ValueError):
            max_lifted_residual = np.inf
            max_projected_residual = np.inf
            max_angular_momentum = np.inf
            max_position_scale = np.inf
            sampled_count = 0
    if (
        primitive_cauchy_inputs_present
        and isolation_certified
        and tau_interval_straddles
        and finite_tolerances
    ):
        try:
            (
                max_interval_lifted_residual,
                interval_lifted_residual_detail,
            ) = _max_interval_fuchsian_lifted_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_lifted_residual_certified = bool(
                np.isfinite(max_interval_lifted_residual)
                and max_interval_lifted_residual <= residual_tolerance
            )
            (
                max_interval_projected_residual,
                interval_projected_residual_detail,
            ) = _max_interval_fuchsian_projected_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_projected_residual_certified = bool(
                np.isfinite(max_interval_projected_residual)
                and max_interval_projected_residual <= residual_tolerance
            )
            (
                max_interval_angular_momentum,
                interval_angular_detail,
            ) = _max_interval_fuchsian_zero_angular_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_angular_certified = bool(
                np.isfinite(max_interval_angular_momentum)
                and max_interval_angular_momentum <= angular_tolerance
            )
            (
                max_interval_center_of_mass,
                max_interval_linear_momentum,
                interval_com_momentum_detail,
            ) = _max_interval_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_com_momentum_certified = bool(
                np.isfinite(max_interval_center_of_mass)
                and np.isfinite(max_interval_linear_momentum)
                and max_interval_center_of_mass <= angular_tolerance
                and max_interval_linear_momentum <= angular_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_lifted_residual = np.inf
            max_interval_projected_residual = np.inf
            max_interval_angular_momentum = np.inf
            max_interval_center_of_mass = np.inf
            max_interval_linear_momentum = np.inf
            interval_lifted_residual_detail = (
                "interval lifted residual computation failed"
            )
            interval_projected_residual_detail = (
                "interval projected residual computation failed"
            )
            interval_angular_detail = "interval zero-angular computation failed"
            interval_com_momentum_detail = (
                "interval COM/linear-momentum computation failed"
            )

    obligations = [
        CertificateCheckObligation(
            "total_collision_fuchsian_stop_chart_type",
            certificate.chart_type == "total_collision_fuchsian_stop",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "total_collision_fuchsian_shape",
            shape_ok,
            f"central_shape={central_shape.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_coefficients,
            "central shape, scale coefficient, and finite Fuchsian-log terms are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_tau_interval",
            finite_intervals,
            f"tau_interval={tau_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "punctured_tau_interval_straddles_total_collision",
            tau_interval_straddles,
            f"tau_interval={tau_interval}; total_collision_tau={total_collision_tau}",
        ),
        CertificateCheckObligation(
            "fuchsian_physical_time_interval_matches_cubic_time",
            physical_time_matches,
            f"event_physical_time={event_time}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"residual_tolerance={residual_tolerance}; "
                f"angular_momentum_tolerance={angular_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "central_shape_collision_free",
            central_shape_collision_free,
            f"minimum_pair_distance={_minimum_pair_distance(central_shape) if shape_ok else np.inf}",
        ),
        CertificateCheckObligation(
            "punctured_total_collision_isolation_data",
            isolation_certified,
            (
                f"central_floor_gap={central_floor_gap}; "
                f"shape_floor_gap={shape_floor_gap}; deviation_gap={deviation_gap}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_lifted_residual_on_punctured_shells",
            interval_lifted_residual_certified,
            (
                f"max_interval_lifted_residual={max_interval_lifted_residual}; "
                f"{interval_lifted_residual_detail}; "
                f"diagnostic_max_sampled_lifted_residual={max_lifted_residual}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_projected_residual_on_punctured_shells",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"{interval_projected_residual_detail}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_zero_angular_momentum_on_punctured_shells",
            interval_angular_certified,
            (
                f"max_interval_angular_momentum={max_interval_angular_momentum}; "
                f"{interval_angular_detail}; "
                f"diagnostic_max_sampled_angular_momentum={max_angular_momentum}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_center_of_mass_and_linear_momentum_on_punctured_shells",
            interval_com_momentum_certified,
            (
                f"max_interval_center_of_mass={max_interval_center_of_mass}; "
                f"max_interval_linear_momentum={max_interval_linear_momentum}; "
                f"{interval_com_momentum_detail}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_supported_scale_finite_energy_matching",
            finite_energy_certified,
            f"finite_energy_limit={finite_energy_limit}; {finite_energy_detail}",
        ),
        CertificateCheckObligation(
            "interval_total_collision_endpoint_collapse_envelope",
            endpoint_collapse_certified,
            (
                "first_shell_position_bound="
                f"{endpoint_first_shell_position_bound}; "
                f"{endpoint_collapse_detail}; "
                f"diagnostic_max_position_over_tau_squared={max_position_scale}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
        CertificateCheckObligation(
            "fuchsian_primitive_cauchy_inputs_present",
            primitive_cauchy_inputs_present,
            (
                "finite Fuchsian-log stop charts require serialized primitive "
                "Cauchy inputs; sampled residuals alone are not proof-grade"
            ),
        ),
    ]
    if primitive_cauchy_inputs_present:
        obligations.extend(
            (
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_inputs_certify",
                    primitive_cauchy_inputs_certified,
                    primitive_cauchy_detail,
                ),
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_shell_inside_isolation",
                    primitive_cauchy_shell_inside_isolation,
                    (
                        "initial_radius="
                        f"{certificate.primitive_cauchy_inputs.initial_radius}; "
                        f"isolation_radius={isolation_radius}"
                    ),
                ),
                CertificateCheckObligation(
                    "fuchsian_tail_bound_covers_primitive_cauchy_tail",
                    primitive_cauchy_tail_covered,
                    (
                        f"tail_bound={tail_bound}; "
                        f"primitive_tail_bound={primitive_cauchy_tail_bound}"
                    ),
                ),
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_residual_tail_within_tolerance",
                    primitive_residual_tail_certified,
                    (
                        f"residual_tolerance={residual_tolerance}; "
                        f"primitive_residual_tail_bound={primitive_residual_tail_bound}; "
                        f"{primitive_residual_tail_detail}"
                    ),
                ),
            )
        )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="total_collision_fuchsian_stop",
        checker_id="independent_total_collision_fuchsian_stop_checker_interval_v2",
        obligations=tuple(obligations),
        max_coefficient_residual=float(max_interval_lifted_residual),
        max_sampled_newton_residual=float(max_interval_projected_residual),
    )


def check_total_collision_generalized_fuchsian_stop_chart(
    certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    """Verify a serialized generalized Fuchsian total-collision stop chart."""

    branch: FuchsianShapeBranch | None = None
    branch_error = ""
    try:
        branch = _generalized_fuchsian_branch_from_certificate(certificate)
    except (FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        branch_error = str(error)

    masses = np.asarray(certificate.masses, dtype=float)
    central_shape = np.asarray(certificate.central_shape, dtype=float)
    tau_interval = tuple(float(value) for value in certificate.tau_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    event_time = float(certificate.event_physical_time)
    total_collision_tau = float(certificate.total_collision_tau)
    isolation_radius = float(certificate.isolation_radius)
    residual_tolerance = float(certificate.residual_tolerance)
    angular_tolerance = float(certificate.angular_momentum_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)

    shape_ok = bool(
        central_shape.ndim == 2
        and central_shape.shape[0] == 3
        and central_shape.shape[1] in (2, 3)
        and masses.shape == (3,)
    )
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    selected_coefficients_finite = bool(
        shape_ok
        and all(
            len(item.index) == len(certificate.powers)
            and sum(item.index) <= int(certificate.max_total_degree)
            and np.asarray(item.coefficient, dtype=float).shape == central_shape.shape
            and np.all(np.isfinite(np.asarray(item.coefficient, dtype=float)))
            for item in certificate.selected_coefficients
        )
    )
    finite_coefficients = bool(
        shape_ok
        and np.all(np.isfinite(central_shape))
        and all(np.isfinite(float(power)) and float(power) > 0.0 for power in certificate.powers)
        and int(certificate.max_total_degree) >= 0
        and selected_coefficients_finite
    )
    finite_intervals = bool(
        _finite_nonempty_interval(tau_interval)
        and _finite_nonempty_interval(physical_time_interval)
        and np.isfinite(event_time)
        and np.isfinite(total_collision_tau)
    )
    tau_interval_straddles = bool(
        finite_intervals
        and tau_interval[0] < total_collision_tau < tau_interval[1]
        and abs(total_collision_tau) <= 1.0e-15
    )
    physical_time_matches = bool(
        finite_intervals
        and np.isclose(
            physical_time_interval[0],
            event_time + tau_interval[0] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and np.isclose(
            physical_time_interval[1],
            event_time + tau_interval[1] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and physical_time_interval[0] < event_time < physical_time_interval[1]
    )
    finite_tolerances = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(angular_tolerance)
        and angular_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    branch_reconstructs = branch is not None
    recurrence_certified = bool(branch is not None and branch.recurrence_certified)
    finite_energy_certified = bool(
        branch is not None
        and branch.scale_index is not None
        and np.isfinite(branch.finite_energy_limit)
    )
    finite_energy_limit = (
        float(branch.finite_energy_limit) if branch is not None else float("inf")
    )

    central_shape_collision_free = bool(
        shape_ok and _minimum_pair_distance(central_shape) > 0.0
    )
    central_floor_gap = np.inf
    shape_floor_gap = np.inf
    deviation_gap = np.inf
    isolation_certified = False
    if (
        branch is not None
        and shape_ok
        and finite_coefficients
        and positive_masses
        and central_shape_collision_free
        and np.isfinite(isolation_radius)
        and 0.0 < isolation_radius < 1.0
    ):
        try:
            central_floor = _minimum_pair_distance(branch.central_shape)
            deviation = _generalized_fuchsian_shape_deviation_bound_for_checker(
                branch,
                isolation_radius,
            )
            dimension = int(branch.central_shape.shape[1])
            shape_floor = central_floor - 2.0 * np.sqrt(float(dimension)) * deviation
            central_floor_gap = abs(
                central_floor - float(certificate.central_shape_pair_distance_floor)
            )
            shape_floor_gap = abs(
                shape_floor - float(certificate.shape_pair_distance_floor)
            )
            deviation_gap = abs(
                deviation - float(certificate.shape_deviation_bound)
            )
            isolation_certified = bool(
                np.isfinite(central_floor)
                and central_floor > 0.0
                and np.isfinite(deviation)
                and shape_floor > 0.0
                and _finite_close(
                    float(certificate.central_shape_pair_distance_floor),
                    central_floor,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                and _finite_close(
                    float(certificate.shape_pair_distance_floor),
                    shape_floor,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                and _finite_close(
                    float(certificate.shape_deviation_bound),
                    deviation,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            )
        except (FloatingPointError, ValueError):
            central_floor_gap = np.inf
            shape_floor_gap = np.inf
            deviation_gap = np.inf

    majorant_checks = _check_generalized_remainder_majorant(
        certificate.remainder_majorant,
        residual_tolerance=residual_tolerance,
        tail_bound=tail_bound,
    )
    majorant_initial_radius = float(majorant_checks["initial_radius"])
    remainder_shell_inside_isolation = bool(
        np.isfinite(majorant_initial_radius)
        and np.isfinite(isolation_radius)
        and 0.0 < majorant_initial_radius <= isolation_radius
    )

    max_lifted_residual = np.inf
    max_interval_lifted_residual = np.inf
    max_interval_projected_residual = np.inf
    max_interval_angular_momentum = np.inf
    max_interval_center_of_mass = np.inf
    max_interval_linear_momentum = np.inf
    max_angular_momentum = np.inf
    max_position_scale = np.inf
    sampled_count = 0
    sampled_residual_certified = False
    interval_lifted_residual_certified = False
    cauchy_projected_residual_certified = False
    interval_angular_certified = False
    interval_com_momentum_certified = False
    interval_lifted_residual_detail = "remainder majorant not supplied"
    interval_projected_residual_detail = "remainder majorant not supplied"
    interval_angular_detail = "remainder majorant not supplied"
    interval_com_momentum_detail = "remainder majorant not supplied"
    angular_certified = False
    scaling_certified = False
    if (
        branch is not None
        and isolation_certified
        and tau_interval_straddles
        and physical_time_matches
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            (
                max_lifted_residual,
                max_angular_momentum,
                max_position_scale,
                sampled_count,
            ) = _max_sampled_generalized_fuchsian_stop_chart_quantities(
                branch,
                tau_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                sampled_count > 0
                and np.isfinite(max_lifted_residual)
                and max_lifted_residual + float(majorant_checks["lifted_residual_tail_bound"])
                <= residual_tolerance * (1.0 + 1.0e-12)
            )
            angular_certified = bool(
                sampled_count > 0 and max_angular_momentum <= angular_tolerance
            )
            scaling_certified = bool(
                sampled_count > 0
                and np.isfinite(max_position_scale)
                and max_position_scale > 0.0
            )
        except (FloatingPointError, ValueError):
            max_lifted_residual = np.inf
            max_angular_momentum = np.inf
            max_position_scale = np.inf
            sampled_count = 0
    if (
        branch is not None
        and certificate.remainder_majorant is not None
        and isolation_certified
        and tau_interval_straddles
        and finite_tolerances
    ):
        try:
            (
                max_interval_lifted_residual,
                interval_lifted_residual_detail,
            ) = _max_interval_generalized_fuchsian_lifted_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_lifted_residual_certified = bool(
                np.isfinite(max_interval_lifted_residual)
                and max_interval_lifted_residual
                + float(majorant_checks["lifted_residual_tail_bound"])
                <= residual_tolerance * (1.0 + 1.0e-12)
            )
            (
                max_interval_projected_residual,
                interval_projected_residual_detail,
            ) = _max_interval_generalized_fuchsian_projected_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            cauchy_projected_residual_certified = bool(
                majorant_checks["required_residual_components_present"]
                and majorant_checks["physical_residual_tail_certified"]
            )
            (
                max_interval_angular_momentum,
                interval_angular_detail,
            ) = _max_interval_generalized_fuchsian_zero_angular_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_angular_certified = bool(
                np.isfinite(max_interval_angular_momentum)
                and max_interval_angular_momentum <= angular_tolerance
            )
            (
                max_interval_center_of_mass,
                max_interval_linear_momentum,
                interval_com_momentum_detail,
            ) = _max_interval_generalized_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_com_momentum_certified = bool(
                np.isfinite(max_interval_center_of_mass)
                and np.isfinite(max_interval_linear_momentum)
                and max_interval_center_of_mass <= angular_tolerance
                and max_interval_linear_momentum <= angular_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_lifted_residual = np.inf
            max_interval_projected_residual = np.inf
            max_interval_angular_momentum = np.inf
            max_interval_center_of_mass = np.inf
            max_interval_linear_momentum = np.inf
            interval_lifted_residual_detail = (
                "interval generalized lifted residual computation failed"
            )
            interval_projected_residual_detail = (
                "diagnostic interval generalized projected residual computation failed"
            )
            interval_angular_detail = "interval generalized angular computation failed"
            interval_com_momentum_detail = (
                "interval generalized COM/linear-momentum computation failed"
            )

    obligations = [
        CertificateCheckObligation(
            "total_collision_generalized_fuchsian_stop_chart_type",
            certificate.chart_type == "total_collision_generalized_fuchsian_stop",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "total_collision_generalized_fuchsian_shape",
            shape_ok,
            f"central_shape={central_shape.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_coefficients,
            "central shape, powers, and selected generalized coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_tau_interval",
            finite_intervals,
            f"tau_interval={tau_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "punctured_tau_interval_straddles_total_collision",
            tau_interval_straddles,
            f"tau_interval={tau_interval}; total_collision_tau={total_collision_tau}",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_physical_time_interval_matches_cubic_time",
            physical_time_matches,
            f"event_physical_time={event_time}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"residual_tolerance={residual_tolerance}; "
                f"angular_momentum_tolerance={angular_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_branch_reconstructs",
            branch_reconstructs,
            branch_error or "constructor replay accepted serialized selected rows",
        ),
        CertificateCheckObligation(
            "nonresonant_generalized_fuchsian_recurrence",
            recurrence_certified,
            (
                "max_recurrence_residual="
                f"{branch.max_recurrence_residual_norm if branch is not None else np.inf}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_finite_energy_scale_row",
            finite_energy_certified,
            (
                f"scale_index={certificate.scale_index!r}; "
                f"finite_energy_limit={finite_energy_limit}"
            ),
        ),
        CertificateCheckObligation(
            "central_shape_collision_free",
            central_shape_collision_free,
            f"minimum_pair_distance={_minimum_pair_distance(central_shape) if shape_ok else np.inf}",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_punctured_isolation_data",
            isolation_certified,
            (
                f"central_floor_gap={central_floor_gap}; "
                f"shape_floor_gap={shape_floor_gap}; deviation_gap={deviation_gap}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_majorant_present",
            bool(certificate.remainder_majorant is not None),
            "serialized generalized stop chart requires a Banach remainder majorant",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_constants_finite",
            bool(majorant_checks["constants_finite"]),
            str(majorant_checks["constant_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_contraction_factor",
            bool(majorant_checks["contraction_certified"]),
            str(majorant_checks["contraction_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_self_map",
            bool(majorant_checks["self_map_certified"]),
            str(majorant_checks["self_map_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_component_inputs_certify",
            bool(majorant_checks["component_inputs_certified"]),
            str(majorant_checks["component_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_majorant_certifies",
            bool(majorant_checks["certified"]),
            str(majorant_checks["detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_required_residual_components",
            bool(majorant_checks["required_residual_components_present"]),
            str(majorant_checks["required_residual_component_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_shell_inside_isolation",
            remainder_shell_inside_isolation,
            (
                f"initial_radius={majorant_initial_radius}; "
                f"isolation_radius={isolation_radius}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_tail_bound_covers_remainder_tail",
            bool(majorant_checks["tail_covered"]),
            (
                f"tail_bound={tail_bound}; "
                f"remainder_tail_bound={majorant_checks['tail_bound']}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_fuchsian_lifted_residual_on_punctured_shells",
            interval_lifted_residual_certified,
            (
                f"max_interval_lifted_residual={max_interval_lifted_residual}; "
                f"{interval_lifted_residual_detail}; "
                f"diagnostic_max_sampled_lifted_residual={max_lifted_residual}; "
                f"diagnostic_lifted_residual_tail={majorant_checks['lifted_residual_tail_bound']}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_residual_tail_within_tolerance",
            bool(majorant_checks["lifted_residual_tail_certified"]),
            (
                f"residual_tolerance={residual_tolerance}; "
                f"lifted_residual_tail={majorant_checks['lifted_residual_tail_bound']}"
            ),
        ),
        CertificateCheckObligation(
            "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells",
            cauchy_projected_residual_certified,
            (
                f"residual_tolerance={residual_tolerance}; "
                f"physical_residual_tail={majorant_checks['physical_residual_tail_bound']}; "
                f"{majorant_checks['required_residual_component_detail']}; "
                "diagnostic_direct_interval_projected_residual="
                f"{max_interval_projected_residual}; "
                f"{interval_projected_residual_detail}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_zero_angular_momentum_on_punctured_shells",
            interval_angular_certified,
            (
                f"max_interval_angular_momentum={max_interval_angular_momentum}; "
                f"{interval_angular_detail}; "
                f"diagnostic_max_sampled_angular_momentum={max_angular_momentum}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells",
            interval_com_momentum_certified,
            (
                f"max_interval_center_of_mass={max_interval_center_of_mass}; "
                f"max_interval_linear_momentum={max_interval_linear_momentum}; "
                f"{interval_com_momentum_detail}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_endpoint_collapse_envelope",
            bool(majorant_checks["endpoint_collapse_certified"]),
            (
                f"{majorant_checks['endpoint_detail']}; "
                f"diagnostic_max_position_over_tau_squared={max_position_scale}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    ]
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="total_collision_generalized_fuchsian_stop",
        checker_id="independent_total_collision_generalized_fuchsian_stop_checker_interval_cauchy_projected_v3",
        obligations=tuple(obligations),
        max_coefficient_residual=float(
            max(
                max_interval_lifted_residual,
                float(majorant_checks["lifted_residual_tail_bound"]),
                float(majorant_checks["physical_residual_tail_bound"]),
            )
        ),
        max_sampled_newton_residual=float(
            max(
                max_interval_lifted_residual,
                float(majorant_checks["physical_residual_tail_bound"]),
            )
        ),
    )


def check_ordinary_chart_transition(
    transition: OrdinaryChartTransitionCertificate,
    charts: Iterable[OrdinaryTaylorChartCertificate],
) -> TransitionCheckResult:
    """Verify an ordinary chart handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(handoff_time)
    )
    transition_type_supported = bool(
        transition.transition_type == "ordinary_overlap_handoff"
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _physical_time_in_chart(source_chart, handoff_time)
        and _physical_time_in_chart(target_chart, handoff_time)
    )
    chart_types_supported = bool(
        endpoint_charts_present
        and source_chart.chart_type == "ordinary_taylor"
        and target_chart.chart_type == "ordinary_taylor"
    )

    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and chart_types_supported
        and transition_type_supported
        and finite_tolerances
        and handoff_inside
    ):
        try:
            source_parameter = _physical_to_parameter(source_chart, handoff_time)
            target_parameter = _physical_to_parameter(target_chart, handoff_time)
            source_q = _evaluate_coefficients(
                _coefficient_array(source_chart.position_coefficients),
                source_parameter,
            )
            target_q = _evaluate_coefficients(
                _coefficient_array(target_chart.position_coefficients),
                target_parameter,
            )
            source_v = _evaluate_coefficients(
                _coefficient_array(source_chart.velocity_coefficients),
                source_parameter,
            )
            target_v = _evaluate_coefficients(
                _coefficient_array(target_chart.velocity_coefficients),
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=0.0,
            )
        except (FloatingPointError, ValueError):
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            chart_types_supported,
            "ordinary transition checker currently supports ordinary_taylor endpoints",
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "supported transition_type is ordinary_overlap_handoff",
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_ordinary_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_planar_levi_civita_transition(
    transition: PlanarLeviCivitaTransitionCertificate,
    charts: Iterable[
        OrdinaryTaylorChartCertificate | PlanarLeviCivitaBinaryChartCertificate
    ],
) -> TransitionCheckResult:
    """Verify an ordinary/LC binary handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    source_parameter = float(transition.source_parameter)
    target_parameter = float(transition.target_parameter)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)
    physical_time_tolerance = float(transition.physical_time_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(physical_time_tolerance)
        and physical_time_tolerance >= 0.0
        and np.isfinite(handoff_time)
        and np.isfinite(source_parameter)
        and np.isfinite(target_parameter)
    )
    source_is_ordinary = isinstance(source_chart, OrdinaryTaylorChartCertificate)
    target_is_ordinary = isinstance(target_chart, OrdinaryTaylorChartCertificate)
    source_is_lc = isinstance(source_chart, PlanarLeviCivitaBinaryChartCertificate)
    target_is_lc = isinstance(target_chart, PlanarLeviCivitaBinaryChartCertificate)
    ordinary_to_lc = source_is_ordinary and target_is_lc
    lc_to_ordinary = source_is_lc and target_is_ordinary
    endpoint_types_supported = bool(ordinary_to_lc or lc_to_ordinary)
    transition_type_supported = bool(
        (
            ordinary_to_lc
            and transition.transition_type
            in {
                "ordinary_to_binary_event_handoff",
                "ordinary_to_planar_lc_binary_event_handoff",
            }
        )
        or (
            lc_to_ordinary
            and transition.transition_type
            in {
                "binary_to_ordinary_event_handoff",
                "planar_lc_binary_to_ordinary_event_handoff",
            }
        )
    )
    parameters_inside = bool(
        endpoint_charts_present
        and _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _time_in_interval(source_chart.physical_time_interval, handoff_time)
        and _time_in_interval(target_chart.physical_time_interval, handoff_time)
    )

    max_physical_time_gap = np.inf
    physical_time_match_certified = False
    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and endpoint_types_supported
        and transition_type_supported
        and finite_tolerances
        and parameters_inside
        and handoff_inside
    ):
        try:
            source_time = _chart_physical_time_at_parameter(source_chart, source_parameter)
            target_time = _chart_physical_time_at_parameter(target_chart, target_parameter)
            max_physical_time_gap = max(
                abs(source_time - handoff_time),
                abs(target_time - handoff_time),
            )
            physical_time_match_certified = bool(
                max_physical_time_gap <= physical_time_tolerance
            )
            source_q, source_v = _chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            target_q, target_v = _chart_projected_state_at_parameter(
                target_chart,
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                physical_time_match_certified
                and max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=physical_time_tolerance,
            )
        except (FloatingPointError, ValueError):
            max_physical_time_gap = np.inf
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            endpoint_types_supported,
            "LC transition checker supports ordinary_taylor <-> planar_levi_civita_binary",
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "transition type must match ordinary->LC or LC->ordinary direction",
        ),
        CertificateCheckObligation(
            "transition_parameters_inside_charts",
            parameters_inside,
            (
                f"source_parameter={source_parameter}; "
                f"target_parameter={target_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}; "
                f"physical_time_tolerance={physical_time_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_physical_time_match",
            physical_time_match_certified,
            f"max_physical_time_gap={max_physical_time_gap}",
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_planar_lc_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_spatial_ks_transition(
    transition: SpatialKSTransitionCertificate,
    charts: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
    ],
) -> TransitionCheckResult:
    """Verify an ordinary/KS binary handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    source_parameter = float(transition.source_parameter)
    target_parameter = float(transition.target_parameter)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)
    physical_time_tolerance = float(transition.physical_time_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(physical_time_tolerance)
        and physical_time_tolerance >= 0.0
        and np.isfinite(handoff_time)
        and np.isfinite(source_parameter)
        and np.isfinite(target_parameter)
    )
    source_is_ordinary = isinstance(source_chart, OrdinaryTaylorChartCertificate)
    target_is_ordinary = isinstance(target_chart, OrdinaryTaylorChartCertificate)
    source_is_ks = isinstance(source_chart, SpatialKSBinaryChartCertificate)
    target_is_ks = isinstance(target_chart, SpatialKSBinaryChartCertificate)
    ordinary_to_ks = source_is_ordinary and target_is_ks
    ks_to_ordinary = source_is_ks and target_is_ordinary
    ks_to_ks = source_is_ks and target_is_ks
    endpoint_types_supported = bool(ordinary_to_ks or ks_to_ordinary or ks_to_ks)
    transition_type_supported = bool(
        (
            ordinary_to_ks
            and transition.transition_type
            in {
                "spatial_ordinary_to_ks_decreasing_distance_entry",
                "ordinary_to_spatial_ks_binary_event_handoff",
            }
        )
        or (
            ks_to_ordinary
            and transition.transition_type
            in {
                "spatial_ks_to_ordinary_rho_positive_endpoint_projection",
                "spatial_ks_exit_event_to_ordinary_rho_positive_projection",
                "spatial_ks_to_ordinary_safe_handoff",
            }
        )
        or (
            ks_to_ks
            and transition.transition_type
            in {
                "spatial_ks_to_ks_competing_binary_entry",
            }
        )
    )
    parameters_inside = bool(
        endpoint_charts_present
        and _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _time_in_interval(source_chart.physical_time_interval, handoff_time)
        and _time_in_interval(target_chart.physical_time_interval, handoff_time)
    )

    max_physical_time_gap = np.inf
    physical_time_match_certified = False
    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and endpoint_types_supported
        and transition_type_supported
        and finite_tolerances
        and parameters_inside
        and handoff_inside
    ):
        try:
            source_time = _chart_physical_time_at_parameter(source_chart, source_parameter)
            target_time = _chart_physical_time_at_parameter(target_chart, target_parameter)
            max_physical_time_gap = max(
                abs(source_time - handoff_time),
                abs(target_time - handoff_time),
            )
            physical_time_match_certified = bool(
                max_physical_time_gap <= physical_time_tolerance
            )
            source_q, source_v = _chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            target_q, target_v = _chart_projected_state_at_parameter(
                target_chart,
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                physical_time_match_certified
                and max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=physical_time_tolerance,
            )
        except (FloatingPointError, ValueError):
            max_physical_time_gap = np.inf
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            endpoint_types_supported,
            "KS transition checker supports ordinary_taylor <-> spatial_ks_binary and spatial_ks_binary -> spatial_ks_binary",
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "transition type must match ordinary->KS, KS->ordinary, or KS->KS direction",
        ),
        CertificateCheckObligation(
            "transition_parameters_inside_charts",
            parameters_inside,
            (
                f"source_parameter={source_parameter}; "
                f"target_parameter={target_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}; "
                f"physical_time_tolerance={physical_time_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_physical_time_match",
            physical_time_match_certified,
            f"max_physical_time_gap={max_physical_time_gap}",
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_spatial_ks_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_event_isolation(
    certificate: EventIsolationCertificate,
) -> EventIsolationCheckResult:
    """Verify a scalar polynomial event isolation from serialized intervals."""

    pattern = _event_isolation_sign_pattern(certificate.event_type)
    coefficient_intervals = _event_interval_coefficients(
        certificate.coefficient_intervals,
    )
    rational_coefficient_intervals = ()
    search_interval = tuple(float(value) for value in certificate.search_interval)
    root_interval = tuple(float(value) for value in certificate.root_interval)
    root = float(certificate.root)
    event_value = float(certificate.event_value)

    identity_ok = bool(certificate.certificate_id and certificate.event_id)
    event_type_supported = pattern is not None
    coefficient_source_supported = bool(
        event_type_supported
        and certificate.coefficient_source
        in _event_isolation_supported_sources(certificate.event_type)
    )
    finite_event_value = bool(np.isfinite(event_value) and event_value > 0.0)
    pair_valid = _event_isolation_pair_valid(certificate)
    finite_coefficients = bool(
        len(coefficient_intervals) >= 2
        and all(
            np.isfinite(interval.lower)
            and np.isfinite(interval.upper)
            and interval.lower <= interval.upper
            for interval in coefficient_intervals
        )
    )
    finite_intervals = bool(
        _finite_nonempty_interval(search_interval)
        and _finite_nonempty_interval(root_interval)
        and np.isfinite(root)
    )
    root_contained = bool(
        finite_intervals
        and search_interval[0] <= root <= search_interval[1]
        and root_interval[0] <= root <= root_interval[1]
        and search_interval[0] <= root_interval[0] <= root_interval[1] <= search_interval[1]
    )

    left_sign_certified = False
    right_sign_certified = False
    derivative_sign_certified = False
    excludes_earlier_roots = False
    exact_interval_arithmetic_certified = False
    if pattern is not None and finite_coefficients and finite_intervals:
        try:
            rational_coefficient_intervals = _event_rational_interval_coefficients(
                certificate.coefficient_intervals,
            )
            left_value = rational_interval_polynomial_eval(
                rational_coefficient_intervals,
                RationalInterval.from_float_interval(root_interval[0]),
            )
            right_value = rational_interval_polynomial_eval(
                rational_coefficient_intervals,
                RationalInterval.from_float_interval(root_interval[1]),
            )
            derivative_value = rational_interval_polynomial_eval(
                rational_interval_polyder(rational_coefficient_intervals),
                RationalInterval.from_float_interval(*root_interval),
            )
            left_sign_certified = rational_interval_sign(left_value) == pattern["left"]
            right_sign_certified = rational_interval_sign(right_value) == pattern["right"]
            derivative_sign_certified = (
                rational_interval_sign(derivative_value) == pattern["derivative"]
            )
            excludes_earlier_roots = _signed_rational_polynomial_range_certified(
                rational_coefficient_intervals,
                search_interval[0],
                root_interval[0],
                sign=pattern["pre_event"],
            )
            exact_interval_arithmetic_certified = True
        except (FloatingPointError, ValueError):
            left_sign_certified = False
            right_sign_certified = False
            derivative_sign_certified = False
            excludes_earlier_roots = False
            exact_interval_arithmetic_certified = False

    root_interval_width = (
        float(root_interval[1] - root_interval[0])
        if finite_intervals
        else float("inf")
    )
    obligations = (
        CertificateCheckObligation(
            "event_identity_present",
            identity_ok,
            (
                f"certificate_id={certificate.certificate_id!r}; "
                f"event_id={certificate.event_id!r}"
            ),
        ),
        CertificateCheckObligation(
            "event_type_supported",
            event_type_supported,
            f"event_type={certificate.event_type!r}",
        ),
        CertificateCheckObligation(
            "event_coefficient_source_supported",
            coefficient_source_supported,
            f"coefficient_source={certificate.coefficient_source!r}",
        ),
        CertificateCheckObligation(
            "event_value_positive",
            finite_event_value,
            f"event_value={event_value}",
        ),
        CertificateCheckObligation(
            "event_pair_valid",
            pair_valid,
            f"pair={certificate.pair!r}",
        ),
        CertificateCheckObligation(
            "event_coefficients_finite",
            finite_coefficients,
            f"coefficient_count={len(coefficient_intervals)}",
        ),
        CertificateCheckObligation(
            "event_intervals_finite_and_nested",
            root_contained,
            (
                f"search_interval={search_interval}; "
                f"root_interval={root_interval}; root={root}"
            ),
        ),
        CertificateCheckObligation(
            "event_exact_rational_interval_arithmetic",
            exact_interval_arithmetic_certified,
            (
                "polynomial endpoint, derivative, and pre-event sign checks "
                "use exact rational interval arithmetic over the serialized "
                f"binary-float endpoints; coefficient_count={len(rational_coefficient_intervals)}"
            ),
        ),
        CertificateCheckObligation(
            "event_root_endpoint_signs",
            bool(left_sign_certified and right_sign_certified),
            f"expected_pattern={pattern}",
        ),
        CertificateCheckObligation(
            "event_root_derivative_sign",
            derivative_sign_certified,
            f"expected_pattern={pattern}",
        ),
        CertificateCheckObligation(
            "event_excludes_earlier_roots",
            excludes_earlier_roots,
            f"search_interval={search_interval}; root_interval={root_interval}",
        ),
    )
    return EventIsolationCheckResult(
        event_id=str(certificate.event_id),
        event_type=str(certificate.event_type),
        checker_id="independent_polynomial_event_isolation_checker_v2",
        obligations=obligations,
        root_interval_width=root_interval_width,
    )


def check_branch_union(
    certificate: BranchUnionCertificate,
    checked_leaf_results: Iterable[object],
) -> BranchUnionCheckResult:
    """Verify a finite branch union against independently checked leaves."""

    union_type = str(certificate.union_type)
    leaf_ids = tuple(str(value) for value in certificate.leaf_response_certificate_ids)
    leaf_intervals = tuple(
        tuple(float(value) for value in interval)
        for interval in certificate.leaf_target_intervals
    )
    aggregate = tuple(float(value) for value in certificate.aggregate_target_interval)
    checked = tuple(checked_leaf_results)
    certified_leaf_ids = {
        response_id
        for response_id in (
            _checked_response_identifier(result)
            for result in checked
            if getattr(result, "certified", False)
        )
        if response_id
    }
    supported_union_type = union_type in {
        "finite_time_branch_union",
        "finite_time_event_order_branch_union",
        "finite_time_stratified_branch_union",
    }
    leaf_kinds = tuple(str(value) for value in certificate.leaf_kinds)
    stratified_union = union_type == "finite_time_stratified_branch_union"
    leaf_ids_present = bool(leaf_ids)
    leaf_ids_unique = len(set(leaf_ids)) == len(leaf_ids)
    leaf_intervals_match = len(leaf_intervals) == len(leaf_ids)
    leaf_kinds_match = bool(
        not leaf_kinds
        or len(leaf_kinds) == len(leaf_ids)
    )
    stratified_leaf_kinds_present = bool(not stratified_union or leaf_kinds)
    stratified_leaf_kinds_supported = bool(
        not stratified_union
        or (
            leaf_kinds_match
            and all(kind in SUPPORTED_STRATIFIED_LEAF_KINDS for kind in leaf_kinds)
        )
    )
    stratified_unsupported_absent = bool(
        not stratified_union
        or all(kind != "unsupported_analytic_stratum" for kind in leaf_kinds)
    )
    aggregate_finite = _finite_nonempty_interval(aggregate)
    leaf_intervals_finite = bool(
        leaf_intervals
        and all(_finite_nonempty_interval(interval) for interval in leaf_intervals)
    )
    leaf_responses_checked = bool(
        leaf_ids_present
        and all(leaf_id in certified_leaf_ids for leaf_id in leaf_ids)
    )
    aggregate_contains_leaves = bool(
        aggregate_finite
        and leaf_intervals_finite
        and all(_interval_contains_interval(aggregate, interval) for interval in leaf_intervals)
    )
    (
        exact_rational_interval_aggregation,
        exact_rational_interval_aggregation_detail,
    ) = _exact_rational_branch_union_interval_aggregation(
        aggregate,
        leaf_intervals,
    )
    obligations = (
        CertificateCheckObligation(
            "branch_union_type_supported",
            supported_union_type,
            f"union_type={union_type}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_ids_present",
            leaf_ids_present,
            f"leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_ids_unique",
            leaf_ids_unique,
            f"unique_leaf_count={len(set(leaf_ids))}; leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_interval_count_matches",
            leaf_intervals_match,
            (
                f"leaf_interval_count={len(leaf_intervals)}; "
                f"leaf_count={len(leaf_ids)}"
            ),
        ),
        CertificateCheckObligation(
            "branch_union_leaf_kinds_match",
            leaf_kinds_match,
            f"leaf_kind_count={len(leaf_kinds)}; leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "stratified_branch_union_leaf_kinds_present",
            stratified_leaf_kinds_present,
            f"union_type={union_type}; leaf_kinds={leaf_kinds}",
        ),
        CertificateCheckObligation(
            "stratified_branch_union_leaf_kinds_supported",
            stratified_leaf_kinds_supported,
            (
                f"leaf_kinds={leaf_kinds}; "
                f"supported={tuple(sorted(SUPPORTED_STRATIFIED_LEAF_KINDS))}"
            ),
        ),
        CertificateCheckObligation(
            "stratified_branch_union_unsupported_strata_absent",
            stratified_unsupported_absent,
            f"leaf_kinds={leaf_kinds}",
        ),
        CertificateCheckObligation(
            "branch_union_intervals_finite",
            bool(aggregate_finite and leaf_intervals_finite),
            f"aggregate={aggregate}; leaf_intervals={leaf_intervals}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_responses_independently_checked",
            leaf_responses_checked,
            (
                f"leaf_ids={leaf_ids}; "
                f"certified_leaf_response_ids={tuple(sorted(certified_leaf_ids))}"
            ),
        ),
        CertificateCheckObligation(
            "branch_union_aggregate_contains_leaf_targets",
            aggregate_contains_leaves,
            f"aggregate={aggregate}; leaf_intervals={leaf_intervals}",
        ),
        CertificateCheckObligation(
            "branch_union_exact_rational_interval_aggregation",
            exact_rational_interval_aggregation,
            exact_rational_interval_aggregation_detail,
        ),
    )
    return BranchUnionCheckResult(
        union_id=str(certificate.union_id),
        union_type=union_type,
        checker_id="branch_union_checker_v1",
        obligations=obligations,
        leaf_count=len(leaf_ids),
    )


def _checked_response_identifier(result: object) -> str:
    for attribute in ("certificate_id", "chain_id", "union_id"):
        value = getattr(result, attribute, "")
        if value:
            return str(value)
    return ""


def check_chart_chain(
    certificate: ChartChainCertificate,
    chart_certificates: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
        | TotalCollisionFuchsianStopChartCertificate
        | TotalCollisionGeneralizedFuchsianStopChartCertificate
    ],
    transition_certificates: Iterable[
        OrdinaryChartTransitionCertificate
        | PlanarLeviCivitaTransitionCertificate
        | SpatialKSTransitionCertificate
    ],
    checked_chart_results: Iterable[CertificateCheckResult],
    checked_transition_results: Iterable[TransitionCheckResult],
) -> ChartChainCheckResult:
    """Verify finite chart-chain grammar and physical target coverage."""

    chain_type = str(certificate.chain_type)
    chart_ids = tuple(str(value) for value in certificate.chart_ids)
    transition_ids = tuple(str(value) for value in certificate.transition_ids)
    target_interval = tuple(float(value) for value in certificate.target_physical_time_interval)
    chart_by_id = {str(chart.chart_id): chart for chart in chart_certificates}
    transition_by_id = {
        str(transition.transition_id): transition
        for transition in transition_certificates
    }
    certified_chart_ids = {
        result.certificate_id
        for result in checked_chart_results
        if result.certified and result.certificate_id
    }
    certified_transition_ids = {
        result.transition_id
        for result in checked_transition_results
        if result.certified and result.transition_id
    }

    supported_chain_type = chain_type in {
        "finite_time_chart_chain",
        "ordinary_chart_chain",
        "regularized_atlas_chart_chain",
    }
    identity_present = bool(certificate.certificate_id and certificate.chain_id)
    chart_ids_present = bool(chart_ids)
    chart_ids_unique = len(set(chart_ids)) == len(chart_ids)
    transition_count_matches = len(transition_ids) == max(0, len(chart_ids) - 1)
    charts_present = all(chart_id in chart_by_id for chart_id in chart_ids)
    transitions_present = all(
        transition_id in transition_by_id
        for transition_id in transition_ids
    )
    chart_certificates_checked = bool(
        chart_ids_present
        and charts_present
        and all(
            chart_by_id[chart_id].certificate_id in certified_chart_ids
            for chart_id in chart_ids
        )
    )
    transition_certificates_checked = bool(
        transition_count_matches
        and transitions_present
        and all(transition_id in certified_transition_ids for transition_id in transition_ids)
    )
    transition_adjacency = bool(
        transition_count_matches
        and transitions_present
        and all(
            str(transition_by_id[transition_ids[index]].source_chart_id)
            == chart_ids[index]
            and str(transition_by_id[transition_ids[index]].target_chart_id)
            == chart_ids[index + 1]
            for index in range(len(transition_ids))
        )
    )
    chart_intervals = tuple(
        tuple(float(value) for value in getattr(chart_by_id[chart_id], "physical_time_interval", ()))
        for chart_id in chart_ids
        if chart_id in chart_by_id
    )
    target_interval_finite = _finite_nonempty_interval(target_interval)
    chart_intervals_finite = bool(
        len(chart_intervals) == len(chart_ids)
        and all(_finite_nonempty_interval(interval) for interval in chart_intervals)
    )
    monotone_no_gaps = bool(
        chart_intervals_finite
        and all(
            chart_intervals[index][0] <= chart_intervals[index + 1][0]
            and chart_intervals[index + 1][0] <= chart_intervals[index][1]
            for index in range(len(chart_intervals) - 1)
        )
    )
    if chart_intervals_finite and monotone_no_gaps:
        covered_interval = (
            chart_intervals[0][0],
            max(interval[1] for interval in chart_intervals),
        )
    elif chart_intervals_finite and len(chart_intervals) == 1:
        covered_interval = chart_intervals[0]
    else:
        covered_interval = (float("inf"), float("-inf"))
    target_interval_covered = bool(
        target_interval_finite
        and _finite_nonempty_interval(covered_interval)
        and _interval_contains_interval(covered_interval, target_interval)
    )
    (
        exact_rational_time_coverage,
        exact_rational_time_coverage_detail,
    ) = _exact_rational_chart_chain_time_coverage(
        chart_intervals,
        target_interval,
    )
    obligations = (
        CertificateCheckObligation(
            "chart_chain_identity_present",
            identity_present,
            f"certificate_id={certificate.certificate_id!r}; chain_id={certificate.chain_id!r}",
        ),
        CertificateCheckObligation(
            "chart_chain_type_supported",
            supported_chain_type,
            f"chain_type={chain_type!r}",
        ),
        CertificateCheckObligation(
            "chart_chain_chart_ids_present",
            chart_ids_present,
            f"chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_chart_ids_unique",
            chart_ids_unique,
            f"unique_chart_count={len(set(chart_ids))}; chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_transition_count_matches",
            transition_count_matches,
            f"transition_count={len(transition_ids)}; chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_charts_independently_checked",
            chart_certificates_checked,
            f"chart_ids={chart_ids}; certified_chart_certificate_ids={tuple(sorted(certified_chart_ids))}",
        ),
        CertificateCheckObligation(
            "chart_chain_transitions_independently_checked",
            transition_certificates_checked,
            f"transition_ids={transition_ids}; certified_transition_ids={tuple(sorted(certified_transition_ids))}",
        ),
        CertificateCheckObligation(
            "chart_chain_transition_adjacency",
            transition_adjacency,
            f"chart_ids={chart_ids}; transition_ids={transition_ids}",
        ),
        CertificateCheckObligation(
            "chart_chain_intervals_finite",
            bool(target_interval_finite and chart_intervals_finite),
            f"target_interval={target_interval}; chart_intervals={chart_intervals}",
        ),
        CertificateCheckObligation(
            "chart_chain_no_physical_time_gaps",
            bool(len(chart_intervals) <= 1 or monotone_no_gaps),
            f"chart_intervals={chart_intervals}",
        ),
        CertificateCheckObligation(
            "chart_chain_target_interval_covered",
            target_interval_covered,
            f"covered_interval={covered_interval}; target_interval={target_interval}",
        ),
        CertificateCheckObligation(
            "chart_chain_exact_rational_time_coverage",
            exact_rational_time_coverage,
            exact_rational_time_coverage_detail,
        ),
    )
    return ChartChainCheckResult(
        chain_id=str(certificate.chain_id),
        chain_type=chain_type,
        checker_id="chart_chain_checker_v1",
        obligations=obligations,
        chart_count=len(chart_ids),
    )


def verify_chart_certificates(
    certificates: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
        | TotalCollisionFuchsianStopChartCertificate
        | TotalCollisionGeneralizedFuchsianStopChartCertificate
    ],
    transitions: Iterable[
        OrdinaryChartTransitionCertificate
        | PlanarLeviCivitaTransitionCertificate
        | SpatialKSTransitionCertificate
    ] = (),
    events: Iterable[EventIsolationCertificate] = (),
    branch_unions: Iterable[BranchUnionCertificate] = (),
    chart_chains: Iterable[ChartChainCertificate] = (),
) -> IndependentChartVerifierCertificate:
    """Run the independent checker over a finite serialized chart bundle."""

    chart_tuple = tuple(certificates)
    transition_tuple = tuple(transitions)
    results = tuple(_check_chart_certificate(certificate) for certificate in chart_tuple)
    transition_results = tuple(
        _check_transition_certificate(transition, chart_tuple)
        for transition in transition_tuple
    )
    event_results = tuple(check_event_isolation(event) for event in events)
    chart_chain_results = tuple(
        check_chart_chain(
            chart_chain,
            chart_tuple,
            transition_tuple,
            results,
            transition_results,
        )
        for chart_chain in chart_chains
    )
    branch_union_results = tuple(
        check_branch_union(
            branch_union,
            (*results, *chart_chain_results),
        )
        for branch_union in branch_unions
    )
    return IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=results,
        transition_results=transition_results,
        event_results=event_results,
        branch_union_results=branch_union_results,
        chart_chain_results=chart_chain_results,
    )


def _check_chart_certificate(
    certificate: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate
    | TotalCollisionFuchsianStopChartCertificate
    | TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    if isinstance(certificate, OrdinaryTaylorChartCertificate):
        return check_ordinary_taylor_chart(certificate)
    if isinstance(certificate, PlanarLeviCivitaBinaryChartCertificate):
        return check_planar_levi_civita_binary_chart(certificate)
    if isinstance(certificate, SpatialKSBinaryChartCertificate):
        return check_spatial_ks_binary_chart(certificate)
    if isinstance(certificate, TotalCollisionFuchsianStopChartCertificate):
        return check_total_collision_fuchsian_stop_chart(certificate)
    if isinstance(certificate, TotalCollisionGeneralizedFuchsianStopChartCertificate):
        return check_total_collision_generalized_fuchsian_stop_chart(certificate)
    raise TypeError(f"unsupported chart certificate type: {type(certificate)!r}")


def _check_transition_certificate(
    transition: OrdinaryChartTransitionCertificate
    | PlanarLeviCivitaTransitionCertificate
    | SpatialKSTransitionCertificate,
    charts: tuple[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate,
        ...,
    ],
) -> TransitionCheckResult:
    if isinstance(transition, OrdinaryChartTransitionCertificate):
        ordinary_charts = tuple(
            chart for chart in charts if isinstance(chart, OrdinaryTaylorChartCertificate)
        )
        return check_ordinary_chart_transition(transition, ordinary_charts)
    if isinstance(transition, PlanarLeviCivitaTransitionCertificate):
        return check_planar_levi_civita_transition(transition, charts)
    if isinstance(transition, SpatialKSTransitionCertificate):
        return check_spatial_ks_transition(transition, charts)
    raise TypeError(f"unsupported transition certificate type: {type(transition)!r}")


def attach_independent_chart_verifier(
    theorem_certificate: object,
    verifier_certificate: IndependentChartVerifierCertificate,
) -> object:
    """Attach checker evidence to an open-time theorem certificate.

    The returned object is still the same dataclass type as the theorem
    certificate, but it carries the verifier evidence fields consumed by the
    closed-form audit.
    """

    if getattr(theorem_certificate, "theorem_id", None) != "open_time_locally_finite_atlas":
        raise TypeError("independent chart verifier can only be attached to open-time atlas theorem certificates")
    return replace(
        theorem_certificate,
        independent_chart_verifier_certificate=verifier_certificate,
        independent_chart_verifier_certified=verifier_certificate.certified,
    )


def _coefficient_array(values: object) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.asarray((), dtype=float)


def _regularized_binary_solution_from_certificate(
    certificate: PlanarLeviCivitaBinaryChartCertificate,
) -> RegularizedBinaryTaylorSolution:
    return RegularizedBinaryTaylorSolution(
        masses=np.asarray(certificate.masses, dtype=float),
        pair=tuple(int(value) for value in certificate.pair),
        z=_coefficient_array(certificate.z_coefficients),
        z_velocity=_coefficient_array(certificate.z_velocity_coefficients),
        pair_energy=_coefficient_array(certificate.pair_energy_coefficients),
        binary_center=_coefficient_array(certificate.binary_center_coefficients),
        binary_center_velocity=_coefficient_array(
            certificate.binary_center_velocity_coefficients,
        ),
        third_offset=_coefficient_array(certificate.third_offset_coefficients),
        third_offset_velocity=_coefficient_array(
            certificate.third_offset_velocity_coefficients,
        ),
        physical_time=_coefficient_array(certificate.physical_time_coefficients),
    )


def _spatial_ks_solution_from_certificate(
    certificate: SpatialKSBinaryChartCertificate,
) -> SpatialKSBinaryTaylorSolution:
    return SpatialKSBinaryTaylorSolution(
        masses=np.asarray(certificate.masses, dtype=float),
        pair=tuple(int(value) for value in certificate.pair),
        u=_coefficient_array(certificate.u_coefficients),
        u_velocity=_coefficient_array(certificate.u_velocity_coefficients),
        pair_energy=_coefficient_array(certificate.pair_energy_coefficients),
        binary_center=_coefficient_array(certificate.binary_center_coefficients),
        binary_center_velocity=_coefficient_array(
            certificate.binary_center_velocity_coefficients,
        ),
        third_offset=_coefficient_array(certificate.third_offset_coefficients),
        third_offset_velocity=_coefficient_array(
            certificate.third_offset_velocity_coefficients,
        ),
        physical_time=_coefficient_array(certificate.physical_time_coefficients),
    )


def _fuchsian_log_branch_from_certificate(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> FiniteFuchsianLogBranch:
    return FiniteFuchsianLogBranch(
        masses=np.asarray(certificate.masses, dtype=float),
        central_shape=np.asarray(certificate.central_shape, dtype=float),
        scale_coefficient=float(certificate.scale_coefficient),
        terms=tuple(
            _fuchsian_log_term_from_certificate(term)
            for term in certificate.terms
        ),
    )


def _generalized_fuchsian_branch_from_certificate(
    certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> FuchsianShapeBranch:
    selected_coefficients = {
        tuple(int(value) for value in item.index): np.asarray(
            item.coefficient,
            dtype=float,
        )
        for item in certificate.selected_coefficients
    }
    return construct_fuchsian_shape_branch(
        masses=np.asarray(certificate.masses, dtype=float),
        central_shape=np.asarray(certificate.central_shape, dtype=float),
        powers=tuple(float(power) for power in certificate.powers),
        selected_coefficients=selected_coefficients,
        max_total_degree=int(certificate.max_total_degree),
        scale_index=(
            None
            if certificate.scale_index is None
            else tuple(int(value) for value in certificate.scale_index)
        ),
    )


def _generalized_fuchsian_shape_deviation_bound_for_checker(
    branch: FuchsianShapeBranch,
    radius: float,
) -> float:
    radius = float(radius)
    if radius <= 0.0:
        return float("inf")
    bound = 0.0
    for index, coefficient in branch.coefficients.items():
        if index == branch.zero_index:
            continue
        exponent = branch.exponent(index)
        if not np.isfinite(exponent) or exponent <= 0.0:
            return float("inf")
        bound += (
            float(np.linalg.norm(np.asarray(coefficient, dtype=float), ord=np.inf))
            * radius**exponent
        )
    return float(bound)


def _max_sampled_generalized_fuchsian_stop_chart_quantities(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, float, float, int]:
    max_lifted_residual = 0.0
    max_angular_momentum = 0.0
    max_position_scale = 0.0
    checked = 0
    for tau in _punctured_tau_samples(tau_interval, sample_count=sample_count):
        lifted_residual = branch.shape_equation_residual_at_tau(tau)
        positions = branch.positions_at_tau(tau)
        max_lifted_residual = max(
            max_lifted_residual,
            _array_sup_norm(lifted_residual),
        )
        if branch.central_shape.shape[1] == 2:
            angular = abs(branch.centered_angular_momentum_scalar_at_tau(tau))
        else:
            angular = _spatial_centered_angular_momentum_norm(branch, tau)
        max_angular_momentum = max(max_angular_momentum, float(angular))
        max_position_scale = max(
            max_position_scale,
            float(np.max(np.abs(positions))) / tau**2,
        )
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, np.inf, 0)
    return (
        float(max_lifted_residual),
        float(max_angular_momentum),
        float(max_position_scale),
        int(checked),
    )


def _check_generalized_remainder_majorant(
    majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
    *,
    residual_tolerance: float,
    tail_bound: float,
) -> dict[str, object]:
    if majorant is None:
        return {
            "initial_radius": np.inf,
            "constants_finite": False,
            "contraction_certified": False,
            "self_map_certified": False,
            "component_inputs_certified": False,
            "certified": False,
            "tail_bound": np.inf,
            "tail_covered": False,
            "lifted_residual_tail_bound": np.inf,
            "lifted_residual_tail_certified": False,
            "physical_residual_tail_bound": np.inf,
            "physical_residual_tail_certified": False,
            "required_residual_components_present": False,
            "required_residual_component_detail": "remainder majorant not supplied",
            "endpoint_collapse_certified": False,
            "constant_detail": "remainder majorant not supplied",
            "contraction_detail": "remainder majorant not supplied",
            "self_map_detail": "remainder majorant not supplied",
            "component_detail": "remainder majorant not supplied",
            "endpoint_detail": "remainder majorant not supplied",
            "detail": "remainder majorant not supplied",
        }

    initial_radius = float(majorant.initial_radius)
    shell_contraction = float(majorant.shell_contraction)
    analytic_disk_fraction = float(majorant.analytic_disk_fraction)
    defect_bound = float(majorant.defect_bound)
    linear_inverse_bound = float(majorant.linear_inverse_bound)
    nonlinear_lipschitz_bound = float(majorant.nonlinear_lipschitz_bound)
    remainder_ball_radius = float(majorant.remainder_ball_radius)
    contraction_factor = float(linear_inverse_bound * nonlinear_lipschitz_bound)
    self_map_bound = float(
        linear_inverse_bound * defect_bound
        + contraction_factor * remainder_ball_radius
    )
    contraction_slack = float(1.0 - contraction_factor)
    self_map_margin = float(remainder_ball_radius - self_map_bound)
    component_inputs = dict(majorant.component_inputs)
    exponents = dict(majorant.component_effective_exponents)
    header_certified = bool(
        np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
        and component_inputs
        and exponents
    )
    constants_finite = bool(
        np.isfinite(defect_bound)
        and defect_bound >= 0.0
        and np.isfinite(linear_inverse_bound)
        and linear_inverse_bound >= 0.0
        and np.isfinite(nonlinear_lipschitz_bound)
        and nonlinear_lipschitz_bound >= 0.0
        and np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
    )
    contraction_certified = bool(
        constants_finite
        and np.isfinite(contraction_factor)
        and contraction_factor < 1.0
    )
    self_map_certified = bool(
        constants_finite
        and contraction_certified
        and np.isfinite(self_map_bound)
        and self_map_bound <= remainder_ball_radius * (1.0 + 1.0e-12)
    )
    key_sets_match = bool(
        component_inputs and set(component_inputs) == set(exponents)
    )

    component_failures: list[str] = []
    envelope_failures: list[str] = []
    first_shell_bounds: list[float] = []
    primitive_by_component: dict[str, PrimitiveCauchyTailInput] = {}
    for component, serialized in component_inputs.items():
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        primitive_by_component[component] = primitive
        recomputed_first_shell = primitive.first_shell_tail_bound
        recomputed_shell_ratio = primitive.shell_ratio
        first_shell_bounds.append(float(recomputed_first_shell))
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        shell_ratio_matches = _finite_close(
            float(serialized.shell_ratio),
            recomputed_shell_ratio,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        try:
            expected_initial, expected_growth = (
                _generalized_remainder_shell_majorant_for_checker(
                    remainder_ball_radius=remainder_ball_radius,
                    effective_exponent=float(exponents[component]),
                    initial_radius=initial_radius,
                    shell_contraction=shell_contraction,
                    analytic_disk_fraction=analytic_disk_fraction,
                )
            )
            envelope_covers = bool(
                float(serialized.majorant_initial) + 1.0e-15 >= expected_initial
                and float(serialized.majorant_growth) + 1.0e-15 >= expected_growth
            )
        except (FloatingPointError, KeyError, TypeError, ValueError):
            envelope_covers = False
        if not envelope_covers:
            envelope_failures.append(component)
        if not (
            primitive.certified
            and first_shell_matches
            and shell_ratio_matches
            and envelope_covers
        ):
            component_failures.append(component)

    remainder_tail_bound = float(max(first_shell_bounds, default=np.inf))
    lifted_residual = primitive_by_component.get("lifted_residual")
    lifted_residual_tail_bound = (
        float(lifted_residual.first_shell_tail_bound)
        if lifted_residual is not None
        else float("inf")
    )
    lifted_residual_tail_certified = bool(
        lifted_residual is not None
        and lifted_residual.certified
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(lifted_residual_tail_bound)
        and lifted_residual_tail_bound <= residual_tolerance
    )
    physical_residual = primitive_by_component.get("physical_residual")
    physical_residual_tail_bound = (
        float(physical_residual.first_shell_tail_bound)
        if physical_residual is not None
        else float("inf")
    )
    physical_residual_tail_certified = bool(
        physical_residual is not None
        and physical_residual.certified
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(physical_residual_tail_bound)
        and physical_residual_tail_bound <= residual_tolerance
    )
    required_residual_components = ("lifted_residual", "physical_residual")
    missing_residual_components = tuple(
        component
        for component in required_residual_components
        if component not in primitive_by_component
    )
    required_residual_components_present = bool(not missing_residual_components)
    endpoint_component = (
        "regularized_position_value"
        if "regularized_position_value" in primitive_by_component
        else "value"
    )
    endpoint_input = primitive_by_component.get(endpoint_component)
    endpoint_collapse_certified = bool(
        endpoint_input is not None
        and endpoint_input.certified
        and np.isfinite(endpoint_input.first_shell_tail_bound)
        and endpoint_input.first_shell_tail_bound >= 0.0
        and np.isfinite(endpoint_input.shell_ratio)
        and endpoint_input.shell_ratio < 1.0
    )
    component_inputs_certified = bool(
        header_certified
        and key_sets_match
        and not component_failures
        and not envelope_failures
        and np.isfinite(remainder_tail_bound)
    )
    tail_covered = bool(
        np.isfinite(tail_bound)
        and tail_bound >= 0.0
        and np.isfinite(remainder_tail_bound)
        and tail_bound + 1.0e-15 >= remainder_tail_bound
    )
    certified = bool(
        constants_finite
        and contraction_certified
        and self_map_certified
        and component_inputs_certified
    )
    return {
        "initial_radius": initial_radius,
        "constants_finite": constants_finite,
        "contraction_certified": contraction_certified,
        "self_map_certified": self_map_certified,
        "component_inputs_certified": component_inputs_certified,
        "certified": certified,
        "tail_bound": remainder_tail_bound,
        "tail_covered": tail_covered,
        "lifted_residual_tail_bound": lifted_residual_tail_bound,
        "lifted_residual_tail_certified": lifted_residual_tail_certified,
        "physical_residual_tail_bound": physical_residual_tail_bound,
        "physical_residual_tail_certified": physical_residual_tail_certified,
        "required_residual_components_present": (
            required_residual_components_present
        ),
        "required_residual_component_detail": (
            f"required_components={required_residual_components}; "
            f"missing_components={missing_residual_components}; "
            f"lifted_residual_tail={lifted_residual_tail_bound}; "
            f"physical_residual_tail={physical_residual_tail_bound}"
        ),
        "endpoint_collapse_certified": endpoint_collapse_certified,
        "constant_detail": (
            f"defect={defect_bound}; inverse={linear_inverse_bound}; "
            f"lipschitz={nonlinear_lipschitz_bound}; radius={remainder_ball_radius}"
        ),
        "contraction_detail": (
            f"q=B*L={contraction_factor}; slack=1-q={contraction_slack}"
        ),
        "self_map_detail": (
            f"B*D+q*R={self_map_bound}; R={remainder_ball_radius}; "
            f"margin=R-(B*D+q*R)={self_map_margin}"
        ),
        "component_detail": (
            f"component_count={len(component_inputs)}; "
            f"key_sets_match={key_sets_match}; "
            f"component_failures={tuple(component_failures)}; "
            f"envelope_failures={tuple(envelope_failures)}"
        ),
        "endpoint_detail": (
            f"endpoint_component={endpoint_component}; "
            f"tail={endpoint_input.first_shell_tail_bound if endpoint_input is not None else np.inf}; "
            f"shell_ratio={endpoint_input.shell_ratio if endpoint_input is not None else np.inf}"
        ),
        "detail": (
            f"header_certified={header_certified}; constants_finite={constants_finite}; "
            f"contraction={contraction_certified}; self_map={self_map_certified}; "
            f"component_inputs={component_inputs_certified}"
        ),
    }


def _generalized_remainder_shell_majorant_for_checker(
    *,
    remainder_ball_radius: float,
    effective_exponent: float,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
) -> tuple[float, float]:
    remainder_ball_radius = float(remainder_ball_radius)
    effective_exponent = float(effective_exponent)
    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    analytic_disk_fraction = float(analytic_disk_fraction)
    if not (
        np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
        and np.isfinite(effective_exponent)
        and np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
    ):
        raise ValueError("invalid generalized remainder shell-majorant inputs")
    radius_factor = (
        initial_radius * (1.0 + analytic_disk_fraction)
        if effective_exponent >= 0.0
        else initial_radius * (1.0 - analytic_disk_fraction)
    )
    return (
        float(remainder_ball_radius * radius_factor**effective_exponent),
        float(shell_contraction**effective_exponent),
    )


def _fuchsian_log_term_from_certificate(
    term: FuchsianLogTermCertificate,
) -> FuchsianLogTerm:
    return FuchsianLogTerm(
        power=float(term.power),
        coefficients_by_log_power={
            int(log_power): np.asarray(coefficient, dtype=float)
            for log_power, coefficient in term.coefficients_by_log_power
        },
        selector_basis=tuple(
            np.asarray(basis, dtype=float) for basis in term.selector_basis
        ),
    )


def _fuchsian_log_terms_finite(
    terms: tuple[FuchsianLogTermCertificate, ...],
    expected_shape: tuple[int, ...],
) -> bool:
    for term in terms:
        if not np.isfinite(term.power) or term.power <= 0.0:
            return False
        for log_power, coefficient in term.coefficients_by_log_power:
            array = np.asarray(coefficient, dtype=float)
            if int(log_power) < 0 or array.shape != expected_shape:
                return False
            if not np.all(np.isfinite(array)):
                return False
        for basis in term.selector_basis:
            basis_array = np.asarray(basis, dtype=float)
            if basis_array.shape != expected_shape:
                return False
            if not np.all(np.isfinite(basis_array)):
                return False
    return True


def _fuchsian_log_term_coefficient_sup_norm(term: FuchsianLogTerm) -> float:
    values = [
        _array_sup_norm(np.asarray(coefficient, dtype=float))
        for coefficient in term.coefficients_by_log_power.values()
    ]
    return float(max(values, default=0.0))


def _check_fuchsian_primitive_cauchy_inputs(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    *,
    branch: FiniteFuchsianLogBranch | None = None,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    component_multipliers = dict(inputs.component_multipliers)
    header_certified = bool(
        np.isfinite(inputs.initial_radius)
        and inputs.initial_radius > 0.0
        and np.isfinite(inputs.shell_contraction)
        and 0.0 < inputs.shell_contraction < 1.0
        and np.isfinite(inputs.analytic_disk_fraction)
        and 0.0 < inputs.analytic_disk_fraction < 1.0
        and np.isfinite(inputs.log_growth_factor)
        and inputs.log_growth_factor > 1.0
        and component_inputs
    )
    key_sets_match = bool(
        component_inputs
        and set(component_inputs) == set(derivative_orders)
        and set(component_inputs) == set(component_multipliers)
    )
    component_failures: list[str] = []
    envelope_failures: list[str] = []
    first_shell_bounds: list[float] = []
    for component, serialized in component_inputs.items():
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        recomputed_first_shell = primitive.first_shell_tail_bound
        recomputed_shell_ratio = primitive.shell_ratio
        first_shell_bounds.append(float(recomputed_first_shell))
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        shell_ratio_matches = _finite_close(
            float(serialized.shell_ratio),
            recomputed_shell_ratio,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        derivative_order = derivative_orders.get(component, -1)
        multiplier = component_multipliers.get(component, np.inf)
        branch_envelope_covers = True
        if branch is not None:
            try:
                expected_initial, expected_growth = (
                    _finite_fuchsian_log_branch_derivative_envelope(
                        branch,
                        derivative_order=int(derivative_order),
                        initial_radius=float(inputs.initial_radius),
                        shell_contraction=float(inputs.shell_contraction),
                        analytic_disk_fraction=float(inputs.analytic_disk_fraction),
                        log_growth_factor=float(inputs.log_growth_factor),
                    )
                )
                expected_initial *= float(multiplier)
                branch_envelope_covers = bool(
                    np.isfinite(expected_initial)
                    and np.isfinite(expected_growth)
                    and float(serialized.majorant_initial) + 1.0e-15
                    >= expected_initial
                    and float(serialized.majorant_growth) + 1.0e-15
                    >= expected_growth
                )
            except (FloatingPointError, TypeError, ValueError):
                branch_envelope_covers = False
            if not branch_envelope_covers:
                envelope_failures.append(component)
        if not (
            primitive.certified
            and first_shell_matches
            and shell_ratio_matches
            and int(derivative_order) >= 0
            and np.isfinite(float(multiplier))
            and float(multiplier) >= 0.0
            and branch_envelope_covers
        ):
            component_failures.append(component)
    tail_bound = float(max(first_shell_bounds, default=np.inf))
    certified = bool(
        header_certified
        and key_sets_match
        and np.isfinite(tail_bound)
        and not envelope_failures
        and not component_failures
    )
    detail = (
        f"component_count={len(component_inputs)}; "
        f"tail_bound={tail_bound}; "
        f"header_certified={header_certified}; "
        f"key_sets_match={key_sets_match}; "
        f"envelope_failures={tuple(envelope_failures)}; "
        f"component_failures={tuple(component_failures)}"
    )
    return certified, tail_bound, detail


def _check_fuchsian_primitive_cauchy_residual_tail(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    *,
    residual_tolerance: float,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    component_multipliers = dict(inputs.component_multipliers)
    required_components = ("lifted_residual", "physical_residual")
    missing_components = tuple(
        component
        for component in required_components
        if component not in component_inputs
    )
    metadata_failures: list[str] = []
    component_failures: list[str] = []
    residual_bounds: list[float] = []
    for component in required_components:
        serialized = component_inputs.get(component)
        if serialized is None:
            continue
        derivative_order = int(derivative_orders.get(component, -1))
        multiplier = float(component_multipliers.get(component, np.inf))
        if not (
            derivative_order == 0
            and np.isfinite(multiplier)
            and multiplier >= 1.0
        ):
            metadata_failures.append(component)
            continue
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        recomputed_first_shell = primitive.first_shell_tail_bound
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        if not (primitive.certified and first_shell_matches):
            component_failures.append(component)
            continue
        residual_bounds.append(float(recomputed_first_shell))
    residual_tail_bound = float(max(residual_bounds, default=np.inf))
    tolerance_ok = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(residual_tail_bound)
        and residual_tail_bound <= residual_tolerance
    )
    certified = bool(
        not missing_components
        and not metadata_failures
        and not component_failures
        and tolerance_ok
    )
    detail = (
        f"required_components={required_components}; "
        f"missing_components={missing_components}; "
        f"metadata_failures={tuple(metadata_failures)}; "
        f"component_failures={tuple(component_failures)}"
    )
    return certified, residual_tail_bound, detail


def _check_fuchsian_endpoint_collapse_envelope(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    value_input = component_inputs.get("value")
    if value_input is None:
        return (
            False,
            np.inf,
            "missing value component for endpoint-collapse envelope",
        )
    initial_radius = float(inputs.initial_radius)
    shell_contraction = float(inputs.shell_contraction)
    value_majorant_growth = float(value_input.majorant_growth)
    value_tail_shell_ratio = float(value_input.shell_ratio)
    value_first_shell_tail_bound = float(value_input.first_shell_tail_bound)
    value_derivative_order = int(derivative_orders.get("value", -1))
    first_shell_shape_bound = float(
        float(value_input.majorant_initial) + value_first_shell_tail_bound
    )
    first_shell_position_bound = float(initial_radius**2 * first_shell_shape_bound)
    finite_part_position_ratio = float(shell_contraction**2 * value_majorant_growth)
    tail_position_ratio = float(shell_contraction**2 * value_tail_shell_ratio)
    primitive = PrimitiveCauchyTailInput(
        majorant_initial=float(value_input.majorant_initial),
        majorant_growth=value_majorant_growth,
        step_ratio_bound=float(value_input.step_ratio_bound),
        retained_order_initial=int(value_input.retained_order_initial),
        retained_order_increment=int(value_input.retained_order_increment),
    )
    certified = bool(
        primitive.certified
        and value_derivative_order == 0
        and np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(first_shell_shape_bound)
        and first_shell_shape_bound >= 0.0
        and np.isfinite(first_shell_position_bound)
        and np.isfinite(finite_part_position_ratio)
        and finite_part_position_ratio < 1.0
        and np.isfinite(tail_position_ratio)
        and tail_position_ratio < 1.0
    )
    detail = (
        f"value_derivative_order={value_derivative_order}; "
        f"first_shell_shape_bound={first_shell_shape_bound}; "
        f"finite_part_position_ratio={finite_part_position_ratio}; "
        f"tail_position_ratio={tail_position_ratio}"
    )
    return certified, first_shell_position_bound, detail


def _check_fuchsian_supported_scale_finite_energy_matching(
    branch: FiniteFuchsianLogBranch,
) -> tuple[bool, float, str]:
    masses = np.asarray(branch.masses, dtype=float)
    central_shape = _interval_point_array(np.asarray(branch.central_shape, dtype=float))
    unsupported_terms = tuple(
        index
        for index, term in enumerate(branch.terms)
        if _fuchsian_log_term_coefficient_sup_norm(term) > 1.0e-15
    )
    try:
        inertia = _interval_weighted_squared_norm(central_shape, masses)
        potential = _interval_pairwise_newtonian_potential(central_shape, masses)
        leading_gap = inertia.scale(2.0 / 9.0) - potential
        finite_energy = inertia.scale((10.0 / 9.0) * float(branch.scale_coefficient))
    except (FloatingPointError, ValueError):
        return (
            False,
            np.inf,
            "interval finite-energy computation failed",
        )
    leading_gap_bound = _interval_abs_sup(leading_gap)
    finite_energy_bound = _interval_abs_sup(finite_energy)
    certified = bool(
        not unsupported_terms
        and np.isfinite(finite_energy_bound)
        and np.isfinite(float(branch.scale_coefficient))
        and leading_gap_bound <= 1.0e-10
    )
    detail = (
        f"leading_energy_gap_bound={leading_gap_bound}; "
        f"unsupported_non_scale_terms={unsupported_terms}"
    )
    return certified, finite_energy_bound, detail


def _finite_close(
    left: float,
    right: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and abs(left - right) <= max(atol, rtol * max(abs(left), abs(right)))
    )


def _fuchsian_log_total_collision_isolation_from_certificate(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> FiniteFuchsianLogTotalCollisionIsolationCertificate:
    return certify_finite_fuchsian_log_total_collision_isolation(
        _fuchsian_log_branch_from_certificate(certificate),
        radius=float(certificate.isolation_radius),
    )


def _finite_nonempty_interval(interval: tuple[float, float]) -> bool:
    return bool(
        len(interval) == 2
        and np.isfinite(interval[0])
        and np.isfinite(interval[1])
        and interval[0] <= interval[1]
    )


def _interval_contains_interval(
    outer: tuple[float, float],
    inner: tuple[float, float],
    *,
    tolerance: float = 0.0,
) -> bool:
    return bool(
        _finite_nonempty_interval(outer)
        and _finite_nonempty_interval(inner)
        and outer[0] - tolerance <= inner[0]
        and inner[1] <= outer[1] + tolerance
    )


def _exact_rational_interval_from_floats(
    interval: tuple[float, float],
) -> tuple[Fraction, Fraction] | None:
    if not _finite_nonempty_interval(interval):
        return None
    lower = Fraction(float(interval[0]))
    upper = Fraction(float(interval[1]))
    if lower > upper:
        return None
    return lower, upper


def _exact_rational_chart_chain_time_coverage(
    chart_intervals: tuple[tuple[float, float], ...],
    target_interval: tuple[float, float],
) -> tuple[bool, str]:
    target = _exact_rational_interval_from_floats(target_interval)
    charts = tuple(
        rational_interval
        for rational_interval in (
            _exact_rational_interval_from_floats(interval)
            for interval in chart_intervals
        )
        if rational_interval is not None
    )
    if target is None or len(charts) != len(chart_intervals) or not charts:
        return (
            False,
            (
                "target_interval or chart physical-time intervals are not "
                "finite nonempty serialized intervals"
            ),
        )
    no_gaps = all(
        charts[index][0] <= charts[index + 1][0]
        and charts[index + 1][0] <= charts[index][1]
        for index in range(len(charts) - 1)
    )
    covered = (charts[0][0], max(interval[1] for interval in charts))
    covered_target = covered[0] <= target[0] and target[1] <= covered[1]
    certified = bool(no_gaps and covered_target)
    return (
        certified,
        (
            "exact rational interval arithmetic over serialized binary-float "
            f"time endpoints; no_gaps={no_gaps}; "
            f"covered_interval=({covered[0]}, {covered[1]}); "
            f"target_interval=({target[0]}, {target[1]})"
        ),
    )


def _exact_rational_branch_union_interval_aggregation(
    aggregate_interval: tuple[float, float],
    leaf_intervals: tuple[tuple[float, float], ...],
) -> tuple[bool, str]:
    aggregate = _exact_rational_interval_from_floats(aggregate_interval)
    leaves = tuple(
        rational_interval
        for rational_interval in (
            _exact_rational_interval_from_floats(interval)
            for interval in leaf_intervals
        )
        if rational_interval is not None
    )
    if (
        aggregate is None
        or len(leaves) != len(leaf_intervals)
        or not leaves
    ):
        return (
            False,
            (
                "aggregate or leaf target intervals are not finite nonempty "
                "serialized intervals"
            ),
        )
    contains_leaves = all(
        aggregate[0] <= leaf[0] and leaf[1] <= aggregate[1]
        for leaf in leaves
    )
    leaf_hull = (
        min(leaf[0] for leaf in leaves),
        max(leaf[1] for leaf in leaves),
    )
    return (
        bool(contains_leaves),
        (
            "exact rational interval arithmetic over serialized binary-float "
            f"target endpoints; contains_leaves={contains_leaves}; "
            f"aggregate=({aggregate[0]}, {aggregate[1]}); "
            f"leaf_hull=({leaf_hull[0]}, {leaf_hull[1]}); "
            f"leaf_count={len(leaves)}"
        ),
    )


def _exact_rational_transition_state_continuity(
    source_chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    target_chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    *,
    source_parameter: float,
    target_parameter: float,
    handoff_time: float,
    position_tolerance: float,
    velocity_tolerance: float,
    physical_time_tolerance: float,
) -> tuple[bool, str]:
    try:
        source_parameter_q = Fraction(float(source_parameter))
        target_parameter_q = Fraction(float(target_parameter))
        handoff_time_q = Fraction(float(handoff_time))
        position_tolerance_q = Fraction(float(position_tolerance))
        velocity_tolerance_q = Fraction(float(velocity_tolerance))
        physical_time_tolerance_q = Fraction(float(physical_time_tolerance))
    except (OverflowError, ValueError):
        return False, "transition parameters or tolerances are not finite rationalizable floats"
    try:
        source_time = _exact_rational_chart_physical_time_at_parameter(
            source_chart,
            source_parameter_q,
        )
        target_time = _exact_rational_chart_physical_time_at_parameter(
            target_chart,
            target_parameter_q,
        )
        source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
            source_chart,
            source_parameter_q,
        )
        target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
            target_chart,
            target_parameter_q,
        )
        max_time_gap = max(
            abs(source_time - handoff_time_q),
            abs(target_time - handoff_time_q),
        )
        max_position_gap = _fraction_array_max_abs_difference(source_q, target_q)
        max_velocity_gap = _fraction_array_max_abs_difference(source_v, target_v)
    except (ZeroDivisionError, ValueError, TypeError):
        return False, "exact rational transition state evaluation failed"
    certified = bool(
        max_time_gap <= physical_time_tolerance_q
        and max_position_gap <= position_tolerance_q
        and max_velocity_gap <= velocity_tolerance_q
    )
    return (
        certified,
        (
            "exact rational arithmetic over serialized binary-float chart "
            "coefficients, parameters, masses, and tolerances; "
            f"max_time_gap={max_time_gap}; "
            f"max_position_gap={max_position_gap}; "
            f"max_velocity_gap={max_velocity_gap}; "
            f"physical_time_tolerance={physical_time_tolerance_q}; "
            f"position_tolerance={position_tolerance_q}; "
            f"velocity_tolerance={velocity_tolerance_q}"
        ),
    )


def _exact_rational_chart_physical_time_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> Fraction:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        physical = _exact_rational_interval_from_floats(chart.physical_time_interval)
        domain = _exact_rational_interval_from_floats(chart.parameter_interval)
        if physical is None or domain is None:
            raise ValueError("ordinary chart time interval is invalid")
        parameter_width = domain[1] - domain[0]
        physical_width = physical[1] - physical[0]
        if parameter_width == 0:
            if parameter != domain[0]:
                raise ValueError("parameter is outside point ordinary chart")
            return physical[0]
        return physical[0] + (parameter - domain[0]) * physical_width / parameter_width
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _evaluate_fraction_coefficients(chart.physical_time_coefficients, parameter)
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _evaluate_fraction_coefficients(chart.physical_time_coefficients, parameter)
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _exact_rational_chart_projected_state_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        return (
            _evaluate_fraction_coefficients(chart.position_coefficients, parameter),
            _evaluate_fraction_coefficients(chart.velocity_coefficients, parameter),
        )
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _exact_rational_planar_lc_projected_state_at_parameter(
            chart,
            parameter,
        )
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _exact_rational_spatial_ks_projected_state_at_parameter(
            chart,
            parameter,
        )
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _exact_rational_planar_lc_projected_state_at_parameter(
    chart: PlanarLeviCivitaBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    masses = tuple(Fraction(float(value)) for value in chart.masses)
    if len(masses) != 3:
        raise ValueError("LC chart masses must have length 3")
    first, second = chart.pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = masses[first] + masses[second]
    if pair_mass == 0:
        raise ValueError("invalid pair mass")
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    z = _evaluate_fraction_coefficients(chart.z_coefficients, parameter)
    z_velocity = _evaluate_fraction_coefficients(
        chart.z_velocity_coefficients,
        parameter,
    )
    binary_center = _evaluate_fraction_coefficients(
        chart.binary_center_coefficients,
        parameter,
    )
    binary_center_velocity = _evaluate_fraction_coefficients(
        chart.binary_center_velocity_coefficients,
        parameter,
    )
    third_offset = _evaluate_fraction_coefficients(
        chart.third_offset_coefficients,
        parameter,
    )
    third_offset_velocity = _evaluate_fraction_coefficients(
        chart.third_offset_velocity_coefficients,
        parameter,
    )
    relative_position = _exact_rational_lc_square(z)
    relative_velocity = _exact_rational_lc_velocity(z, z_velocity)
    positions = _fraction_zero_array((3, 2))
    velocities = _fraction_zero_array((3, 2))
    positions[first] = binary_center - alpha * relative_position
    positions[second] = binary_center + beta * relative_position
    positions[third] = binary_center + third_offset
    velocities[first] = binary_center_velocity - alpha * relative_velocity
    velocities[second] = binary_center_velocity + beta * relative_velocity
    velocities[third] = binary_center_velocity + third_offset_velocity
    return positions, velocities


def _exact_rational_spatial_ks_projected_state_at_parameter(
    chart: SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    masses = tuple(Fraction(float(value)) for value in chart.masses)
    if len(masses) != 3:
        raise ValueError("KS chart masses must have length 3")
    first, second = chart.pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = masses[first] + masses[second]
    if pair_mass == 0:
        raise ValueError("invalid pair mass")
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    u = _evaluate_fraction_coefficients(chart.u_coefficients, parameter)
    u_velocity = _evaluate_fraction_coefficients(
        chart.u_velocity_coefficients,
        parameter,
    )
    binary_center = _evaluate_fraction_coefficients(
        chart.binary_center_coefficients,
        parameter,
    )
    binary_center_velocity = _evaluate_fraction_coefficients(
        chart.binary_center_velocity_coefficients,
        parameter,
    )
    third_offset = _evaluate_fraction_coefficients(
        chart.third_offset_coefficients,
        parameter,
    )
    third_offset_velocity = _evaluate_fraction_coefficients(
        chart.third_offset_velocity_coefficients,
        parameter,
    )
    relative_position = _exact_rational_ks_project(u)
    relative_velocity = _exact_rational_ks_velocity(u, u_velocity)
    positions = _fraction_zero_array((3, 3))
    velocities = _fraction_zero_array((3, 3))
    positions[first] = binary_center - alpha * relative_position
    positions[second] = binary_center + beta * relative_position
    positions[third] = binary_center + third_offset
    velocities[first] = binary_center_velocity - alpha * relative_velocity
    velocities[second] = binary_center_velocity + beta * relative_velocity
    velocities[third] = binary_center_velocity + third_offset_velocity
    return positions, velocities


def _evaluate_fraction_coefficients(
    coefficients: object,
    parameter: Fraction,
) -> np.ndarray | Fraction:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim == 0:
        return Fraction(float(values))
    if values.ndim == 1:
        out = Fraction(0)
        for coefficient in values[::-1]:
            out = out * parameter + Fraction(float(coefficient))
        return out
    out = _fraction_zero_array(values.shape[1:])
    for coefficient in values[::-1]:
        out = out * parameter + _fraction_array(coefficient)
    return out


def _fraction_array(values: object) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    out = np.empty(numeric.shape, dtype=object)
    for index in np.ndindex(numeric.shape):
        out[index] = Fraction(float(numeric[index]))
    return out


def _fraction_zero_array(shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = Fraction(0)
    return out


def _exact_rational_lc_square(z: np.ndarray) -> np.ndarray:
    x, y = z
    return np.array((x * x - y * y, 2 * x * y), dtype=object)


def _exact_rational_lc_velocity(
    z: np.ndarray,
    z_velocity: np.ndarray,
) -> np.ndarray:
    x, y = z
    vx, vy = z_velocity
    rho = x * x + y * y
    if rho == 0:
        raise ValueError("LC physical velocity is singular at binary collision")
    return np.array(
        (
            (2 * x * vx - 2 * y * vy) / rho,
            (2 * y * vx + 2 * x * vy) / rho,
        ),
        dtype=object,
    )


def _exact_rational_ks_project(u: np.ndarray) -> np.ndarray:
    a, b, c, d = u
    return np.array(
        (
            a * a - b * b - c * c + d * d,
            2 * (a * b - c * d),
            2 * (a * c + b * d),
        ),
        dtype=object,
    )


def _exact_rational_ks_velocity(
    u: np.ndarray,
    u_velocity: np.ndarray,
) -> np.ndarray:
    a, b, c, d = u
    va, vb, vc, vd = u_velocity
    rho = a * a + b * b + c * c + d * d
    if rho == 0:
        raise ValueError("KS physical velocity is singular at binary collision")
    return np.array(
        (
            (2 * a * va - 2 * b * vb - 2 * c * vc + 2 * d * vd) / rho,
            (2 * b * va + 2 * a * vb - 2 * d * vc - 2 * c * vd) / rho,
            (2 * c * va + 2 * d * vb + 2 * a * vc + 2 * b * vd) / rho,
        ),
        dtype=object,
    )


def _fraction_array_max_abs_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> Fraction:
    if left.shape != right.shape:
        raise ValueError("fraction arrays have incompatible shapes")
    maximum = Fraction(0)
    for index in np.ndindex(left.shape):
        maximum = max(maximum, abs(left[index] - right[index]))
    return maximum


def _sampled_physical_time_contained(
    physical_time_coefficients: np.ndarray,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    sample_count: int,
) -> bool:
    if physical_time_coefficients.ndim != 1:
        return False
    lower, upper = physical_time_interval
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        value = _evaluate_scalar_coefficients(
            physical_time_coefficients,
            float(parameter),
        )
        if not (lower <= value <= upper):
            return False
    return True


def _interval_physical_time_contained(
    physical_time_coefficients: np.ndarray,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    tolerance: float,
    pieces: int = 64,
) -> tuple[bool, tuple[float, float]]:
    """Enclose a serialized physical-time polynomial over the full chart slab.

    This is intentionally independent of the certificate sample count.  The
    checker subdivides the serialized parameter interval exactly as rational
    subintervals of the binary-float endpoints, then evaluates the polynomial
    with rational interval Horner arithmetic on every piece.
    """

    coefficients = np.asarray(physical_time_coefficients, dtype=float)
    if (
        coefficients.ndim != 1
        or not np.all(np.isfinite(coefficients))
        or not _finite_nonempty_interval(parameter_interval)
        or not _finite_nonempty_interval(physical_time_interval)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
        or pieces <= 0
    ):
        return False, (float("inf"), float("inf"))
    try:
        parameter = RationalInterval.from_float_interval(
            float(parameter_interval[0]),
            float(parameter_interval[1]),
        )
        coefficient_intervals = tuple(
            RationalInterval.from_float_interval(float(coefficient))
            for coefficient in coefficients
        )
        width = parameter.upper - parameter.lower
        lower = None
        upper = None
        piece_count = int(pieces)
        for index in range(piece_count):
            left = parameter.lower + width * Fraction(index, piece_count)
            right = parameter.lower + width * Fraction(index + 1, piece_count)
            value = rational_interval_polynomial_eval(
                coefficient_intervals,
                RationalInterval(left, right),
            )
            lower = value.lower if lower is None else min(lower, value.lower)
            upper = value.upper if upper is None else max(upper, value.upper)
        if lower is None or upper is None:
            return False, (float("inf"), float("inf"))
        enclosure = RationalInterval(lower, upper)
        outward = enclosure.as_float_interval().as_tuple()
        tol = RationalInterval.from_float_interval(float(tolerance))
        allowed_lower = (
            RationalInterval.from_float_interval(float(physical_time_interval[0])).lower
            - tol.upper
        )
        allowed_upper = (
            RationalInterval.from_float_interval(float(physical_time_interval[1])).upper
            + tol.upper
        )
        contained = bool(
            enclosure.lower >= allowed_lower
            and enclosure.upper <= allowed_upper
        )
        return contained, outward
    except (FloatingPointError, OverflowError, ValueError):
        return False, (float("inf"), float("inf"))


def _ordinary_unit_physical_parameter_speed(
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    tolerance: float,
) -> bool:
    """Check that an ordinary chart parameter is physical time up to translation."""

    parameter_width = parameter_interval[1] - parameter_interval[0]
    physical_width = physical_time_interval[1] - physical_time_interval[0]
    if parameter_width < 0.0 or physical_width < 0.0:
        return False
    return bool(abs(parameter_width - physical_width) <= float(tolerance))


def _physical_time_in_chart(
    chart: OrdinaryTaylorChartCertificate,
    physical_time: float,
) -> bool:
    lower, upper = chart.physical_time_interval
    return bool(np.isfinite(physical_time) and lower <= physical_time <= upper)


def _time_in_interval(interval: tuple[float, float], value: float) -> bool:
    lower, upper = interval
    return bool(np.isfinite(value) and lower <= value <= upper)


def _parameter_in_interval(interval: tuple[float, float], value: float) -> bool:
    lower, upper = interval
    return bool(np.isfinite(value) and lower <= value <= upper)


def _physical_to_parameter(
    chart: OrdinaryTaylorChartCertificate,
    physical_time: float,
) -> float:
    physical_lower, physical_upper = chart.physical_time_interval
    parameter_lower, parameter_upper = chart.parameter_interval
    physical_width = physical_upper - physical_lower
    parameter_width = parameter_upper - parameter_lower
    if physical_width == 0.0:
        if physical_time != physical_lower:
            raise ValueError("handoff time is outside point chart")
        return float(parameter_lower)
    return float(
        parameter_lower
        + (physical_time - physical_lower) * parameter_width / physical_width
    )


def _chart_physical_time_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: float,
) -> float:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        physical_lower, physical_upper = chart.physical_time_interval
        parameter_lower, parameter_upper = chart.parameter_interval
        parameter_width = parameter_upper - parameter_lower
        physical_width = physical_upper - physical_lower
        if parameter_width == 0.0:
            if parameter != parameter_lower:
                raise ValueError("parameter is outside point chart")
            return float(physical_lower)
        return float(physical_lower + (parameter - parameter_lower) * physical_width / parameter_width)
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _evaluate_scalar_coefficients(
            _coefficient_array(chart.physical_time_coefficients),
            parameter,
        )
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _evaluate_scalar_coefficients(
            _coefficient_array(chart.physical_time_coefficients),
            parameter,
        )
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _chart_projected_state_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        q = _evaluate_coefficients(
            _coefficient_array(chart.position_coefficients),
            parameter,
        )
        v = _evaluate_coefficients(
            _coefficient_array(chart.velocity_coefficients),
            parameter,
        )
        return q, v
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        solution = _regularized_binary_solution_from_certificate(chart)
        state = solution.state_at(parameter)
        return regularized_binary_collision_chart_to_planar(state)
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        solution = _spatial_ks_solution_from_certificate(chart)
        state = solution.state_at(parameter)
        return ks_binary_chart_to_spatial(state)
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _initial_noncollision(positions: np.ndarray) -> bool:
    for i in range(3):
        for j in range(i + 1, 3):
            if not np.linalg.norm(positions[i] - positions[j]) > 0.0:
                return False
    return True


def _minimum_pair_distance(positions: np.ndarray) -> float:
    if np.asarray(positions).ndim != 2 or positions.shape[0] < 2:
        return float("inf")
    return float(
        min(
            np.linalg.norm(positions[j] - positions[i])
            for i in range(positions.shape[0])
            for j in range(i + 1, positions.shape[0])
        )
    )


def _max_coefficient_recurrence_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
) -> float:
    order = q.shape[0] - 1
    acc = acceleration_coefficients(q, masses, order - 1)
    position_residual = 0.0
    velocity_residual = 0.0
    for n in range(order):
        position_residual = max(
            position_residual,
            float(np.max(np.abs((n + 1) * q[n + 1] - v[n]))),
        )
        velocity_residual = max(
            velocity_residual,
            float(np.max(np.abs((n + 1) * v[n + 1] - acc[n]))),
        )
    return max(position_residual, velocity_residual)


def _max_regularized_binary_coefficient_residual(
    solution: RegularizedBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = lc_regularized_rhs_coefficients(solution, n)
        worst = max(
            worst,
            _array_sup_norm((n + 1) * solution.z[n + 1] - rhs.z[n]),
            _array_sup_norm(
                (n + 1) * solution.z_velocity[n + 1] - rhs.z_velocity[n],
            ),
            abs(float((n + 1) * solution.pair_energy[n + 1] - rhs.pair_energy[n])),
            _array_sup_norm(
                (n + 1) * solution.binary_center[n + 1] - rhs.binary_center[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.binary_center_velocity[n + 1]
                - rhs.binary_center_velocity[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset[n + 1] - rhs.third_offset[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset_velocity[n + 1]
                - rhs.third_offset_velocity[n],
            ),
            abs(float((n + 1) * solution.physical_time[n + 1] - rhs.physical_time[n])),
        )
    return float(worst)


def _max_interval_planar_lc_taylor_model_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = solution.order
    rhs = lc_regularized_rhs_coefficients(solution, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    residual_blocks = (
        _lc_vector_residual_coefficients(solution.z, rhs.z, order),
        _lc_vector_residual_coefficients(
            solution.z_velocity,
            rhs.z_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.pair_energy,
            rhs.pair_energy,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center,
            rhs.binary_center,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center_velocity,
            rhs.binary_center_velocity,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset,
            rhs.third_offset,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset_velocity,
            rhs.third_offset_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.physical_time,
            rhs.physical_time,
            order,
        ),
    )
    worst = RationalInterval.point(0).upper
    for block in residual_blocks:
        block = np.asarray(block, dtype=float)
        if block.ndim == 1:
            worst = max(
                worst,
                _rational_interval_polynomial_abs_sup(block, variable),
            )
        else:
            for index in np.ndindex(block.shape[1:]):
                worst = max(
                    worst,
                    _rational_interval_polynomial_abs_sup(
                        block[(slice(None), *index)],
                        variable,
                    ),
                )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _max_interval_planar_lc_projected_newton_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
    rho_lower_bound: float,
) -> float:
    """Conservative interval check for projected Newton residuals.

    This evaluates the LC chart projection and Newton acceleration over the
    whole parameter interval using interval arithmetic.  It is intentionally a
    small trusted-checker slice: it only certifies slabs whose entire LC rho
    interval is away from the binary collision floor.
    """

    variable = FloatInterval(float(parameter_interval[0]), float(parameter_interval[1]))
    z = interval_array_series_eval(solution.z, variable)
    z_velocity = interval_array_series_eval(solution.z_velocity, variable)
    pair_energy = interval_polynomial_eval(solution.pair_energy, variable)
    binary_center = interval_array_series_eval(solution.binary_center, variable)
    binary_center_velocity = interval_array_series_eval(
        solution.binary_center_velocity,
        variable,
    )
    third_offset = interval_array_series_eval(solution.third_offset, variable)
    third_offset_velocity = interval_array_series_eval(
        solution.third_offset_velocity,
        variable,
    )
    rho = _interval_dot(z, z)
    if rho.lower <= max(0.0, float(rho_lower_bound)):
        return float("inf")

    center_acceleration, third_offset_acceleration, relative_perturbation = (
        _interval_planar_lc_analytic_coordinate_accelerations(
            solution,
            z,
            binary_center,
            third_offset,
        )
    )
    z_acceleration = _interval_vector_add(
        _interval_vector_scale_by_interval(z, pair_energy.scale(0.5)),
        _interval_vector_scale_by_interval(
            _interval_lc_transpose_matvec(z, relative_perturbation),
            rho.scale(0.25),
        ),
    )
    projected = _interval_planar_lc_projected_accelerations(
        solution,
        z,
        z_velocity,
        z_acceleration,
        center_acceleration,
        third_offset_acceleration,
    )
    positions = _interval_planar_lc_positions(
        solution,
        z,
        binary_center,
        third_offset,
    )
    newton = _interval_newton_accelerations(positions, solution.masses)
    worst = 0.0
    for index in np.ndindex(projected.shape):
        worst = max(worst, _interval_abs_sup(projected[index] - newton[index]))
    return float(worst + max(0.0, float(tail_bound)))


def _interval_planar_lc_analytic_coordinate_accelerations(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    relative_position = _interval_lc_square(z)
    from_first_to_third = _interval_vector_add(
        third_offset,
        _interval_vector_scale(relative_position, alpha),
    )
    from_second_to_third = _interval_vector_sub(
        third_offset,
        _interval_vector_scale(relative_position, beta),
    )
    field_first = _interval_inverse_square_field(from_first_to_third)
    field_second = _interval_inverse_square_field(from_second_to_third)
    binary_center_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        float(masses[third] / pair_mass),
    )
    third_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        -1.0,
    )
    third_offset_acceleration = _interval_vector_sub(
        third_acceleration,
        binary_center_acceleration,
    )
    relative_perturbation = _interval_vector_scale(
        _interval_vector_sub(field_second, field_first),
        float(masses[third]),
    )
    return (
        binary_center_acceleration,
        third_offset_acceleration,
        relative_perturbation,
    )


def _interval_planar_lc_positions(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_position = _interval_lc_square(z)
    positions = _interval_zero_array((3, 2))
    positions[first] = _interval_vector_sub(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[second] / pair_mass)),
    )
    positions[second] = _interval_vector_add(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[first] / pair_mass)),
    )
    positions[third] = _interval_vector_add(binary_center, third_offset)
    return positions


def _interval_planar_lc_projected_accelerations(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    z_velocity: np.ndarray,
    z_acceleration: np.ndarray,
    binary_center_acceleration: np.ndarray,
    third_offset_acceleration: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_acceleration = _interval_relative_acceleration_from_lc_acceleration(
        z,
        z_velocity,
        z_acceleration,
    )
    acceleration = _interval_zero_array((3, 2))
    acceleration[first] = _interval_vector_sub(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[second] / pair_mass)),
    )
    acceleration[second] = _interval_vector_add(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[first] / pair_mass)),
    )
    acceleration[third] = _interval_vector_add(
        binary_center_acceleration,
        third_offset_acceleration,
    )
    return acceleration


def _lc_vector_residual_coefficients(
    values: np.ndarray,
    rhs_values: np.ndarray,
    order: int,
) -> np.ndarray:
    return np.asarray(
        [
            (n + 1) * values[n + 1] - rhs_values[n]
            for n in range(order)
        ],
        dtype=float,
    )


def _lc_scalar_residual_coefficients(
    values: np.ndarray,
    rhs_values: np.ndarray,
    order: int,
) -> np.ndarray:
    return np.asarray(
        [
            float((n + 1) * values[n + 1] - rhs_values[n])
            for n in range(order)
        ],
        dtype=float,
    )


def _max_spatial_ks_coefficient_residual(
    solution: SpatialKSBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = ks_regularized_rhs_coefficients(solution, n)
        worst = max(
            worst,
            _array_sup_norm((n + 1) * solution.u[n + 1] - rhs.u[n]),
            _array_sup_norm(
                (n + 1) * solution.u_velocity[n + 1] - rhs.u_velocity[n],
            ),
            abs(float((n + 1) * solution.pair_energy[n + 1] - rhs.pair_energy[n])),
            _array_sup_norm(
                (n + 1) * solution.binary_center[n + 1] - rhs.binary_center[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.binary_center_velocity[n + 1]
                - rhs.binary_center_velocity[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset[n + 1] - rhs.third_offset[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset_velocity[n + 1]
                - rhs.third_offset_velocity[n],
            ),
            abs(float((n + 1) * solution.physical_time[n + 1] - rhs.physical_time[n])),
        )
    return float(worst)


def _max_spatial_ks_relative_coefficient_residual(
    solution: SpatialKSBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = ks_regularized_rhs_coefficients(solution, n)
        blocks = (
            ((n + 1) * solution.u[n + 1], rhs.u[n]),
            ((n + 1) * solution.u_velocity[n + 1], rhs.u_velocity[n]),
            (
                np.array([(n + 1) * solution.pair_energy[n + 1]], dtype=float),
                np.array([rhs.pair_energy[n]], dtype=float),
            ),
            ((n + 1) * solution.binary_center[n + 1], rhs.binary_center[n]),
            (
                (n + 1) * solution.binary_center_velocity[n + 1],
                rhs.binary_center_velocity[n],
            ),
            ((n + 1) * solution.third_offset[n + 1], rhs.third_offset[n]),
            (
                (n + 1) * solution.third_offset_velocity[n + 1],
                rhs.third_offset_velocity[n],
            ),
            (
                np.array([(n + 1) * solution.physical_time[n + 1]], dtype=float),
                np.array([rhs.physical_time[n]], dtype=float),
            ),
        )
        for left, right in blocks:
            left_array = np.asarray(left, dtype=float)
            right_array = np.asarray(right, dtype=float)
            residual = float(np.max(np.abs(left_array - right_array)))
            scale = max(
                1.0,
                float(np.max(np.abs(left_array))),
                float(np.max(np.abs(right_array))),
            )
            worst = max(worst, residual / scale)
    return float(worst)


def _max_interval_spatial_ks_constraint_residuals(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
) -> tuple[float, float]:
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    pair_energy_constraint = ks_pair_energy_constraint_coefficients(
        solution,
        solution.order,
    )
    horizontal_constraint = ks_horizontal_constraint_coefficients(
        solution,
        solution.order,
    )
    return (
        float(
            _rational_interval_polynomial_abs_sup(
                pair_energy_constraint,
                variable,
            )
        ),
        float(
            _rational_interval_polynomial_abs_sup(
                horizontal_constraint,
                variable,
            )
        ),
    )


def _max_interval_spatial_ks_taylor_model_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = solution.order
    rhs = ks_regularized_rhs_coefficients(solution, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    residual_blocks = (
        _lc_vector_residual_coefficients(solution.u, rhs.u, order),
        _lc_vector_residual_coefficients(
            solution.u_velocity,
            rhs.u_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.pair_energy,
            rhs.pair_energy,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center,
            rhs.binary_center,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center_velocity,
            rhs.binary_center_velocity,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset,
            rhs.third_offset,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset_velocity,
            rhs.third_offset_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.physical_time,
            rhs.physical_time,
            order,
        ),
    )
    worst = RationalInterval.point(0).upper
    for block in residual_blocks:
        block = np.asarray(block, dtype=float)
        if block.ndim == 1:
            worst = max(
                worst,
                _rational_interval_polynomial_abs_sup(block, variable),
            )
        else:
            for index in np.ndindex(block.shape[1:]):
                worst = max(
                    worst,
                    _rational_interval_polynomial_abs_sup(
                        block[(slice(None), *index)],
                        variable,
                    ),
                )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _max_interval_spatial_ks_projected_newton_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
    rho_lower_bound: float,
    pieces: int = 128,
) -> float:
    """Conservative projected residual check on nonsingular KS slabs.

    A regularized KS chart may legitimately pass through binary collision,
    where the projected Newtonian coordinates are singular.  The regularized
    RHS is checked on the full chart elsewhere; this projected obligation is
    only meaningful on resolved subintervals whose interval ``rho`` stays
    bounded away from zero.
    """

    lower = float(parameter_interval[0])
    upper = float(parameter_interval[1])
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        return float("inf")
    pieces = max(1, int(pieces))
    worst = 0.0
    checked_piece_count = 0
    for index in range(pieces):
        piece_lower = lower + (upper - lower) * index / pieces
        piece_upper = lower + (upper - lower) * (index + 1) / pieces
        try:
            piece_value = _interval_spatial_ks_projected_newton_residual_piece(
                solution,
                FloatInterval(piece_lower, piece_upper),
                tail_bound=tail_bound,
                rho_lower_bound=rho_lower_bound,
            )
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(piece_value):
            worst = max(worst, piece_value)
            checked_piece_count += 1
    if checked_piece_count == 0:
        return float("inf")
    return float(worst)


def _interval_spatial_ks_projected_newton_residual_piece(
    solution: SpatialKSBinaryTaylorSolution,
    variable: FloatInterval,
    *,
    tail_bound: float,
    rho_lower_bound: float,
) -> float:
    """Projected Newton residual over one nonsingular KS interval piece."""

    u = interval_array_series_eval(solution.u, variable)
    u_velocity = interval_array_series_eval(solution.u_velocity, variable)
    pair_energy = interval_polynomial_eval(solution.pair_energy, variable)
    binary_center = interval_array_series_eval(solution.binary_center, variable)
    third_offset = interval_array_series_eval(solution.third_offset, variable)
    rho = _interval_dot(u, u)
    if rho.lower <= max(0.0, float(rho_lower_bound)):
        return float("inf")

    center_acceleration, third_offset_acceleration, relative_perturbation = (
        _interval_spatial_ks_analytic_coordinate_accelerations(
            solution,
            u,
            third_offset,
        )
    )
    u_acceleration = _interval_vector_add(
        _interval_vector_scale_by_interval(u, pair_energy.scale(0.5)),
        _interval_vector_scale_by_interval(
            _interval_ks_transpose_matvec(u, relative_perturbation),
            rho.scale(0.25),
        ),
    )
    projected = _interval_spatial_ks_projected_accelerations(
        solution,
        u,
        u_velocity,
        u_acceleration,
        center_acceleration,
        third_offset_acceleration,
    )
    positions = _interval_spatial_ks_positions(
        solution,
        u,
        binary_center,
        third_offset,
    )
    newton = _interval_newton_accelerations(positions, solution.masses)
    worst = 0.0
    for index in np.ndindex(projected.shape):
        worst = max(worst, _interval_abs_sup(projected[index] - newton[index]))
    return float(worst + max(0.0, float(tail_bound)))


def _interval_spatial_ks_analytic_coordinate_accelerations(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    third_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    relative_position = _interval_ks_project(u)
    from_first_to_third = _interval_vector_add(
        third_offset,
        _interval_vector_scale(relative_position, alpha),
    )
    from_second_to_third = _interval_vector_sub(
        third_offset,
        _interval_vector_scale(relative_position, beta),
    )
    field_first = _interval_inverse_square_field(from_first_to_third)
    field_second = _interval_inverse_square_field(from_second_to_third)
    binary_center_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        float(masses[third] / pair_mass),
    )
    third_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        -1.0,
    )
    third_offset_acceleration = _interval_vector_sub(
        third_acceleration,
        binary_center_acceleration,
    )
    relative_perturbation = _interval_vector_scale(
        _interval_vector_sub(field_second, field_first),
        float(masses[third]),
    )
    return (
        binary_center_acceleration,
        third_offset_acceleration,
        relative_perturbation,
    )


def _interval_spatial_ks_positions(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_position = _interval_ks_project(u)
    positions = _interval_zero_array((3, 3))
    positions[first] = _interval_vector_sub(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[second] / pair_mass)),
    )
    positions[second] = _interval_vector_add(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[first] / pair_mass)),
    )
    positions[third] = _interval_vector_add(binary_center, third_offset)
    return positions


def _interval_spatial_ks_projected_accelerations(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    u_velocity: np.ndarray,
    u_acceleration: np.ndarray,
    binary_center_acceleration: np.ndarray,
    third_offset_acceleration: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_acceleration = _interval_relative_acceleration_from_ks_acceleration(
        u,
        u_velocity,
        u_acceleration,
    )
    acceleration = _interval_zero_array((3, 3))
    acceleration[first] = _interval_vector_sub(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[second] / pair_mass)),
    )
    acceleration[second] = _interval_vector_add(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[first] / pair_mass)),
    )
    acceleration[third] = _interval_vector_add(
        binary_center_acceleration,
        third_offset_acceleration,
    )
    return acceleration


def _max_sampled_newton_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    q_derivative = _derivative_coefficients(q)
    v_derivative = _derivative_coefficients(v)
    worst = 0.0
    for time in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        positions = _evaluate_coefficients(q, float(time))
        velocities = _evaluate_coefficients(v, float(time))
        dqdt = _evaluate_coefficients(q_derivative, float(time))
        dvdt = _evaluate_coefficients(v_derivative, float(time))
        acceleration = accelerations(positions, masses)
        worst = max(
            worst,
            float(np.max(np.abs(dqdt - velocities))),
            float(np.max(np.abs(dvdt - acceleration))),
        )
    return worst


def _max_interval_ordinary_newton_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
) -> float:
    variable = FloatInterval(float(parameter_interval[0]), float(parameter_interval[1]))
    positions = interval_array_series_eval(q, variable)
    velocities = interval_array_series_eval(v, variable)
    position_derivative = interval_array_series_eval(
        _derivative_coefficients(q),
        variable,
    )
    velocity_derivative = interval_array_series_eval(
        _derivative_coefficients(v),
        variable,
    )
    acceleration = _interval_newton_accelerations(positions, masses)
    return max(
        _interval_array_difference_sup(position_derivative, velocities),
        _interval_array_difference_sup(velocity_derivative, acceleration),
    )


def _max_interval_ordinary_taylor_model_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = q.shape[0] - 1
    acc = acceleration_coefficients(q, masses, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    worst = RationalInterval.point(0).upper
    for index in np.ndindex(q.shape[1:]):
        position_residual = tuple(
            RationalInterval.from_float_interval(
                float((n + 1) * q[(n + 1, *index)] - v[(n, *index)])
            )
            for n in range(order)
        )
        velocity_residual = tuple(
            RationalInterval.from_float_interval(
                float((n + 1) * v[(n + 1, *index)] - acc[(n, *index)])
            )
            for n in range(order)
        )
        worst = max(
            worst,
            _rational_interval_abs_sup(
                rational_interval_polynomial_eval(position_residual, variable)
            ),
            _rational_interval_abs_sup(
                rational_interval_polynomial_eval(velocity_residual, variable)
            ),
        )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _interval_newton_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != masses.size:
        raise ValueError("position interval array shape does not match masses")
    accelerations_out = np.empty(positions.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        accelerations_out[index] = FloatInterval.point(0.0)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i == j:
                continue
            diff = tuple(positions[j, axis] - positions[i, axis] for axis in range(positions.shape[1]))
            distance_squared = FloatInterval.point(0.0)
            for component in diff:
                distance_squared = distance_squared + _interval_square(component)
            inv_distance_cubed = distance_squared.positive_power(-1.5)
            for axis, component in enumerate(diff):
                accelerations_out[i, axis] = (
                    accelerations_out[i, axis]
                    + component * inv_distance_cubed.scale(float(masses[j]))
                )
    return accelerations_out


def _interval_square(interval: FloatInterval) -> FloatInterval:
    if interval.lower <= 0.0 <= interval.upper:
        lower = 0.0
        upper = max(interval.lower * interval.lower, interval.upper * interval.upper)
    else:
        values = (interval.lower * interval.lower, interval.upper * interval.upper)
        lower = min(values)
        upper = max(values)
    return FloatInterval(
        float(np.nextafter(lower, -np.inf)),
        float(np.nextafter(upper, np.inf)),
    )


def _interval_array_difference_sup(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval arrays must have matching shapes")
    worst = 0.0
    for index in np.ndindex(left.shape):
        interval = left[index] - right[index]
        worst = max(worst, _interval_abs_sup(interval))
    return float(worst)


def _interval_abs_sup(interval: FloatInterval) -> float:
    return float(max(abs(interval.lower), abs(interval.upper)))


def _rational_interval_abs_sup(interval: RationalInterval):
    return max(abs(interval.lower), abs(interval.upper))


def _rational_interval_polynomial_abs_sup(
    coefficients: np.ndarray,
    variable: RationalInterval,
):
    coefficient_intervals = tuple(
        RationalInterval.from_float_interval(float(coefficient))
        for coefficient in np.asarray(coefficients, dtype=float)
    )
    return _rational_interval_abs_sup(
        rational_interval_polynomial_eval(coefficient_intervals, variable)
    )


def _interval_zero_array(shape: tuple[int, ...]) -> np.ndarray:
    values = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        values[index] = FloatInterval.point(0.0)
    return values


def _interval_vector_add(*vectors: np.ndarray) -> np.ndarray:
    if not vectors:
        raise ValueError("at least one vector is required")
    out = _interval_zero_array(np.asarray(vectors[0], dtype=object).shape)
    for vector in vectors:
        vector = np.asarray(vector, dtype=object)
        if vector.shape != out.shape:
            raise ValueError("interval vectors must have matching shapes")
        for index in np.ndindex(out.shape):
            out[index] = out[index] + _coerce_float_interval(vector[index])
    return out


def _interval_vector_sub(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    out = _interval_zero_array(left.shape)
    for index in np.ndindex(left.shape):
        out[index] = (
            _coerce_float_interval(left[index])
            - _coerce_float_interval(right[index])
        )
    return out


def _interval_vector_scale(vector: np.ndarray, factor: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=object)
    out = _interval_zero_array(vector.shape)
    for index in np.ndindex(vector.shape):
        out[index] = _coerce_float_interval(vector[index]).scale(float(factor))
    return out


def _interval_vector_scale_by_interval(
    vector: np.ndarray,
    factor: FloatInterval,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=object)
    factor = _coerce_float_interval(factor)
    out = _interval_zero_array(vector.shape)
    for index in np.ndindex(vector.shape):
        out[index] = _coerce_float_interval(vector[index]) * factor
    return out


def _interval_dot(left: np.ndarray, right: np.ndarray) -> FloatInterval:
    left = np.asarray(left, dtype=object).reshape(-1)
    right = np.asarray(right, dtype=object).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    value = FloatInterval.point(0.0)
    for index in range(left.shape[0]):
        value = value + (
            _coerce_float_interval(left[index])
            * _coerce_float_interval(right[index])
        )
    return value


def _interval_centered_rows(values: np.ndarray, masses: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    center = _interval_zero_array((values.shape[1],))
    for body, mass in enumerate(masses):
        for axis in range(values.shape[1]):
            center[axis] = center[axis] + _coerce_float_interval(
                values[body, axis],
            ).scale(float(mass / total_mass))
    out = _interval_zero_array(values.shape)
    for body in range(values.shape[0]):
        for axis in range(values.shape[1]):
            out[body, axis] = _coerce_float_interval(values[body, axis]) - center[axis]
    return out


def _interval_planar_wedge(left: np.ndarray, right: np.ndarray) -> FloatInterval:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("planar wedge expects two 2D vectors")
    return _coerce_float_interval(left[0]) * _coerce_float_interval(
        right[1],
    ) - _coerce_float_interval(left[1]) * _coerce_float_interval(right[0])


def _interval_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("cross product expects two 3D vectors")
    return np.array(
        [
            _coerce_float_interval(left[1]) * _coerce_float_interval(right[2])
            - _coerce_float_interval(left[2]) * _coerce_float_interval(right[1]),
            _coerce_float_interval(left[2]) * _coerce_float_interval(right[0])
            - _coerce_float_interval(left[0]) * _coerce_float_interval(right[2]),
            _coerce_float_interval(left[0]) * _coerce_float_interval(right[1])
            - _coerce_float_interval(left[1]) * _coerce_float_interval(right[0]),
        ],
        dtype=object,
    )


def _coerce_float_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_lc_square(z: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    return np.array(
        [
            x * x - y * y,
            (x * y).scale(2.0),
        ],
        dtype=object,
    )


def _interval_lc_matvec(z: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    return np.array(
        [
            x.scale(2.0) * v0 - y.scale(2.0) * v1,
            y.scale(2.0) * v0 + x.scale(2.0) * v1,
        ],
        dtype=object,
    )


def _interval_lc_transpose_matvec(z: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    return np.array(
        [
            x.scale(2.0) * v0 + y.scale(2.0) * v1,
            y.scale(-2.0) * v0 + x.scale(2.0) * v1,
        ],
        dtype=object,
    )


def _interval_relative_acceleration_from_lc_acceleration(
    z: np.ndarray,
    z_velocity: np.ndarray,
    z_acceleration: np.ndarray,
) -> np.ndarray:
    rho = _interval_dot(z, z)
    if rho.lower <= 0.0:
        raise ValueError("LC projection interval reaches binary collision")
    rho_inv = rho.reciprocal()
    rho_inv_squared = rho_inv * rho_inv
    rho_prime = _interval_dot(z, z_velocity).scale(2.0)
    q_prime = _interval_lc_matvec(z, z_velocity)
    velocity_prime = _interval_vector_sub(
        _interval_vector_scale_by_interval(
            _interval_vector_add(
                _interval_lc_matvec(z_velocity, z_velocity),
                _interval_lc_matvec(z, z_acceleration),
            ),
            rho_inv,
        ),
        _interval_vector_scale_by_interval(
            q_prime,
            rho_prime * rho_inv_squared,
        ),
    )
    return _interval_vector_scale_by_interval(velocity_prime, rho_inv)


def _interval_ks_project(u: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    return np.array(
        [
            a * a - b * b - c * c + d * d,
            (a * b - c * d).scale(2.0),
            (a * c + b * d).scale(2.0),
        ],
        dtype=object,
    )


def _interval_ks_matvec(u: np.ndarray, vector: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    v2 = _coerce_float_interval(vector[2])
    v3 = _coerce_float_interval(vector[3])
    return np.array(
        [
            a.scale(2.0) * v0
            - b.scale(2.0) * v1
            - c.scale(2.0) * v2
            + d.scale(2.0) * v3,
            b.scale(2.0) * v0
            + a.scale(2.0) * v1
            - d.scale(2.0) * v2
            - c.scale(2.0) * v3,
            c.scale(2.0) * v0
            + d.scale(2.0) * v1
            + a.scale(2.0) * v2
            + b.scale(2.0) * v3,
        ],
        dtype=object,
    )


def _interval_ks_transpose_matvec(u: np.ndarray, vector: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    x = _coerce_float_interval(vector[0])
    y = _coerce_float_interval(vector[1])
    z = _coerce_float_interval(vector[2])
    return np.array(
        [
            a.scale(2.0) * x + b.scale(2.0) * y + c.scale(2.0) * z,
            b.scale(-2.0) * x + a.scale(2.0) * y + d.scale(2.0) * z,
            c.scale(-2.0) * x + d.scale(-2.0) * y + a.scale(2.0) * z,
            d.scale(2.0) * x + c.scale(-2.0) * y + b.scale(2.0) * z,
        ],
        dtype=object,
    )


def _interval_relative_acceleration_from_ks_acceleration(
    u: np.ndarray,
    u_velocity: np.ndarray,
    u_acceleration: np.ndarray,
) -> np.ndarray:
    rho = _interval_dot(u, u)
    if rho.lower <= 0.0:
        raise ValueError("KS projection interval reaches binary collision")
    rho_inv = rho.reciprocal()
    rho_inv_squared = rho_inv * rho_inv
    rho_inv_cubed = rho_inv_squared * rho_inv
    rho_prime = _interval_dot(u, u_velocity).scale(2.0)
    q_prime = _interval_ks_matvec(u, u_velocity)
    q_second = _interval_vector_add(
        _interval_ks_matvec(u_velocity, u_velocity),
        _interval_ks_matvec(u, u_acceleration),
    )
    return _interval_vector_sub(
        _interval_vector_scale_by_interval(q_second, rho_inv_squared),
        _interval_vector_scale_by_interval(
            q_prime,
            rho_prime * rho_inv_cubed,
        ),
    )


def _interval_inverse_square_field(vector: np.ndarray) -> np.ndarray:
    norm_square = _interval_dot(vector, vector)
    if norm_square.lower <= 0.0:
        raise ValueError("inverse-square field interval reaches collision")
    inverse_cube = norm_square.positive_power(-1.5)
    return _interval_vector_scale_by_interval(vector, inverse_cube)


def _interval_newton_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three positions and three masses")
    accelerations_interval = _interval_zero_array(positions.shape)
    for first in range(3):
        for second in range(first + 1, 3):
            delta = _interval_vector_sub(positions[second], positions[first])
            direction_over_radius_cubed = _interval_inverse_square_field(delta)
            accelerations_interval[first] = _interval_vector_add(
                accelerations_interval[first],
                _interval_vector_scale(
                    direction_over_radius_cubed,
                    float(masses[second]),
                ),
            )
            accelerations_interval[second] = _interval_vector_sub(
                accelerations_interval[second],
                _interval_vector_scale(
                    direction_over_radius_cubed,
                    float(masses[first]),
                ),
            )
    return accelerations_interval


def _max_sampled_regularized_binary_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    derivatives = {
        "z": _derivative_coefficients(solution.z),
        "z_velocity": _derivative_coefficients(solution.z_velocity),
        "pair_energy": _derivative_scalar_coefficients(solution.pair_energy),
        "binary_center": _derivative_coefficients(solution.binary_center),
        "binary_center_velocity": _derivative_coefficients(solution.binary_center_velocity),
        "third_offset": _derivative_coefficients(solution.third_offset),
        "third_offset_velocity": _derivative_coefficients(solution.third_offset_velocity),
        "physical_time": _derivative_scalar_coefficients(solution.physical_time),
    }
    worst = 0.0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        s_value = float(parameter)
        state = solution.state_at(s_value)
        rhs = regularized_binary_collision_chart_rhs(state)
        worst = max(
            worst,
            _array_sup_norm(_evaluate_coefficients(derivatives["z"], s_value) - rhs.z),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["z_velocity"], s_value)
                - rhs.z_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["pair_energy"], s_value)
                - rhs.pair_energy
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center"], s_value)
                - rhs.binary_center,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center_velocity"], s_value)
                - rhs.binary_center_velocity,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["third_offset"], s_value)
                - rhs.third_offset,
            ),
            _array_sup_norm(
                _evaluate_coefficients(
                    derivatives["third_offset_velocity"],
                    s_value,
                )
                - rhs.third_offset_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["physical_time"], s_value)
                - rhs.physical_time
            ),
        )
    return float(worst)


def _max_sampled_projected_binary_newton_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
    rho_lower_bound: float,
) -> tuple[float, int]:
    worst = 0.0
    checked = 0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        state = solution.state_at(float(parameter))
        if state.rho <= rho_lower_bound:
            continue
        derivative = regularized_binary_collision_chart_rhs(state)
        positions, _velocities = regularized_binary_collision_chart_to_planar(state)
        projected_acceleration = planar_accelerations_from_regularized_chart_rhs(
            state,
            derivative,
        )
        newton_acceleration = accelerations(positions, solution.masses)
        worst = max(
            worst,
            _array_sup_norm(projected_acceleration - newton_acceleration),
        )
        checked += 1
    return float(worst if checked else np.inf), checked


def _max_sampled_spatial_ks_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    derivatives = {
        "u": _derivative_coefficients(solution.u),
        "u_velocity": _derivative_coefficients(solution.u_velocity),
        "pair_energy": _derivative_scalar_coefficients(solution.pair_energy),
        "binary_center": _derivative_coefficients(solution.binary_center),
        "binary_center_velocity": _derivative_coefficients(solution.binary_center_velocity),
        "third_offset": _derivative_coefficients(solution.third_offset),
        "third_offset_velocity": _derivative_coefficients(solution.third_offset_velocity),
        "physical_time": _derivative_scalar_coefficients(solution.physical_time),
    }
    worst = 0.0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        s_value = float(parameter)
        state = solution.state_at(s_value)
        rhs = regularized_ks_binary_chart_rhs(state)
        worst = max(
            worst,
            _array_sup_norm(_evaluate_coefficients(derivatives["u"], s_value) - rhs.u),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["u_velocity"], s_value)
                - rhs.u_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["pair_energy"], s_value)
                - rhs.pair_energy
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center"], s_value)
                - rhs.binary_center,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center_velocity"], s_value)
                - rhs.binary_center_velocity,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["third_offset"], s_value)
                - rhs.third_offset,
            ),
            _array_sup_norm(
                _evaluate_coefficients(
                    derivatives["third_offset_velocity"],
                    s_value,
                )
                - rhs.third_offset_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["physical_time"], s_value)
                - rhs.physical_time
            ),
        )
    return float(worst)


def _max_sampled_spatial_ks_projected_newton_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
    rho_lower_bound: float,
) -> tuple[float, int]:
    worst = 0.0
    checked = 0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        state = solution.state_at(float(parameter))
        if state.rho <= rho_lower_bound:
            continue
        derivative = regularized_ks_binary_chart_rhs(state)
        positions, _velocities = ks_binary_chart_to_spatial(state)
        projected_acceleration = spatial_accelerations_from_ks_binary_rhs(
            state,
            derivative,
        )
        newton_acceleration = accelerations(positions, solution.masses)
        worst = max(
            worst,
            _array_sup_norm(projected_acceleration - newton_acceleration),
        )
        checked += 1
    return float(worst if checked else np.inf), checked


def _max_sampled_fuchsian_stop_chart_quantities(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, float, float, float, int]:
    max_lifted_residual = 0.0
    max_projected_residual = 0.0
    max_angular_momentum = 0.0
    max_position_scale = 0.0
    checked = 0
    for tau in _punctured_tau_samples(tau_interval, sample_count=sample_count):
        shape, first, second = branch.tau_derivatives(tau)
        positions = branch.positions_at_tau(tau)
        projected_acceleration = (
            tau**2 * second + 2.0 * tau * first - 2.0 * shape
        ) / (9.0 * tau**4)
        lifted_residual = (
            tau**2 * second
            + 2.0 * tau * first
            - 2.0 * shape
            - 9.0 * accelerations(shape, branch.masses)
        )
        projected_residual = projected_acceleration - accelerations(
            positions,
            branch.masses,
        )
        max_lifted_residual = max(
            max_lifted_residual,
            _array_sup_norm(lifted_residual),
        )
        max_projected_residual = max(
            max_projected_residual,
            _array_sup_norm(projected_residual),
        )
        if branch.central_shape.shape[1] == 2:
            max_angular_momentum = max(
                max_angular_momentum,
                abs(branch.centered_angular_momentum_scalar_at_tau(tau)),
            )
        else:
            max_angular_momentum = max(
                max_angular_momentum,
                _spatial_centered_angular_momentum_norm(branch, tau),
            )
        max_position_scale = max(
            max_position_scale,
            float(np.max(np.abs(positions))) / tau**2,
        )
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, np.inf, np.inf, 0)
    return (
        float(max_lifted_residual),
        float(max_projected_residual),
        float(max_angular_momentum),
        float(max_position_scale),
        int(checked),
    )


def _max_interval_fuchsian_lifted_residual_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for the lifted Fuchsian equation.

    The primitive Cauchy certificate covers the geometric tail near tau=0.
    This checker covers the resolved compact punctured slabs between that
    tail radius and the declared isolation radius, on both time sides.
    """

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        lifted_residual = _interval_fuchsian_lifted_residual(branch, tau_slab)
        for index in np.ndindex(lifted_residual.shape):
            worst = max(worst, _interval_abs_sup(lifted_residual[index]))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_projected_residual_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for projected Fuchsian Newton residuals."""

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        lifted_residual = _interval_fuchsian_lifted_residual(branch, tau_slab)
        tau_fourth = (tau_slab * tau_slab) * (tau_slab * tau_slab)
        scale = tau_fourth.scale(9.0).reciprocal()
        for index in np.ndindex(lifted_residual.shape):
            projected = _coerce_float_interval(lifted_residual[index]) * scale
            worst = max(worst, _interval_abs_sup(projected))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_zero_angular_momentum_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for zero angular momentum near total collision."""

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        angular = _interval_fuchsian_centered_angular_momentum(branch, tau_slab)
        if isinstance(angular, FloatInterval):
            worst = max(worst, _interval_abs_sup(angular))
        else:
            for component in angular:
                worst = max(worst, _interval_abs_sup(component))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, float, str]:
    """Conservative interval COM and total-momentum check on punctured slabs."""

    if primitive_cauchy_inputs is None:
        return (np.inf, np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst_center = 0.0
    worst_momentum = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        center, momentum = _interval_fuchsian_center_of_mass_and_linear_momentum(
            branch,
            tau_slab,
        )
        for component in center:
            worst_center = max(worst_center, _interval_abs_sup(component))
        for component in momentum:
            worst_momentum = max(worst_momentum, _interval_abs_sup(component))
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, "no punctured shell slabs were available")
    return (
        float(worst_center),
        float(worst_momentum),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_generalized_fuchsian_lifted_residual_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        width = (tau_slab.upper - tau_slab.lower) / pieces
        if width <= 0.0:
            continue
        for piece in range(pieces):
            sub_slab = FloatInterval(
                tau_slab.lower + piece * width,
                tau_slab.lower + (piece + 1) * width,
            )
            lifted_residual = _interval_generalized_fuchsian_lifted_residual(
                branch,
                sub_slab,
            )
            for index in np.ndindex(lifted_residual.shape):
                worst = max(worst, _interval_abs_sup(lifted_residual[index]))
            checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}"


def _max_interval_generalized_fuchsian_projected_residual_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        lifted_residual = _interval_generalized_fuchsian_lifted_residual(
            branch,
            tau_slab,
        )
        tau_fourth = (tau_slab * tau_slab) * (tau_slab * tau_slab)
        scale = tau_fourth.scale(9.0).reciprocal()
        for index in np.ndindex(lifted_residual.shape):
            projected = _coerce_float_interval(lifted_residual[index]) * scale
            worst = max(worst, _interval_abs_sup(projected))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_slabs={checked}; {detail}"


def _max_interval_generalized_fuchsian_zero_angular_momentum_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        for sub_slab in _subdivide_float_interval(tau_slab, pieces):
            angular = _interval_generalized_fuchsian_centered_angular_momentum(
                branch,
                sub_slab,
            )
            if isinstance(angular, FloatInterval):
                worst = max(worst, _interval_abs_sup(angular))
            else:
                for component in angular:
                    worst = max(worst, _interval_abs_sup(component))
            checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}"


def _max_interval_generalized_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, float, str]:
    if remainder_majorant is None:
        return (np.inf, np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst_center = 0.0
    worst_momentum = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        for sub_slab in _subdivide_float_interval(tau_slab, pieces):
            center, momentum = (
                _interval_generalized_fuchsian_center_of_mass_and_linear_momentum(
                    branch,
                    sub_slab,
                )
            )
            for component in center:
                worst_center = max(worst_center, _interval_abs_sup(component))
            for component in momentum:
                worst_momentum = max(worst_momentum, _interval_abs_sup(component))
            checked += 1
    if checked == 0:
        return (np.inf, np.inf, "no punctured generalized slabs were available")
    return (
        float(worst_center),
        float(worst_momentum),
        f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}",
    )


def _subdivide_float_interval(
    interval: FloatInterval,
    pieces: int,
) -> tuple[FloatInterval, ...]:
    pieces = int(pieces)
    if pieces <= 0:
        raise ValueError("pieces must be positive")
    width = (interval.upper - interval.lower) / pieces
    if width <= 0.0:
        return ()
    return tuple(
        FloatInterval(
            interval.lower + piece * width,
            interval.lower + (piece + 1) * width,
        )
        for piece in range(pieces)
    )


def _generalized_fuchsian_punctured_tau_slabs(
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    initial_radius: float,
    shell_contraction: float,
) -> tuple[tuple[FloatInterval, ...], str]:
    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius <= radius_upper
    ):
        raise ValueError(
            "invalid generalized Fuchsian punctured shell radii: "
            f"initial_radius={initial_radius}; "
            f"shell_contraction={shell_contraction}; radius_upper={radius_upper}"
        )
    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))
    return (
        tuple(tau_slabs),
        f"initial_radius={initial_radius}; radius_upper={radius_upper}",
    )


def _interval_generalized_fuchsian_lifted_residual(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> np.ndarray:
    shape, first, second = _interval_generalized_fuchsian_tau_derivatives(
        branch,
        tau_slab,
    )
    tau_squared = tau_slab * tau_slab
    left = _interval_vector_add(
        _interval_vector_scale_by_interval(second, tau_squared),
        _interval_vector_scale_by_interval(first, tau_slab.scale(2.0)),
        _interval_vector_scale(shape, -2.0),
    )
    acceleration = _interval_newton_accelerations(shape, branch.masses)
    return _interval_vector_sub(left, _interval_vector_scale(acceleration, 9.0))


def _interval_generalized_fuchsian_center_of_mass_and_linear_momentum(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_shape, weighted_first = (
        _interval_generalized_fuchsian_weighted_shape_and_first(
            branch,
            tau_slab,
        )
    )
    masses = np.asarray(branch.masses, dtype=float)
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    tau_squared = tau_slab * tau_slab
    center = _interval_vector_scale_by_interval(
        weighted_shape,
        tau_squared.scale(1.0 / total_mass),
    )
    numerator = _interval_vector_add(
        _interval_vector_scale_by_interval(weighted_shape, tau_slab.scale(2.0)),
        _interval_vector_scale_by_interval(weighted_first, tau_squared),
    )
    momentum = _interval_vector_scale_by_interval(
        numerator,
        tau_squared.scale(3.0).reciprocal(),
    )
    return center, momentum


def _interval_generalized_fuchsian_weighted_shape_and_first(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("generalized Fuchsian interval check needs a punctured tau slab")
    masses = np.asarray(branch.masses, dtype=float)
    dimension = int(branch.central_shape.shape[1])
    weighted_shape = _interval_zero_array((dimension,))
    weighted_first_radius = _interval_zero_array((dimension,))
    for index, coefficient in branch.coefficients.items():
        exponent = branch.exponent(index)
        if not np.isfinite(exponent):
            raise ValueError("generalized Fuchsian exponent must be finite")
        weighted_coefficient = np.zeros(dimension, dtype=float)
        for body, mass in enumerate(masses):
            weighted_coefficient += float(mass) * np.asarray(
                coefficient[body],
                dtype=float,
            )
        coefficient_interval = _interval_point_array(weighted_coefficient)
        weighted_shape = _interval_vector_add(
            weighted_shape,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent),
            ),
        )
        if index == branch.zero_index:
            continue
        weighted_first_radius = _interval_vector_add(
            weighted_first_radius,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 1.0).scale(exponent),
            ),
        )
    return weighted_shape, _interval_vector_scale(weighted_first_radius, sign)


def _interval_generalized_fuchsian_centered_angular_momentum(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> FloatInterval | np.ndarray:
    shape, first, _second = _interval_generalized_fuchsian_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    centered_shape = _interval_centered_rows(shape, masses)
    centered_first = _interval_centered_rows(first, masses)
    tau_squared_over_three = (tau_slab * tau_slab).scale(1.0 / 3.0)
    if centered_shape.shape[1] == 2:
        angular = FloatInterval.point(0.0)
        for body, mass in enumerate(masses):
            angular = angular + _interval_planar_wedge(
                centered_shape[body],
                centered_first[body],
            ).scale(float(mass))
        return angular * tau_squared_over_three
    if centered_shape.shape[1] == 3:
        angular_vector = _interval_zero_array((3,))
        for body, mass in enumerate(masses):
            angular_vector = _interval_vector_add(
                angular_vector,
                _interval_vector_scale(
                    _interval_cross(centered_shape[body], centered_first[body]),
                    float(mass),
                ),
            )
        return _interval_vector_scale_by_interval(
            angular_vector,
            tau_squared_over_three,
        )
    raise ValueError("generalized Fuchsian angular check supports only 2D and 3D shapes")


def _interval_generalized_fuchsian_tau_derivatives(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("generalized Fuchsian interval check needs a punctured tau slab")
    shape, radius_first, second = _interval_generalized_fuchsian_radius_derivatives(
        branch,
        radius,
    )
    return shape, _interval_vector_scale(radius_first, sign), second


def _interval_generalized_fuchsian_radius_derivatives(
    branch: FuchsianShapeBranch,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius.lower <= 0.0:
        raise ValueError("generalized Fuchsian radius interval must be positive")
    shape = _interval_zero_array(branch.central_shape.shape)
    first = _interval_zero_array(branch.central_shape.shape)
    second = _interval_zero_array(branch.central_shape.shape)
    for index, coefficient in branch.coefficients.items():
        exponent = branch.exponent(index)
        if not np.isfinite(exponent):
            raise ValueError("generalized Fuchsian exponent must be finite")
        coefficient_interval = _interval_point_array(np.asarray(coefficient, dtype=float))
        shape = _interval_vector_add(
            shape,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent),
            ),
        )
        if index == branch.zero_index:
            continue
        first = _interval_vector_add(
            first,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 1.0).scale(exponent),
            ),
        )
        second = _interval_vector_add(
            second,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 2.0).scale(
                    exponent * (exponent - 1.0),
                ),
            ),
        )
    return shape, first, second


def _interval_fuchsian_center_of_mass_and_linear_momentum(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    shape, first, _second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    weighted_shape = _interval_weighted_row_sum(shape, masses)
    weighted_first = _interval_weighted_row_sum(first, masses)
    tau_squared = tau_slab * tau_slab
    center = _interval_vector_scale_by_interval(
        weighted_shape,
        tau_squared.scale(1.0 / total_mass),
    )
    numerator = _interval_vector_add(
        _interval_vector_scale_by_interval(weighted_shape, tau_slab.scale(2.0)),
        _interval_vector_scale_by_interval(weighted_first, tau_squared),
    )
    momentum = _interval_vector_scale_by_interval(
        numerator,
        tau_squared.scale(3.0).reciprocal(),
    )
    return center, momentum


def _interval_weighted_row_sum(values: np.ndarray, masses: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    out = _interval_zero_array((values.shape[1],))
    for body, mass in enumerate(masses):
        for axis in range(values.shape[1]):
            out[axis] = out[axis] + _coerce_float_interval(
                values[body, axis],
            ).scale(float(mass))
    return out


def _interval_weighted_squared_norm(values: np.ndarray, masses: np.ndarray) -> FloatInterval:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    out = FloatInterval.point(0.0)
    for body, mass in enumerate(masses):
        out = out + _interval_dot(values[body], values[body]).scale(float(mass))
    return out


def _interval_pairwise_newtonian_potential(
    positions: np.ndarray,
    masses: np.ndarray,
) -> FloatInterval:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != masses.size:
        raise ValueError("positions must have one row per mass")
    potential = FloatInterval.point(0.0)
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            delta = _interval_vector_sub(positions[second], positions[first])
            radius_squared = _interval_dot(delta, delta)
            inverse_radius = radius_squared.positive_power(-0.5)
            potential = potential + inverse_radius.scale(
                float(masses[first] * masses[second]),
            )
    return potential


def _interval_fuchsian_centered_angular_momentum(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> FloatInterval | np.ndarray:
    shape, first, _second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    centered_shape = _interval_centered_rows(shape, masses)
    centered_first = _interval_centered_rows(first, masses)
    tau_squared_over_three = (tau_slab * tau_slab).scale(1.0 / 3.0)
    if centered_shape.shape[1] == 2:
        angular = FloatInterval.point(0.0)
        for body, mass in enumerate(masses):
            angular = angular + _interval_planar_wedge(
                centered_shape[body],
                centered_first[body],
            ).scale(float(mass))
        return angular * tau_squared_over_three
    if centered_shape.shape[1] == 3:
        angular_vector = _interval_zero_array((3,))
        for body, mass in enumerate(masses):
            angular_vector = _interval_vector_add(
                angular_vector,
                _interval_vector_scale(
                    _interval_cross(centered_shape[body], centered_first[body]),
                    float(mass),
                ),
            )
        return _interval_vector_scale_by_interval(
            angular_vector,
            tau_squared_over_three,
        )
    raise ValueError("Fuchsian angular check supports only 2D and 3D shapes")


def _interval_fuchsian_lifted_residual(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> np.ndarray:
    shape, first, second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    tau_squared = tau_slab * tau_slab
    left = _interval_vector_add(
        _interval_vector_scale_by_interval(second, tau_squared),
        _interval_vector_scale_by_interval(first, tau_slab.scale(2.0)),
        _interval_vector_scale(shape, -2.0),
    )
    acceleration = _interval_newton_accelerations(shape, branch.masses)
    return _interval_vector_sub(left, _interval_vector_scale(acceleration, 9.0))


def _interval_fuchsian_log_branch_tau_derivatives(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("Fuchsian interval check needs a punctured tau slab")
    shape, radius_first, second = _interval_fuchsian_log_branch_radius_derivatives(
        branch,
        radius,
    )
    return shape, _interval_vector_scale(radius_first, sign), second


def _interval_fuchsian_log_branch_radius_derivatives(
    branch: FiniteFuchsianLogBranch,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius.lower <= 0.0:
        raise ValueError("Fuchsian radius interval must be positive")
    central = np.asarray(branch.central_shape, dtype=float)
    shape = _interval_point_array(central)
    first = _interval_vector_scale(
        _interval_point_array(central),
        0.0,
    )
    second = _interval_vector_scale(
        _interval_point_array(central),
        0.0,
    )
    scale = float(branch.scale_coefficient)
    radius_squared = radius * radius
    shape = _interval_vector_add(
        shape,
        _interval_vector_scale_by_interval(
            _interval_point_array(central),
            radius_squared.scale(scale),
        ),
    )
    first = _interval_vector_add(
        first,
        _interval_vector_scale_by_interval(
            _interval_point_array(central),
            radius.scale(2.0 * scale),
        ),
    )
    second = _interval_vector_add(
        second,
        _interval_vector_scale(_interval_point_array(central), 2.0 * scale),
    )
    for term in branch.terms:
        term_value, term_first, term_second = (
            _interval_fuchsian_log_term_radius_derivatives(term, radius)
        )
        shape = _interval_vector_add(shape, term_value)
        first = _interval_vector_add(first, term_first)
        second = _interval_vector_add(second, term_second)
    return shape, first, second


def _interval_fuchsian_log_term_radius_derivatives(
    term: FuchsianLogTerm,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logarithm = _interval_log_positive(radius)
    polynomial, log_first, log_second = (
        _interval_fuchsian_log_polynomial_derivatives(term, logarithm)
    )
    power = float(term.power)
    value_factor = radius.positive_power(power)
    first_factor = radius.positive_power(power - 1.0)
    second_factor = radius.positive_power(power - 2.0)
    first_poly = _interval_vector_add(
        _interval_vector_scale(polynomial, power),
        log_first,
    )
    second_poly = _interval_vector_add(
        _interval_vector_scale(polynomial, power * (power - 1.0)),
        _interval_vector_scale(log_first, 2.0 * power - 1.0),
        log_second,
    )
    return (
        _interval_vector_scale_by_interval(polynomial, value_factor),
        _interval_vector_scale_by_interval(first_poly, first_factor),
        _interval_vector_scale_by_interval(second_poly, second_factor),
    )


def _interval_fuchsian_log_polynomial_derivatives(
    term: FuchsianLogTerm,
    logarithm: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polynomial = _interval_zero_array(term.body_shape)
    first = _interval_zero_array(term.body_shape)
    second = _interval_zero_array(term.body_shape)
    for log_power, coefficient in term.coefficients_by_log_power.items():
        coefficient = np.asarray(coefficient, dtype=float)
        polynomial = _interval_vector_add(
            polynomial,
            _interval_vector_scale_by_interval(
                _interval_point_array(coefficient),
                _interval_nonnegative_integer_power(logarithm, int(log_power)),
            ),
        )
        if log_power >= 1:
            first = _interval_vector_add(
                first,
                _interval_vector_scale_by_interval(
                    _interval_point_array(coefficient),
                    _interval_nonnegative_integer_power(
                        logarithm,
                        int(log_power) - 1,
                    ).scale(float(log_power)),
                ),
            )
        if log_power >= 2:
            second = _interval_vector_add(
                second,
                _interval_vector_scale_by_interval(
                    _interval_point_array(coefficient),
                    _interval_nonnegative_integer_power(
                        logarithm,
                        int(log_power) - 2,
                    ).scale(float(log_power * (log_power - 1))),
                ),
            )
    return polynomial, first, second


def _interval_point_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval.point(float(values[index]))
    return out


def _interval_log_positive(value: FloatInterval) -> FloatInterval:
    if value.lower <= 0.0:
        raise ValueError("log interval must be positive")
    return FloatInterval(
        float(np.nextafter(np.log(value.lower), -np.inf)),
        float(np.nextafter(np.log(value.upper), np.inf)),
    )


def _interval_nonnegative_integer_power(
    value: FloatInterval,
    exponent: int,
) -> FloatInterval:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("integer interval power must be nonnegative")
    result = FloatInterval.point(1.0)
    for _ in range(exponent):
        result = result * value
    return result


def _punctured_tau_samples(
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, ...]:
    lower, upper = tau_interval
    if sample_count < 2 or not lower < 0.0 < upper:
        return ()
    raw_samples = np.linspace(lower, upper, int(sample_count))
    epsilon = max(1.0e-15, 1.0e-12 * max(abs(lower), abs(upper)))
    samples = tuple(float(value) for value in raw_samples if abs(value) > epsilon)
    if samples:
        return samples
    return (float(0.5 * lower), float(0.5 * upper))


def _spatial_centered_angular_momentum_norm(
    branch: FiniteFuchsianLogBranch,
    tau: float,
) -> float:
    positions = branch.positions_at_tau(tau)
    velocities = branch.velocities_at_tau(tau)
    masses = np.asarray(branch.masses, dtype=float)
    center = np.average(positions, axis=0, weights=masses)
    center_velocity = np.average(velocities, axis=0, weights=masses)
    angular = np.zeros(3, dtype=float)
    for mass, position, velocity in zip(masses, positions, velocities):
        angular += mass * np.cross(position - center, velocity - center_velocity)
    return float(np.linalg.norm(angular))


def _derivative_coefficients(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.shape[0] <= 1:
        return np.zeros_like(coefficients)
    out = np.empty((coefficients.shape[0] - 1, *coefficients.shape[1:]), dtype=float)
    for n in range(out.shape[0]):
        out[n] = (n + 1) * coefficients[n + 1]
    return out


def _derivative_scalar_coefficients(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.shape[0] <= 1:
        return np.zeros_like(coefficients)
    out = np.empty((coefficients.shape[0] - 1,), dtype=float)
    for n in range(out.shape[0]):
        out[n] = (n + 1) * coefficients[n + 1]
    return out


def _evaluate_coefficients(coefficients: np.ndarray, time: float) -> np.ndarray:
    if coefficients.size == 0:
        return np.asarray((), dtype=float)
    value = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        value = value * time + coefficient
    return value


def _evaluate_scalar_coefficients(coefficients: np.ndarray, time: float) -> float:
    if coefficients.size == 0:
        return np.inf
    value = 0.0
    for coefficient in coefficients[::-1]:
        value = value * time + float(coefficient)
    return float(value)


def _event_isolation_sign_pattern(event_type: str) -> dict[str, int] | None:
    if event_type == "spatial_ks_rho_exit":
        return {"left": -1, "right": 1, "derivative": 1, "pre_event": -1}
    if event_type in {
        "spatial_ordinary_ks_entry",
        "spatial_ks_competing_binary_entry",
    }:
        return {"left": 1, "right": -1, "derivative": -1, "pre_event": 1}
    return None


def _event_isolation_supported_sources(event_type: str) -> set[str]:
    if event_type == "spatial_ks_rho_exit":
        return {"spatial_ks_binary_interval_taylor"}
    if event_type == "spatial_ordinary_ks_entry":
        return {"spatial_ordinary_interval_taylor"}
    if event_type == "spatial_ks_competing_binary_entry":
        return {"spatial_ks_binary_interval_taylor"}
    return set()


def _event_isolation_pair_valid(certificate: EventIsolationCertificate) -> bool:
    if certificate.event_type == "spatial_ks_rho_exit":
        return tuple(certificate.pair) == (-1, -1)
    if certificate.event_type in {
        "spatial_ordinary_ks_entry",
        "spatial_ks_competing_binary_entry",
    }:
        pair = tuple(certificate.pair)
        return bool(
            len(pair) == 2
            and pair[0] != pair[1]
            and set(pair).issubset({0, 1, 2})
        )
    return False


def _event_interval_coefficients(
    coefficient_intervals: tuple[tuple[float, float], ...],
) -> tuple[FloatInterval, ...]:
    return tuple(FloatInterval(float(lower), float(upper)) for lower, upper in coefficient_intervals)


def _event_rational_interval_coefficients(
    coefficient_intervals: tuple[tuple[float, float], ...],
) -> tuple[RationalInterval, ...]:
    return tuple(
        RationalInterval.from_float_interval(float(lower), float(upper))
        for lower, upper in coefficient_intervals
    )


def _signed_polynomial_range_certified(
    coefficients: tuple[FloatInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 512,
) -> bool:
    if sign == 0 or not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return False
    if lower == upper:
        value = interval_polynomial_eval(coefficients, FloatInterval.point(lower))
        return interval_sign(value) == sign
    pieces = 1
    while pieces <= max_subintervals:
        certified = True
        for index in range(pieces):
            piece_lower = lower + (upper - lower) * index / pieces
            piece_upper = lower + (upper - lower) * (index + 1) / pieces
            value = interval_polynomial_eval(
                coefficients,
                FloatInterval(piece_lower, piece_upper),
            )
            if interval_sign(value) != sign:
                certified = False
                break
        if certified:
            return True
        pieces *= 2
    return False


def _signed_rational_polynomial_range_certified(
    coefficients: tuple[RationalInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 512,
) -> bool:
    if sign == 0 or not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return False
    lower_interval = RationalInterval.from_float_interval(lower)
    upper_interval = RationalInterval.from_float_interval(upper)
    rational_lower = lower_interval.lower
    rational_upper = upper_interval.lower
    if rational_lower == rational_upper:
        value = rational_interval_polynomial_eval(
            coefficients,
            RationalInterval(rational_lower, rational_upper),
        )
        return rational_interval_sign(value) == sign
    pieces = 1
    while pieces <= max_subintervals:
        certified = True
        width = rational_upper - rational_lower
        for index in range(pieces):
            piece_lower = rational_lower + width * index / pieces
            piece_upper = rational_lower + width * (index + 1) / pieces
            value = rational_interval_polynomial_eval(
                coefficients,
                RationalInterval(piece_lower, piece_upper),
            )
            if rational_interval_sign(value) != sign:
                certified = False
                break
        if certified:
            return True
        pieces *= 2
    return False


def _array_sup_norm(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.inf
    return float(np.max(np.abs(array)))
