"""Proof-grade replay of one binary64 ordinary-IVP root.

This deliberately small profile is the direct-evidence counterpart of the
historical ``check_validated_ordinary_ivp_chart`` composition.  The latter
requires a claimed Taylor-tail chart replay.  Here that replay is retained as
an authenticated *diagnostic* only: the decisive root ledger is discharged by
the direct IVP-binding and a-posteriori-tube replays, with exact binary64
``Fraction`` comparisons at their composition boundary.

No constructor, compatibility checker, or continuation layer imports this
module.  It is a separate proof-grade boundary with a frozen fresh-replay
snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .certificate_checker import (
    CertificateCheckObligation,
    CertificateCheckResult,
    InitialValueBindingCheckResult,
    OrdinaryAposterioriTubeCheckResult,
    check_initial_value_problem_binding,
    check_ordinary_aposteriori_tube,
    check_ordinary_taylor_chart,
)
from .certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
)


_CHECKER_ID = "proof_grade_validated_ordinary_root_checker_v04"
_PROFILE_ID = "binary64_outward_validated_ordinary_root_v04"

PROOF_GRADE_VALIDATED_ORDINARY_ROOT_CHECKER_ID = _CHECKER_ID
PROOF_GRADE_VALIDATED_ORDINARY_ROOT_PROFILE_ID = _PROFILE_ID
BINARY64_OUTWARD_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID = _PROFILE_ID

VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS = (
    "validated_ordinary_chart_serialization_admissible",
    "validated_ordinary_chart_exact_unit_speed",
    "validated_ordinary_root_exact_time_anchor",
    "validated_ordinary_ivp_binding_checked",
    "validated_ordinary_tube_checked",
    "validated_ordinary_component_chart_ids_match",
    "validated_ordinary_anchor_parameter_matches_binding",
    "validated_ordinary_actual_initial_error_covered",
)
PROOF_GRADE_VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS = (
    VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS
)

_BINDING_OBLIGATION_IDS = (
    "initial_value_binding_identity_present",
    "initial_value_binding_chart_present",
    "initial_value_problem_finite_positive_mass_state",
    "initial_value_binding_tolerances_finite",
    "initial_value_binding_chart_masses_match",
    "initial_value_binding_parameter_inside_chart",
    "initial_value_binding_state_shape_matches",
    "initial_value_binding_polynomial_state_matches",
)
_TUBE_OBLIGATION_IDS = (
    "ordinary_tube_identity_matches_chart",
    "ordinary_tube_inputs_finite",
    "ordinary_tube_polynomial_defect_within_cap",
    "ordinary_tube_collision_free",
    "ordinary_tube_lipschitz_within_cap",
    "ordinary_tube_gronwall_self_consistent",
)


@dataclass(frozen=True)
class ProofGradeValidatedOrdinaryRootResult:
    """Frozen direct-evidence replay for one ordinary root chart.

    ``chart_diagnostic`` is intentionally authenticated despite being excluded
    from ``validated_root_satisfied``.  Thus a negative claimed tail can
    produce a freshly replayed false diagnostic without weakening the direct
    root proof, while a substituted diagnostic cannot be smuggled into a
    certified snapshot.
    """

    checker_id: str
    raw_binding: InitialValueProblemBindingCertificate
    raw_tube: OrdinaryAposterioriTubeCertificate
    raw_chart: OrdinaryTaylorChartCertificate
    chart_diagnostic: CertificateCheckResult | None
    binding_result: InitialValueBindingCheckResult | None
    tube_result: OrdinaryAposterioriTubeCheckResult | None
    obligations: tuple[CertificateCheckObligation, ...]
    actual_initial_error: Fraction | None
    root_clock_origin: Fraction | None

    @property
    def profile_id(self) -> str:
        """The immutable binary64-outward profile identifier."""

        return _PROFILE_ID

    @property
    def chart_result(self) -> CertificateCheckResult | None:
        """Compatibility-friendly spelling for the stored chart diagnostic."""

        return self.chart_diagnostic

    @property
    def diagnostic(self) -> CertificateCheckResult | None:
        """The non-decisive claimed-tail chart replay diagnostic."""

        return self.chart_diagnostic

    @property
    def validated_root_satisfied(self) -> bool:
        """Whether the eight decisive obligations themselves are all true."""

        return _root_ledger_satisfied(self.obligations)

    def _snapshot_certified(self) -> bool:
        """Validate the exact-class, exact-shape stored replay snapshot."""

        return bool(
            type(self) is ProofGradeValidatedOrdinaryRootResult
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.raw_binding) is InitialValueProblemBindingCertificate
            and type(self.raw_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.raw_chart) is OrdinaryTaylorChartCertificate
            and _binding_schema(self.raw_binding)
            and _tube_schema(self.raw_tube)
            and _chart_schema(self.raw_chart)
            and _canonical_chart_diagnostic(
                self.chart_diagnostic,
                self.raw_chart,
            )
            and _canonical_binding_result(
                self.binding_result,
                self.raw_binding,
                self.raw_chart,
            )
            and _canonical_tube_result(
                self.tube_result,
                self.raw_tube,
                self.raw_chart,
            )
            and _root_ledger_schema(self.obligations)
            and _actual_error_schema(
                self.actual_initial_error,
                self.binding_result,
            )
            and _root_clock_schema(
                self.root_clock_origin,
                self.raw_binding,
                self.raw_chart,
                self.binding_result,
            )
        )

    @property
    def certified(self) -> bool:
        """Freshly replay every raw input and exact-compare this snapshot."""

        try:
            if (
                type(self) is not ProofGradeValidatedOrdinaryRootResult
                or not self._snapshot_certified()
                or not self.validated_root_satisfied
            ):
                return False
            fresh = check_proof_grade_validated_ordinary_root(
                self.raw_binding,
                self.raw_tube,
                self.raw_chart,
            )
            return bool(
                type(fresh) is ProofGradeValidatedOrdinaryRootResult
                and fresh._snapshot_certified()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        """The ordered decisive root obligations which remain unsatisfied."""

        if not _root_ledger_schema(self.obligations):
            return ("validated_ordinary_root_malformed_obligation_ledger",)
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.certified is not True
        )


# The Rust name is useful to callers comparing the two independent profiles.
# This is an alias, rather than a subclass, so exact-type checking remains
# single-valued.
ValidatedOrdinaryRootReplay = ProofGradeValidatedOrdinaryRootResult


def check_proof_grade_validated_ordinary_root(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> ProofGradeValidatedOrdinaryRootResult:
    """Replay the proof-grade ordinary root with binary64-outward evidence.

    The public boundary rejects subclasses.  Malformed values *inside* an
    exact certificate class are fail-closed as false obligations where direct
    replay permits a result; they never become a certified snapshot.
    """

    expected = (
        (binding, InitialValueProblemBindingCertificate),
        (tube, OrdinaryAposterioriTubeCertificate),
        (chart, OrdinaryTaylorChartCertificate),
    )
    if any(type(value) is not expected_type for value, expected_type in expected):
        raise TypeError(
            "proof-grade validated ordinary root inputs must have exact classes"
        )

    # Do not condition these direct calls on claimed-tail validity.  They are
    # the evidence replays for this profile and their results are retained in
    # full, including non-certified direct diagnostics for malformed evidence.
    chart_diagnostic = _safe_chart_diagnostic(chart)
    binding_result = _safe_binding_replay(binding, chart)
    tube_result = _safe_tube_replay(tube, chart)

    chart_serialization_admissible = _chart_schema(chart)
    exact_unit_speed = _exact_unit_speed(chart)
    exact_time_anchor = _exact_time_anchor(binding_result)
    binding_checked = _binding_certified(binding_result)
    tube_checked = _tube_certified(tube_result)
    component_chart_ids_match = _component_chart_ids_match(binding, tube, chart)
    anchor_parameter_matches_binding = _anchor_matches(binding, tube)
    actual_initial_error = _actual_initial_error(binding_result)
    actual_initial_error_covered = _error_covered(actual_initial_error, tube)
    root_clock_origin = _root_clock_origin(
        binding,
        chart,
        exact_time_anchor,
        exact_unit_speed,
    )

    satisfied = (
        chart_serialization_admissible,
        exact_unit_speed,
        exact_time_anchor,
        binding_checked,
        tube_checked,
        component_chart_ids_match,
        anchor_parameter_matches_binding,
        actual_initial_error_covered,
    )
    details = (
        f"chart_id={_safe_repr(chart.chart_id)}",
        _unit_speed_detail(chart),
        _time_anchor_detail(binding_result),
        f"binding_id={_safe_repr(binding.binding_id)}",
        f"tube_id={_safe_repr(tube.tube_id)}",
        (
            f"binding_chart={_safe_repr(binding.chart_id)}; "
            f"tube_chart={_safe_repr(tube.chart_id)}; "
            f"chart={_safe_repr(chart.chart_id)}"
        ),
        (
            f"binding_parameter={_safe_repr(binding.chart_parameter)}; "
            f"tube_anchor={_safe_repr(tube.anchor_parameter)}"
        ),
        (
            f"actual_initial_error={_safe_repr(actual_initial_error)}; "
            f"tube_initial_error={_safe_repr(tube.initial_error_bound)}"
        ),
    )
    obligations = tuple(
        CertificateCheckObligation(name, bool(value), detail)
        for name, value, detail in zip(
            VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS,
            satisfied,
            details,
            strict=True,
        )
    )
    return ProofGradeValidatedOrdinaryRootResult(
        checker_id=_CHECKER_ID,
        raw_binding=binding,
        raw_tube=tube,
        raw_chart=chart,
        chart_diagnostic=chart_diagnostic,
        binding_result=binding_result,
        tube_result=tube_result,
        obligations=obligations,
        actual_initial_error=actual_initial_error,
        root_clock_origin=root_clock_origin,
    )


# A concise spelling is intentionally an alias to the same checker path.
check_validated_ordinary_root = check_proof_grade_validated_ordinary_root


def _safe_chart_diagnostic(
    chart: OrdinaryTaylorChartCertificate,
) -> CertificateCheckResult | None:
    try:
        result = check_ordinary_taylor_chart(chart)
        return result if type(result) is CertificateCheckResult else None
    except Exception:
        return None


def _safe_binding_replay(
    binding: InitialValueProblemBindingCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> InitialValueBindingCheckResult | None:
    try:
        result = check_initial_value_problem_binding(binding, (chart,))
        return result if type(result) is InitialValueBindingCheckResult else None
    except Exception:
        return None


def _safe_tube_replay(
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> OrdinaryAposterioriTubeCheckResult | None:
    try:
        result = check_ordinary_aposteriori_tube(tube, chart)
        return result if type(result) is OrdinaryAposterioriTubeCheckResult else None
    except Exception:
        return None


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _float_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and _finite_float(value[0])
        and _finite_float(value[1])
        and value[0] < value[1]
    )


def _planar_matrix(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 3
        and all(
            type(row) is tuple
            and len(row) == 2
            and all(_finite_float(component) for component in row)
            for row in value
        )
    )


def _coefficient_series(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) >= 2
        and all(_planar_matrix(coefficient) for coefficient in value)
    )


def _chart_schema(value: object) -> bool:
    """Strict serial form admitted by the proof-grade planar root profile.

    In particular, the claimed tail is finite but need not be nonnegative: it
    belongs to the stored chart diagnostic, not the decisive direct proof.
    """

    return bool(
        type(value) is OrdinaryTaylorChartCertificate
        and all(
            type(item) is str and bool(item)
            for item in (
                value.certificate_id,
                value.chart_id,
                value.chart_type,
                value.source,
            )
        )
        and value.chart_type == "ordinary_taylor"
        and type(value.masses) is tuple
        and len(value.masses) == 3
        and all(_finite_float(mass) and mass > 0.0 for mass in value.masses)
        and _coefficient_series(value.position_coefficients)
        and _coefficient_series(value.velocity_coefficients)
        and len(value.position_coefficients) == len(value.velocity_coefficients)
        and _float_interval(value.parameter_interval)
        and _float_interval(value.physical_time_interval)
        and all(
            _finite_float(item)
            for item in (
                value.coefficient_tolerance,
                value.residual_tolerance,
                value.tail_bound,
            )
        )
        and type(value.sample_count) is int
        and value.sample_count >= 1
    )


def _binding_schema(value: object) -> bool:
    return bool(
        type(value) is InitialValueProblemBindingCertificate
        and all(
            type(item) is str and bool(item)
            for item in (value.binding_id, value.chart_id, value.source)
        )
        and type(value.masses) is tuple
        and len(value.masses) == 3
        and all(_finite_float(mass) and mass > 0.0 for mass in value.masses)
        and _finite_float(value.initial_time)
        and _finite_float(value.chart_parameter)
        and _planar_matrix(value.positions)
        and _planar_matrix(value.velocities)
        and all(
            _finite_float(item) and item >= 0.0
            for item in (
                value.time_tolerance,
                value.position_tolerance,
                value.velocity_tolerance,
            )
        )
    )


def _tube_schema(value: object) -> bool:
    return bool(
        type(value) is OrdinaryAposterioriTubeCertificate
        and all(
            type(item) is str and bool(item)
            for item in (value.tube_id, value.chart_id, value.source)
        )
        and _finite_float(value.anchor_parameter)
        and _finite_float(value.initial_error_bound)
        and value.initial_error_bound >= 0.0
        and _finite_float(value.tube_radius)
        and value.tube_radius > 0.0
        and _finite_float(value.max_defect_bound)
        and value.max_defect_bound >= 0.0
        and _finite_float(value.max_lipschitz_bound)
        and value.max_lipschitz_bound >= 0.0
    )


def _exact_unit_speed(chart: OrdinaryTaylorChartCertificate) -> bool:
    if not (
        type(chart) is OrdinaryTaylorChartCertificate
        and _float_interval(chart.parameter_interval)
        and _float_interval(chart.physical_time_interval)
    ):
        return False
    parameter_width = (
        Fraction.from_float(chart.parameter_interval[1])
        - Fraction.from_float(chart.parameter_interval[0])
    )
    physical_width = (
        Fraction.from_float(chart.physical_time_interval[1])
        - Fraction.from_float(chart.physical_time_interval[0])
    )
    return parameter_width == physical_width


def _exact_time_anchor(result: InitialValueBindingCheckResult | None) -> bool:
    return bool(
        type(result) is InitialValueBindingCheckResult
        and type(result.time_gap) is float
        and math.isfinite(result.time_gap)
        and Fraction.from_float(result.time_gap) == 0
    )


def _binding_certified(result: InitialValueBindingCheckResult | None) -> bool:
    return bool(
        type(result) is InitialValueBindingCheckResult and result.certified
    )


def _tube_certified(result: OrdinaryAposterioriTubeCheckResult | None) -> bool:
    return bool(
        type(result) is OrdinaryAposterioriTubeCheckResult and result.certified
    )


def _component_chart_ids_match(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    return bool(
        all(
            type(value) is str and bool(value)
            for value in (binding.chart_id, tube.chart_id, chart.chart_id)
        )
        and binding.chart_id == tube.chart_id == chart.chart_id
    )


def _anchor_matches(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    return bool(
        _finite_float(binding.chart_parameter)
        and _finite_float(tube.anchor_parameter)
        and Fraction.from_float(binding.chart_parameter)
        == Fraction.from_float(tube.anchor_parameter)
    )


def _actual_initial_error(
    result: InitialValueBindingCheckResult | None,
) -> Fraction | None:
    if not (
        type(result) is InitialValueBindingCheckResult
        and type(result.max_position_gap) is float
        and type(result.max_velocity_gap) is float
        and math.isfinite(result.max_position_gap)
        and math.isfinite(result.max_velocity_gap)
        and result.max_position_gap >= 0.0
        and result.max_velocity_gap >= 0.0
    ):
        return None
    return max(
        Fraction.from_float(result.max_position_gap),
        Fraction.from_float(result.max_velocity_gap),
    )


def _error_covered(
    actual_error: Fraction | None,
    tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    return bool(
        type(actual_error) is Fraction
        and actual_error >= 0
        and _finite_float(tube.initial_error_bound)
        and tube.initial_error_bound >= 0.0
        and actual_error <= Fraction.from_float(tube.initial_error_bound)
    )


def _root_clock_origin(
    binding: InitialValueProblemBindingCertificate,
    chart: OrdinaryTaylorChartCertificate,
    exact_time_anchor: bool,
    exact_unit_speed: bool,
) -> Fraction | None:
    if not (
        exact_time_anchor
        and exact_unit_speed
        and _finite_float(binding.initial_time)
        and _finite_float(binding.chart_parameter)
    ):
        return None
    return (
        Fraction.from_float(binding.initial_time)
        - Fraction.from_float(binding.chart_parameter)
    )


def _generic_ledger(
    value: object,
    expected_ids: tuple[str, ...],
) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(expected_ids)
        and all(
            type(item) is CertificateCheckObligation
            and type(item.obligation) is str
            and item.obligation == expected
            and type(item.certified) is bool
            and type(item.detail) is str
            for item, expected in zip(value, expected_ids, strict=True)
        )
    )


def _root_ledger_schema(value: object) -> bool:
    return _generic_ledger(value, VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS)


def _root_ledger_satisfied(value: object) -> bool:
    return bool(
        _root_ledger_schema(value)
        and all(item.certified is True for item in value)
    )


def _canonical_chart_diagnostic(
    result: object,
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    return bool(
        type(result) is CertificateCheckResult
        and type(result.certificate_id) is str
        and result.certificate_id == chart.certificate_id
        and type(result.certificate_type) is str
        and result.certificate_type == "ordinary_taylor"
        and type(result.checker_id) is str
        and result.checker_id == "independent_ordinary_taylor_checker_interval_v2"
        and _generic_ledger(
            result.obligations,
            (
                "ordinary_chart_type",
                "certificate_identity_present",
                "coefficient_array_shape",
                "finite_coefficients",
                "positive_masses",
                "finite_nonempty_time_intervals",
                "ordinary_physical_parameter_unit_speed",
                "finite_checker_tolerances",
                "initial_noncollision",
                "ordinary_taylor_coefficient_recurrence",
                "ordinary_taylor_exact_rational_residual_polynomials",
                "interval_taylor_model_newton_residual",
                "tail_bound_admissible",
            ),
        )
        and type(result.max_coefficient_residual) is float
        and type(result.max_sampled_newton_residual) is float
    )


def _canonical_binding_result(
    result: object,
    binding: InitialValueProblemBindingCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    return bool(
        type(result) is InitialValueBindingCheckResult
        and type(result.binding_id) is str
        and result.binding_id == binding.binding_id
        and type(result.chart_id) is str
        and result.chart_id == binding.chart_id
        and type(result.checker_id) is str
        and result.checker_id == "independent_initial_value_binding_checker_v1"
        and _generic_ledger(result.obligations, _BINDING_OBLIGATION_IDS)
        and all(
            type(item) is float
            for item in (
                result.max_position_gap,
                result.max_velocity_gap,
                result.time_gap,
            )
        )
        and chart.chart_id == result.chart_id
    )


def _canonical_tube_result(
    result: object,
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    return bool(
        type(result) is OrdinaryAposterioriTubeCheckResult
        and type(result.tube_id) is str
        and result.tube_id == tube.tube_id
        and type(result.chart_id) is str
        and result.chart_id == tube.chart_id == chart.chart_id
        and type(result.checker_id) is str
        and result.checker_id == "independent_ordinary_aposteriori_tube_checker_v1"
        and _generic_ledger(result.obligations, _TUBE_OBLIGATION_IDS)
        and all(
            type(item) is float
            for item in (
                result.defect_bound,
                result.lipschitz_bound,
                result.gronwall_error_bound,
                result.nominal_pair_distance_floor,
                result.tube_pair_distance_floor,
            )
        )
    )


def _actual_error_schema(
    actual_error: object,
    binding_result: InitialValueBindingCheckResult | None,
) -> bool:
    expected = _actual_initial_error(binding_result)
    return bool(
        (actual_error is None and expected is None)
        or (
            type(actual_error) is Fraction
            and type(expected) is Fraction
            and actual_error == expected
        )
    )


def _root_clock_schema(
    root_clock: object,
    binding: InitialValueProblemBindingCertificate,
    chart: OrdinaryTaylorChartCertificate,
    binding_result: InitialValueBindingCheckResult | None,
) -> bool:
    expected = _root_clock_origin(
        binding,
        chart,
        _exact_time_anchor(binding_result),
        _exact_unit_speed(chart),
    )
    return bool(
        (root_clock is None and expected is None)
        or (
            type(root_clock) is Fraction
            and type(expected) is Fraction
            and root_clock == expected
        )
    )


def _unit_speed_detail(chart: OrdinaryTaylorChartCertificate) -> str:
    if not _chart_schema(chart):
        return "chart serialization is not admissible"
    parameter_width = (
        Fraction.from_float(chart.parameter_interval[1])
        - Fraction.from_float(chart.parameter_interval[0])
    )
    physical_width = (
        Fraction.from_float(chart.physical_time_interval[1])
        - Fraction.from_float(chart.physical_time_interval[0])
    )
    return (
        f"parameter_width={parameter_width}; "
        f"physical_width={physical_width}"
    )


def _time_anchor_detail(result: InitialValueBindingCheckResult | None) -> str:
    if type(result) is not InitialValueBindingCheckResult:
        return "direct binding replay was unavailable"
    return f"direct_binding_time_gap={result.time_gap!r}"


def _safe_repr(value: object) -> str:
    """Keep malformed exact-class inputs from raising while forming details."""

    try:
        return repr(value)
    except Exception:
        return "<unrepresentable>"
