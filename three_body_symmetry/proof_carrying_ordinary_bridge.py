"""Private carried ordinary-to-ordinary bridge composition.

This module proves one local ``N -> N`` handoff for a solution already carried
by a parent continuation checker.  The source clock-origin interval is parent
state, not independent wire evidence: if the source clock is ``t = s+B_s``,
the bridge derives

``B_t = B_s + source_parameter - target_parameter``.

No public certificate may use this lemma to invent a clock origin.  A parent
must first derive and retain ``B_s`` from its preceding certified prefix.

The result pins ``ordinary_autonomous_uniqueness_bridge_kernel_v1`` for the
analytic fact that endpoint containment, local uniqueness, and autonomy make
the target tube a continuation of the same carried Newtonian branch.  Finite
replay checks the kernel's hypotheses; it does not re-prove that analytic
lemma on every run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from fractions import Fraction
import json
import math
from typing import Any

import numpy as np

from .certificate_checker import (
    CertificateCheckObligation,
    OrdinaryAposterioriTubeCheckResult,
    _exact_rational_chart_projected_state_at_parameter,
    check_ordinary_aposteriori_tube,
)
from .certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
)


_SCHEMA_VERSION = 1
_RECORD_TYPE = "ordinary_bridge_transition"
_RECORD_SOURCE = "private_carried_ordinary_bridge_v1"
_CHECKER_ID = "carried_ordinary_bridge_checker_v1"
_ANALYTIC_KERNEL_ID = "ordinary_autonomous_uniqueness_bridge_kernel_v1"
_NESTED_TUBE_CHECKER_ID = "independent_ordinary_aposteriori_tube_checker_v1"
_OBLIGATION_NAMES = (
    "ordinary_bridge_exact_raw_schemas",
    "ordinary_bridge_identifiers_match_and_are_unique",
    "ordinary_bridge_parent_clock_origin_is_exact_interval",
    "ordinary_bridge_common_planar_mass_problem",
    "ordinary_bridge_source_and_target_tubes_freshly_certified",
    "ordinary_bridge_exact_right_to_left_endpoint_handoff",
    "ordinary_bridge_complete_source_endpoint_enclosure_reconstructed",
    "ordinary_bridge_target_initial_ball_contains_complete_source_endpoint",
    "ordinary_bridge_target_clock_origin_exactly_derived",
)

FractionInterval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class OrdinaryBridgeTransitionRecord:
    """Strict private wire record for one ordinary endpoint handoff."""

    transition_id: str
    source_chart_id: str
    source_tube_id: str
    target_chart_id: str
    target_tube_id: str
    source_parameter: float
    target_parameter: float
    schema_version: int = _SCHEMA_VERSION
    record_type: str = _RECORD_TYPE
    source: str = _RECORD_SOURCE

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON data model for this record."""

        return {
            "transition_id": self.transition_id,
            "source_chart_id": self.source_chart_id,
            "source_tube_id": self.source_tube_id,
            "target_chart_id": self.target_chart_id,
            "target_tube_id": self.target_tube_id,
            "source_parameter": self.source_parameter,
            "target_parameter": self.target_parameter,
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryBridgeTransitionRecord":
        """Parse the exact canonical record shape without scalar coercion."""

        if type(data) is not dict:
            raise TypeError("ordinary bridge transition must be a dict")
        expected = {field.name for field in dataclass_fields(cls)}
        actual = set(data)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            unknown = tuple(sorted(actual - expected))
            raise ValueError(
                "ordinary bridge transition fields are noncanonical; "
                f"missing={missing!r}; unknown={unknown!r}"
            )
        if not (
            all(
                type(data[name]) is str
                for name in (
                    "transition_id",
                    "source_chart_id",
                    "source_tube_id",
                    "target_chart_id",
                    "target_tube_id",
                    "record_type",
                    "source",
                )
            )
            and type(data["source_parameter"]) is float
            and math.isfinite(data["source_parameter"])
            and type(data["target_parameter"]) is float
            and math.isfinite(data["target_parameter"])
            and type(data["schema_version"]) is int
        ):
            raise ValueError("ordinary bridge transition scalars are noncanonical")
        record = cls(
            transition_id=data["transition_id"],
            source_chart_id=data["source_chart_id"],
            source_tube_id=data["source_tube_id"],
            target_chart_id=data["target_chart_id"],
            target_tube_id=data["target_tube_id"],
            source_parameter=data["source_parameter"],
            target_parameter=data["target_parameter"],
            schema_version=data["schema_version"],
            record_type=data["record_type"],
            source=data["source"],
        )
        if _canonical_json(data) != _canonical_json(record.to_dict()):
            raise ValueError("ordinary bridge transition encoding is noncanonical")
        return record


@dataclass(frozen=True)
class CarriedOrdinaryBridgeResult:
    """Fresh replay result for a parent-carried ordinary bridge."""

    transition_id: str
    checker_id: str
    raw_transition: OrdinaryBridgeTransitionRecord
    raw_source_chart: OrdinaryTaylorChartCertificate
    raw_source_tube: OrdinaryAposterioriTubeCertificate
    raw_target_chart: OrdinaryTaylorChartCertificate
    raw_target_tube: OrdinaryAposterioriTubeCertificate
    parent_source_clock_origin_interval: FractionInterval
    target_clock_origin_interval: FractionInterval | tuple[()]
    source_tube_result: OrdinaryAposterioriTubeCheckResult | None
    target_tube_result: OrdinaryAposterioriTubeCheckResult | None
    source_endpoint_state_intervals: tuple[FractionInterval, ...]
    target_anchor_centers: tuple[Fraction, ...]
    maximum_target_anchor_gap: Fraction | None
    obligations: tuple[CertificateCheckObligation, ...]
    analytic_kernel_id: str = _ANALYTIC_KERNEL_ID

    def _snapshot_certified(self) -> bool:
        return bool(
            type(self.transition_id) is str
            and bool(self.transition_id)
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.analytic_kernel_id) is str
            and self.analytic_kernel_id == _ANALYTIC_KERNEL_ID
            and type(self.raw_transition) is OrdinaryBridgeTransitionRecord
            and _transition_schema(self.raw_transition)
            and self.transition_id == self.raw_transition.transition_id
            and type(self.raw_source_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_source_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.raw_target_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_target_tube) is OrdinaryAposterioriTubeCertificate
            and _ordinary_chart_schema(self.raw_source_chart)
            and _ordinary_tube_schema(self.raw_source_tube)
            and _ordinary_chart_schema(self.raw_target_chart)
            and _ordinary_tube_schema(self.raw_target_tube)
            and _fraction_interval(self.parent_source_clock_origin_interval)
            and _fraction_interval(self.target_clock_origin_interval)
            and _canonical_tube_result(
                self.source_tube_result,
                self.raw_source_tube,
                self.raw_source_chart,
            )
            and _canonical_tube_result(
                self.target_tube_result,
                self.raw_target_tube,
                self.raw_target_chart,
            )
            and _fraction_box(self.source_endpoint_state_intervals, 12)
            and type(self.target_anchor_centers) is tuple
            and len(self.target_anchor_centers) == 12
            and all(type(value) is Fraction for value in self.target_anchor_centers)
            and type(self.maximum_target_anchor_gap) is Fraction
            and self.maximum_target_anchor_gap >= 0
            and _exact_obligation_manifest(self.obligations)
        )

    @property
    def certified(self) -> bool:
        """Freshly replay the raw bridge and exact-compare its full snapshot."""

        try:
            if not self._snapshot_certified():
                return False
            fresh = check_carried_ordinary_bridge(
                self.raw_transition,
                self.raw_source_chart,
                self.raw_source_tube,
                self.raw_target_chart,
                self.raw_target_tube,
                self.parent_source_clock_origin_interval,
            )
            return bool(
                type(fresh) is CarriedOrdinaryBridgeResult
                and fresh._snapshot_certified()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = []
        for index, obligation in enumerate(self.obligations):
            if type(obligation) is not CertificateCheckObligation:
                missing.append(f"ordinary_bridge_malformed_obligation:{index}")
            elif obligation.certified is not True:
                missing.append(obligation.obligation)
        return tuple(missing)


def check_carried_ordinary_bridge(
    transition: OrdinaryBridgeTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
    parent_source_clock_origin_interval: FractionInterval,
) -> CarriedOrdinaryBridgeResult:
    """Check one carried ``N -> N`` endpoint handoff and transport its clock."""

    expected_types = (
        (transition, OrdinaryBridgeTransitionRecord),
        (source_chart, OrdinaryTaylorChartCertificate),
        (source_tube, OrdinaryAposterioriTubeCertificate),
        (target_chart, OrdinaryTaylorChartCertificate),
        (target_tube, OrdinaryAposterioriTubeCertificate),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("all ordinary bridge raw inputs must have exact classes")

    raw_schemas = bool(
        _transition_schema(transition)
        and _ordinary_chart_schema(source_chart)
        and _ordinary_tube_schema(source_tube)
        and _ordinary_chart_schema(target_chart)
        and _ordinary_tube_schema(target_tube)
    )
    identifiers = _identifiers_match_and_unique(
        transition,
        source_chart,
        source_tube,
        target_chart,
        target_tube,
    )
    clock_valid = _fraction_interval(parent_source_clock_origin_interval)
    common_problem = _common_planar_problem(source_chart, target_chart)

    source_result: OrdinaryAposterioriTubeCheckResult | None = None
    target_result: OrdinaryAposterioriTubeCheckResult | None = None
    try:
        if raw_schemas:
            source_result = check_ordinary_aposteriori_tube(
                source_tube, source_chart
            )
    except Exception:
        pass
    try:
        if raw_schemas:
            target_result = check_ordinary_aposteriori_tube(
                target_tube, target_chart
            )
    except Exception:
        pass
    source_certified = _canonical_tube_result(
        source_result, source_tube, source_chart
    )
    target_certified = _canonical_tube_result(
        target_result, target_tube, target_chart
    )
    tubes_certified = bool(source_certified and target_certified)
    endpoint_handoff = _exact_endpoint_handoff(
        transition,
        source_chart,
        source_tube,
        target_chart,
        target_tube,
    )

    source_box: tuple[FractionInterval, ...] = ()
    target_centers: tuple[Fraction, ...] = ()
    maximum_gap: Fraction | None = None
    endpoint_reconstructed = False
    target_contains = False
    target_clock: FractionInterval | tuple[()] = ()
    clock_derived = False
    if all(
        (
            raw_schemas,
            identifiers,
            clock_valid,
            common_problem,
            tubes_certified,
            endpoint_handoff,
        )
    ):
        try:
            source_parameter = Fraction.from_float(transition.source_parameter)
            target_parameter = Fraction.from_float(transition.target_parameter)
            source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
                target_chart,
                target_parameter,
            )
            source_centers = _flatten_fraction_matrices(source_q, source_v)
            target_centers = _flatten_fraction_matrices(target_q, target_v)
            if source_result is None:
                raise ValueError("source replay result missing")
            source_radius = Fraction.from_float(
                source_result.gronwall_error_bound
            )
            target_radius = Fraction.from_float(target_tube.initial_error_bound)
            source_box = tuple(
                (center - source_radius, center + source_radius)
                for center in source_centers
            )
            endpoint_reconstructed = _fraction_box(source_box, 12)
            gaps = tuple(
                max(abs(lower - center), abs(upper - center))
                for (lower, upper), center in zip(source_box, target_centers)
            )
            maximum_gap = max(gaps, default=Fraction(0))
            target_contains = bool(
                endpoint_reconstructed
                and len(target_centers) == 12
                and maximum_gap <= target_radius
            )
            delta = source_parameter - target_parameter
            target_clock = (
                parent_source_clock_origin_interval[0] + delta,
                parent_source_clock_origin_interval[1] + delta,
            )
            clock_derived = _fraction_interval(target_clock)
        except Exception:
            pass

    obligations = (
        _obligation(
            "ordinary_bridge_exact_raw_schemas",
            raw_schemas,
            "exact built-in chart/tube fields and canonical private transition",
        ),
        _obligation(
            "ordinary_bridge_identifiers_match_and_are_unique",
            identifiers,
            "transition IDs bind both distinct chart/tube vertices",
        ),
        _obligation(
            "ordinary_bridge_parent_clock_origin_is_exact_interval",
            clock_valid,
            (
                "B_source is derived parent state, not independent bridge wire "
                f"evidence: {parent_source_clock_origin_interval!s}"
            ),
        ),
        _obligation(
            "ordinary_bridge_common_planar_mass_problem",
            common_problem,
            f"source_masses={source_chart.masses!r}; target_masses={target_chart.masses!r}",
        ),
        _obligation(
            "ordinary_bridge_source_and_target_tubes_freshly_certified",
            tubes_certified,
            (
                f"source={_nested_detail(source_result)}; "
                f"target={_nested_detail(target_result)}"
            ),
        ),
        _obligation(
            "ordinary_bridge_exact_right_to_left_endpoint_handoff",
            endpoint_handoff,
            (
                f"source_parameter={transition.source_parameter!r}; "
                f"target_parameter={transition.target_parameter!r}; "
                f"source_tube_anchor={source_tube.anchor_parameter!r}; "
                f"target_tube_anchor={target_tube.anchor_parameter!r}"
            ),
        ),
        _obligation(
            "ordinary_bridge_complete_source_endpoint_enclosure_reconstructed",
            endpoint_reconstructed,
            f"components={len(source_box)}",
        ),
        _obligation(
            "ordinary_bridge_target_initial_ball_contains_complete_source_endpoint",
            target_contains,
            (
                f"maximum_gap={maximum_gap!s}; "
                f"target_radius={target_tube.initial_error_bound!r}"
            ),
        ),
        _obligation(
            "ordinary_bridge_target_clock_origin_exactly_derived",
            clock_derived,
            "B_target=B_source+source_parameter-target_parameter",
        ),
    )
    transition_id = (
        transition.transition_id if type(transition.transition_id) is str else ""
    )
    return CarriedOrdinaryBridgeResult(
        transition_id=transition_id,
        checker_id=_CHECKER_ID,
        raw_transition=transition,
        raw_source_chart=source_chart,
        raw_source_tube=source_tube,
        raw_target_chart=target_chart,
        raw_target_tube=target_tube,
        parent_source_clock_origin_interval=parent_source_clock_origin_interval,
        target_clock_origin_interval=target_clock,
        source_tube_result=source_result,
        target_tube_result=target_result,
        source_endpoint_state_intervals=source_box,
        target_anchor_centers=target_centers,
        maximum_target_anchor_gap=maximum_gap,
        obligations=obligations,
    )


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _transition_schema(value: object) -> bool:
    return bool(
        type(value) is OrdinaryBridgeTransitionRecord
        and type(value.schema_version) is int
        and value.schema_version == _SCHEMA_VERSION
        and type(value.record_type) is str
        and value.record_type == _RECORD_TYPE
        and type(value.source) is str
        and value.source == _RECORD_SOURCE
        and all(
            type(item) is str and bool(item)
            for item in (
                value.transition_id,
                value.source_chart_id,
                value.source_tube_id,
                value.target_chart_id,
                value.target_tube_id,
                value.source,
            )
        )
        and _finite_float(value.source_parameter)
        and _finite_float(value.target_parameter)
    )


def _ordinary_chart_schema(value: object) -> bool:
    def matrix(item: object) -> bool:
        return bool(
            type(item) is tuple
            and len(item) == 3
            and all(
                type(row) is tuple
                and len(row) == 2
                and all(_finite_float(component) for component in row)
                for row in item
            )
        )

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
        and all(_finite_float(item) and item > 0 for item in value.masses)
        and type(value.position_coefficients) is tuple
        and type(value.velocity_coefficients) is tuple
        and len(value.position_coefficients) >= 2
        and len(value.position_coefficients) == len(value.velocity_coefficients)
        and all(matrix(item) for item in value.position_coefficients)
        and all(matrix(item) for item in value.velocity_coefficients)
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


def _ordinary_tube_schema(value: object) -> bool:
    return bool(
        type(value) is OrdinaryAposterioriTubeCertificate
        and all(
            type(item) is str and bool(item)
            for item in (value.tube_id, value.chart_id, value.source)
        )
        and all(
            _finite_float(item)
            for item in (
                value.anchor_parameter,
                value.initial_error_bound,
                value.tube_radius,
                value.max_defect_bound,
                value.max_lipschitz_bound,
            )
        )
        and value.initial_error_bound >= 0
        and value.tube_radius > 0
        and value.max_defect_bound >= 0
        and value.max_lipschitz_bound >= 0
    )


def _identifiers_match_and_unique(
    transition: OrdinaryBridgeTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    try:
        identifiers = (
            transition.transition_id,
            source_chart.certificate_id,
            source_chart.chart_id,
            source_tube.tube_id,
            target_chart.certificate_id,
            target_chart.chart_id,
            target_tube.tube_id,
        )
        return bool(
            all(type(item) is str and bool(item) for item in identifiers)
            and len(set(identifiers)) == len(identifiers)
            and transition.source_chart_id == source_chart.chart_id
            and transition.source_tube_id == source_tube.tube_id
            and transition.target_chart_id == target_chart.chart_id
            and transition.target_tube_id == target_tube.tube_id
            and source_tube.chart_id == source_chart.chart_id
            and target_tube.chart_id == target_chart.chart_id
        )
    except Exception:
        return False


def _common_planar_problem(
    source: OrdinaryTaylorChartCertificate,
    target: OrdinaryTaylorChartCertificate,
) -> bool:
    try:
        return bool(
            _ordinary_chart_schema(source)
            and _ordinary_chart_schema(target)
            and source.masses == target.masses
            and source.dimension == target.dimension == 2
        )
    except Exception:
        return False


def _exact_endpoint_handoff(
    transition: OrdinaryBridgeTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    try:
        source_domain = tuple(
            Fraction.from_float(value) for value in source_chart.parameter_interval
        )
        target_domain = tuple(
            Fraction.from_float(value) for value in target_chart.parameter_interval
        )
        return bool(
            len(source_domain) == 2
            and len(target_domain) == 2
            and Fraction.from_float(transition.source_parameter) == source_domain[1]
            and Fraction.from_float(transition.target_parameter) == target_domain[0]
            and Fraction.from_float(source_tube.anchor_parameter)
            == source_domain[0]
            and Fraction.from_float(target_tube.anchor_parameter) == target_domain[0]
        )
    except Exception:
        return False


def _flatten_fraction_matrices(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[Fraction, ...]:
    values = tuple(
        value
        for matrix in (positions, velocities)
        for value in np.asarray(matrix, dtype=object).reshape(-1)
    )
    if len(values) != 12 or not all(type(value) is Fraction for value in values):
        raise ValueError("ordinary bridge state must have 12 exact components")
    return values


def _canonical_tube_result(
    result: object,
    raw_tube: OrdinaryAposterioriTubeCertificate,
    raw_chart: OrdinaryTaylorChartCertificate,
) -> bool:
    return bool(
        type(result) is OrdinaryAposterioriTubeCheckResult
        and type(result.tube_id) is str
        and bool(result.tube_id)
        and result.tube_id == raw_tube.tube_id
        and type(result.chart_id) is str
        and bool(result.chart_id)
        and result.chart_id == raw_chart.chart_id == raw_tube.chart_id
        and type(result.checker_id) is str
        and result.checker_id == _NESTED_TUBE_CHECKER_ID
        and type(result.obligations) is tuple
        and bool(result.obligations)
        and all(
            type(item) is CertificateCheckObligation
            and type(item.obligation) is str
            and bool(item.obligation)
            and type(item.certified) is bool
            and item.certified is True
            and type(item.detail) is str
            for item in result.obligations
        )
        and all(
            _finite_float(value)
            for value in (
                result.defect_bound,
                result.lipschitz_bound,
                result.gronwall_error_bound,
                result.nominal_pair_distance_floor,
                result.tube_pair_distance_floor,
            )
        )
        and result.gronwall_error_bound >= 0
        and result.certified
    )


def _exact_obligation_manifest(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_OBLIGATION_NAMES)
        and all(
            type(item) is CertificateCheckObligation
            and type(item.obligation) is str
            and type(item.certified) is bool
            and item.certified is True
            and type(item.detail) is str
            for item in value
        )
        and tuple(item.obligation for item in value) == _OBLIGATION_NAMES
    )


def _fraction_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(endpoint) is Fraction for endpoint in value)
        and value[0] <= value[1]
    )


def _fraction_box(value: object, length: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == length
        and all(_fraction_interval(item) for item in value)
    )


def _float_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(_finite_float(item) for item in value)
        and value[0] < value[1]
    )


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _obligation(
    name: str,
    certified: bool,
    detail: str,
) -> CertificateCheckObligation:
    return CertificateCheckObligation(name, bool(certified), detail)


def _nested_detail(result: object | None) -> str:
    if result is None:
        return "fresh replay did not produce a result"
    try:
        return f"checker={result.checker_id!r}; missing={result.missing_obligations!r}"
    except Exception as error:
        return f"malformed nested result:{type(error).__name__}"
