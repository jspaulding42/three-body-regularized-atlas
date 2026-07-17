"""Replayable ordinary-only continuation evidence, schema version 1.

This module is the completed ordinary-only Milestone-2 vertical slice toward a
mixed planar ordinary/Levi-Civita proof-carrying continuation checker.  Version
1 intentionally supports ordinary planar charts only; it makes no claim of
LC-chart support.

The evidence digest is SHA-256 over UTF-8 canonical JSON.  Canonical JSON uses
``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)``.  Invalid non-finite float evidence is first represented by
the deterministic tagged object ``{"__nonfinite_float__": <name>}``, so even a
rejected certificate has a stable digest without admitting NaN JSON tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from .certificate_checker import (
    OrdinaryEnclosureTransitionCheckResult,
    ValidatedOrdinaryIVPChainCheckResult,
    ValidatedOrdinaryIVPChartCheckResult,
    _exact_rational_chart_projected_state_at_parameter,
    _fraction_array_tube_intervals,
    _fraction_upper_float,
    check_validated_ordinary_ivp_chain,
)
from .certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryEnclosureTransitionCertificate,
    OrdinaryTaylorChartCertificate,
    ValidatedOrdinaryIVPChainCertificate,
)


CERTIFIED_TO_T = "CERTIFIED_TO_T"
UNRESOLVED = "UNRESOLVED"

_CERTIFICATE_TYPE = "raw_ordinary_continuation"
_SCHEMA_VERSION = 1
_CHECKER_ID = "raw_ordinary_continuation_replay_checker_v1"
_NESTED_CHECKER_ID = "validated_ordinary_ivp_chain_checker_v1"
_OBLIGATION_NAMES = (
    "raw_ordinary_schema_version_supported",
    "raw_ordinary_component_manifest_exact",
    "raw_ordinary_identifiers_unique_and_nonempty",
    "raw_ordinary_exact_point_ivp",
    "raw_ordinary_planar_common_mass_problem",
    "raw_ordinary_forward_positive_intervals",
    "raw_ordinary_exact_affine_physical_clocks",
    "raw_ordinary_first_chart_matches_binding_origin",
    "raw_ordinary_tube_anchors_match_chart_left_endpoints",
    "raw_ordinary_transitions_are_exact_endpoint_handoffs",
    "raw_ordinary_requested_target_finite_and_covered",
    "raw_ordinary_requested_component_width_admissible",
    "raw_ordinary_nested_replay_certified",
    "raw_ordinary_target_enclosure_computed",
    "raw_ordinary_final_component_width_within_requested_bound",
)


@dataclass(frozen=True)
class RawOrdinaryContinuationCertificate:
    """Raw ordered evidence for one finite ordinary continuation chain."""

    certificate_id: str
    binding: InitialValueProblemBindingCertificate
    charts: tuple[OrdinaryTaylorChartCertificate, ...]
    tubes: tuple[OrdinaryAposterioriTubeCertificate, ...]
    transitions: tuple[OrdinaryEnclosureTransitionCertificate, ...]
    requested_target_time: float
    requested_maximum_component_width: float
    schema_version: int = _SCHEMA_VERSION
    certificate_type: str = _CERTIFICATE_TYPE
    source: str = "raw_ordinary_continuation_v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON data model for this raw evidence."""

        _require_raw_component_types(self)
        return {
            "certificate_id": self.certificate_id,
            "binding": _json_container(self.binding.to_dict()),
            "charts": [_json_container(chart.to_dict()) for chart in self.charts],
            "tubes": [_json_container(tube.to_dict()) for tube in self.tubes],
            "transitions": [
                _json_container(transition.to_dict())
                for transition in self.transitions
            ],
            "requested_target_time": self.requested_target_time,
            "requested_maximum_component_width": (
                self.requested_maximum_component_width
            ),
            "schema_version": self.schema_version,
            "certificate_type": self.certificate_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawOrdinaryContinuationCertificate":
        """Rebuild raw evidence from its deterministic dictionary form."""

        if type(data) is not dict:
            raise TypeError("raw ordinary continuation evidence must be a dict")
        _require_exact_wire_fields(
            data,
            RawOrdinaryContinuationCertificate,
            "raw ordinary continuation certificate",
        )
        binding_data = _mapping_field(data["binding"], "binding")
        chart_data = _mapping_sequence_field(data["charts"], "charts")
        tube_data = _mapping_sequence_field(data["tubes"], "tubes")
        transition_data = _mapping_sequence_field(
            data["transitions"],
            "transitions",
        )
        _require_exact_wire_fields(
            binding_data,
            InitialValueProblemBindingCertificate,
            "initial-value binding",
        )
        for index, item in enumerate(chart_data):
            _require_exact_wire_fields(
                item,
                OrdinaryTaylorChartCertificate,
                f"ordinary chart {index}",
            )
        for index, item in enumerate(tube_data):
            _require_exact_wire_fields(
                item,
                OrdinaryAposterioriTubeCertificate,
                f"ordinary tube {index}",
            )
        for index, item in enumerate(transition_data):
            _require_exact_wire_fields(
                item,
                OrdinaryEnclosureTransitionCertificate,
                f"ordinary transition {index}",
            )
        if not (
            type(data["certificate_id"]) is str
            and type(data["certificate_type"]) is str
            and type(data["source"]) is str
            and type(data["schema_version"]) is int
            and type(data["requested_target_time"]) is float
            and type(data["requested_maximum_component_width"]) is float
        ):
            raise ValueError("raw ordinary continuation wire scalars are noncanonical")
        certificate = cls(
            certificate_id=data["certificate_id"],
            binding=InitialValueProblemBindingCertificate.from_dict(binding_data),
            charts=tuple(
                OrdinaryTaylorChartCertificate.from_dict(item) for item in chart_data
            ),
            tubes=tuple(
                OrdinaryAposterioriTubeCertificate.from_dict(item) for item in tube_data
            ),
            transitions=tuple(
                OrdinaryEnclosureTransitionCertificate.from_dict(item)
                for item in transition_data
            ),
            requested_target_time=data["requested_target_time"],
            requested_maximum_component_width=data[
                "requested_maximum_component_width"
            ],
            schema_version=data["schema_version"],
            certificate_type=data["certificate_type"],
            source=data["source"],
        )
        if _canonical_json_from_data(data) != canonical_evidence_json(certificate):
            raise ValueError("raw ordinary continuation wire encoding is noncanonical")
        return certificate


@dataclass(frozen=True)
class RawOrdinaryContinuationObligation:
    """One deterministic v1 wrapper or nested-replay obligation."""

    obligation: str
    certified: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation,
            "certified": self.certified,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OrdinaryStateEnclosure:
    """A point-time planar state enclosure retained by the replay result."""

    enclosure_type: str
    physical_time_interval: tuple[float, float]
    position_intervals: tuple[tuple[tuple[float, float], ...], ...]
    velocity_intervals: tuple[tuple[tuple[float, float], ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_container(
            {
                "enclosure_type": self.enclosure_type,
                "physical_time_interval": self.physical_time_interval,
                "position_intervals": self.position_intervals,
                "velocity_intervals": self.velocity_intervals,
            }
        )


@dataclass(frozen=True)
class RawOrdinaryContinuationReplayResult:
    """Frozen fail-closed result of freshly replaying one raw certificate."""

    certificate_id: str
    raw_certificate: RawOrdinaryContinuationCertificate
    status: str
    evidence_sha256: str
    obligations: tuple[RawOrdinaryContinuationObligation, ...]
    first_failed_obligation: str | None
    covered_physical_time_interval: tuple[float, float]
    requested_target_time: float
    requested_maximum_component_width: float
    maximum_final_component_width: float
    final_enclosure: OrdinaryStateEnclosure | None
    retained_safe_regions: tuple[OrdinaryStateEnclosure, ...]
    nested_checker_id: str
    nested_missing_obligations: tuple[str, ...]
    schema_version: int = _SCHEMA_VERSION
    checker_id: str = _CHECKER_ID

    def _snapshot_certified(self) -> bool:
        return bool(
            type(self.schema_version) is int
            and self.schema_version == _SCHEMA_VERSION
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and self.status == CERTIFIED_TO_T
            and self.first_failed_obligation is None
            and type(self.raw_certificate) is RawOrdinaryContinuationCertificate
            and self.certificate_id == self.raw_certificate.certificate_id
            and self.requested_target_time
            == self.raw_certificate.requested_target_time
            and self.requested_maximum_component_width
            == self.raw_certificate.requested_maximum_component_width
            and type(self.evidence_sha256) is str
            and len(self.evidence_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.evidence_sha256
            )
            and type(self.obligations) is tuple
            and bool(self.obligations)
            and tuple(
                obligation.obligation
                for obligation in self.obligations
                if type(obligation) is RawOrdinaryContinuationObligation
            )
            == _OBLIGATION_NAMES
            and all(
                type(obligation) is RawOrdinaryContinuationObligation
                and type(obligation.obligation) is str
                and type(obligation.certified) is bool
                and obligation.certified is True
                and type(obligation.detail) is str
                for obligation in self.obligations
            )
            and type(self.final_enclosure) is OrdinaryStateEnclosure
            and self.final_enclosure.physical_time_interval
            == (self.requested_target_time, self.requested_target_time)
            and _finite_float(self.maximum_final_component_width)
            and self.maximum_final_component_width >= 0.0
            and self.nested_checker_id == _NESTED_CHECKER_ID
            and type(self.nested_missing_obligations) is tuple
            and self.nested_missing_obligations == ()
        )

    @property
    def certified(self) -> bool:
        """Freshly replay the retained raw evidence and bind every result field."""

        if not self._snapshot_certified():
            return False
        try:
            fresh = check_raw_ordinary_continuation(self.raw_certificate)
            return bool(
                type(fresh) is RawOrdinaryContinuationReplayResult
                and fresh._snapshot_certified()
                and fresh == self
            )
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "raw_certificate": self.raw_certificate.to_dict(),
            "status": self.status,
            "evidence_sha256": self.evidence_sha256,
            "obligations": [obligation.to_dict() for obligation in self.obligations],
            "first_failed_obligation": self.first_failed_obligation,
            "covered_physical_time_interval": list(
                self.covered_physical_time_interval
            ),
            "requested_target_time": self.requested_target_time,
            "requested_maximum_component_width": (
                self.requested_maximum_component_width
            ),
            "maximum_final_component_width": self.maximum_final_component_width,
            "final_enclosure": (
                None if self.final_enclosure is None else self.final_enclosure.to_dict()
            ),
            "retained_safe_regions": [
                region.to_dict() for region in self.retained_safe_regions
            ],
            "nested_checker_id": self.nested_checker_id,
            "nested_missing_obligations": list(self.nested_missing_obligations),
            "schema_version": self.schema_version,
            "checker_id": self.checker_id,
        }


def canonical_evidence_json(
    certificate: RawOrdinaryContinuationCertificate,
) -> str:
    """Return sorted compact canonical JSON used by the evidence digest.

    Finite valid evidence is exactly the certificate's ``to_dict`` data encoded
    with sorted keys, compact separators, UTF-8 text, and ``allow_nan=False``.
    Non-finite floats are tagged before encoding solely so invalid evidence can
    still receive a deterministic rejection digest.
    """

    if type(certificate) is not RawOrdinaryContinuationCertificate:
        raise TypeError("certificate must be RawOrdinaryContinuationCertificate")
    _require_raw_component_types(certificate)
    canonical_data = _canonical_json_value(certificate.to_dict())
    return json.dumps(
        canonical_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def raw_ordinary_continuation_evidence_sha256(
    certificate: RawOrdinaryContinuationCertificate,
) -> str:
    """Return the lowercase SHA-256 hex digest of canonical raw evidence."""

    payload = canonical_evidence_json(certificate).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_raw_ordinary_continuation(
    certificate: RawOrdinaryContinuationCertificate,
) -> RawOrdinaryContinuationReplayResult:
    """Freshly replay raw ordinary evidence and fail closed to ``UNRESOLVED``.

    No supplied checker result or boolean is accepted as evidence.  The legacy
    chain manifest is reconstructed here and the existing binding, chart, tube,
    transition, and chain checker path is invoked afresh.
    """

    if type(certificate) is not RawOrdinaryContinuationCertificate:
        raise TypeError("certificate must be RawOrdinaryContinuationCertificate")
    _require_raw_component_types(certificate)
    digest = raw_ordinary_continuation_evidence_sha256(certificate)

    schema_supported = bool(
        type(certificate.schema_version) is int
        and certificate.schema_version == _SCHEMA_VERSION
        and type(certificate.certificate_type) is str
        and certificate.certificate_type == _CERTIFICATE_TYPE
    )
    component_manifest = _component_manifest_exact(certificate)
    identifiers_valid = _primary_identifiers_unique(certificate)
    exact_point_ivp = _binding_is_exact_point_ivp(certificate.binding)
    planar_common_problem = _planar_common_mass_problem(certificate)
    forward_intervals = _forward_positive_intervals(certificate.charts)
    exact_clocks = _exact_affine_clocks(certificate.charts)
    first_origin = _first_chart_matches_binding_origin(certificate)
    tube_anchors = _tube_anchors_match_left_endpoints(certificate)
    exact_handoffs = _transitions_are_exact_endpoint_handoffs(certificate)
    target_covered = _requested_target_is_finite_and_covered(certificate)
    requested_width_admissible = _requested_component_width_admissible(certificate)

    nested_result: ValidatedOrdinaryIVPChainCheckResult | None = None
    nested_error = ""
    try:
        nested_result = check_validated_ordinary_ivp_chain(
            _legacy_chain_manifest(certificate),
            certificate.binding,
            certificate.charts,
            certificate.tubes,
            certificate.transitions,
        )
    except Exception as error:  # Invalid mathematical evidence must fail closed.
        nested_error = type(error).__name__

    nested_certified = bool(
        type(nested_result) is ValidatedOrdinaryIVPChainCheckResult
        and nested_result.certified
    )
    nested_missing = _nested_missing_obligations(nested_result, nested_error)
    final_enclosure = _target_enclosure(certificate, nested_result)
    target_enclosure_computed = final_enclosure is not None
    maximum_width_q = _maximum_component_width_fraction(final_enclosure)
    maximum_width = (
        _fraction_upper_float(maximum_width_q)
        if maximum_width_q is not None
        else math.inf
    )
    width_within_request = bool(
        requested_width_admissible
        and maximum_width_q is not None
        and maximum_width_q
        <= Fraction.from_float(certificate.requested_maximum_component_width)
    )

    obligations = (
        RawOrdinaryContinuationObligation(
            "raw_ordinary_schema_version_supported",
            schema_supported,
            (
                f"schema_version={certificate.schema_version!r}; "
                f"certificate_type={certificate.certificate_type!r}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_component_manifest_exact",
            component_manifest,
            (
                f"charts={len(certificate.charts)}; tubes={len(certificate.tubes)}; "
                f"transitions={len(certificate.transitions)}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_identifiers_unique_and_nonempty",
            identifiers_valid,
            "all primary wrapper, binding, chart, tube, and transition ids",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_exact_point_ivp",
            exact_point_ivp,
            (
                f"time_tolerance={certificate.binding.time_tolerance!r}; "
                f"position_tolerance={certificate.binding.position_tolerance!r}; "
                f"velocity_tolerance={certificate.binding.velocity_tolerance!r}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_planar_common_mass_problem",
            planar_common_problem,
            f"binding_masses={certificate.binding.masses!r}",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_forward_positive_intervals",
            forward_intervals,
            "every parameter and physical-time interval has positive exact width",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_exact_affine_physical_clocks",
            exact_clocks,
            "every ordinary chart has exact serialized clock t=s+c",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_first_chart_matches_binding_origin",
            first_origin,
            (
                f"binding_parameter={certificate.binding.chart_parameter!r}; "
                f"binding_time={certificate.binding.initial_time!r}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_tube_anchors_match_chart_left_endpoints",
            tube_anchors,
            "each positional tube anchor equals its chart parameter left endpoint",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_transitions_are_exact_endpoint_handoffs",
            exact_handoffs,
            "each transition is source-right to target-left with zero exact time gap",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_requested_target_finite_and_covered",
            target_covered,
            f"requested_target_time={certificate.requested_target_time!r}",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_requested_component_width_admissible",
            requested_width_admissible,
            (
                "requested_maximum_component_width="
                f"{certificate.requested_maximum_component_width!r}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_nested_replay_certified",
            nested_certified,
            (
                f"checker_id={getattr(nested_result, 'checker_id', '')!r}; "
                f"missing={nested_missing!r}"
            ),
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_target_enclosure_computed",
            target_enclosure_computed,
            f"requested_target_time={certificate.requested_target_time!r}",
        ),
        RawOrdinaryContinuationObligation(
            "raw_ordinary_final_component_width_within_requested_bound",
            width_within_request,
            (
                f"maximum_final_component_width={maximum_width!r}; "
                "requested_maximum_component_width="
                f"{certificate.requested_maximum_component_width!r}"
            ),
        ),
    )
    nested_obligation_index = _OBLIGATION_NAMES.index(
        "raw_ordinary_nested_replay_certified"
    )
    first_wrapper_failure = next(
        (
            obligation.obligation
            for obligation in obligations[:nested_obligation_index]
            if obligation.certified is not True
        ),
        None,
    )
    if first_wrapper_failure is not None:
        first_failed = first_wrapper_failure
    elif not nested_certified:
        nested_blocker = (
            nested_missing[0]
            if nested_missing
            else "raw_ordinary_nested_replay_certified"
        )
        first_failed = f"nested:{nested_blocker}"
    elif not target_enclosure_computed:
        first_failed = "raw_ordinary_target_enclosure_computed"
    elif not width_within_request:
        first_failed = "raw_ordinary_final_component_width_within_requested_bound"
    else:
        first_failed = None
    succeeded = bool(
        first_failed is None
        and final_enclosure is not None
        and width_within_request
    )
    if succeeded:
        covered_interval = (
            float(certificate.binding.initial_time),
            float(certificate.requested_target_time),
        )
        retained = (final_enclosure,)
    else:
        retained, covered_interval = _freshly_certified_prefix(
            certificate,
            nested_result,
        )

    return RawOrdinaryContinuationReplayResult(
        certificate_id=str(certificate.certificate_id),
        raw_certificate=certificate,
        status=CERTIFIED_TO_T if succeeded else UNRESOLVED,
        evidence_sha256=digest,
        obligations=obligations,
        first_failed_obligation=first_failed,
        covered_physical_time_interval=covered_interval,
        requested_target_time=float(certificate.requested_target_time),
        requested_maximum_component_width=float(
            certificate.requested_maximum_component_width
        ),
        maximum_final_component_width=float(maximum_width),
        final_enclosure=final_enclosure if succeeded else None,
        retained_safe_regions=retained,
        nested_checker_id=(
            nested_result.checker_id
            if type(nested_result) is ValidatedOrdinaryIVPChainCheckResult
            else ""
        ),
        nested_missing_obligations=nested_missing,
    )


def _require_raw_component_types(
    certificate: RawOrdinaryContinuationCertificate,
) -> None:
    if type(certificate.binding) is not InitialValueProblemBindingCertificate:
        raise TypeError("binding must be InitialValueProblemBindingCertificate")
    if type(certificate.charts) is not tuple or not all(
        type(chart) is OrdinaryTaylorChartCertificate for chart in certificate.charts
    ):
        raise TypeError("charts must be a tuple of OrdinaryTaylorChartCertificate")
    if type(certificate.tubes) is not tuple or not all(
        type(tube) is OrdinaryAposterioriTubeCertificate for tube in certificate.tubes
    ):
        raise TypeError("tubes must be a tuple of OrdinaryAposterioriTubeCertificate")
    if type(certificate.transitions) is not tuple or not all(
        type(transition) is OrdinaryEnclosureTransitionCertificate
        for transition in certificate.transitions
    ):
        raise TypeError(
            "transitions must be a tuple of OrdinaryEnclosureTransitionCertificate"
        )


def _component_manifest_exact(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    charts = certificate.charts
    tubes = certificate.tubes
    transitions = certificate.transitions
    return bool(
        charts
        and len(tubes) == len(charts)
        and len(transitions) == len(charts) - 1
        and certificate.binding.chart_id == charts[0].chart_id
        and all(tube.chart_id == chart.chart_id for chart, tube in zip(charts, tubes))
        and all(
            transition.source_chart_id == charts[index].chart_id
            and transition.target_chart_id == charts[index + 1].chart_id
            for index, transition in enumerate(transitions)
        )
    )


def _primary_identifiers_unique(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    identifiers = (
        certificate.certificate_id,
        certificate.binding.binding_id,
        *(chart.certificate_id for chart in certificate.charts),
        *(chart.chart_id for chart in certificate.charts),
        *(tube.tube_id for tube in certificate.tubes),
        *(transition.transition_id for transition in certificate.transitions),
    )
    return bool(
        identifiers
        and all(type(identifier) is str and bool(identifier) for identifier in identifiers)
        and len(set(identifiers)) == len(identifiers)
    )


def _binding_is_exact_point_ivp(
    binding: InitialValueProblemBindingCertificate,
) -> bool:
    tolerances = (
        binding.time_tolerance,
        binding.position_tolerance,
        binding.velocity_tolerance,
    )
    return bool(
        all(_finite_float(value) for value in tolerances)
        and all(Fraction.from_float(value) == 0 for value in tolerances)
    )


def _planar_common_mass_problem(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    binding = certificate.binding
    masses = binding.masses
    return bool(
        _positive_mass_triple(masses)
        and _finite_matrix(binding.positions, rows=3, columns=2)
        and _finite_matrix(binding.velocities, rows=3, columns=2)
        and certificate.charts
        and all(
            _ordinary_chart_has_planar_shape(chart)
            and _positive_mass_triple(chart.masses)
            and chart.masses == masses
            for chart in certificate.charts
        )
    )


def _forward_positive_intervals(
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
) -> bool:
    return bool(
        charts
        and all(
            _positive_float_interval(chart.parameter_interval)
            and _positive_float_interval(chart.physical_time_interval)
            for chart in charts
        )
    )


def _exact_affine_clocks(
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
) -> bool:
    if not charts:
        return False
    for chart in charts:
        parameter = _fraction_interval(chart.parameter_interval)
        physical = _fraction_interval(chart.physical_time_interval)
        if parameter is None or physical is None:
            return False
        if physical[1] - physical[0] != parameter[1] - parameter[0]:
            return False
    return True


def _first_chart_matches_binding_origin(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    if not certificate.charts:
        return False
    parameter = _fraction_interval(certificate.charts[0].parameter_interval)
    physical = _fraction_interval(certificate.charts[0].physical_time_interval)
    binding_parameter = _fraction_float(certificate.binding.chart_parameter)
    binding_time = _fraction_float(certificate.binding.initial_time)
    return bool(
        certificate.binding.chart_id == certificate.charts[0].chart_id
        and parameter is not None
        and physical is not None
        and binding_parameter is not None
        and binding_time is not None
        and parameter[0] == binding_parameter
        and physical[0] == binding_time
    )


def _tube_anchors_match_left_endpoints(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    if len(certificate.tubes) != len(certificate.charts) or not certificate.charts:
        return False
    return _tube_anchor_sequence_matches_left_endpoints(
        certificate.charts,
        certificate.tubes,
    )


def _transitions_are_exact_endpoint_handoffs(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    if len(certificate.transitions) != len(certificate.charts) - 1:
        return False
    return _transition_sequence_is_exact_endpoint_handoffs(
        certificate.charts,
        certificate.transitions,
    )


def _requested_target_is_finite_and_covered(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    target = _fraction_float(certificate.requested_target_time)
    initial = _fraction_float(certificate.binding.initial_time)
    if target is None or initial is None or target < initial:
        return False
    return any(
        interval is not None and interval[0] <= target <= interval[1]
        for interval in (
            _fraction_interval(chart.physical_time_interval)
            for chart in certificate.charts
        )
    )


def _requested_component_width_admissible(
    certificate: RawOrdinaryContinuationCertificate,
) -> bool:
    width = certificate.requested_maximum_component_width
    return bool(_finite_float(width) and Fraction.from_float(width) >= 0)


def _legacy_chain_manifest(
    certificate: RawOrdinaryContinuationCertificate,
) -> ValidatedOrdinaryIVPChainCertificate:
    return ValidatedOrdinaryIVPChainCertificate(
        chain_id=f"{certificate.certificate_id}:legacy-chain-v1",
        chart_ids=tuple(chart.chart_id for chart in certificate.charts),
        transition_ids=tuple(
            transition.transition_id for transition in certificate.transitions
        ),
        target_physical_time_interval=(
            certificate.binding.initial_time,
            certificate.requested_target_time,
        ),
        orientation="forward",
        source="raw_ordinary_continuation_v1_internal_manifest",
    )


def _target_enclosure(
    certificate: RawOrdinaryContinuationCertificate,
    nested_result: ValidatedOrdinaryIVPChainCheckResult | None,
) -> OrdinaryStateEnclosure | None:
    if not (
        type(nested_result) is ValidatedOrdinaryIVPChainCheckResult
        and nested_result.target_state_enclosure_certified
        and _fraction_float(nested_result.target_time)
        == _fraction_float(certificate.requested_target_time)
        and _valid_state_intervals(nested_result.target_position_intervals)
        and _valid_state_intervals(nested_result.target_velocity_intervals)
    ):
        return None
    target = float(certificate.requested_target_time)
    return OrdinaryStateEnclosure(
        enclosure_type="validated_ordinary_target_tube",
        physical_time_interval=(target, target),
        position_intervals=nested_result.target_position_intervals,
        velocity_intervals=nested_result.target_velocity_intervals,
    )


def _maximum_component_width_fraction(
    enclosure: OrdinaryStateEnclosure | None,
) -> Fraction | None:
    if enclosure is None or not (
        _valid_state_intervals(enclosure.position_intervals)
        and _valid_state_intervals(enclosure.velocity_intervals)
    ):
        return None
    widths = tuple(
        Fraction.from_float(interval[1]) - Fraction.from_float(interval[0])
        for components in (
            enclosure.position_intervals,
            enclosure.velocity_intervals,
        )
        for body in components
        for interval in body
    )
    if not widths or any(width < 0 for width in widths):
        return None
    return max(widths)


def _freshly_certified_prefix(
    certificate: RawOrdinaryContinuationCertificate,
    nested_result: ValidatedOrdinaryIVPChainCheckResult | None,
) -> tuple[
    tuple[OrdinaryStateEnclosure, ...],
    tuple[float, float],
]:
    unresolved_interval = (math.inf, -math.inf)
    if not (
        type(nested_result) is ValidatedOrdinaryIVPChainCheckResult
        and type(nested_result.first_chart_result)
        is ValidatedOrdinaryIVPChartCheckResult
        and nested_result.first_chart_result.certified
        and _prefix_wrapper_certified(certificate, 0)
    ):
        return (), unresolved_interval

    last_index = 0
    for index, result in enumerate(nested_result.transition_results):
        target_index = index + 1
        if not (
            type(result) is OrdinaryEnclosureTransitionCheckResult
            and result.certified
            and _prefix_wrapper_certified(certificate, target_index)
        ):
            break
        last_index = target_index

    enclosure = _right_endpoint_enclosure(
        certificate,
        nested_result,
        last_index,
    )
    if enclosure is None:
        return (), unresolved_interval
    return (
        (enclosure,),
        (
            float(certificate.binding.initial_time),
            enclosure.physical_time_interval[1],
        ),
    )


def _prefix_wrapper_certified(
    certificate: RawOrdinaryContinuationCertificate,
    chart_index: int,
) -> bool:
    if not (
        type(chart_index) is int
        and 0 <= chart_index < len(certificate.charts)
        and chart_index < len(certificate.tubes)
        and chart_index <= len(certificate.transitions)
    ):
        return False
    charts = certificate.charts[: chart_index + 1]
    tubes = certificate.tubes[: chart_index + 1]
    transitions = certificate.transitions[:chart_index]
    return bool(
        type(certificate.schema_version) is int
        and certificate.schema_version == _SCHEMA_VERSION
        and type(certificate.certificate_type) is str
        and certificate.certificate_type == _CERTIFICATE_TYPE
        and certificate.binding.chart_id == charts[0].chart_id
        and all(tube.chart_id == chart.chart_id for chart, tube in zip(charts, tubes))
        and all(
            transition.source_chart_id == charts[index].chart_id
            and transition.target_chart_id == charts[index + 1].chart_id
            for index, transition in enumerate(transitions)
        )
        and _prefix_primary_identifiers_unique(
            certificate,
            charts,
            tubes,
            transitions,
        )
        and _binding_is_exact_point_ivp(certificate.binding)
        and _prefix_planar_common_mass_problem(certificate.binding, charts)
        and _forward_positive_intervals(charts)
        and _exact_affine_clocks(charts)
        and _first_chart_matches_binding_origin(certificate)
        and _tube_anchor_sequence_matches_left_endpoints(charts, tubes)
        and _transition_sequence_is_exact_endpoint_handoffs(charts, transitions)
    )


def _prefix_primary_identifiers_unique(
    certificate: RawOrdinaryContinuationCertificate,
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
    tubes: tuple[OrdinaryAposterioriTubeCertificate, ...],
    transitions: tuple[OrdinaryEnclosureTransitionCertificate, ...],
) -> bool:
    identifiers = (
        certificate.certificate_id,
        certificate.binding.binding_id,
        *(chart.certificate_id for chart in charts),
        *(chart.chart_id for chart in charts),
        *(tube.tube_id for tube in tubes),
        *(transition.transition_id for transition in transitions),
    )
    return bool(
        identifiers
        and all(type(identifier) is str and bool(identifier) for identifier in identifiers)
        and len(set(identifiers)) == len(identifiers)
    )


def _prefix_planar_common_mass_problem(
    binding: InitialValueProblemBindingCertificate,
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
) -> bool:
    masses = binding.masses
    return bool(
        _positive_mass_triple(masses)
        and _finite_matrix(binding.positions, rows=3, columns=2)
        and _finite_matrix(binding.velocities, rows=3, columns=2)
        and charts
        and all(
            _ordinary_chart_has_planar_shape(chart)
            and _positive_mass_triple(chart.masses)
            and chart.masses == masses
            for chart in charts
        )
    )


def _tube_anchor_sequence_matches_left_endpoints(
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
    tubes: tuple[OrdinaryAposterioriTubeCertificate, ...],
) -> bool:
    if not charts or len(charts) != len(tubes):
        return False
    for chart, tube in zip(charts, tubes):
        anchor = _fraction_float(tube.anchor_parameter)
        parameter = _fraction_interval(chart.parameter_interval)
        if anchor is None or parameter is None or anchor != parameter[0]:
            return False
    return True


def _transition_sequence_is_exact_endpoint_handoffs(
    charts: tuple[OrdinaryTaylorChartCertificate, ...],
    transitions: tuple[OrdinaryEnclosureTransitionCertificate, ...],
) -> bool:
    if not charts or len(transitions) != len(charts) - 1:
        return False
    for index, transition in enumerate(transitions):
        source = charts[index]
        target = charts[index + 1]
        source_parameter = _fraction_float(transition.source_parameter)
        target_parameter = _fraction_float(transition.target_parameter)
        handoff_time = _fraction_float(transition.handoff_time)
        max_time_gap = _fraction_float(transition.max_time_gap)
        source_domain = _fraction_interval(source.parameter_interval)
        target_domain = _fraction_interval(target.parameter_interval)
        source_time = _fraction_interval(source.physical_time_interval)
        target_time = _fraction_interval(target.physical_time_interval)
        if None in {
            source_parameter,
            target_parameter,
            handoff_time,
            max_time_gap,
            source_domain,
            target_domain,
            source_time,
            target_time,
        }:
            return False
        if not (
            transition.source_chart_id == source.chart_id
            and transition.target_chart_id == target.chart_id
            and source_parameter == source_domain[1]  # type: ignore[index]
            and target_parameter == target_domain[0]  # type: ignore[index]
            and handoff_time == source_time[1]  # type: ignore[index]
            and handoff_time == target_time[0]  # type: ignore[index]
            and max_time_gap == 0
        ):
            return False
    return True


def _right_endpoint_enclosure(
    certificate: RawOrdinaryContinuationCertificate,
    nested_result: ValidatedOrdinaryIVPChainCheckResult,
    chart_index: int,
) -> OrdinaryStateEnclosure | None:
    try:
        chart = certificate.charts[chart_index]
        if chart_index == 0:
            tube_result = nested_result.first_chart_result.tube_result
        else:
            transition_result = nested_result.transition_results[chart_index - 1]
            if type(transition_result) is not OrdinaryEnclosureTransitionCheckResult:
                return None
            tube_result = transition_result.target_tube_result
        parameter = Fraction.from_float(chart.parameter_interval[1])
        positions, velocities = _exact_rational_chart_projected_state_at_parameter(
            chart,
            parameter,
        )
        radius = float(tube_result.gronwall_error_bound)
        position_intervals = _fraction_array_tube_intervals(positions, radius)
        velocity_intervals = _fraction_array_tube_intervals(velocities, radius)
        if not (
            _valid_state_intervals(position_intervals)
            and _valid_state_intervals(velocity_intervals)
        ):
            return None
        physical_time = float(chart.physical_time_interval[1])
        return OrdinaryStateEnclosure(
            enclosure_type="validated_ordinary_prefix_right_endpoint",
            physical_time_interval=(physical_time, physical_time),
            position_intervals=position_intervals,
            velocity_intervals=velocity_intervals,
        )
    except (IndexError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        return None


def _nested_missing_obligations(
    nested_result: ValidatedOrdinaryIVPChainCheckResult | None,
    nested_error: str,
) -> tuple[str, ...]:
    if type(nested_result) is ValidatedOrdinaryIVPChainCheckResult:
        try:
            return tuple(nested_result.missing_obligations)
        except Exception as error:
            return (f"nested_missing_obligations_error:{type(error).__name__}",)
    if nested_error:
        return (f"nested_replay_exception:{nested_error}",)
    return ("nested_replay_result_missing",)


def _ordinary_chart_has_planar_shape(
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    positions = chart.position_coefficients
    velocities = chart.velocity_coefficients
    return bool(
        chart.chart_type == "ordinary_taylor"
        and type(positions) is tuple
        and type(velocities) is tuple
        and len(positions) >= 2
        and len(positions) == len(velocities)
        and all(_finite_matrix(degree, rows=3, columns=2) for degree in positions)
        and all(_finite_matrix(degree, rows=3, columns=2) for degree in velocities)
    )


def _positive_mass_triple(values: object) -> bool:
    return bool(
        type(values) is tuple
        and len(values) == 3
        and all(_finite_float(value) and value > 0.0 for value in values)
    )


def _finite_matrix(values: object, *, rows: int, columns: int) -> bool:
    return bool(
        type(values) is tuple
        and len(values) == rows
        and all(
            type(row) is tuple
            and len(row) == columns
            and all(_finite_float(value) for value in row)
            for row in values
        )
    )


def _positive_float_interval(values: object) -> bool:
    interval = _fraction_interval(values)
    return bool(interval is not None and interval[0] < interval[1])


def _fraction_interval(values: object) -> tuple[Fraction, Fraction] | None:
    if not (
        type(values) is tuple
        and len(values) == 2
        and all(_finite_float(value) for value in values)
    ):
        return None
    return Fraction.from_float(values[0]), Fraction.from_float(values[1])


def _fraction_float(value: object) -> Fraction | None:
    if not _finite_float(value):
        return None
    return Fraction.from_float(value)


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _valid_state_intervals(values: object) -> bool:
    return bool(
        type(values) is tuple
        and len(values) == 3
        and all(
            type(body) is tuple
            and len(body) == 2
            and all(
                type(interval) is tuple
                and len(interval) == 2
                and _finite_float(interval[0])
                and _finite_float(interval[1])
                and interval[0] <= interval[1]
                for interval in body
            )
            for body in values
        )
    )


def _mapping_field(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a dict")
    return value


def _mapping_sequence_field(
    value: object,
    name: str,
) -> tuple[dict[str, Any], ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{name} must be a list")
    if not all(type(item) is dict for item in value):
        raise TypeError(f"every {name} item must be a dict")
    return tuple(value)


def _require_exact_wire_fields(
    data: dict[str, Any],
    certificate_type: type[object],
    name: str,
) -> None:
    expected = {field.name for field in dataclass_fields(certificate_type)}
    actual = set(data)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        unknown = tuple(sorted(actual - expected))
        raise ValueError(
            f"{name} fields are noncanonical; "
            f"missing={missing!r}; unknown={unknown!r}"
        )


def _canonical_json_from_data(data: dict[str, Any]) -> str:
    return json.dumps(
        _canonical_json_value(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_container(value: Any) -> Any:
    if type(value) is dict:
        return {key: _json_container(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_json_container(item) for item in value]
    return value


def _canonical_json_value(value: Any) -> Any:
    if type(value) is float and not math.isfinite(value):
        if math.isnan(value):
            name = "NaN"
        elif value > 0.0:
            name = "Infinity"
        else:
            name = "-Infinity"
        return {"__nonfinite_float__": name}
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("canonical evidence JSON requires string dictionary keys")
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_canonical_json_value(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise TypeError(f"unsupported canonical JSON value type: {type(value)!r}")
