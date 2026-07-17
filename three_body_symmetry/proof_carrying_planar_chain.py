"""Private raw-replaying finite planar continuation chains.

Version 1 folds one exact serialized planar three-body IVP through an ordered
finite tagged union of ordinary ``N -> N`` bridges and carried
``N -> LC_ij -> N`` passages.  The wire contains no later IVP binding, clock
origin, gauge choice, checker result, or state enclosure.  Every such object
is derived during fresh replay.

This is a supplied-certificate soundness theorem.  It does not claim that a
producer finds a chain for every initial condition, that an LC passage
contains a collision, or that finite chains establish an all-time or closed-
form solution.  Gauges from distinct LC passages are retained separately and
are never equated across different binary pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, TypeAlias

from .certificate_checker import (
    OrdinaryAposterioriTubeCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    ValidatedOrdinaryIVPChartCheckResult,
    _exact_rational_chart_projected_state_at_parameter,
    _planar_lc_state_intervals,
    _regularized_binary_solution_from_certificate,
    check_ordinary_aposteriori_tube,
    check_planar_lc_aposteriori_tube,
    check_validated_ordinary_ivp_chart,
)
from .certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
)
from .intervals import RationalInterval
from .lc_gauge_gluing import PlanarLCGaugeOverlapEdge
from .planar_lc_mass_coefficients import PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
from .proof_carrying_carried_planar_lc_entry import (
    CarriedPlanarLCEntryResult,
    CarriedPlanarLCEntryTransitionRecord,
    _lc_chart_schema as _entry_lc_chart_schema,
    _lc_tube_schema as _entry_lc_tube_schema,
    _ordinary_chart_schema as _entry_ordinary_chart_schema,
    _ordinary_tube_schema as _entry_ordinary_tube_schema,
    _transition_schema as _entry_transition_schema,
    check_carried_planar_lc_entry,
)
from .proof_carrying_carried_planar_lc_exit import (
    CarriedPlanarLCExitResult,
    _exit_schema as _lc_exit_transition_schema,
    check_carried_planar_lc_exit,
)
from .proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
    _canonical_json_value,
    _json_container,
    _mapping_field,
    _require_exact_wire_fields,
)
from .proof_carrying_ordinary_bridge import (
    CarriedOrdinaryBridgeResult,
    OrdinaryBridgeTransitionRecord,
    _transition_schema as _ordinary_bridge_transition_schema,
    check_carried_ordinary_bridge,
)


_SCHEMA_VERSION = 1
_CERTIFICATE_TYPE = "raw_planar_continuation_chain"
_CERTIFICATE_SOURCE = "raw_planar_continuation_chain_v1"
_ORDINARY_SEGMENT_TYPE = "ordinary_bridge_v1"
_LC_SEGMENT_TYPE = "planar_lc_passage_v1"
_CHECKER_ID = "raw_planar_continuation_chain_replay_checker_v2"
_ROOT_CHECKER_ID = "validated_ordinary_ivp_chart_checker_v1"
_ORDINARY_CHECKER_ID = "carried_ordinary_bridge_checker_v1"
_LC_CHECKER_ID = "carried_planar_lc_exit_checker_v2"
_FIXED_TIME_KERNEL_ID = "exact_rational_horner_fixed_time_kernel_v1"
_CLOCK_LEDGER_ID = "forward_interval_clock_origin_ledger_v1"
_INDUCTION_KERNEL_ID = (
    "exact_root_and_conditional_planar_segment_induction_kernel_v1"
)
_INVALID_DIGEST_PAYLOAD = b"invalid-noncanonical-planar-chain-evidence-v1"

_OBLIGATION_NAMES = (
    "raw_planar_chain_outer_schema_exact",
    "raw_planar_chain_global_identifier_namespace_unique",
    "raw_planar_chain_canonical_evidence_serializable",
    "raw_planar_chain_requested_target_finite",
    "raw_planar_chain_requested_width_admissible",
    "raw_planar_chain_root_exact_point_left_anchor",
    "raw_planar_chain_root_freshly_certified",
    "raw_planar_chain_all_segments_freshly_folded",
    "raw_planar_chain_target_not_before_current_left_clock",
    "raw_planar_chain_fixed_time_preimage_exactly_derived",
    "raw_planar_chain_fixed_time_preimage_inside_forward_current_domain",
    "raw_planar_chain_target_state_exactly_evaluated_and_inflated",
    "raw_planar_chain_final_component_width_within_requested_bound",
)

FractionInterval = tuple[Fraction, Fraction]
FractionMatrix = tuple[tuple[FractionInterval, ...], ...]


@dataclass(frozen=True)
class OrdinaryBridgeV1Segment:
    """Strict raw union arm for one carried ordinary bridge."""

    transition: OrdinaryBridgeTransitionRecord
    target_chart: OrdinaryTaylorChartCertificate
    target_tube: OrdinaryAposterioriTubeCertificate
    segment_type: str = _ORDINARY_SEGMENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        _require_ordinary_segment_types(self)
        return {
            "transition": _json_container(self.transition.to_dict()),
            "target_chart": _json_container(self.target_chart.to_dict()),
            "target_tube": _json_container(self.target_tube.to_dict()),
            "segment_type": self.segment_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryBridgeV1Segment":
        if type(data) is not dict:
            raise TypeError("ordinary bridge segment must be a dict")
        _require_exact_wire_fields(data, cls, "ordinary bridge segment")
        transition = _mapping_field(data["transition"], "transition")
        target_chart = _mapping_field(data["target_chart"], "target_chart")
        target_tube = _mapping_field(data["target_tube"], "target_tube")
        _require_exact_wire_fields(
            transition,
            OrdinaryBridgeTransitionRecord,
            "ordinary bridge transition",
        )
        _require_exact_wire_fields(
            target_chart,
            OrdinaryTaylorChartCertificate,
            "ordinary bridge target chart",
        )
        _require_exact_wire_fields(
            target_tube,
            OrdinaryAposterioriTubeCertificate,
            "ordinary bridge target tube",
        )
        if type(data["segment_type"]) is not str:
            raise ValueError("ordinary segment tag is noncanonical")
        value = cls(
            transition=OrdinaryBridgeTransitionRecord.from_dict(transition),
            target_chart=OrdinaryTaylorChartCertificate.from_dict(target_chart),
            target_tube=OrdinaryAposterioriTubeCertificate.from_dict(target_tube),
            segment_type=data["segment_type"],
        )
        if _canonical_json(data) != _canonical_json(value.to_dict()):
            raise ValueError("ordinary bridge segment encoding is noncanonical")
        return value


@dataclass(frozen=True)
class PlanarLCPassageV1Segment:
    """Strict raw union arm for one carried ``N -> LC_ij -> N`` passage."""

    entry_transition: CarriedPlanarLCEntryTransitionRecord
    lc_chart: PlanarLeviCivitaBinaryChartCertificate
    lc_tube: PlanarLCAposterioriTubeCertificate
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate
    target_chart: OrdinaryTaylorChartCertificate
    target_tube: OrdinaryAposterioriTubeCertificate
    segment_type: str = _LC_SEGMENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        _require_lc_segment_types(self)
        return {
            "entry_transition": _json_container(
                self.entry_transition.to_dict()
            ),
            "lc_chart": _json_container(self.lc_chart.to_dict()),
            "lc_tube": _json_container(self.lc_tube.to_dict()),
            "exit_transition": _json_container(self.exit_transition.to_dict()),
            "target_chart": _json_container(self.target_chart.to_dict()),
            "target_tube": _json_container(self.target_tube.to_dict()),
            "segment_type": self.segment_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLCPassageV1Segment":
        if type(data) is not dict:
            raise TypeError("planar LC passage segment must be a dict")
        _require_exact_wire_fields(data, cls, "planar LC passage segment")
        specs = (
            (
                "entry_transition",
                CarriedPlanarLCEntryTransitionRecord,
            ),
            ("lc_chart", PlanarLeviCivitaBinaryChartCertificate),
            ("lc_tube", PlanarLCAposterioriTubeCertificate),
            (
                "exit_transition",
                PlanarLCToOrdinaryEnclosureTransitionCertificate,
            ),
            ("target_chart", OrdinaryTaylorChartCertificate),
            ("target_tube", OrdinaryAposterioriTubeCertificate),
        )
        mappings: dict[str, dict[str, Any]] = {}
        for name, expected in specs:
            mapping = _mapping_field(data[name], name)
            _require_exact_wire_fields(mapping, expected, name)
            mappings[name] = mapping
        if type(data["segment_type"]) is not str:
            raise ValueError("LC passage segment tag is noncanonical")
        value = cls(
            entry_transition=CarriedPlanarLCEntryTransitionRecord.from_dict(
                mappings["entry_transition"]
            ),
            lc_chart=PlanarLeviCivitaBinaryChartCertificate.from_dict(
                mappings["lc_chart"]
            ),
            lc_tube=PlanarLCAposterioriTubeCertificate.from_dict(
                mappings["lc_tube"]
            ),
            exit_transition=(
                PlanarLCToOrdinaryEnclosureTransitionCertificate.from_dict(
                    mappings["exit_transition"]
                )
            ),
            target_chart=OrdinaryTaylorChartCertificate.from_dict(
                mappings["target_chart"]
            ),
            target_tube=OrdinaryAposterioriTubeCertificate.from_dict(
                mappings["target_tube"]
            ),
            segment_type=data["segment_type"],
        )
        if _canonical_json(data) != _canonical_json(value.to_dict()):
            raise ValueError("planar LC passage segment encoding is noncanonical")
        return value


RawPlanarChainSegment: TypeAlias = (
    OrdinaryBridgeV1Segment | PlanarLCPassageV1Segment
)


@dataclass(frozen=True)
class RawPlanarChainCertificate:
    """Canonical raw evidence for a finite repeated planar chain."""

    certificate_id: str
    root_binding: InitialValueProblemBindingCertificate
    initial_chart: OrdinaryTaylorChartCertificate
    initial_tube: OrdinaryAposterioriTubeCertificate
    segments: tuple[RawPlanarChainSegment, ...]
    requested_target_time: float
    requested_maximum_component_width: float
    schema_version: int = _SCHEMA_VERSION
    certificate_type: str = _CERTIFICATE_TYPE
    source: str = _CERTIFICATE_SOURCE

    def to_dict(self) -> dict[str, Any]:
        _require_raw_certificate_types(self)
        return {
            "certificate_id": self.certificate_id,
            "root_binding": _json_container(self.root_binding.to_dict()),
            "initial_chart": _json_container(self.initial_chart.to_dict()),
            "initial_tube": _json_container(self.initial_tube.to_dict()),
            "segments": [segment.to_dict() for segment in self.segments],
            "requested_target_time": self.requested_target_time,
            "requested_maximum_component_width": (
                self.requested_maximum_component_width
            ),
            "schema_version": self.schema_version,
            "certificate_type": self.certificate_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawPlanarChainCertificate":
        if type(data) is not dict:
            raise TypeError("raw planar chain evidence must be a dict")
        _require_exact_wire_fields(data, cls, "raw planar chain certificate")
        binding = _mapping_field(data["root_binding"], "root_binding")
        initial_chart = _mapping_field(data["initial_chart"], "initial_chart")
        initial_tube = _mapping_field(data["initial_tube"], "initial_tube")
        _require_exact_wire_fields(
            binding,
            InitialValueProblemBindingCertificate,
            "root binding",
        )
        _require_exact_wire_fields(
            initial_chart,
            OrdinaryTaylorChartCertificate,
            "initial chart",
        )
        _require_exact_wire_fields(
            initial_tube,
            OrdinaryAposterioriTubeCertificate,
            "initial tube",
        )
        if type(data["segments"]) is not list:
            raise ValueError("raw planar chain segments must be a canonical list")
        segments: list[RawPlanarChainSegment] = []
        for index, raw_segment in enumerate(data["segments"]):
            segment = _mapping_field(raw_segment, f"segments[{index}]")
            tag = segment.get("segment_type")
            if type(tag) is not str:
                raise ValueError(f"segment {index} tag is noncanonical")
            if tag == _ORDINARY_SEGMENT_TYPE:
                segments.append(OrdinaryBridgeV1Segment.from_dict(segment))
            elif tag == _LC_SEGMENT_TYPE:
                segments.append(PlanarLCPassageV1Segment.from_dict(segment))
            else:
                raise ValueError(f"segment {index} has unknown tag {tag!r}")
        if not (
            type(data["certificate_id"]) is str
            and type(data["requested_target_time"]) is float
            and type(data["requested_maximum_component_width"]) is float
            and type(data["schema_version"]) is int
            and type(data["certificate_type"]) is str
            and type(data["source"]) is str
        ):
            raise ValueError("raw planar chain outer scalars are noncanonical")
        value = cls(
            certificate_id=data["certificate_id"],
            root_binding=InitialValueProblemBindingCertificate.from_dict(binding),
            initial_chart=OrdinaryTaylorChartCertificate.from_dict(initial_chart),
            initial_tube=OrdinaryAposterioriTubeCertificate.from_dict(initial_tube),
            segments=tuple(segments),
            requested_target_time=data["requested_target_time"],
            requested_maximum_component_width=data[
                "requested_maximum_component_width"
            ],
            schema_version=data["schema_version"],
            certificate_type=data["certificate_type"],
            source=data["source"],
        )
        if _canonical_json(data) != canonical_planar_chain_evidence_json(value):
            raise ValueError("raw planar chain encoding is noncanonical")
        return value


@dataclass(frozen=True)
class PlanarChainObligation:
    obligation: str
    certified: bool
    detail: str


@dataclass(frozen=True)
class ClockOriginLedgerEntry:
    vertex_index: int
    chart_id: str
    clock_origin_interval: FractionInterval


@dataclass(frozen=True)
class OrdinaryBridgeCocycleRecord:
    segment_index: int
    transition_id: str
    source_chart_id: str
    target_chart_id: str
    source_clock_origin_interval: FractionInterval
    target_clock_origin_interval: FractionInterval
    exact_parameter_translation: Fraction
    cocycle_type: str = _ORDINARY_SEGMENT_TYPE


@dataclass(frozen=True)
class PlanarLCPassageCocycleRecord:
    segment_index: int
    entry_transition_id: str
    exit_transition_id: str
    source_chart_id: str
    lc_chart_id: str
    target_chart_id: str
    source_clock_origin_interval: FractionInterval
    entry_time_interval: FractionInterval
    exit_time_interval: FractionInterval
    target_clock_origin_interval: FractionInterval
    canonical_pair: tuple[int, int]
    gauge_assignment: tuple[tuple[str, int], ...]
    gauge_edges: tuple[PlanarLCGaugeOverlapEdge, ...]
    cocycle_type: str = _LC_SEGMENT_TYPE


PlanarChainCocycleRecord: TypeAlias = (
    OrdinaryBridgeCocycleRecord | PlanarLCPassageCocycleRecord
)


@dataclass(frozen=True)
class PlanarChainOrdinaryStateEnclosure:
    enclosure_type: str
    physical_time_interval: FractionInterval
    parameter_preimage_interval: FractionInterval
    position_intervals: FractionMatrix
    velocity_intervals: FractionMatrix


@dataclass(frozen=True)
class PlanarChainRetainedRegion:
    region_type: str
    coordinate_system: str
    physical_time_interval: FractionInterval
    parameter_interval: FractionInterval
    component_intervals: tuple[FractionInterval, ...]
    certified_segment_count: int
    provenance_checker_id: str


@dataclass(frozen=True)
class RawPlanarChainReplayResult:
    """Exact-self replay result for success and structured failure."""

    certificate_id: str
    raw_certificate: RawPlanarChainCertificate
    status: str
    evidence_sha256: str
    obligations: tuple[PlanarChainObligation, ...]
    first_failed_obligation: str | None
    certified_segment_count: int
    failed_segment_index: int | None
    failed_segment_missing_obligations: tuple[str, ...]
    certified_transition_checker_ids: tuple[str, ...]
    clock_origin_ledger: tuple[ClockOriginLedgerEntry, ...]
    cocycle_records: tuple[PlanarChainCocycleRecord, ...]
    current_chart_id: str
    current_clock_origin_interval: FractionInterval | tuple[()]
    covered_physical_time_interval: FractionInterval | tuple[()]
    target_parameter_preimage_interval: FractionInterval | tuple[()]
    maximum_final_component_width: Fraction | None
    final_enclosure: PlanarChainOrdinaryStateEnclosure | None
    retained_regions: tuple[PlanarChainRetainedRegion, ...]
    requested_target_time: float | None
    requested_maximum_component_width: float | None
    schema_version: int = _SCHEMA_VERSION
    checker_id: str = _CHECKER_ID
    induction_kernel_id: str = _INDUCTION_KERNEL_ID
    fixed_time_kernel_id: str = _FIXED_TIME_KERNEL_ID
    clock_ledger_id: str = _CLOCK_LEDGER_ID
    mass_arithmetic_kernel_id: str = PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID

    def _snapshot_well_formed(self) -> bool:
        if not (
            type(self) is RawPlanarChainReplayResult
            and type(self.raw_certificate) is RawPlanarChainCertificate
            and type(self.certificate_id) is str
            and self.certificate_id == self.raw_certificate.certificate_id
            and type(self.status) is str
            and self.status in {CERTIFIED_TO_T, UNRESOLVED}
            and type(self.schema_version) is int
            and self.schema_version == _SCHEMA_VERSION
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.induction_kernel_id) is str
            and self.induction_kernel_id == _INDUCTION_KERNEL_ID
            and type(self.fixed_time_kernel_id) is str
            and self.fixed_time_kernel_id == _FIXED_TIME_KERNEL_ID
            and type(self.clock_ledger_id) is str
            and self.clock_ledger_id == _CLOCK_LEDGER_ID
            and type(self.mass_arithmetic_kernel_id) is str
            and self.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and _sha256_hex(self.evidence_sha256)
            and _obligation_ledger_schema(self.obligations)
            and type(self.certified_segment_count) is int
            and 0 <= self.certified_segment_count <= len(self.raw_certificate.segments)
            and (
                self.failed_segment_index is None
                or (
                    type(self.failed_segment_index) is int
                    and 0 <= self.failed_segment_index < len(self.raw_certificate.segments)
                )
            )
            and type(self.failed_segment_missing_obligations) is tuple
            and all(
                type(value) is str and bool(value)
                for value in self.failed_segment_missing_obligations
            )
            and type(self.certified_transition_checker_ids) is tuple
            and len(self.certified_transition_checker_ids)
            == self.certified_segment_count
            and all(
                type(value) is str
                and value in {_ORDINARY_CHECKER_ID, _LC_CHECKER_ID}
                for value in self.certified_transition_checker_ids
            )
            and _clock_ledger_schema(
                self.clock_origin_ledger,
                self.certified_segment_count,
            )
            and _cocycle_ledger_schema(
                self.cocycle_records,
                self.certified_segment_count,
            )
            and type(self.current_chart_id) is str
            and _optional_fraction_interval(self.current_clock_origin_interval)
            and _optional_fraction_interval(self.covered_physical_time_interval)
            and _optional_fraction_interval(
                self.target_parameter_preimage_interval
            )
            and (
                self.maximum_final_component_width is None
                or (
                    type(self.maximum_final_component_width) is Fraction
                    and self.maximum_final_component_width >= 0
                )
            )
            and (
                self.final_enclosure is None
                or _ordinary_enclosure_schema(self.final_enclosure)
            )
            and type(self.retained_regions) is tuple
            and len(self.retained_regions) <= 1
            and all(_retained_region_schema(item) for item in self.retained_regions)
            and (
                self.requested_target_time is None
                or _finite_float(self.requested_target_time)
            )
            and (
                self.requested_maximum_component_width is None
                or (
                    _finite_float(self.requested_maximum_component_width)
                    and self.requested_maximum_component_width >= 0
                )
            )
        ):
            return False
        try:
            if self.evidence_sha256 != _evidence_digest(self.raw_certificate):
                return False
        except Exception:
            return False

        all_true = all(item.certified is True for item in self.obligations)
        earliest_false = next(
            (
                item.obligation
                for item in self.obligations
                if item.certified is not True
            ),
            None,
        )
        expected_first = earliest_false
        if earliest_false == _OBLIGATION_NAMES[6]:
            expected_first = (
                self.first_failed_obligation
                if type(self.first_failed_obligation) is str
                and self.first_failed_obligation.startswith("root:")
                else ""
            )
        elif earliest_false == _OBLIGATION_NAMES[7]:
            prefix = f"segment[{self.failed_segment_index}]:"
            expected_first = (
                self.first_failed_obligation
                if type(self.first_failed_obligation) is str
                and self.first_failed_obligation.startswith(prefix)
                else ""
            )
        if self.first_failed_obligation != expected_first:
            return False
        root_certified = self.obligations[6].certified is True
        if root_certified:
            if not (
                bool(self.clock_origin_ledger)
                and bool(self.current_chart_id)
                and _fraction_interval(self.current_clock_origin_interval)
                and self.clock_origin_ledger[-1].chart_id
                == self.current_chart_id
                and self.clock_origin_ledger[-1].clock_origin_interval
                == self.current_clock_origin_interval
                and _prefix_ledgers_bind_raw(self)
            ):
                return False
        elif not (
            self.certified_segment_count == 0
            and self.clock_origin_ledger == ()
            and self.cocycle_records == ()
            and self.certified_transition_checker_ids == ()
            and self.current_chart_id == ""
            and self.current_clock_origin_interval == ()
            and self.retained_regions == ()
        ):
            return False
        if self.failed_segment_index is None:
            if self.failed_segment_missing_obligations != ():
                return False
        elif not (
            self.failed_segment_index == self.certified_segment_count
            and bool(self.failed_segment_missing_obligations)
        ):
            return False
        if any(
            region.certified_segment_count != self.certified_segment_count
            for region in self.retained_regions
        ):
            return False
        if self.requested_target_time != (
            self.raw_certificate.requested_target_time
            if _finite_float(self.raw_certificate.requested_target_time)
            else None
        ):
            return False
        raw_width = self.raw_certificate.requested_maximum_component_width
        if self.requested_maximum_component_width != (
            raw_width
            if _finite_float(raw_width) and raw_width >= 0
            else None
        ):
            return False
        if self.final_enclosure is None:
            if self.maximum_final_component_width is not None:
                return False
        elif not (
            type(self.maximum_final_component_width) is Fraction
            and self.maximum_final_component_width
            == _maximum_component_width(self.final_enclosure)
            and self.target_parameter_preimage_interval
            == self.final_enclosure.parameter_preimage_interval
        ):
            return False
        if self.status == CERTIFIED_TO_T:
            return bool(
                all_true
                and self.first_failed_obligation is None
                and self.failed_segment_index is None
                and self.failed_segment_missing_obligations == ()
                and self.certified_segment_count
                == len(self.raw_certificate.segments)
                and _fraction_interval(self.current_clock_origin_interval)
                and _fraction_interval(self.covered_physical_time_interval)
                and _fraction_interval(self.target_parameter_preimage_interval)
                and type(self.maximum_final_component_width) is Fraction
                and type(self.final_enclosure)
                is PlanarChainOrdinaryStateEnclosure
                and _ordinary_enclosure_schema(self.final_enclosure)
                and self.retained_regions == ()
            )
        unresolved_coherent = bool(
            self.status == UNRESOLVED
            and type(self.first_failed_obligation) is str
            and bool(self.first_failed_obligation)
            and not all_true
        )
        if not unresolved_coherent:
            return False
        if self.final_enclosure is not None:
            return bool(
                len(self.retained_regions) == 1
                and self.retained_regions[0].region_type
                == "certified_ordinary_fixed_time_enclosure"
                and self.retained_regions[0].physical_time_interval
                == self.final_enclosure.physical_time_interval
                and self.retained_regions[0].parameter_interval
                == self.final_enclosure.parameter_preimage_interval
            )
        return True

    @property
    def replay_consistent(self) -> bool:
        try:
            if (
                type(self) is not RawPlanarChainReplayResult
                or not self._snapshot_well_formed()
            ):
                return False
            fresh = check_raw_planar_chain(self.raw_certificate)
            return bool(
                type(fresh) is RawPlanarChainReplayResult
                and fresh._snapshot_well_formed()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def certified(self) -> bool:
        return bool(self.status == CERTIFIED_TO_T and self.replay_consistent)


def canonical_planar_chain_evidence_json(
    certificate: RawPlanarChainCertificate,
) -> str:
    if type(certificate) is not RawPlanarChainCertificate:
        raise TypeError("certificate must be RawPlanarChainCertificate")
    _require_raw_certificate_types(certificate)
    return json.dumps(
        _canonical_json_value(certificate.to_dict()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def raw_planar_chain_evidence_sha256(
    certificate: RawPlanarChainCertificate,
) -> str:
    return hashlib.sha256(
        canonical_planar_chain_evidence_json(certificate).encode("utf-8")
    ).hexdigest()


def check_raw_planar_chain(
    certificate: RawPlanarChainCertificate,
) -> RawPlanarChainReplayResult:
    """Freshly replay a finite ordinary/planar-LC continuation chain.

    The only induction state is the current ordinary chart/tube together with
    its derived rational clock-origin interval.  A segment advances that state
    only after the corresponding private helper returns an exact-class,
    exact-self-certified snapshot bound to those very raw inputs.
    """

    if type(certificate) is not RawPlanarChainCertificate:
        raise TypeError("certificate must be RawPlanarChainCertificate")
    _require_raw_certificate_types(certificate)

    serializable = True
    try:
        digest = raw_planar_chain_evidence_sha256(certificate)
    except Exception:
        serializable = False
        digest = hashlib.sha256(_INVALID_DIGEST_PAYLOAD).hexdigest()

    outer_schema = _raw_outer_schema(certificate)
    namespace_unique = _global_identifier_namespace_unique(certificate)
    requested_target_valid = _finite_float(certificate.requested_target_time)
    requested_width_valid = bool(
        _finite_float(certificate.requested_maximum_component_width)
        and Fraction.from_float(
            certificate.requested_maximum_component_width
        )
        >= 0
    )

    root_exact = _root_exact_point_left_anchor(certificate)
    root_result: ValidatedOrdinaryIVPChartCheckResult | None = None
    root_error = ""
    try:
        if root_exact:
            root_result = check_validated_ordinary_ivp_chart(
                certificate.root_binding,
                certificate.initial_tube,
                certificate.initial_chart,
            )
    except Exception as error:
        root_error = type(error).__name__
    root_certified = _fresh_root_certified(certificate, root_result)

    current_chart = certificate.initial_chart
    current_tube = certificate.initial_tube
    current_clock: FractionInterval | tuple[()] = ()
    clock_ledger: list[ClockOriginLedgerEntry] = []
    cocycles: list[PlanarChainCocycleRecord] = []
    checker_ids: list[str] = []
    certified_count = 0
    failed_segment_index: int | None = None
    failed_segment_missing: tuple[str, ...] = ()
    failed_lc_segment: PlanarLCPassageV1Segment | None = None

    if root_certified:
        initial = Fraction.from_float(certificate.root_binding.initial_time)
        parameter = Fraction.from_float(
            certificate.root_binding.chart_parameter
        )
        current_clock = (initial - parameter, initial - parameter)
        clock_ledger.append(
            ClockOriginLedgerEntry(
                vertex_index=0,
                chart_id=current_chart.chart_id,
                clock_origin_interval=current_clock,
            )
        )

        for index, segment in enumerate(certificate.segments):
            if type(segment) is OrdinaryBridgeV1Segment:
                result: CarriedOrdinaryBridgeResult | None = None
                error = ""
                try:
                    result = check_carried_ordinary_bridge(
                        segment.transition,
                        current_chart,
                        current_tube,
                        segment.target_chart,
                        segment.target_tube,
                        current_clock,
                    )
                except Exception as exception:
                    error = type(exception).__name__
                if not _fresh_ordinary_advance_certified(
                    result,
                    segment,
                    current_chart,
                    current_tube,
                    current_clock,
                ):
                    failed_segment_index = index
                    failed_segment_missing = _nested_missing(
                        result,
                        error,
                        "ordinary_bridge_result_missing",
                    )
                    break
                assert result is not None
                target_clock = result.target_clock_origin_interval
                source_parameter = Fraction.from_float(
                    segment.transition.source_parameter
                )
                target_parameter = Fraction.from_float(
                    segment.transition.target_parameter
                )
                cocycles.append(
                    OrdinaryBridgeCocycleRecord(
                        segment_index=index,
                        transition_id=segment.transition.transition_id,
                        source_chart_id=current_chart.chart_id,
                        target_chart_id=segment.target_chart.chart_id,
                        source_clock_origin_interval=current_clock,
                        target_clock_origin_interval=target_clock,
                        exact_parameter_translation=(
                            source_parameter - target_parameter
                        ),
                    )
                )
                current_chart = segment.target_chart
                current_tube = segment.target_tube
                current_clock = target_clock
                checker_ids.append(_ORDINARY_CHECKER_ID)
            elif type(segment) is PlanarLCPassageV1Segment:
                lc_result: CarriedPlanarLCExitResult | None = None
                error = ""
                try:
                    lc_result = check_carried_planar_lc_exit(
                        segment.entry_transition,
                        current_chart,
                        current_tube,
                        segment.lc_chart,
                        segment.lc_tube,
                        segment.exit_transition,
                        segment.target_chart,
                        segment.target_tube,
                        current_clock,
                    )
                except Exception as exception:
                    error = type(exception).__name__
                if not _fresh_lc_advance_certified(
                    lc_result,
                    segment,
                    current_chart,
                    current_tube,
                    current_clock,
                ):
                    failed_segment_index = index
                    failed_segment_missing = _nested_missing(
                        lc_result,
                        error,
                        "carried_lc_exit_result_missing",
                    )
                    failed_lc_segment = segment
                    break
                assert lc_result is not None
                assert lc_result.entry_result is not None
                entry_result = lc_result.entry_result
                target_clock = lc_result.target_clock_origin_interval
                cocycles.append(
                    PlanarLCPassageCocycleRecord(
                        segment_index=index,
                        entry_transition_id=(
                            segment.entry_transition.transition_id
                        ),
                        exit_transition_id=(
                            segment.exit_transition.transition_id
                        ),
                        source_chart_id=current_chart.chart_id,
                        lc_chart_id=segment.lc_chart.chart_id,
                        target_chart_id=segment.target_chart.chart_id,
                        source_clock_origin_interval=current_clock,
                        entry_time_interval=entry_result.entry_time_interval,
                        exit_time_interval=lc_result.exit_time_interval,
                        target_clock_origin_interval=target_clock,
                        canonical_pair=segment.lc_chart.pair,
                        gauge_assignment=entry_result.selected_assignment,
                        gauge_edges=entry_result.derived_edges,
                    )
                )
                current_chart = segment.target_chart
                current_tube = segment.target_tube
                current_clock = target_clock
                checker_ids.append(_LC_CHECKER_ID)
            else:  # guarded by the exact raw grammar, retained fail-closed
                failed_segment_index = index
                failed_segment_missing = ("unknown_segment_exact_type",)
                break

            certified_count += 1
            clock_ledger.append(
                ClockOriginLedgerEntry(
                    vertex_index=certified_count,
                    chart_id=current_chart.chart_id,
                    clock_origin_interval=current_clock,
                )
            )

    all_segments_folded = bool(
        root_certified
        and failed_segment_index is None
        and certified_count == len(certificate.segments)
    )

    target_after_left = False
    preimage_derived = False
    preimage_inside = False
    state_evaluated = False
    width_within = False
    preimage: FractionInterval | tuple[()] = ()
    final_enclosure: PlanarChainOrdinaryStateEnclosure | None = None
    maximum_width: Fraction | None = None
    final_tube_result: OrdinaryAposterioriTubeCheckResult | None = None
    target: Fraction | None = None

    if all_segments_folded and _fraction_interval(current_clock):
        try:
            domain = tuple(
                Fraction.from_float(value)
                for value in current_chart.parameter_interval
            )
            if len(domain) != 2:
                raise ValueError("current ordinary domain is malformed")
            left, right = domain
            if requested_target_valid:
                target = Fraction.from_float(certificate.requested_target_time)
                target_after_left = bool(
                    target >= left + current_clock[1]
                )
                candidate = (
                    target - current_clock[1],
                    target - current_clock[0],
                )
                preimage_derived = bool(
                    target_after_left and _fraction_interval(candidate)
                )
                if preimage_derived:
                    preimage = candidate
                    preimage_inside = bool(
                        left <= preimage[0]
                        and preimage[0] <= preimage[1]
                        and preimage[1] <= right
                    )
            if preimage_inside:
                final_tube_result = check_ordinary_aposteriori_tube(
                    current_tube,
                    current_chart,
                )
                if not (
                    type(final_tube_result)
                    is OrdinaryAposterioriTubeCheckResult
                    and final_tube_result.tube_id == current_tube.tube_id
                    and final_tube_result.chart_id == current_chart.chart_id
                    and final_tube_result.certified
                ):
                    raise ValueError("current ordinary tube fresh replay failed")
                radius = Fraction.from_float(
                    final_tube_result.gronwall_error_bound
                )
                positions = _evaluate_and_inflate_matrix(
                    current_chart.position_coefficients,
                    preimage,
                    radius,
                )
                velocities = _evaluate_and_inflate_matrix(
                    current_chart.velocity_coefficients,
                    preimage,
                    radius,
                )
                assert target is not None
                final_enclosure = PlanarChainOrdinaryStateEnclosure(
                    enclosure_type=(
                        "validated_planar_chain_fixed_physical_time"
                    ),
                    physical_time_interval=(target, target),
                    parameter_preimage_interval=preimage,
                    position_intervals=positions,
                    velocity_intervals=velocities,
                )
                state_evaluated = _ordinary_enclosure_schema(final_enclosure)
                if state_evaluated:
                    maximum_width = _maximum_component_width(final_enclosure)
                    width_within = bool(
                        requested_width_valid
                        and maximum_width
                        <= Fraction.from_float(
                            certificate.requested_maximum_component_width
                        )
                    )
        except Exception:
            pass

    obligations = (
        _obligation(
            _OBLIGATION_NAMES[0],
            outer_schema,
            "exact v1 outer manifest and strict tagged segment union",
        ),
        _obligation(
            _OBLIGATION_NAMES[1],
            namespace_unique,
            "all primary and reserved derived identifiers are globally unique",
        ),
        _obligation(
            _OBLIGATION_NAMES[2],
            serializable,
            f"digest={digest}",
        ),
        _obligation(
            _OBLIGATION_NAMES[3],
            requested_target_valid,
            f"T={certificate.requested_target_time!r}",
        ),
        _obligation(
            _OBLIGATION_NAMES[4],
            requested_width_valid,
            f"width={certificate.requested_maximum_component_width!r}",
        ),
        _obligation(
            _OBLIGATION_NAMES[5],
            root_exact,
            "zero-tolerance point IVP; binding and tube anchored at N-left",
        ),
        _obligation(
            _OBLIGATION_NAMES[6],
            root_certified,
            (
                f"checker={getattr(root_result, 'checker_id', '')!r}; "
                f"missing={_root_missing(root_result, root_error)!r}"
            ),
        ),
        _obligation(
            _OBLIGATION_NAMES[7],
            all_segments_folded,
            (
                f"certified_prefix={certified_count}/"
                f"{len(certificate.segments)}; "
                f"failed={failed_segment_missing!r}"
            ),
        ),
        _obligation(
            _OBLIGATION_NAMES[8],
            target_after_left,
            "T >= upper(a + B_current)",
        ),
        _obligation(
            _OBLIGATION_NAMES[9],
            preimage_derived,
            f"J=[T-B_hi,T-B_lo]={preimage!s}",
        ),
        _obligation(
            _OBLIGATION_NAMES[10],
            preimage_inside,
            (
                f"J={preimage!s}; "
                f"domain={current_chart.parameter_interval!r}"
            ),
        ),
        _obligation(
            _OBLIGATION_NAMES[11],
            state_evaluated,
            "rational interval Horner plus fresh ordinary Gronwall radius",
        ),
        _obligation(
            _OBLIGATION_NAMES[12],
            width_within,
            (
                f"maximum={maximum_width!s}; requested="
                f"{certificate.requested_maximum_component_width!r}"
            ),
        ),
    )

    first_failed = next(
        (item.obligation for item in obligations if item.certified is not True),
        None,
    )
    if first_failed == _OBLIGATION_NAMES[6]:
        missing = _root_missing(root_result, root_error)
        if missing:
            first_failed = f"root:{missing[0]}"
    elif first_failed == _OBLIGATION_NAMES[7] and failed_segment_index is not None:
        nested = failed_segment_missing or ("nested_result_missing",)
        first_failed = f"segment[{failed_segment_index}]:{nested[0]}"

    mathematical_to_t = bool(
        outer_schema
        and namespace_unique
        and serializable
        and requested_target_valid
        and root_exact
        and root_certified
        and all_segments_folded
        and target_after_left
        and preimage_derived
        and preimage_inside
        and state_evaluated
        and type(final_enclosure) is PlanarChainOrdinaryStateEnclosure
        and type(maximum_width) is Fraction
        and target is not None
    )
    succeeded = bool(
        mathematical_to_t
        and requested_width_valid
        and width_within
        and first_failed is None
    )

    retained: tuple[PlanarChainRetainedRegion, ...]
    prefix_ordinary_frontier: PlanarChainRetainedRegion | None = None
    if succeeded:
        retained = ()
    elif mathematical_to_t and final_enclosure is not None:
        retained = (_final_retained_region(final_enclosure, certified_count),)
    elif root_certified and _fraction_interval(current_clock):
        lc_frontier = (
            _fresh_lc_right_frontier(
                failed_lc_segment,
                current_chart,
                current_tube,
                current_clock,
                certified_count,
            )
            if failed_lc_segment is not None
            else None
        )
        prefix_ordinary_frontier = _fresh_ordinary_right_frontier(
            current_chart,
            current_tube,
            current_clock,
            certified_count,
        )
        retained = (
            (lc_frontier,)
            if lc_frontier is not None
            else (
                (prefix_ordinary_frontier,)
                if prefix_ordinary_frontier is not None
                else ()
            )
        )
    else:
        retained = ()

    if mathematical_to_t and target is not None:
        initial_time = Fraction.from_float(
            certificate.root_binding.initial_time
        )
        covered: FractionInterval | tuple[()] = (initial_time, target)
    else:
        covered = _covered_interval(
            certificate,
            retained,
            prefix_ordinary_frontier,
        )

    return RawPlanarChainReplayResult(
        certificate_id=(
            certificate.certificate_id
            if type(certificate.certificate_id) is str
            else ""
        ),
        raw_certificate=certificate,
        status=CERTIFIED_TO_T if succeeded else UNRESOLVED,
        evidence_sha256=digest,
        obligations=obligations,
        first_failed_obligation=first_failed,
        certified_segment_count=certified_count,
        failed_segment_index=failed_segment_index,
        failed_segment_missing_obligations=failed_segment_missing,
        certified_transition_checker_ids=tuple(checker_ids),
        clock_origin_ledger=tuple(clock_ledger),
        cocycle_records=tuple(cocycles),
        current_chart_id=current_chart.chart_id if root_certified else "",
        current_clock_origin_interval=current_clock,
        covered_physical_time_interval=covered,
        target_parameter_preimage_interval=preimage,
        maximum_final_component_width=(
            maximum_width if mathematical_to_t else None
        ),
        final_enclosure=final_enclosure if mathematical_to_t else None,
        retained_regions=retained,
        requested_target_time=(
            certificate.requested_target_time
            if requested_target_valid
            else None
        ),
        requested_maximum_component_width=(
            certificate.requested_maximum_component_width
            if requested_width_valid
            else None
        ),
    )


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(
        _canonical_json_value(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_ordinary_segment_types(segment: OrdinaryBridgeV1Segment) -> None:
    if type(segment) is not OrdinaryBridgeV1Segment:
        raise TypeError("ordinary segment must have exact v1 type")
    specs = (
        (segment.transition, OrdinaryBridgeTransitionRecord, "transition"),
        (segment.target_chart, OrdinaryTaylorChartCertificate, "target_chart"),
        (segment.target_tube, OrdinaryAposterioriTubeCertificate, "target_tube"),
    )
    for value, expected, name in specs:
        if type(value) is not expected:
            raise TypeError(f"{name} must have exact type {expected.__name__}")
    if type(segment.segment_type) is not str:
        raise TypeError("ordinary segment tag must be an exact string")


def _require_lc_segment_types(segment: PlanarLCPassageV1Segment) -> None:
    if type(segment) is not PlanarLCPassageV1Segment:
        raise TypeError("LC segment must have exact v1 type")
    specs = (
        (
            segment.entry_transition,
            CarriedPlanarLCEntryTransitionRecord,
            "entry_transition",
        ),
        (
            segment.lc_chart,
            PlanarLeviCivitaBinaryChartCertificate,
            "lc_chart",
        ),
        (segment.lc_tube, PlanarLCAposterioriTubeCertificate, "lc_tube"),
        (
            segment.exit_transition,
            PlanarLCToOrdinaryEnclosureTransitionCertificate,
            "exit_transition",
        ),
        (segment.target_chart, OrdinaryTaylorChartCertificate, "target_chart"),
        (segment.target_tube, OrdinaryAposterioriTubeCertificate, "target_tube"),
    )
    for value, expected, name in specs:
        if type(value) is not expected:
            raise TypeError(f"{name} must have exact type {expected.__name__}")
    if type(segment.segment_type) is not str:
        raise TypeError("LC segment tag must be an exact string")


def _require_raw_certificate_types(
    certificate: RawPlanarChainCertificate,
) -> None:
    if type(certificate) is not RawPlanarChainCertificate:
        raise TypeError("certificate must have exact RawPlanarChainCertificate type")
    specs = (
        (
            certificate.root_binding,
            InitialValueProblemBindingCertificate,
            "root_binding",
        ),
        (
            certificate.initial_chart,
            OrdinaryTaylorChartCertificate,
            "initial_chart",
        ),
        (
            certificate.initial_tube,
            OrdinaryAposterioriTubeCertificate,
            "initial_tube",
        ),
    )
    for value, expected, name in specs:
        if type(value) is not expected:
            raise TypeError(f"{name} must have exact type {expected.__name__}")
    if type(certificate.segments) is not tuple:
        raise TypeError("segments must be an exact tuple")
    for segment in certificate.segments:
        if type(segment) is OrdinaryBridgeV1Segment:
            _require_ordinary_segment_types(segment)
        elif type(segment) is PlanarLCPassageV1Segment:
            _require_lc_segment_types(segment)
        else:
            raise TypeError("segment must be an exact v1 tagged-union arm")
    scalar_specs = (
        (certificate.certificate_id, str),
        (certificate.requested_target_time, float),
        (certificate.requested_maximum_component_width, float),
        (certificate.schema_version, int),
        (certificate.certificate_type, str),
        (certificate.source, str),
    )
    if any(type(value) is not expected for value, expected in scalar_specs):
        raise TypeError("raw planar chain outer scalars are noncanonical")


def _raw_outer_schema(certificate: RawPlanarChainCertificate) -> bool:
    try:
        return bool(
            type(certificate) is RawPlanarChainCertificate
            and type(certificate.schema_version) is int
            and certificate.schema_version == _SCHEMA_VERSION
            and type(certificate.certificate_type) is str
            and certificate.certificate_type == _CERTIFICATE_TYPE
            and type(certificate.source) is str
            and certificate.source == _CERTIFICATE_SOURCE
            and type(certificate.certificate_id) is str
            and bool(certificate.certificate_id)
            and type(certificate.segments) is tuple
            and all(_segment_schema(item) for item in certificate.segments)
        )
    except Exception:
        return False


def _segment_schema(segment: object) -> bool:
    try:
        if type(segment) is OrdinaryBridgeV1Segment:
            return bool(
                segment.segment_type == _ORDINARY_SEGMENT_TYPE
                and _ordinary_bridge_transition_schema(segment.transition)
                and _entry_ordinary_chart_schema(segment.target_chart)
                and _entry_ordinary_tube_schema(segment.target_tube)
                and OrdinaryBridgeV1Segment.from_dict(segment.to_dict())
                == segment
            )
        if type(segment) is PlanarLCPassageV1Segment:
            return bool(
                segment.segment_type == _LC_SEGMENT_TYPE
                and _entry_transition_schema(segment.entry_transition)
                and _entry_lc_chart_schema(segment.lc_chart)
                and _entry_lc_tube_schema(segment.lc_tube)
                and _lc_exit_transition_schema(segment.exit_transition)
                and _entry_ordinary_chart_schema(segment.target_chart)
                and _entry_ordinary_tube_schema(segment.target_tube)
                and PlanarLCPassageV1Segment.from_dict(segment.to_dict())
                == segment
            )
        return False
    except Exception:
        return False


def _binding_schema(value: object) -> bool:
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

    try:
        return bool(
            type(value) is InitialValueProblemBindingCertificate
            and all(
                type(item) is str and bool(item)
                for item in (value.binding_id, value.chart_id, value.source)
            )
            and type(value.masses) is tuple
            and len(value.masses) == 3
            and all(_finite_float(item) and item > 0 for item in value.masses)
            and _finite_float(value.initial_time)
            and _finite_float(value.chart_parameter)
            and matrix(value.positions)
            and matrix(value.velocities)
            and all(
                _finite_float(item)
                for item in (
                    value.time_tolerance,
                    value.position_tolerance,
                    value.velocity_tolerance,
                )
            )
            and InitialValueProblemBindingCertificate.from_dict(value.to_dict())
            == value
        )
    except Exception:
        return False


def _root_exact_point_left_anchor(
    certificate: RawPlanarChainCertificate,
) -> bool:
    try:
        binding = certificate.root_binding
        chart = certificate.initial_chart
        tube = certificate.initial_tube
        left = Fraction.from_float(chart.parameter_interval[0])
        parameter = Fraction.from_float(binding.chart_parameter)
        anchor = Fraction.from_float(tube.anchor_parameter)
        return bool(
            _binding_schema(binding)
            and _entry_ordinary_chart_schema(chart)
            and _entry_ordinary_tube_schema(tube)
            and binding.chart_id == chart.chart_id == tube.chart_id
            and binding.masses == chart.masses
            and all(
                Fraction.from_float(value) == 0
                for value in (
                    binding.time_tolerance,
                    binding.position_tolerance,
                    binding.velocity_tolerance,
                )
            )
            and parameter == left == anchor
        )
    except Exception:
        return False


def _fresh_root_certified(
    certificate: RawPlanarChainCertificate,
    result: ValidatedOrdinaryIVPChartCheckResult | None,
) -> bool:
    try:
        return bool(
            type(result) is ValidatedOrdinaryIVPChartCheckResult
            and result.checker_id == _ROOT_CHECKER_ID
            and result.certified
            and result.binding_result.binding_id
            == certificate.root_binding.binding_id
            and result.binding_result.chart_id
            == certificate.initial_chart.chart_id
            and result.tube_result.tube_id == certificate.initial_tube.tube_id
            and result.tube_result.chart_id
            == certificate.initial_chart.chart_id
            and result.chart_result.certificate_id
            == certificate.initial_chart.certificate_id
        )
    except Exception:
        return False


def _fresh_ordinary_advance_certified(
    result: CarriedOrdinaryBridgeResult | None,
    segment: OrdinaryBridgeV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_clock: FractionInterval,
) -> bool:
    try:
        return bool(
            type(result) is CarriedOrdinaryBridgeResult
            and result.checker_id == _ORDINARY_CHECKER_ID
            and result.raw_transition is segment.transition
            and result.raw_source_chart is source_chart
            and result.raw_source_tube is source_tube
            and result.raw_target_chart is segment.target_chart
            and result.raw_target_tube is segment.target_tube
            and result.parent_source_clock_origin_interval == source_clock
            and _fraction_interval(result.target_clock_origin_interval)
            and result.certified
        )
    except Exception:
        return False


def _fresh_lc_advance_certified(
    result: CarriedPlanarLCExitResult | None,
    segment: PlanarLCPassageV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_clock: FractionInterval,
) -> bool:
    try:
        return bool(
            type(result) is CarriedPlanarLCExitResult
            and result.checker_id == _LC_CHECKER_ID
            and type(result.mass_arithmetic_kernel_id) is str
            and result.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and result.raw_entry_transition is segment.entry_transition
            and result.raw_source_chart is source_chart
            and result.raw_source_tube is source_tube
            and result.raw_lc_chart is segment.lc_chart
            and result.raw_lc_tube is segment.lc_tube
            and result.raw_exit_transition is segment.exit_transition
            and result.raw_target_chart is segment.target_chart
            and result.raw_target_tube is segment.target_tube
            and result.parent_source_clock_origin_interval == source_clock
            and type(result.entry_result) is CarriedPlanarLCEntryResult
            and result.entry_result.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and result.entry_result.raw_transition is segment.entry_transition
            and result.entry_result.parent_source_clock_origin_interval
            == source_clock
            and _fraction_interval(result.entry_result.entry_time_interval)
            and _fraction_interval(result.exit_time_interval)
            and _fraction_interval(result.target_clock_origin_interval)
            and result.certified
        )
    except Exception:
        return False


def _global_identifier_namespace_unique(
    certificate: RawPlanarChainCertificate,
) -> bool:
    try:
        identifiers: list[str] = [
            certificate.certificate_id,
            certificate.root_binding.binding_id,
            certificate.initial_chart.certificate_id,
            certificate.initial_chart.chart_id,
            certificate.initial_tube.tube_id,
        ]
        for segment in certificate.segments:
            if type(segment) is OrdinaryBridgeV1Segment:
                identifiers.extend(
                    (
                        segment.transition.transition_id,
                        segment.target_chart.certificate_id,
                        segment.target_chart.chart_id,
                        segment.target_tube.tube_id,
                    )
                )
            elif type(segment) is PlanarLCPassageV1Segment:
                prefix = segment.entry_transition.transition_id
                identifiers.extend(
                    (
                        prefix,
                        f"{prefix}:derived-gauge-cover",
                        f"{prefix}:patch:0-upper",
                        f"{prefix}:patch:0-lower",
                        f"{prefix}:patch:0-right",
                        f"{prefix}:patch:1-lower",
                        f"{prefix}:negative-axis-overlap",
                        segment.lc_chart.certificate_id,
                        segment.lc_chart.chart_id,
                        segment.lc_tube.tube_id,
                        segment.exit_transition.transition_id,
                        segment.target_chart.certificate_id,
                        segment.target_chart.chart_id,
                        segment.target_tube.tube_id,
                    )
                )
            else:
                return False
        return bool(
            all(type(item) is str and bool(item) for item in identifiers)
            and len(identifiers) == len(set(identifiers))
        )
    except Exception:
        return False


def _evaluate_and_inflate_matrix(
    coefficients: tuple[tuple[tuple[float, ...], ...], ...],
    parameter: FractionInterval,
    radius: Fraction,
) -> FractionMatrix:
    """Exact rational interval Horner evaluation and symmetric inflation."""

    if not (
        _fraction_interval(parameter)
        and type(radius) is Fraction
        and radius >= 0
    ):
        raise ValueError("valid rational parameter interval and radius required")
    if not (
        type(coefficients) is tuple
        and len(coefficients) >= 2
        and all(
            type(degree) is tuple
            and len(degree) == 3
            and all(
                type(row) is tuple
                and len(row) == 2
                and all(_finite_float(value) for value in row)
                for row in degree
            )
            for degree in coefficients
        )
    ):
        raise ValueError("ordinary coefficient matrix has noncanonical shape")
    variable = RationalInterval(parameter[0], parameter[1])
    rows: list[tuple[FractionInterval, ...]] = []
    for body in range(3):
        row: list[FractionInterval] = []
        for axis in range(2):
            value = RationalInterval.point(0)
            for degree in reversed(coefficients):
                coefficient = RationalInterval.point(
                    Fraction.from_float(degree[body][axis])
                )
                value = value * variable + coefficient
            row.append((value.lower - radius, value.upper + radius))
        rows.append(tuple(row))
    return tuple(rows)


def _maximum_component_width(
    enclosure: PlanarChainOrdinaryStateEnclosure,
) -> Fraction:
    if not _ordinary_enclosure_schema(enclosure):
        raise ValueError("valid ordinary enclosure required")
    widths = tuple(
        upper - lower
        for matrix in (enclosure.position_intervals, enclosure.velocity_intervals)
        for row in matrix
        for lower, upper in row
    )
    if not widths or any(width < 0 for width in widths):
        raise ValueError("invalid component width")
    return max(widths)


def _fresh_ordinary_right_frontier(
    chart: OrdinaryTaylorChartCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    clock: FractionInterval,
    certified_segment_count: int,
) -> PlanarChainRetainedRegion | None:
    try:
        if not _fraction_interval(clock):
            return None
        replay = check_ordinary_aposteriori_tube(tube, chart)
        if not (
            type(replay) is OrdinaryAposterioriTubeCheckResult
            and replay.tube_id == tube.tube_id
            and replay.chart_id == chart.chart_id
            and replay.certified
        ):
            return None
        parameter = Fraction.from_float(chart.parameter_interval[1])
        positions, velocities = _exact_rational_chart_projected_state_at_parameter(
            chart,
            parameter,
        )
        radius = Fraction.from_float(replay.gronwall_error_bound)
        components = tuple(
            (center - radius, center + radius)
            for matrix in (positions, velocities)
            for center in matrix.reshape(-1)
        )
        if not _fraction_box(components, 12):
            return None
        return PlanarChainRetainedRegion(
            region_type="certified_current_ordinary_right_frontier",
            coordinate_system="ordinary_cartesian_q_then_v_12",
            physical_time_interval=(
                parameter + clock[0],
                parameter + clock[1],
            ),
            parameter_interval=(parameter, parameter),
            component_intervals=components,
            certified_segment_count=certified_segment_count,
            provenance_checker_id=_CHECKER_ID,
        )
    except Exception:
        return None


def _fresh_lc_right_frontier(
    segment: PlanarLCPassageV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_clock: FractionInterval,
    certified_segment_count: int,
) -> PlanarChainRetainedRegion | None:
    """Retain LC-right only when entry and the LC tube replay independently."""

    try:
        entry = check_carried_planar_lc_entry(
            segment.entry_transition,
            source_chart,
            source_tube,
            segment.lc_chart,
            segment.lc_tube,
            source_clock,
        )
        lc_replay = check_planar_lc_aposteriori_tube(
            segment.lc_tube,
            segment.lc_chart,
        )
        if not (
            type(entry) is CarriedPlanarLCEntryResult
            and entry.raw_transition is segment.entry_transition
            and entry.raw_source_chart is source_chart
            and entry.raw_source_tube is source_tube
            and entry.raw_target_chart is segment.lc_chart
            and entry.raw_target_tube is segment.lc_tube
            and entry.parent_source_clock_origin_interval == source_clock
            and entry.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and entry.certified
            and type(lc_replay) is PlanarLCAposterioriTubeCheckResult
            and lc_replay.checker_id
            == "independent_planar_lc_aposteriori_tube_checker_v2"
            and lc_replay.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and lc_replay.tube_id == segment.lc_tube.tube_id
            and lc_replay.chart_id == segment.lc_chart.chart_id
            and lc_replay.certified
        ):
            return None
        solution = _regularized_binary_solution_from_certificate(
            segment.lc_chart
        )
        raw_parameter = segment.lc_chart.parameter_interval[1]
        parameter = Fraction.from_float(raw_parameter)
        state = _planar_lc_state_intervals(
            solution,
            (raw_parameter, raw_parameter),
            inflate=lc_replay.gronwall_error_bound,
        )
        components = tuple(
            (Fraction.from_float(item.lower), Fraction.from_float(item.upper))
            for item in state
        )
        if not _fraction_box(components, 14):
            return None
        return PlanarChainRetainedRegion(
            region_type="certified_lifted_lc_right_frontier",
            coordinate_system="planar_lc_lifted_14",
            physical_time_interval=components[13],
            parameter_interval=(parameter, parameter),
            component_intervals=components,
            certified_segment_count=certified_segment_count,
            provenance_checker_id=_CHECKER_ID,
        )
    except Exception:
        return None


def _final_retained_region(
    enclosure: PlanarChainOrdinaryStateEnclosure,
    certified_segment_count: int,
) -> PlanarChainRetainedRegion:
    if not _ordinary_enclosure_schema(enclosure):
        raise ValueError("valid final ordinary enclosure required")
    components = tuple(
        interval
        for matrix in (enclosure.position_intervals, enclosure.velocity_intervals)
        for row in matrix
        for interval in row
    )
    return PlanarChainRetainedRegion(
        region_type="certified_ordinary_fixed_time_enclosure",
        coordinate_system="ordinary_cartesian_q_then_v_12",
        physical_time_interval=enclosure.physical_time_interval,
        parameter_interval=enclosure.parameter_preimage_interval,
        component_intervals=components,
        certified_segment_count=certified_segment_count,
        provenance_checker_id=_CHECKER_ID,
    )


def _covered_interval(
    certificate: RawPlanarChainCertificate,
    retained: tuple[PlanarChainRetainedRegion, ...],
    prefix_ordinary_frontier: PlanarChainRetainedRegion | None = None,
) -> FractionInterval | tuple[()]:
    try:
        frontiers = retained + (
            (prefix_ordinary_frontier,)
            if prefix_ordinary_frontier is not None
            else ()
        )
        if not frontiers:
            return ()
        initial = Fraction.from_float(certificate.root_binding.initial_time)
        # A frontier time interval encloses an unknown actual endpoint.  Only
        # its lower endpoint is guaranteed to have been reached.  Take the
        # furthest such lower endpoint among independently certified prefix
        # frontiers, never an upper endpoint.
        frontier = max(
            initial,
            *(item.physical_time_interval[0] for item in frontiers),
        )
        return initial, frontier
    except Exception:
        return ()


def _root_missing(
    result: ValidatedOrdinaryIVPChartCheckResult | None,
    error: str,
) -> tuple[str, ...]:
    if type(result) is ValidatedOrdinaryIVPChartCheckResult:
        try:
            return tuple(result.missing_obligations)
        except Exception as exception:
            return (f"root_missing_error:{type(exception).__name__}",)
    if error:
        return (f"root_replay_exception:{error}",)
    return ("validated_root_result_missing",)


def _nested_missing(
    result: object | None,
    error: str,
    missing_name: str,
) -> tuple[str, ...]:
    if type(result) in {CarriedOrdinaryBridgeResult, CarriedPlanarLCExitResult}:
        try:
            return tuple(result.missing_obligations)  # type: ignore[union-attr]
        except Exception as exception:
            return (f"nested_missing_error:{type(exception).__name__}",)
    if error:
        return (f"nested_replay_exception:{error}",)
    return (missing_name,)


def _obligation(name: str, certified: bool, detail: str) -> PlanarChainObligation:
    return PlanarChainObligation(name, bool(certified), detail)


def _finite_float(value: object) -> bool:
    return bool(type(value) is float and math.isfinite(value))


def _sha256_hex(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is Fraction for item in value)
        and value[0] <= value[1]
    )


def _optional_fraction_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and (len(value) == 0 or _fraction_interval(value))
    )


def _fraction_box(value: object, length: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == length
        and all(_fraction_interval(item) for item in value)
    )


def _fraction_matrix(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 3
        and all(
            type(row) is tuple
            and len(row) == 2
            and all(_fraction_interval(item) for item in row)
            for row in value
        )
    )


def _obligation_ledger_schema(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_OBLIGATION_NAMES)
        and all(type(item) is PlanarChainObligation for item in value)
        and tuple(item.obligation for item in value) == _OBLIGATION_NAMES
        and all(
            type(item.obligation) is str
            and type(item.certified) is bool
            and type(item.detail) is str
            for item in value
        )
    )


def _clock_ledger_schema(value: object, certified_count: int) -> bool:
    if type(value) is not tuple:
        return False
    if not value:
        return certified_count == 0
    return bool(
        len(value) == certified_count + 1
        and all(type(item) is ClockOriginLedgerEntry for item in value)
        and all(
            type(item.vertex_index) is int
            and item.vertex_index == index
            and type(item.chart_id) is str
            and bool(item.chart_id)
            and _fraction_interval(item.clock_origin_interval)
            for index, item in enumerate(value)
        )
    )


def _edge_schema(value: object) -> bool:
    return bool(
        type(value) is PlanarLCGaugeOverlapEdge
        and all(
            type(item) is str and bool(item)
            for item in (
                value.overlap_id,
                value.source_chart_id,
                value.target_chart_id,
            )
        )
        and type(value.parity) is int
        and value.parity in (0, 1)
    )


def _assignment_schema(value: object) -> bool:
    return bool(
        type(value) is tuple
        and bool(value)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and bool(item[0])
            and type(item[1]) is int
            and item[1] in (0, 1)
            for item in value
        )
        and len({item[0] for item in value}) == len(value)
    )


def _ordinary_cocycle_schema(
    value: OrdinaryBridgeCocycleRecord,
    expected_index: int,
) -> bool:
    return bool(
        type(value) is OrdinaryBridgeCocycleRecord
        and type(value.segment_index) is int
        and value.segment_index == expected_index
        and all(
            type(item) is str and bool(item)
            for item in (
                value.transition_id,
                value.source_chart_id,
                value.target_chart_id,
            )
        )
        and _fraction_interval(value.source_clock_origin_interval)
        and _fraction_interval(value.target_clock_origin_interval)
        and type(value.exact_parameter_translation) is Fraction
        and type(value.cocycle_type) is str
        and value.cocycle_type == _ORDINARY_SEGMENT_TYPE
        and value.target_clock_origin_interval
        == (
            value.source_clock_origin_interval[0]
            + value.exact_parameter_translation,
            value.source_clock_origin_interval[1]
            + value.exact_parameter_translation,
        )
    )


def _lc_cocycle_schema(
    value: PlanarLCPassageCocycleRecord,
    expected_index: int,
) -> bool:
    return bool(
        type(value) is PlanarLCPassageCocycleRecord
        and type(value.segment_index) is int
        and value.segment_index == expected_index
        and all(
            type(item) is str and bool(item)
            for item in (
                value.entry_transition_id,
                value.exit_transition_id,
                value.source_chart_id,
                value.lc_chart_id,
                value.target_chart_id,
            )
        )
        and _fraction_interval(value.source_clock_origin_interval)
        and _fraction_interval(value.entry_time_interval)
        and _fraction_interval(value.exit_time_interval)
        and _fraction_interval(value.target_clock_origin_interval)
        and type(value.canonical_pair) is tuple
        and value.canonical_pair in ((0, 1), (0, 2), (1, 2))
        and _assignment_schema(value.gauge_assignment)
        and type(value.gauge_edges) is tuple
        and all(_edge_schema(edge) for edge in value.gauge_edges)
        and type(value.cocycle_type) is str
        and value.cocycle_type == _LC_SEGMENT_TYPE
    )


def _cocycle_ledger_schema(value: object, certified_count: int) -> bool:
    if not (type(value) is tuple and len(value) == certified_count):
        return False
    return all(
        _ordinary_cocycle_schema(item, index)
        if type(item) is OrdinaryBridgeCocycleRecord
        else _lc_cocycle_schema(item, index)
        if type(item) is PlanarLCPassageCocycleRecord
        else False
        for index, item in enumerate(value)
    )


def _ordinary_enclosure_schema(value: object) -> bool:
    return bool(
        type(value) is PlanarChainOrdinaryStateEnclosure
        and type(value.enclosure_type) is str
        and value.enclosure_type
        == "validated_planar_chain_fixed_physical_time"
        and _fraction_interval(value.physical_time_interval)
        and value.physical_time_interval[0] == value.physical_time_interval[1]
        and _fraction_interval(value.parameter_preimage_interval)
        and _fraction_matrix(value.position_intervals)
        and _fraction_matrix(value.velocity_intervals)
    )


def _retained_region_schema(value: object) -> bool:
    if type(value) is not PlanarChainRetainedRegion:
        return False
    expected_components = {
        "ordinary_cartesian_q_then_v_12": 12,
        "planar_lc_lifted_14": 14,
    }.get(value.coordinate_system)
    expected_region_types = {
        "certified_current_ordinary_right_frontier",
        "certified_lifted_lc_right_frontier",
        "certified_ordinary_fixed_time_enclosure",
    }
    return bool(
        type(value.region_type) is str
        and value.region_type in expected_region_types
        and type(value.coordinate_system) is str
        and expected_components is not None
        and _fraction_interval(value.physical_time_interval)
        and _fraction_interval(value.parameter_interval)
        and _fraction_box(value.component_intervals, expected_components)
        and type(value.certified_segment_count) is int
        and value.certified_segment_count >= 0
        and type(value.provenance_checker_id) is str
        and value.provenance_checker_id == _CHECKER_ID
    )


def _evidence_digest(certificate: RawPlanarChainCertificate) -> str:
    try:
        return raw_planar_chain_evidence_sha256(certificate)
    except Exception:
        return hashlib.sha256(_INVALID_DIGEST_PAYLOAD).hexdigest()


def _prefix_ledgers_bind_raw(result: RawPlanarChainReplayResult) -> bool:
    """Check that every retained induction vertex/cocycle binds the raw prefix."""

    try:
        certificate = result.raw_certificate
        if not (
            result.clock_origin_ledger[0].chart_id
            == certificate.initial_chart.chart_id
            and result.clock_origin_ledger[0].clock_origin_interval
            == (
                Fraction.from_float(certificate.root_binding.initial_time)
                - Fraction.from_float(certificate.root_binding.chart_parameter),
            )
            * 2
        ):
            return False
        source_chart_id = certificate.initial_chart.chart_id
        for index in range(result.certified_segment_count):
            segment = certificate.segments[index]
            cocycle = result.cocycle_records[index]
            source_clock = result.clock_origin_ledger[index]
            target_clock = result.clock_origin_ledger[index + 1]
            if type(segment) is OrdinaryBridgeV1Segment:
                if not (
                    result.certified_transition_checker_ids[index]
                    == _ORDINARY_CHECKER_ID
                    and type(cocycle) is OrdinaryBridgeCocycleRecord
                    and cocycle.segment_index == index
                    and cocycle.transition_id
                    == segment.transition.transition_id
                    and cocycle.source_chart_id == source_chart_id
                    and cocycle.target_chart_id
                    == segment.target_chart.chart_id
                    and cocycle.source_clock_origin_interval
                    == source_clock.clock_origin_interval
                    and cocycle.target_clock_origin_interval
                    == target_clock.clock_origin_interval
                    and target_clock.chart_id == segment.target_chart.chart_id
                ):
                    return False
                source_chart_id = segment.target_chart.chart_id
            elif type(segment) is PlanarLCPassageV1Segment:
                source_right = Fraction.from_float(
                    segment.entry_transition.source_right_parameter
                )
                target_anchor = Fraction.from_float(
                    segment.exit_transition.target_parameter
                )
                if not (
                    result.certified_transition_checker_ids[index]
                    == _LC_CHECKER_ID
                    and type(cocycle) is PlanarLCPassageCocycleRecord
                    and cocycle.segment_index == index
                    and cocycle.entry_transition_id
                    == segment.entry_transition.transition_id
                    and cocycle.exit_transition_id
                    == segment.exit_transition.transition_id
                    and cocycle.source_chart_id == source_chart_id
                    and cocycle.lc_chart_id == segment.lc_chart.chart_id
                    and cocycle.target_chart_id
                    == segment.target_chart.chart_id
                    and cocycle.canonical_pair == segment.lc_chart.pair
                    and cocycle.source_clock_origin_interval
                    == source_clock.clock_origin_interval
                    and cocycle.entry_time_interval
                    == (
                        source_right
                        + source_clock.clock_origin_interval[0],
                        source_right
                        + source_clock.clock_origin_interval[1],
                    )
                    and cocycle.target_clock_origin_interval
                    == (
                        cocycle.exit_time_interval[0] - target_anchor,
                        cocycle.exit_time_interval[1] - target_anchor,
                    )
                    and cocycle.target_clock_origin_interval
                    == target_clock.clock_origin_interval
                    and target_clock.chart_id == segment.target_chart.chart_id
                ):
                    return False
                source_chart_id = segment.target_chart.chart_id
            else:
                return False
        return bool(source_chart_id == result.current_chart_id)
    except Exception:
        return False
