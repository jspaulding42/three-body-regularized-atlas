"""Proof-grade replay for raw v1 planar continuation chains.

This is deliberately separate from :mod:`proof_carrying_planar_chain`'s
historical claimed-tail composition.  The raw wire grammar is shared, but the
root and planar-LC legs are re-established with their binary64-outward,
direct-evidence profiles.  Claimed Taylor-tail checks retained by those local
profiles are diagnostics, not proof prerequisites.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from typing import TypeAlias

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
from .planar_lc_mass_coefficients import PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
from .proof_carrying_carried_planar_lc_exit import (
    ProofGradeCarriedPlanarLCExitResult,
    check_proof_grade_carried_planar_lc_exit,
)
from .proof_carrying_continuation import CERTIFIED_TO_T, UNRESOLVED
from .proof_carrying_ordinary_bridge import (
    CarriedOrdinaryBridgeResult,
    check_carried_ordinary_bridge,
)
from .proof_carrying_planar_chain import (
    ClockOriginLedgerEntry,
    OrdinaryBridgeCocycleRecord,
    OrdinaryBridgeV1Segment,
    PlanarChainCocycleRecord,
    PlanarLCPassageCocycleRecord,
    PlanarLCPassageV1Segment,
    RawPlanarChainCertificate,
    _evaluate_and_inflate_matrix,
    _finite_float,
    _fraction_box,
    _fraction_interval,
    _global_identifier_namespace_unique,
    _raw_outer_schema,
    _require_raw_certificate_types,
    _root_exact_point_left_anchor,
    canonical_planar_chain_evidence_json,
)
from .proof_carrying_proof_grade_ordinary_root import (
    ProofGradeValidatedOrdinaryRootResult,
    check_proof_grade_validated_ordinary_root,
)


_CHECKER_ID = "proof_grade_raw_planar_chain_checker_v04"
_PROFILE_ID = "binary64_outward_proof_grade_raw_planar_chain_v04"
_FIXED_TIME_KERNEL_ID = "exact_rational_horner_fixed_time_kernel_v1"
_CLOCK_LEDGER_ID = "forward_interval_clock_origin_ledger_v1"
_INDUCTION_KERNEL_ID = "proof_grade_direct_evidence_planar_segment_induction_v04"
_HARD_MAX_SEGMENTS = 256
_INVALID_DIGEST_PAYLOAD = b"invalid-proof-grade-raw-planar-chain-evidence-v04"

PROOF_GRADE_RAW_PLANAR_CHAIN_CHECKER_ID = _CHECKER_ID
PROOF_GRADE_RAW_PLANAR_CHAIN_PROFILE_ID = _PROFILE_ID
BINARY64_OUTWARD_PROOF_GRADE_RAW_PLANAR_CHAIN_V04_PROFILE_ID = _PROFILE_ID
HARD_MAX_PROOF_GRADE_RAW_PLANAR_CHAIN_SEGMENTS = _HARD_MAX_SEGMENTS

PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS = (
    "proof_grade_raw_planar_chain_outer_schema_exact",
    "proof_grade_raw_planar_chain_global_identifier_namespace_unique",
    "proof_grade_raw_planar_chain_canonical_evidence_serializable",
    "proof_grade_raw_planar_chain_requested_target_finite",
    "proof_grade_raw_planar_chain_requested_width_admissible",
    "proof_grade_raw_planar_chain_root_exact_point_left_anchor",
    "proof_grade_raw_planar_chain_root_direct_tube_and_binding_certified",
    "proof_grade_raw_planar_chain_all_segments_direct_evidence_folded",
    "proof_grade_raw_planar_chain_target_not_before_current_left_clock",
    "proof_grade_raw_planar_chain_fixed_time_preimage_exactly_derived",
    "proof_grade_raw_planar_chain_fixed_time_preimage_inside_forward_current_domain",
    "proof_grade_raw_planar_chain_target_state_directly_evaluated_and_inflated",
    "proof_grade_raw_planar_chain_final_component_width_within_requested_bound",
)

FractionInterval = tuple[Fraction, Fraction]
FractionMatrix = tuple[tuple[FractionInterval, ...], ...]
ProofGradeLocalSegmentResult: TypeAlias = (
    CarriedOrdinaryBridgeResult | ProofGradeCarriedPlanarLCExitResult
)


@dataclass(frozen=True)
class ProofGradePlanarChainOrdinaryStateEnclosure:
    """A direct ordinary-tube enclosure at the requested physical time."""

    enclosure_type: str
    physical_time_interval: FractionInterval
    parameter_preimage_interval: FractionInterval
    position_intervals: FractionMatrix
    velocity_intervals: FractionMatrix


@dataclass(frozen=True)
class ProofGradePlanarChainRetainedRegion:
    """A proof-specific retained frontier for an unresolved raw chain."""

    region_type: str
    coordinate_system: str
    physical_time_interval: FractionInterval
    parameter_interval: FractionInterval
    component_intervals: tuple[FractionInterval, ...]
    certified_segment_count: int
    provenance_checker_id: str
    failed_segment_index: int | None
    chart_id: str
    pair: tuple[int, int] | None


@dataclass(frozen=True)
class ProofGradeRawPlanarChainReplayResult:
    """Frozen proof-grade whole-chain replay, including authentic failures."""

    certificate_id: str
    raw_certificate: RawPlanarChainCertificate
    status: str
    evidence_sha256: str
    obligations: tuple[CertificateCheckObligation, ...]
    first_failed_obligation: str | None
    certified_segment_count: int
    failed_segment_index: int | None
    failed_segment_missing_obligations: tuple[str, ...]
    replay_failure: str | None
    root_result: ProofGradeValidatedOrdinaryRootResult | None
    local_segment_results: tuple[ProofGradeLocalSegmentResult, ...]
    certified_transition_checker_ids: tuple[str, ...]
    clock_origin_ledger: tuple[ClockOriginLedgerEntry, ...]
    cocycle_records: tuple[PlanarChainCocycleRecord, ...]
    current_chart: OrdinaryTaylorChartCertificate | None
    current_tube: OrdinaryAposterioriTubeCertificate | None
    current_chart_id: str
    current_clock_origin_interval: FractionInterval | tuple[()]
    covered_physical_time_interval: FractionInterval | tuple[()]
    target_parameter_preimage_interval: FractionInterval | tuple[()]
    final_tube_result: OrdinaryAposterioriTubeCheckResult | None
    maximum_final_component_width: Fraction | None
    final_enclosure: ProofGradePlanarChainOrdinaryStateEnclosure | None
    retained_regions: tuple[ProofGradePlanarChainRetainedRegion, ...]
    requested_target_time: float | None
    requested_maximum_component_width: float | None
    schema_version: int = 1
    checker_id: str = _CHECKER_ID
    profile_id: str = _PROFILE_ID
    induction_kernel_id: str = _INDUCTION_KERNEL_ID
    fixed_time_kernel_id: str = _FIXED_TIME_KERNEL_ID
    clock_ledger_id: str = _CLOCK_LEDGER_ID
    mass_arithmetic_kernel_id: str = PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID

    def _snapshot_well_formed(self) -> bool:
        """Check shape before exact whole-chain replay authenticates values."""

        try:
            if not (
                type(self) is ProofGradeRawPlanarChainReplayResult
                and type(self.raw_certificate) is RawPlanarChainCertificate
                and type(self.certificate_id) is str
                and self.certificate_id == self.raw_certificate.certificate_id
                and self.status in {CERTIFIED_TO_T, UNRESOLVED}
                and type(self.schema_version) is int
                and self.schema_version == 1
                and self.checker_id == _CHECKER_ID
                and self.profile_id == _PROFILE_ID
                and self.induction_kernel_id == _INDUCTION_KERNEL_ID
                and self.fixed_time_kernel_id == _FIXED_TIME_KERNEL_ID
                and self.clock_ledger_id == _CLOCK_LEDGER_ID
                and self.mass_arithmetic_kernel_id
                == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
                and _sha256_hex(self.evidence_sha256)
                and _ledger_schema(self.obligations)
                and (
                    self.replay_failure is None
                    or (
                        type(self.replay_failure) is str
                        and bool(self.replay_failure)
                    )
                )
                and type(self.certified_segment_count) is int
                and 0 <= self.certified_segment_count <= len(self.raw_certificate.segments)
                and type(self.failed_segment_missing_obligations) is tuple
                and all(type(item) is str and bool(item) for item in self.failed_segment_missing_obligations)
                and type(self.local_segment_results) is tuple
                and len(self.local_segment_results)
                in {self.certified_segment_count, self.certified_segment_count + 1}
                and all(type(item) in {CarriedOrdinaryBridgeResult, ProofGradeCarriedPlanarLCExitResult} for item in self.local_segment_results)
                and type(self.certified_transition_checker_ids) is tuple
                and len(self.certified_transition_checker_ids) == self.certified_segment_count
                and all(item in {"carried_ordinary_bridge_checker_v1", "proof_grade_carried_planar_lc_exit_checker_v04"} for item in self.certified_transition_checker_ids)
                and _clock_ledger_schema(self.clock_origin_ledger, self.certified_segment_count)
                and len(self.cocycle_records) == self.certified_segment_count
                and all(type(item) in {OrdinaryBridgeCocycleRecord, PlanarLCPassageCocycleRecord} for item in self.cocycle_records)
                and (self.current_chart is None or type(self.current_chart) is OrdinaryTaylorChartCertificate)
                and (self.current_tube is None or type(self.current_tube) is OrdinaryAposterioriTubeCertificate)
                and type(self.current_chart_id) is str
                and _optional_fraction_interval(self.current_clock_origin_interval)
                and _optional_fraction_interval(self.covered_physical_time_interval)
                and _optional_fraction_interval(self.target_parameter_preimage_interval)
                and (self.final_tube_result is None or type(self.final_tube_result) is OrdinaryAposterioriTubeCheckResult)
                and (self.maximum_final_component_width is None or (type(self.maximum_final_component_width) is Fraction and self.maximum_final_component_width >= 0))
                and (self.final_enclosure is None or _enclosure_schema(self.final_enclosure))
                and type(self.retained_regions) is tuple
                and len(self.retained_regions) <= 1
                and all(_retained_region_schema(item) for item in self.retained_regions)
                and (self.requested_target_time is None or _finite_float(self.requested_target_time))
                and (self.requested_maximum_component_width is None or _finite_float(self.requested_maximum_component_width))
            ):
                return False
            if self.evidence_sha256 != _evidence_digest(self.raw_certificate):
                return False
            if self.requested_target_time != (
                self.raw_certificate.requested_target_time
                if _finite_float(self.raw_certificate.requested_target_time)
                else None
            ):
                return False
            if self.requested_maximum_component_width != (
                self.raw_certificate.requested_maximum_component_width
                if _finite_float(self.raw_certificate.requested_maximum_component_width)
                else None
            ):
                return False
            if self.failed_segment_index is not None and not (
                type(self.failed_segment_index) is int
                and self.failed_segment_index == self.certified_segment_count
                and 0 <= self.failed_segment_index < len(self.raw_certificate.segments)
                and bool(self.failed_segment_missing_obligations)
            ):
                return False
            if self.failed_segment_index is None and len(self.local_segment_results) != self.certified_segment_count:
                return False
            if self.root_result is not None and type(self.root_result) is not ProofGradeValidatedOrdinaryRootResult:
                return False
            root_ok = bool(
                self.obligations[5].certified is True
                and self.obligations[6].certified is True
            )
            if root_ok:
                if not (
                    type(self.root_result) is ProofGradeValidatedOrdinaryRootResult
                    and self.current_chart is not None
                    and self.current_tube is not None
                    and _fraction_interval(self.current_clock_origin_interval)
                    and bool(self.clock_origin_ledger)
                    and self.clock_origin_ledger[-1].chart_id == self.current_chart_id
                    and self.clock_origin_ledger[-1].clock_origin_interval == self.current_clock_origin_interval
                ):
                    return False
            elif any((self.clock_origin_ledger, self.cocycle_records, self.certified_transition_checker_ids)):
                return False
            if self.final_enclosure is None:
                if self.maximum_final_component_width is not None:
                    return False
            elif not (
                self.maximum_final_component_width == _maximum_component_width(self.final_enclosure)
                and self.target_parameter_preimage_interval == self.final_enclosure.parameter_preimage_interval
            ):
                return False
            all_true = all(item.certified is True for item in self.obligations)
            if self.status == CERTIFIED_TO_T:
                return bool(
                    all_true
                    and self.first_failed_obligation is None
                    and self.failed_segment_index is None
                    and self.replay_failure is None
                    and self.certified_segment_count == len(self.raw_certificate.segments)
                    and self.final_enclosure is not None
                    and self.retained_regions == ()
                )
            return bool(
                not all_true
                and type(self.first_failed_obligation) is str
                and bool(self.first_failed_obligation)
            )
        except Exception:
            return False

    @property
    def replay_consistent(self) -> bool:
        """Authenticate success *and* structured unresolved snapshots anew."""

        try:
            if not self._snapshot_well_formed():
                return False
            fresh = check_proof_grade_raw_planar_chain(self.raw_certificate)
            return bool(
                type(fresh) is ProofGradeRawPlanarChainReplayResult
                and fresh._snapshot_well_formed()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def certified(self) -> bool:
        return bool(self.status == CERTIFIED_TO_T and self.replay_consistent)


def check_proof_grade_raw_planar_chain(
    certificate: RawPlanarChainCertificate,
) -> ProofGradeRawPlanarChainReplayResult:
    """Replay a raw v1 chain with proof-grade root and LC exit evidence.

    Segment commits are transactional.  A failed proof-grade LC leg is still
    freshly recomputed and exact-compared before its false local obligations
    and independently reconstructed lifted right slice are retained.
    """

    if type(certificate) is not RawPlanarChainCertificate:
        raise TypeError("certificate must be RawPlanarChainCertificate")
    _require_raw_certificate_types(certificate)

    serializable = True
    try:
        canonical = canonical_planar_chain_evidence_json(certificate)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        serializable = False
        digest = hashlib.sha256(_INVALID_DIGEST_PAYLOAD).hexdigest()
    outer_schema = _raw_outer_schema(certificate)
    segment_cap_admissible = len(certificate.segments) <= _HARD_MAX_SEGMENTS
    namespace_unique = bool(
        segment_cap_admissible
        and _global_identifier_namespace_unique(certificate)
    )
    target_valid = _finite_float(certificate.requested_target_time)
    width_valid = bool(
        _finite_float(certificate.requested_maximum_component_width)
        and Fraction.from_float(certificate.requested_maximum_component_width) >= 0
    )
    root_exact = _root_exact_point_left_anchor(certificate)

    root_result: ProofGradeValidatedOrdinaryRootResult | None = None
    if root_exact:
        try:
            root_result = check_proof_grade_validated_ordinary_root(
                certificate.root_binding,
                certificate.initial_tube,
                certificate.initial_chart,
            )
        except Exception:
            pass
    root_direct = bool(root_exact and _fresh_proof_root(root_result, certificate))
    root_certified = bool(root_exact and root_direct)

    current_chart: OrdinaryTaylorChartCertificate | None = None
    current_tube: OrdinaryAposterioriTubeCertificate | None = None
    current_clock: FractionInterval | tuple[()] = ()
    clock_ledger: list[ClockOriginLedgerEntry] = []
    cocycles: list[PlanarChainCocycleRecord] = []
    local_results: list[ProofGradeLocalSegmentResult] = []
    checker_ids: list[str] = []
    certified_count = 0
    failed_index: int | None = None
    failed_missing: tuple[str, ...] = ()
    replay_failure: str | None = None
    failed_lifted: tuple[FractionInterval, ...] = ()

    if root_certified:
        current_chart = certificate.initial_chart
        current_tube = certificate.initial_tube
        origin = Fraction.from_float(certificate.root_binding.initial_time) - Fraction.from_float(certificate.root_binding.chart_parameter)
        current_clock = (origin, origin)
        clock_ledger.append(ClockOriginLedgerEntry(0, current_chart.chart_id, current_clock))

        if not segment_cap_admissible:
            replay_failure = f"segment_word_limit_exceeded:actual={len(certificate.segments)};limit={_HARD_MAX_SEGMENTS}"
        else:
            for index, segment in enumerate(certificate.segments):
                assert current_chart is not None and current_tube is not None
                assert _fraction_interval(current_clock)
                if type(segment) is OrdinaryBridgeV1Segment:
                    result: CarriedOrdinaryBridgeResult | None = None
                    try:
                        result = check_carried_ordinary_bridge(
                            segment.transition, current_chart, current_tube,
                            segment.target_chart, segment.target_tube, current_clock,
                        )
                    except Exception as error:
                        replay_failure = f"ordinary_bridge_exception:{type(error).__name__}"
                    if result is not None:
                        local_results.append(result)
                    if not _fresh_ordinary_commit(result, segment, current_chart, current_tube, current_clock):
                        failed_index = index
                        failed_missing = _local_missing(result, "ordinary_bridge_result_missing")
                        replay_failure = replay_failure or "ordinary_bridge_local_obligations"
                        break
                    assert result is not None
                    target_clock = result.target_clock_origin_interval
                    cocycles.append(OrdinaryBridgeCocycleRecord(
                        index, segment.transition.transition_id, current_chart.chart_id,
                        segment.target_chart.chart_id, current_clock, target_clock,
                        Fraction.from_float(segment.transition.source_parameter)
                        - Fraction.from_float(segment.transition.target_parameter),
                    ))
                    current_chart, current_tube, current_clock = (
                        segment.target_chart, segment.target_tube, target_clock,
                    )
                    checker_ids.append("carried_ordinary_bridge_checker_v1")
                elif type(segment) is PlanarLCPassageV1Segment:
                    result: ProofGradeCarriedPlanarLCExitResult | None = None
                    try:
                        result = check_proof_grade_carried_planar_lc_exit(
                            segment.entry_transition, current_chart, current_tube,
                            segment.lc_chart, segment.lc_tube, segment.exit_transition,
                            segment.target_chart, segment.target_tube, current_clock,
                        )
                    except Exception as error:
                        replay_failure = f"proof_grade_lc_exit_exception:{type(error).__name__}"
                    if result is not None:
                        local_results.append(result)
                    if not _fresh_proof_lc_commit(result, segment, current_chart, current_tube, current_clock):
                        failed_index = index
                        failed_missing = _local_missing(result, "proof_grade_lc_exit_result_missing")
                        replay_failure = replay_failure or "proof_grade_lc_exit_local_obligations"
                        # A false local proof result is non-decisive only for
                        # retention: recompute it and exact-compare before use.
                        fresh = _fresh_proof_lc_equal(result, segment, current_chart, current_tube, current_clock)
                        if fresh is not None and _lc_slice_reconstructed(fresh):
                            failed_lifted = fresh.lifted_exit_slice
                        break
                    assert result is not None and result.entry_result is not None
                    target_clock = result.target_clock_origin_interval
                    entry = result.entry_result
                    cocycles.append(PlanarLCPassageCocycleRecord(
                        index, segment.entry_transition.transition_id,
                        segment.exit_transition.transition_id, current_chart.chart_id,
                        segment.lc_chart.chart_id, segment.target_chart.chart_id,
                        current_clock, entry.entry_time_interval,
                        result.exit_time_interval, target_clock, segment.lc_chart.pair,
                        entry.selected_assignment, entry.derived_edges,
                    ))
                    current_chart, current_tube, current_clock = (
                        segment.target_chart, segment.target_tube, target_clock,
                    )
                    checker_ids.append("proof_grade_carried_planar_lc_exit_checker_v04")
                else:
                    failed_index = index
                    failed_missing = ("unknown_segment_exact_type",)
                    replay_failure = "unknown_segment_exact_type"
                    break
                certified_count += 1
                clock_ledger.append(ClockOriginLedgerEntry(
                    certified_count, current_chart.chart_id, current_clock,
                ))

    all_segments = bool(
        root_certified
        and replay_failure is None
        and failed_index is None
        and certified_count == len(certificate.segments)
    )

    target_after_left = False
    preimage_derived = False
    preimage_inside = False
    state_evaluated = False
    width_within = False
    preimage: FractionInterval | tuple[()] = ()
    target: Fraction | None = None
    final_tube: OrdinaryAposterioriTubeCheckResult | None = None
    final_enclosure: ProofGradePlanarChainOrdinaryStateEnclosure | None = None
    maximum_width: Fraction | None = None
    if all_segments and current_chart is not None and current_tube is not None and _fraction_interval(current_clock):
        try:
            left, right = tuple(Fraction.from_float(value) for value in current_chart.parameter_interval)
            if target_valid:
                target = Fraction.from_float(certificate.requested_target_time)
                target_after_left = target >= left + current_clock[1]
                candidate = (target - current_clock[1], target - current_clock[0])
                preimage_derived = bool(target_after_left and _fraction_interval(candidate))
                if preimage_derived:
                    preimage = candidate
                    preimage_inside = left <= preimage[0] <= preimage[1] <= right
            if preimage_inside:
                final_tube = check_ordinary_aposteriori_tube(current_tube, current_chart)
                if not _fresh_final_tube(final_tube, current_tube, current_chart):
                    raise ValueError("direct final ordinary tube is not certified")
                radius = Fraction.from_float(final_tube.gronwall_error_bound)
                positions = _evaluate_and_inflate_matrix(current_chart.position_coefficients, preimage, radius)
                velocities = _evaluate_and_inflate_matrix(current_chart.velocity_coefficients, preimage, radius)
                assert target is not None
                final_enclosure = ProofGradePlanarChainOrdinaryStateEnclosure(
                    "validated_planar_chain_fixed_physical_time",
                    (target, target), preimage, positions, velocities,
                )
                state_evaluated = _enclosure_schema(final_enclosure)
                if state_evaluated:
                    maximum_width = _maximum_component_width(final_enclosure)
                    width_within = bool(width_valid and maximum_width <= Fraction.from_float(certificate.requested_maximum_component_width))
        except Exception:
            pass

    obligations = tuple(
        CertificateCheckObligation(name, value, detail)
        for name, value, detail in (
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[0], outer_schema, "exact v1 raw outer grammar and tagged segment union"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[1], namespace_unique, "global primary and reserved identifier namespace is unique"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[2], serializable, f"digest={digest}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[3], target_valid, f"T={certificate.requested_target_time!r}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[4], width_valid, f"width={certificate.requested_maximum_component_width!r}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[5], root_exact, "zero-tolerance binding and tube anchor at ordinary left endpoint"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[6], root_direct, _root_detail(root_result)),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[7], all_segments, f"certified_prefix={certified_count}/{len(certificate.segments)}; failed={failed_missing!r}; replay_failure={replay_failure!r}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[8], target_after_left, "T >= upper(a + B_current)"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[9], preimage_derived, f"J=[T-B_hi,T-B_lo]={preimage!s}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[10], preimage_inside, f"J={preimage!s}; domain={getattr(current_chart, 'parameter_interval', ())!r}"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[11], state_evaluated, "exact-rational interval Horner plus fresh direct ordinary Gronwall radius"),
            (PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[12], width_within, f"maximum={maximum_width!s}; requested={certificate.requested_maximum_component_width!r}"),
        )
    )

    first_failed = next((item.obligation for item in obligations if item.certified is not True), None)
    if first_failed == PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[6]:
        missing = _root_missing(root_result)
        if missing:
            first_failed = f"root:{missing[0]}"
    elif first_failed == PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS[7] and failed_index is not None:
        first_failed = f"segment[{failed_index}]:{(failed_missing or ('nested_result_missing',))[0]}"

    local_mathematical_to_target = bool(
        all(item.certified is True for item in obligations[5:12])
    )
    succeeded = bool(
        all(item.certified is True for item in obligations)
        and first_failed is None
    )
    retained: tuple[ProofGradePlanarChainRetainedRegion, ...] = ()
    ordinary_frontier: ProofGradePlanarChainRetainedRegion | None = None
    if (
        not succeeded
        and local_mathematical_to_target
        and final_enclosure is not None
        and not width_within
    ):
        retained = (_final_retained_region(
            final_enclosure,
            certified_count,
            current_chart.chart_id if current_chart is not None else "",
        ),)
    elif (
        not succeeded
        and not local_mathematical_to_target
        and root_certified
        and current_chart is not None
        and current_tube is not None
        and _fraction_interval(current_clock)
    ):
        if _fraction_box(failed_lifted, 14):
            retained = (ProofGradePlanarChainRetainedRegion(
                "certified_lifted_lc_right_frontier", "planar_lc_lifted_14",
                failed_lifted[13],
                (Fraction.from_float(certificate.segments[failed_index].exit_transition.source_parameter),) * 2 if failed_index is not None and type(certificate.segments[failed_index]) is PlanarLCPassageV1Segment else (Fraction(0), Fraction(0)),
                failed_lifted, certified_count, _CHECKER_ID, failed_index,
                certificate.segments[failed_index].lc_chart.chart_id if failed_index is not None and type(certificate.segments[failed_index]) is PlanarLCPassageV1Segment else "",
                certificate.segments[failed_index].lc_chart.pair if failed_index is not None and type(certificate.segments[failed_index]) is PlanarLCPassageV1Segment else None,
            ),)
        else:
            ordinary_frontier = _fresh_ordinary_right_frontier(current_chart, current_tube, current_clock, certified_count)
            retained = (ordinary_frontier,) if ordinary_frontier is not None else ()

    if local_mathematical_to_target and target is not None:
        covered: FractionInterval | tuple[()] = (Fraction.from_float(certificate.root_binding.initial_time), target)
    elif retained:
        covered = (Fraction.from_float(certificate.root_binding.initial_time), max(Fraction.from_float(certificate.root_binding.initial_time), retained[0].physical_time_interval[0]))
    else:
        covered = ()

    return ProofGradeRawPlanarChainReplayResult(
        certificate.certificate_id, certificate, CERTIFIED_TO_T if succeeded else UNRESOLVED,
        digest, obligations, first_failed, certified_count, failed_index,
        failed_missing, replay_failure, root_result, tuple(local_results),
        tuple(checker_ids), tuple(clock_ledger), tuple(cocycles), current_chart,
        current_tube, current_chart.chart_id if current_chart is not None else "",
        current_clock, covered, preimage, final_tube, maximum_width,
        final_enclosure, retained,
        certificate.requested_target_time if target_valid else None,
        certificate.requested_maximum_component_width if _finite_float(certificate.requested_maximum_component_width) else None,
    )


def _fresh_proof_root(result: object, certificate: RawPlanarChainCertificate) -> bool:
    try:
        return bool(
            type(result) is ProofGradeValidatedOrdinaryRootResult
            and result.raw_binding is certificate.root_binding
            and result.raw_tube is certificate.initial_tube
            and result.raw_chart is certificate.initial_chart
            and result.certified
        )
    except Exception:
        return False


def _fresh_ordinary_commit(
    result: object, segment: OrdinaryBridgeV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate, source_clock: FractionInterval,
) -> bool:
    try:
        return bool(
            type(result) is CarriedOrdinaryBridgeResult
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


def _fresh_proof_lc_commit(
    result: object, segment: PlanarLCPassageV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate, source_clock: FractionInterval,
) -> bool:
    try:
        return bool(
            type(result) is ProofGradeCarriedPlanarLCExitResult
            and result.raw_entry_transition is segment.entry_transition
            and result.raw_source_chart is source_chart
            and result.raw_source_tube is source_tube
            and result.raw_lc_chart is segment.lc_chart
            and result.raw_lc_tube is segment.lc_tube
            and result.raw_exit_transition is segment.exit_transition
            and result.raw_target_chart is segment.target_chart
            and result.raw_target_tube is segment.target_tube
            and result.parent_source_clock_origin_interval == source_clock
            and _fraction_interval(result.target_clock_origin_interval)
            and result.certified
        )
    except Exception:
        return False


def _fresh_proof_lc_equal(
    result: object, segment: PlanarLCPassageV1Segment,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate, source_clock: FractionInterval,
) -> ProofGradeCarriedPlanarLCExitResult | None:
    """Authenticate a false proof local result without reading `.certified`."""

    try:
        fresh = check_proof_grade_carried_planar_lc_exit(
            segment.entry_transition, source_chart, source_tube, segment.lc_chart,
            segment.lc_tube, segment.exit_transition, segment.target_chart,
            segment.target_tube, source_clock,
        )
        return fresh if type(result) is ProofGradeCarriedPlanarLCExitResult and fresh == result else None
    except Exception:
        return None


def _fresh_final_tube(
    result: object, tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> bool:
    try:
        return bool(
            type(result) is OrdinaryAposterioriTubeCheckResult
            and result.tube_id == tube.tube_id
            and result.chart_id == chart.chart_id
            and result.certified
        )
    except Exception:
        return False


def _local_missing(result: object, fallback: str) -> tuple[str, ...]:
    try:
        if type(result) in {CarriedOrdinaryBridgeResult, ProofGradeCarriedPlanarLCExitResult}:
            values = tuple(result.missing_obligations)
            return values or (fallback,)
    except Exception:
        pass
    return (fallback,)


def _root_missing(result: object) -> tuple[str, ...]:
    try:
        if type(result) is ProofGradeValidatedOrdinaryRootResult:
            return tuple(result.missing_obligations)
    except Exception:
        pass
    return ("proof_grade_validated_ordinary_root_result_missing",)


def _root_detail(result: object) -> str:
    return f"checker={getattr(result, 'checker_id', '')!r}; missing={_root_missing(result)!r}"


def _lc_slice_reconstructed(result: ProofGradeCarriedPlanarLCExitResult) -> bool:
    try:
        return bool(
            len(result.obligations) == 21
            and result.obligations[14].obligation
            == "proof_grade_carried_lc_exit_complete_inflated_slice_reconstructed"
            and result.obligations[14].certified is True
            and _fraction_box(result.lifted_exit_slice, 14)
        )
    except Exception:
        return False


def _fresh_ordinary_right_frontier(
    chart: OrdinaryTaylorChartCertificate, tube: OrdinaryAposterioriTubeCertificate,
    clock: FractionInterval, count: int,
) -> ProofGradePlanarChainRetainedRegion | None:
    try:
        replay = check_ordinary_aposteriori_tube(tube, chart)
        if not _fresh_final_tube(replay, tube, chart):
            return None
        parameter = Fraction.from_float(chart.parameter_interval[1])
        positions, velocities = _exact_rational_chart_projected_state_at_parameter(chart, parameter)
        radius = Fraction.from_float(replay.gronwall_error_bound)
        components = tuple(
            (center - radius, center + radius)
            for matrix in (positions, velocities)
            for center in matrix.reshape(-1)
        )
        if not _fraction_box(components, 12):
            return None
        return ProofGradePlanarChainRetainedRegion(
            "certified_current_ordinary_right_frontier",
            "planar_cartesian_12",
            (parameter + clock[0], parameter + clock[1]),
            (parameter, parameter), components, count, _CHECKER_ID, None,
            chart.chart_id, None,
        )
    except Exception:
        return None


def _final_retained_region(
    enclosure: ProofGradePlanarChainOrdinaryStateEnclosure,
    count: int,
    chart_id: str,
) -> ProofGradePlanarChainRetainedRegion:
    components = tuple(
        interval
        for matrix in (enclosure.position_intervals, enclosure.velocity_intervals)
        for row in matrix
        for interval in row
    )
    return ProofGradePlanarChainRetainedRegion(
        "certified_ordinary_fixed_time_enclosure",
        "planar_cartesian_12", enclosure.physical_time_interval,
        enclosure.parameter_preimage_interval, components, count, _CHECKER_ID,
        None, chart_id, None,
    )


def _enclosure_schema(value: object) -> bool:
    try:
        return bool(
            type(value) is ProofGradePlanarChainOrdinaryStateEnclosure
            and value.enclosure_type == "validated_planar_chain_fixed_physical_time"
            and _fraction_interval(value.physical_time_interval)
            and _fraction_interval(value.parameter_preimage_interval)
            and _fraction_matrix(value.position_intervals)
            and _fraction_matrix(value.velocity_intervals)
        )
    except Exception:
        return False


def _maximum_component_width(value: ProofGradePlanarChainOrdinaryStateEnclosure) -> Fraction:
    # Keep this implementation independent from the compatibility result
    # class while using the same exact mathematical width convention.
    widths = tuple(
        upper - lower
        for matrix in (value.position_intervals, value.velocity_intervals)
        for row in matrix
        for lower, upper in row
    )
    if not widths or any(width < 0 for width in widths):
        raise ValueError("invalid component width")
    return max(widths)


def _retained_region_schema(value: object) -> bool:
    try:
        common = bool(
            type(value) is ProofGradePlanarChainRetainedRegion
            and type(value.region_type) is str and bool(value.region_type)
            and type(value.coordinate_system) is str and bool(value.coordinate_system)
            and _fraction_interval(value.physical_time_interval)
            and _fraction_interval(value.parameter_interval)
            and type(value.certified_segment_count) is int
            and value.certified_segment_count >= 0
            and type(value.provenance_checker_id) is str
            and value.provenance_checker_id == _CHECKER_ID
            and (value.failed_segment_index is None or (type(value.failed_segment_index) is int and value.failed_segment_index >= 0))
            and type(value.chart_id) is str
            and (value.pair is None or value.pair in {(0, 1), (0, 2), (1, 2)})
        )
        if not common:
            return False
        if value.region_type == "certified_lifted_lc_right_frontier":
            return bool(
                value.coordinate_system == "planar_lc_lifted_14"
                and _fraction_box(value.component_intervals, 14)
                and type(value.failed_segment_index) is int
                and bool(value.chart_id)
                and value.pair in {(0, 1), (0, 2), (1, 2)}
            )
        return bool(
            value.region_type in {
                "certified_current_ordinary_right_frontier",
                "certified_ordinary_fixed_time_enclosure",
            }
            and value.coordinate_system == "planar_cartesian_12"
            and _fraction_box(value.component_intervals, 12)
            and value.failed_segment_index is None
            and bool(value.chart_id)
            and value.pair is None
        )
    except Exception:
        return False


def _ledger_schema(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS)
        and all(
            type(item) is CertificateCheckObligation
            and item.obligation == expected
            and type(item.certified) is bool
            and type(item.detail) is str
            for item, expected in zip(value, PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS)
        )
    )


def _clock_ledger_schema(value: object, count: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == (count + 1 if count >= 0 and value else 0)
        and all(
            type(item) is ClockOriginLedgerEntry
            and type(item.vertex_index) is int
            and item.vertex_index == index
            and type(item.chart_id) is str and bool(item.chart_id)
            and _fraction_interval(item.clock_origin_interval)
            for index, item in enumerate(value)
        )
    )


def _fraction_matrix(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 3
        and all(type(row) is tuple and len(row) == 2 and all(_fraction_interval(item) for item in row) for row in value)
    )


def _optional_fraction_interval(value: object) -> bool:
    return value == () or _fraction_interval(value)


def _sha256_hex(value: object) -> bool:
    return bool(type(value) is str and len(value) == 64 and all(item in "0123456789abcdef" for item in value))


def _evidence_digest(certificate: RawPlanarChainCertificate) -> str:
    try:
        canonical = canonical_planar_chain_evidence_json(certificate)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(_INVALID_DIGEST_PAYLOAD).hexdigest()
