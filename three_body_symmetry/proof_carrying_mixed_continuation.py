"""Raw-replaying one-passage planar ``N -> LC_ij -> N`` continuation.

This private version-1 proof surface consumes exactly the nine raw primitives
used by :mod:`proof_carrying_planar_lc_exit`, plus a finite target physical
time and a finite nonnegative component-width request.  It accepts no supplied
exit result, clock origin, gauge choice, parameter preimage, or state box.

The checker is deliberately a *supplied-certificate* theorem.  It does not
claim that a producer can find evidence for arbitrary inputs, that the LC
segment contains a collision, or that repeated/mixed-pair chains are already
supported.

Pinned trusted lemmas are named independently in every result: the mixed
Newton/LC analytic kernel carries the constrained branch through the chart
changes; the autonomous-clock kernel proves ``b=t_exit-a``; and the fixed-time
kernel is exact rational interval Horner evaluation over ``J=T-[b]`` followed
by inflation with the freshly checked ordinary Gronwall error.  The first two
analytic facts are not re-proved symbolically during replay.  Horner and
inflation are finite checker arithmetic performed on every replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from .certificate_checker import (
    GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult,
    OrdinaryAposterioriTubeCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    ValidatedOrdinaryIVPChartCheckResult,
    _exact_rational_chart_projected_state_at_parameter,
    _planar_lc_mass_ratio_arithmetic_exact,
    _planar_lc_state_intervals,
    _regularized_binary_solution_from_certificate,
    check_gauge_aware_ordinary_to_planar_lc_enclosure_transition,
    check_planar_lc_aposteriori_tube,
    check_validated_ordinary_ivp_chart,
)
from .certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    OrdinaryToPlanarLCEnclosureTransitionCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
)
from .intervals import RationalInterval
from .proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
    _canonical_json_from_data,
    _canonical_json_value,
    _json_container,
    _mapping_field,
    _require_exact_wire_fields,
)
from .proof_carrying_planar_lc_exit import (
    RawGaugeAwarePlanarLCExitContainmentResult,
    check_raw_gauge_aware_planar_lc_exit_containment,
)


_CERTIFICATE_TYPE = "raw_mixed_planar_continuation"
_SCHEMA_VERSION = 1
_CHECKER_ID = "raw_mixed_planar_continuation_replay_checker_v1"
_ANALYTIC_KERNEL_ID = "planar_newton_lc_mixed_analytic_kernel_v1"
_AUTONOMOUS_CLOCK_KERNEL_ID = "autonomous_target_clock_cocycle_kernel_v1"
_FIXED_TIME_KERNEL_ID = "exact_rational_horner_fixed_time_kernel_v1"
_NESTED_EXIT_CHECKER_ID = "raw_gauge_aware_planar_lc_exit_containment_checker_v1"
_PREFIX_CHECKER_ID = "raw_mixed_planar_prefix_replay_checker_v1"
_INVALID_EVIDENCE_DIGEST_PAYLOAD = b"invalid-noncanonical-mixed-evidence-v1"

_OBLIGATION_NAMES = (
    "raw_mixed_schema_version_and_outer_manifest_exact",
    "raw_mixed_global_identifiers_unique_and_manifest_exact",
    "raw_mixed_canonical_evidence_serializable",
    "raw_mixed_requested_target_finite",
    "raw_mixed_requested_component_width_admissible",
    "raw_mixed_exit_freshly_certified",
    "raw_mixed_target_ordinary_tube_freshly_certified",
    "raw_mixed_target_clock_origin_exactly_rederived",
    "raw_mixed_target_not_before_complete_exit_time_interval",
    "raw_mixed_fixed_time_preimage_exactly_derived",
    "raw_mixed_fixed_time_preimage_inside_forward_target_slab",
    "raw_mixed_global_physical_time_coverage_certified",
    "raw_mixed_target_state_interval_evaluated_and_inflated",
    "raw_mixed_final_component_width_within_requested_bound",
)

FractionInterval = tuple[Fraction, Fraction]
FractionMatrix = tuple[tuple[FractionInterval, ...], ...]


@dataclass(frozen=True)
class RawMixedPlanarContinuationCertificate:
    """Canonical raw evidence for one ``N -> LC_ij -> N`` passage."""

    certificate_id: str
    entry_transition: OrdinaryToPlanarLCEnclosureTransitionCertificate
    source_binding: InitialValueProblemBindingCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    source_chart: OrdinaryTaylorChartCertificate
    lc_chart: PlanarLeviCivitaBinaryChartCertificate
    lc_tube: PlanarLCAposterioriTubeCertificate
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate
    target_chart: OrdinaryTaylorChartCertificate
    target_tube: OrdinaryAposterioriTubeCertificate
    requested_target_time: float
    requested_maximum_component_width: float
    schema_version: int = _SCHEMA_VERSION
    certificate_type: str = _CERTIFICATE_TYPE
    source: str = "raw_mixed_planar_continuation_v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON data model for the raw evidence."""

        _require_raw_component_types(self)
        return {
            "certificate_id": self.certificate_id,
            "entry_transition": _json_container(self.entry_transition.to_dict()),
            "source_binding": _json_container(self.source_binding.to_dict()),
            "source_tube": _json_container(self.source_tube.to_dict()),
            "source_chart": _json_container(self.source_chart.to_dict()),
            "lc_chart": _json_container(self.lc_chart.to_dict()),
            "lc_tube": _json_container(self.lc_tube.to_dict()),
            "exit_transition": _json_container(self.exit_transition.to_dict()),
            "target_chart": _json_container(self.target_chart.to_dict()),
            "target_tube": _json_container(self.target_tube.to_dict()),
            "requested_target_time": self.requested_target_time,
            "requested_maximum_component_width": (
                self.requested_maximum_component_width
            ),
            "schema_version": self.schema_version,
            "certificate_type": self.certificate_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawMixedPlanarContinuationCertificate":
        """Parse only the exact versioned wire shape, without coercion leaks."""

        if type(data) is not dict:
            raise TypeError("raw mixed continuation evidence must be a dict")
        _require_exact_wire_fields(
            data,
            RawMixedPlanarContinuationCertificate,
            "raw mixed continuation certificate",
        )
        component_specs = (
            (
                "entry_transition",
                OrdinaryToPlanarLCEnclosureTransitionCertificate,
            ),
            ("source_binding", InitialValueProblemBindingCertificate),
            ("source_tube", OrdinaryAposterioriTubeCertificate),
            ("source_chart", OrdinaryTaylorChartCertificate),
            ("lc_chart", PlanarLeviCivitaBinaryChartCertificate),
            ("lc_tube", PlanarLCAposterioriTubeCertificate),
            (
                "exit_transition",
                PlanarLCToOrdinaryEnclosureTransitionCertificate,
            ),
            ("target_chart", OrdinaryTaylorChartCertificate),
            ("target_tube", OrdinaryAposterioriTubeCertificate),
        )
        components: dict[str, dict[str, Any]] = {}
        for field_name, component_type in component_specs:
            component = _mapping_field(data[field_name], field_name)
            _require_exact_wire_fields(component, component_type, field_name)
            components[field_name] = component
        if not (
            type(data["certificate_id"]) is str
            and type(data["certificate_type"]) is str
            and type(data["source"]) is str
            and type(data["schema_version"]) is int
            and type(data["requested_target_time"]) is float
            and type(data["requested_maximum_component_width"]) is float
        ):
            raise ValueError("raw mixed continuation wire scalars are noncanonical")

        certificate = cls(
            certificate_id=data["certificate_id"],
            entry_transition=(
                OrdinaryToPlanarLCEnclosureTransitionCertificate.from_dict(
                    components["entry_transition"]
                )
            ),
            source_binding=InitialValueProblemBindingCertificate.from_dict(
                components["source_binding"]
            ),
            source_tube=OrdinaryAposterioriTubeCertificate.from_dict(
                components["source_tube"]
            ),
            source_chart=OrdinaryTaylorChartCertificate.from_dict(
                components["source_chart"]
            ),
            lc_chart=PlanarLeviCivitaBinaryChartCertificate.from_dict(
                components["lc_chart"]
            ),
            lc_tube=PlanarLCAposterioriTubeCertificate.from_dict(
                components["lc_tube"]
            ),
            exit_transition=(
                PlanarLCToOrdinaryEnclosureTransitionCertificate.from_dict(
                    components["exit_transition"]
                )
            ),
            target_chart=OrdinaryTaylorChartCertificate.from_dict(
                components["target_chart"]
            ),
            target_tube=OrdinaryAposterioriTubeCertificate.from_dict(
                components["target_tube"]
            ),
            requested_target_time=data["requested_target_time"],
            requested_maximum_component_width=data[
                "requested_maximum_component_width"
            ],
            schema_version=data["schema_version"],
            certificate_type=data["certificate_type"],
            source=data["source"],
        )
        if _canonical_json_from_data(data) != canonical_mixed_evidence_json(
            certificate
        ):
            raise ValueError("raw mixed continuation wire encoding is noncanonical")
        return certificate


@dataclass(frozen=True)
class RawMixedPlanarContinuationObligation:
    """One exact boolean in the deterministic mixed replay ledger."""

    obligation: str
    certified: bool
    detail: str


@dataclass(frozen=True)
class MixedOrdinaryStateEnclosure:
    """Exact rational Cartesian enclosure at one requested physical time."""

    enclosure_type: str
    physical_time_interval: FractionInterval
    target_parameter_preimage_interval: FractionInterval
    position_intervals: FractionMatrix
    velocity_intervals: FractionMatrix


@dataclass(frozen=True)
class MixedRetainedSafeRegion:
    """A freshly certified prefix frontier in ordinary or lifted coordinates."""

    region_type: str
    coordinate_system: str
    physical_time_interval: FractionInterval
    parameter_interval: FractionInterval
    component_intervals: tuple[FractionInterval, ...]
    provenance_checker_id: str


@dataclass(frozen=True)
class RawMixedPlanarContinuationReplayResult:
    """Immutable result bound to a fresh replay of all retained raw evidence."""

    certificate_id: str
    raw_certificate: RawMixedPlanarContinuationCertificate
    status: str
    evidence_sha256: str
    obligations: tuple[RawMixedPlanarContinuationObligation, ...]
    first_failed_obligation: str | None
    covered_physical_time_interval: FractionInterval | tuple[()]
    requested_target_time: float | None
    requested_maximum_component_width: float | None
    exit_time_interval: FractionInterval | tuple[()]
    target_clock_origin_interval: FractionInterval | tuple[()]
    target_parameter_preimage_interval: FractionInterval | tuple[()]
    maximum_final_component_width: Fraction | None
    final_enclosure: MixedOrdinaryStateEnclosure | None
    retained_safe_regions: tuple[MixedRetainedSafeRegion, ...]
    nested_exit_result: RawGaugeAwarePlanarLCExitContainmentResult | None
    nested_exit_checker_id: str
    nested_missing_obligations: tuple[str, ...]
    analytic_kernel_id: str = _ANALYTIC_KERNEL_ID
    autonomous_clock_kernel_id: str = _AUTONOMOUS_CLOCK_KERNEL_ID
    fixed_time_kernel_id: str = _FIXED_TIME_KERNEL_ID
    schema_version: int = _SCHEMA_VERSION
    checker_id: str = _CHECKER_ID

    def _snapshot_well_formed(self) -> bool:
        if not (
            type(self) is RawMixedPlanarContinuationReplayResult
            and type(self.raw_certificate)
            is RawMixedPlanarContinuationCertificate
            and type(self.certificate_id) is str
            and self.certificate_id
            == (
                self.raw_certificate.certificate_id
                if type(self.raw_certificate.certificate_id) is str
                else ""
            )
            and type(self.schema_version) is int
            and self.schema_version == _SCHEMA_VERSION
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.analytic_kernel_id) is str
            and self.analytic_kernel_id == _ANALYTIC_KERNEL_ID
            and type(self.autonomous_clock_kernel_id) is str
            and self.autonomous_clock_kernel_id == _AUTONOMOUS_CLOCK_KERNEL_ID
            and type(self.fixed_time_kernel_id) is str
            and self.fixed_time_kernel_id == _FIXED_TIME_KERNEL_ID
            and type(self.status) is str
            and self.status in {CERTIFIED_TO_T, UNRESOLVED}
            and _sha256_hex(self.evidence_sha256)
            and type(self.obligations) is tuple
            and tuple(
                item.obligation
                for item in self.obligations
                if type(item) is RawMixedPlanarContinuationObligation
            )
            == _OBLIGATION_NAMES
            and all(
                type(item) is RawMixedPlanarContinuationObligation
                and type(item.obligation) is str
                and type(item.certified) is bool
                and type(item.detail) is str
                for item in self.obligations
            )
            and (
                self.requested_target_time is None
                or _finite_float(self.requested_target_time)
            )
            and (
                self.requested_maximum_component_width is None
                or _finite_float(self.requested_maximum_component_width)
            )
            and _optional_fraction_interval(self.covered_physical_time_interval)
            and _optional_fraction_interval(self.exit_time_interval)
            and _optional_fraction_interval(self.target_clock_origin_interval)
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
            and type(self.retained_safe_regions) is tuple
            and len(self.retained_safe_regions) <= 1
            and all(_retained_region_schema(item) for item in self.retained_safe_regions)
            and (
                self.nested_exit_result is None
                or type(self.nested_exit_result)
                is RawGaugeAwarePlanarLCExitContainmentResult
            )
            and type(self.nested_exit_checker_id) is str
            and type(self.nested_missing_obligations) is tuple
            and all(type(item) is str for item in self.nested_missing_obligations)
        ):
            return False
        try:
            if self.evidence_sha256 != raw_mixed_continuation_evidence_sha256(
                self.raw_certificate
            ):
                return False
        except Exception:
            if self.evidence_sha256 != _invalid_evidence_sha256():
                return False

        all_true = all(item.certified is True for item in self.obligations)
        if self.status == CERTIFIED_TO_T:
            return bool(
                all_true
                and self.first_failed_obligation is None
                and _finite_float(self.requested_target_time)
                and _finite_float(self.requested_maximum_component_width)
                and type(self.nested_exit_result)
                is RawGaugeAwarePlanarLCExitContainmentResult
                and self.nested_exit_checker_id == _NESTED_EXIT_CHECKER_ID
                and self.nested_missing_obligations == ()
                and self.nested_exit_result.certified
                and _fraction_interval(self.exit_time_interval)
                and _fraction_interval(self.target_clock_origin_interval)
                and _fraction_interval(self.target_parameter_preimage_interval)
                and type(self.maximum_final_component_width) is Fraction
                and type(self.final_enclosure) is MixedOrdinaryStateEnclosure
                and _ordinary_enclosure_schema(self.final_enclosure)
                and self.final_enclosure.physical_time_interval
                == (
                    Fraction.from_float(self.requested_target_time),
                    Fraction.from_float(self.requested_target_time),
                )
                and self.final_enclosure.target_parameter_preimage_interval
                == self.target_parameter_preimage_interval
                and _fraction_interval(self.covered_physical_time_interval)
                and self.covered_physical_time_interval
                == (
                    Fraction.from_float(self.raw_certificate.source_binding.initial_time),
                    Fraction.from_float(self.requested_target_time),
                )
                and self.retained_safe_regions == ()
            )
        if not (
            self.status == UNRESOLVED
            and type(self.first_failed_obligation) is str
            and bool(self.first_failed_obligation)
            and not all_true
        ):
            return False
        if self.final_enclosure is None:
            return True
        return bool(
            _finite_float(self.requested_target_time)
            and type(self.maximum_final_component_width) is Fraction
            and type(self.nested_exit_result)
            is RawGaugeAwarePlanarLCExitContainmentResult
            and self.nested_exit_checker_id == _NESTED_EXIT_CHECKER_ID
            and self.nested_missing_obligations == ()
            and self.nested_exit_result.certified
            and _ordinary_enclosure_schema(self.final_enclosure)
            and _fraction_interval(self.exit_time_interval)
            and _fraction_interval(self.target_clock_origin_interval)
            and _fraction_interval(self.target_parameter_preimage_interval)
            and self.final_enclosure.target_parameter_preimage_interval
            == self.target_parameter_preimage_interval
            and self.final_enclosure.physical_time_interval
            == (
                Fraction.from_float(self.requested_target_time),
                Fraction.from_float(self.requested_target_time),
            )
            and _fraction_interval(self.covered_physical_time_interval)
            and self.covered_physical_time_interval
            == (
                Fraction.from_float(self.raw_certificate.source_binding.initial_time),
                Fraction.from_float(self.requested_target_time),
            )
            and len(self.retained_safe_regions) == 1
            and self.retained_safe_regions[0]
            == _final_retained_region(self.final_enclosure)
        )

    @property
    def replay_consistent(self) -> bool:
        """Whether a fresh replay reproduces this success or failure exactly."""

        try:
            if (
                type(self) is not RawMixedPlanarContinuationReplayResult
                or not self._snapshot_well_formed()
            ):
                return False
            fresh = check_raw_mixed_planar_continuation(self.raw_certificate)
            return bool(
                type(fresh) is RawMixedPlanarContinuationReplayResult
                and fresh._snapshot_well_formed()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def certified(self) -> bool:
        """True only for an exact fresh-replaying ``CERTIFIED_TO_T`` snapshot."""

        return bool(self.status == CERTIFIED_TO_T and self.replay_consistent)


def canonical_mixed_evidence_json(
    certificate: RawMixedPlanarContinuationCertificate,
) -> str:
    """Return sorted compact canonical JSON for mixed raw evidence."""

    if type(certificate) is not RawMixedPlanarContinuationCertificate:
        raise TypeError("certificate must be RawMixedPlanarContinuationCertificate")
    _require_raw_component_types(certificate)
    return json.dumps(
        _canonical_json_value(certificate.to_dict()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def raw_mixed_continuation_evidence_sha256(
    certificate: RawMixedPlanarContinuationCertificate,
) -> str:
    """Return the lowercase SHA-256 digest of canonical mixed evidence."""

    return hashlib.sha256(
        canonical_mixed_evidence_json(certificate).encode("utf-8")
    ).hexdigest()


def check_raw_mixed_planar_continuation(
    certificate: RawMixedPlanarContinuationCertificate,
) -> RawMixedPlanarContinuationReplayResult:
    """Freshly replay one supplied ``N -> LC_ij -> N`` certificate."""

    if type(certificate) is not RawMixedPlanarContinuationCertificate:
        raise TypeError("certificate must be RawMixedPlanarContinuationCertificate")
    _require_raw_component_types(certificate)

    serializable = True
    try:
        digest = raw_mixed_continuation_evidence_sha256(certificate)
    except Exception:
        serializable = False
        digest = _invalid_evidence_sha256()

    outer_schema = bool(
        type(certificate.schema_version) is int
        and certificate.schema_version == _SCHEMA_VERSION
        and type(certificate.certificate_type) is str
        and certificate.certificate_type == _CERTIFICATE_TYPE
        and type(certificate.certificate_id) is str
        and bool(certificate.certificate_id)
        and type(certificate.source) is str
        and bool(certificate.source)
    )
    global_identifiers = _global_identifiers_unique(certificate)
    requested_target_valid = _finite_float(certificate.requested_target_time)
    requested_width_valid = bool(
        _finite_float(certificate.requested_maximum_component_width)
        and Fraction.from_float(certificate.requested_maximum_component_width) >= 0
    )

    exit_result: RawGaugeAwarePlanarLCExitContainmentResult | None = None
    exit_error = ""
    try:
        exit_result = check_raw_gauge_aware_planar_lc_exit_containment(
            certificate.entry_transition,
            certificate.source_binding,
            certificate.source_tube,
            certificate.source_chart,
            certificate.lc_chart,
            certificate.lc_tube,
            certificate.exit_transition,
            certificate.target_chart,
            certificate.target_tube,
        )
    except Exception as error:
        exit_error = type(error).__name__
    exit_certified = False
    try:
        exit_certified = bool(
            type(exit_result) is RawGaugeAwarePlanarLCExitContainmentResult
            and exit_result.certified
        )
    except Exception:
        exit_certified = False
    nested_missing = _nested_missing(exit_result, exit_error)
    target_tube_certified = False
    try:
        target_tube_certified = bool(
            exit_certified
            and exit_result is not None
            and type(exit_result.target_tube_result)
            is OrdinaryAposterioriTubeCheckResult
            and exit_result.target_tube_result.checker_id
            == "independent_ordinary_aposteriori_tube_checker_v1"
            and exit_result.target_tube_result.tube_id
            == certificate.target_tube.tube_id
            and exit_result.target_tube_result.chart_id
            == certificate.target_chart.chart_id
            and exit_result.target_tube_result.certified
        )
    except Exception:
        target_tube_certified = False

    exit_time: FractionInterval | tuple[()] = ()
    clock_origin: FractionInterval | tuple[()] = ()
    preimage: FractionInterval | tuple[()] = ()
    final_enclosure: MixedOrdinaryStateEnclosure | None = None
    maximum_width: Fraction | None = None
    clock_rederived = False
    target_after_exit = False
    preimage_derived = False
    preimage_inside = False
    global_coverage = False
    state_evaluated = False
    width_within_request = False

    if exit_certified and exit_result is not None:
        try:
            exit_time = exit_result.exit_time_interval
            if not _fraction_interval(exit_time):
                raise ValueError("certified exit did not retain a rational time box")
            anchor = Fraction.from_float(certificate.exit_transition.target_parameter)
            recomputed_clock = (
                exit_time[0] - anchor,
                exit_time[1] - anchor,
            )
            clock_rederived = bool(
                recomputed_clock == exit_result.target_clock_origin_interval
            )
            if clock_rederived:
                clock_origin = recomputed_clock
            if requested_target_valid:
                target = Fraction.from_float(certificate.requested_target_time)
                target_after_exit = target >= exit_time[1]
                recomputed_preimage = (
                    anchor + target - exit_time[1],
                    anchor + target - exit_time[0],
                )
                preimage_derived = bool(
                    clock_rederived
                    and recomputed_preimage
                    == (
                        target - recomputed_clock[1],
                        target - recomputed_clock[0],
                    )
                    and _fraction_interval(recomputed_preimage)
                )
                if preimage_derived:
                    preimage = recomputed_preimage
                    target_domain = tuple(
                        Fraction.from_float(value)
                        for value in certificate.target_chart.parameter_interval
                    )
                    preimage_inside = bool(
                        target_after_exit
                        and len(target_domain) == 2
                        and target_domain[0] == anchor
                        and target_domain[0] <= preimage[0]
                        and preimage[1] <= target_domain[1]
                    )
                    global_coverage = bool(
                        outer_schema
                        and global_identifiers
                        and serializable
                        and target_tube_certified
                        and target_after_exit
                        and preimage_inside
                        and Fraction.from_float(
                            certificate.source_binding.initial_time
                        )
                        <= target
                    )
            if preimage_inside and exit_result.target_tube_result is not None:
                radius = Fraction.from_float(
                    exit_result.target_tube_result.gronwall_error_bound
                )
                positions = _evaluate_and_inflate_matrix(
                    certificate.target_chart.position_coefficients,
                    preimage,
                    radius,
                )
                velocities = _evaluate_and_inflate_matrix(
                    certificate.target_chart.velocity_coefficients,
                    preimage,
                    radius,
                )
                final_enclosure = MixedOrdinaryStateEnclosure(
                    enclosure_type="validated_mixed_fixed_physical_time",
                    physical_time_interval=(target, target),
                    target_parameter_preimage_interval=preimage,
                    position_intervals=positions,
                    velocity_intervals=velocities,
                )
                state_evaluated = _ordinary_enclosure_schema(final_enclosure)
                if state_evaluated:
                    maximum_width = _maximum_component_width(final_enclosure)
                    width_within_request = bool(
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
            "raw_mixed_schema_version_and_outer_manifest_exact",
            outer_schema,
            (
                f"schema_version={certificate.schema_version!r}; "
                f"certificate_type={certificate.certificate_type!r}"
            ),
        ),
        _obligation(
            "raw_mixed_global_identifiers_unique_and_manifest_exact",
            global_identifiers,
            (
                "outer certificate id and all nested primary ids are nonempty "
                "and globally unique"
            ),
        ),
        _obligation(
            "raw_mixed_canonical_evidence_serializable",
            serializable,
            f"digest={digest}",
        ),
        _obligation(
            "raw_mixed_requested_target_finite",
            requested_target_valid,
            f"requested_target_time={certificate.requested_target_time!r}",
        ),
        _obligation(
            "raw_mixed_requested_component_width_admissible",
            requested_width_valid,
            (
                "requested_maximum_component_width="
                f"{certificate.requested_maximum_component_width!r}"
            ),
        ),
        _obligation(
            "raw_mixed_exit_freshly_certified",
            exit_certified,
            (
                f"checker={getattr(exit_result, 'checker_id', '')!r}; "
                f"missing={nested_missing!r}"
            ),
        ),
        _obligation(
            "raw_mixed_target_ordinary_tube_freshly_certified",
            target_tube_certified,
            (
                "the exit replay's exact target chart/tube pair was freshly "
                "checked by the ordinary a-posteriori tube checker"
            ),
        ),
        _obligation(
            "raw_mixed_target_clock_origin_exactly_rederived",
            clock_rederived,
            f"B=D-a={clock_origin!s}",
        ),
        _obligation(
            "raw_mixed_target_not_before_complete_exit_time_interval",
            target_after_exit,
            f"T={certificate.requested_target_time!r}; D={exit_time!s}",
        ),
        _obligation(
            "raw_mixed_fixed_time_preimage_exactly_derived",
            preimage_derived,
            f"J=T-B=a+T-D={preimage!s}",
        ),
        _obligation(
            "raw_mixed_fixed_time_preimage_inside_forward_target_slab",
            preimage_inside,
            (
                f"J={preimage!s}; "
                f"target_domain={certificate.target_chart.parameter_interval!r}"
            ),
        ),
        _obligation(
            "raw_mixed_global_physical_time_coverage_certified",
            global_coverage,
            (
                "the exact bound IVP is carried through the certified exit, "
                "and T>=sup(D) has all of J inside the forward target tube"
            ),
        ),
        _obligation(
            "raw_mixed_target_state_interval_evaluated_and_inflated",
            state_evaluated,
            (
                "exact rational Horner evaluation over all J plus the freshly "
                "replayed target ordinary Gronwall radius"
            ),
        ),
        _obligation(
            "raw_mixed_final_component_width_within_requested_bound",
            width_within_request,
            (
                f"maximum_width={maximum_width!s}; requested="
                f"{certificate.requested_maximum_component_width!r}"
            ),
        ),
    )

    first_failed = next(
        (
            item.obligation
            for item in obligations
            if item.certified is not True
        ),
        None,
    )
    if first_failed == "raw_mixed_exit_freshly_certified" and nested_missing:
        first_failed = f"nested_exit:{nested_missing[0]}"
    mathematical_to_t = bool(
        outer_schema
        and global_identifiers
        and serializable
        and requested_target_valid
        and exit_certified
        and target_tube_certified
        and clock_rederived
        and target_after_exit
        and preimage_derived
        and preimage_inside
        and global_coverage
        and state_evaluated
        and final_enclosure is not None
        and maximum_width is not None
    )
    succeeded = bool(
        mathematical_to_t
        and requested_width_valid
        and width_within_request
        and first_failed is None
    )
    if succeeded:
        retained = ()
    elif mathematical_to_t and final_enclosure is not None:
        retained = (_final_retained_region(final_enclosure),)
    else:
        retained = _fresh_retained_region(certificate, exit_result)
    covered = (
        (
            Fraction.from_float(certificate.source_binding.initial_time),
            Fraction.from_float(certificate.requested_target_time),
        )
        if succeeded or mathematical_to_t
        else _covered_interval(certificate, retained)
    )

    return RawMixedPlanarContinuationReplayResult(
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
        covered_physical_time_interval=covered,
        requested_target_time=(
            certificate.requested_target_time if requested_target_valid else None
        ),
        requested_maximum_component_width=(
            certificate.requested_maximum_component_width
            if requested_width_valid
            else None
        ),
        exit_time_interval=exit_time,
        target_clock_origin_interval=clock_origin,
        target_parameter_preimage_interval=preimage,
        maximum_final_component_width=maximum_width,
        final_enclosure=final_enclosure if mathematical_to_t else None,
        retained_safe_regions=retained,
        nested_exit_result=exit_result if exit_certified else None,
        nested_exit_checker_id=(
            exit_result.checker_id
            if type(exit_result) is RawGaugeAwarePlanarLCExitContainmentResult
            and type(exit_result.checker_id) is str
            else ""
        ),
        nested_missing_obligations=nested_missing,
    )


def _require_raw_component_types(
    certificate: RawMixedPlanarContinuationCertificate,
) -> None:
    specs = (
        (
            certificate.entry_transition,
            OrdinaryToPlanarLCEnclosureTransitionCertificate,
            "entry_transition",
        ),
        (
            certificate.source_binding,
            InitialValueProblemBindingCertificate,
            "source_binding",
        ),
        (
            certificate.source_tube,
            OrdinaryAposterioriTubeCertificate,
            "source_tube",
        ),
        (certificate.source_chart, OrdinaryTaylorChartCertificate, "source_chart"),
        (
            certificate.lc_chart,
            PlanarLeviCivitaBinaryChartCertificate,
            "lc_chart",
        ),
        (certificate.lc_tube, PlanarLCAposterioriTubeCertificate, "lc_tube"),
        (
            certificate.exit_transition,
            PlanarLCToOrdinaryEnclosureTransitionCertificate,
            "exit_transition",
        ),
        (certificate.target_chart, OrdinaryTaylorChartCertificate, "target_chart"),
        (
            certificate.target_tube,
            OrdinaryAposterioriTubeCertificate,
            "target_tube",
        ),
    )
    for value, expected, name in specs:
        if type(value) is not expected:
            raise TypeError(f"{name} must have exact type {expected.__name__}")


def _global_identifiers_unique(
    certificate: RawMixedPlanarContinuationCertificate,
) -> bool:
    try:
        identifiers = (
            certificate.certificate_id,
            certificate.source_binding.binding_id,
            certificate.source_chart.certificate_id,
            certificate.source_chart.chart_id,
            certificate.source_tube.tube_id,
            certificate.entry_transition.transition_id,
            certificate.lc_chart.certificate_id,
            certificate.lc_chart.chart_id,
            certificate.lc_tube.tube_id,
            certificate.exit_transition.transition_id,
            certificate.target_chart.certificate_id,
            certificate.target_chart.chart_id,
            certificate.target_tube.tube_id,
        )
        return bool(
            all(type(value) is str and bool(value) for value in identifiers)
            and len(set(identifiers)) == len(identifiers)
        )
    except Exception:
        return False


def _evaluate_and_inflate_matrix(
    coefficients: tuple[tuple[tuple[float, ...], ...], ...],
    parameter: FractionInterval,
    radius: Fraction,
) -> FractionMatrix:
    if not (_fraction_interval(parameter) and type(radius) is Fraction and radius >= 0):
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
    rows = []
    for body in range(3):
        row = []
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


def _maximum_component_width(enclosure: MixedOrdinaryStateEnclosure) -> Fraction:
    widths = tuple(
        upper - lower
        for matrix in (enclosure.position_intervals, enclosure.velocity_intervals)
        for row in matrix
        for lower, upper in row
    )
    if not widths or any(width < 0 for width in widths):
        raise ValueError("state enclosure widths are invalid")
    return max(widths)


def _fresh_retained_region(
    certificate: RawMixedPlanarContinuationCertificate,
    exit_result: RawGaugeAwarePlanarLCExitContainmentResult | None,
) -> tuple[MixedRetainedSafeRegion, ...]:
    """Return the furthest independently replayed prefix frontier."""

    target_right = _fresh_target_right_frontier(certificate, exit_result)
    if target_right is not None:
        return (target_right,)
    lc_right = _fresh_lc_right_frontier(certificate)
    if lc_right is not None:
        return (lc_right,)
    source_right = _fresh_source_right_frontier(certificate)
    if source_right is not None:
        return (source_right,)
    return ()


def _fresh_target_right_frontier(
    certificate: RawMixedPlanarContinuationCertificate,
    exit_result: RawGaugeAwarePlanarLCExitContainmentResult | None,
) -> MixedRetainedSafeRegion | None:
    try:
        if not (
            type(exit_result) is RawGaugeAwarePlanarLCExitContainmentResult
            and exit_result.certified
            and type(exit_result.target_tube_result)
            is OrdinaryAposterioriTubeCheckResult
            and exit_result.target_tube_result.certified
            and _fraction_interval(exit_result.target_clock_origin_interval)
        ):
            return None
        parameter = Fraction.from_float(
            certificate.target_chart.parameter_interval[1]
        )
        radius = Fraction.from_float(
            exit_result.target_tube_result.gronwall_error_bound
        )
        positions = _evaluate_and_inflate_matrix(
            certificate.target_chart.position_coefficients,
            (parameter, parameter),
            radius,
        )
        velocities = _evaluate_and_inflate_matrix(
            certificate.target_chart.velocity_coefficients,
            (parameter, parameter),
            radius,
        )
        components = tuple(
            interval
            for matrix in (positions, velocities)
            for row in matrix
            for interval in row
        )
        clock = exit_result.target_clock_origin_interval
        return MixedRetainedSafeRegion(
            region_type="certified_ordinary_target_right_frontier",
            coordinate_system="ordinary_cartesian_q_then_v_12",
            physical_time_interval=(parameter + clock[0], parameter + clock[1]),
            parameter_interval=(parameter, parameter),
            component_intervals=components,
            provenance_checker_id=_PREFIX_CHECKER_ID,
        )
    except Exception:
        return None


def _fresh_lc_right_frontier(
    certificate: RawMixedPlanarContinuationCertificate,
) -> MixedRetainedSafeRegion | None:
    try:
        entry = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
            certificate.entry_transition,
            certificate.source_binding,
            certificate.source_tube,
            certificate.source_chart,
            certificate.lc_chart,
            certificate.lc_tube,
        )
        lc_result = check_planar_lc_aposteriori_tube(
            certificate.lc_tube,
            certificate.lc_chart,
        )
        source_domain = tuple(
            Fraction.from_float(value)
            for value in certificate.source_chart.parameter_interval
        )
        lc_domain = tuple(
            Fraction.from_float(value)
            for value in certificate.lc_chart.parameter_interval
        )
        pair = certificate.lc_chart.pair
        if not (
            _binding_is_exact_point(certificate.source_binding)
            and type(entry)
            is GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult
            and entry.certified
            and type(lc_result) is PlanarLCAposterioriTubeCheckResult
            and lc_result.certified
            and entry.raw_transition_certificate is certificate.entry_transition
            and entry.raw_source_binding is certificate.source_binding
            and entry.raw_source_tube is certificate.source_tube
            and entry.raw_source_chart is certificate.source_chart
            and entry.raw_target_chart is certificate.lc_chart
            and entry.raw_target_tube is certificate.lc_tube
            and type(pair) is tuple
            and pair in ((0, 1), (0, 2), (1, 2))
            and _planar_lc_mass_ratio_arithmetic_exact(
                certificate.lc_chart.masses,
                pair,
            )
            and len(source_domain) == len(lc_domain) == 2
            and source_domain[0] < source_domain[1]
            and lc_domain[0] < lc_domain[1]
            and Fraction.from_float(certificate.entry_transition.source_parameter)
            == source_domain[1]
            and Fraction.from_float(certificate.entry_transition.target_parameter)
            == lc_domain[0]
            and Fraction.from_float(certificate.lc_tube.anchor_parameter)
            == lc_domain[0]
        ):
            return None
        solution = _regularized_binary_solution_from_certificate(
            certificate.lc_chart
        )
        state = _planar_lc_state_intervals(
            solution,
            (certificate.lc_chart.parameter_interval[1],) * 2,
            inflate=lc_result.gronwall_error_bound,
        )
        lifted = tuple(
            (
                Fraction.from_float(component.lower),
                Fraction.from_float(component.upper),
            )
            for component in state
        )
        if not _fraction_box(lifted, 14):
            return None
        parameter = lc_domain[1]
        return MixedRetainedSafeRegion(
            region_type="certified_lifted_lc_exit_slice",
            coordinate_system="planar_lc_lifted_14",
            physical_time_interval=lifted[13],
            parameter_interval=(parameter, parameter),
            component_intervals=lifted,
            provenance_checker_id=_PREFIX_CHECKER_ID,
        )
    except Exception:
        return None


def _fresh_source_right_frontier(
    certificate: RawMixedPlanarContinuationCertificate,
) -> MixedRetainedSafeRegion | None:
    try:
        if not _binding_is_exact_point(certificate.source_binding):
            return None
        validation = check_validated_ordinary_ivp_chart(
            certificate.source_binding,
            certificate.source_tube,
            certificate.source_chart,
        )
        if not (
            type(validation) is ValidatedOrdinaryIVPChartCheckResult
            and validation.certified
            and type(validation.tube_result) is OrdinaryAposterioriTubeCheckResult
            and validation.tube_result.certified
        ):
            return None
        parameter = Fraction.from_float(
            certificate.source_chart.parameter_interval[1]
        )
        binding_parameter = Fraction.from_float(
            certificate.source_binding.chart_parameter
        )
        initial_time = Fraction.from_float(certificate.source_binding.initial_time)
        if parameter < binding_parameter:
            return None
        positions, velocities = _exact_rational_chart_projected_state_at_parameter(
            certificate.source_chart,
            parameter,
        )
        radius = Fraction.from_float(
            validation.tube_result.gronwall_error_bound
        )
        components = tuple(
            (center - radius, center + radius)
            for array in (positions, velocities)
            for center in array.reshape(-1)
        )
        if not _fraction_box(components, 12):
            return None
        physical_time = initial_time + parameter - binding_parameter
        return MixedRetainedSafeRegion(
            region_type="certified_ordinary_entry_slice",
            coordinate_system="ordinary_cartesian_q_then_v_12",
            physical_time_interval=(physical_time, physical_time),
            parameter_interval=(parameter, parameter),
            component_intervals=components,
            provenance_checker_id=_PREFIX_CHECKER_ID,
        )
    except Exception:
        return None


def _binding_is_exact_point(
    binding: InitialValueProblemBindingCertificate,
) -> bool:
    try:
        return bool(
            all(
                _finite_float(value)
                and Fraction.from_float(value) == 0
                for value in (
                    binding.time_tolerance,
                    binding.position_tolerance,
                    binding.velocity_tolerance,
                )
            )
        )
    except Exception:
        return False


def _final_retained_region(
    enclosure: MixedOrdinaryStateEnclosure,
) -> MixedRetainedSafeRegion:
    if not _ordinary_enclosure_schema(enclosure):
        raise ValueError("a valid final enclosure is required")
    components = tuple(
        interval
        for matrix in (enclosure.position_intervals, enclosure.velocity_intervals)
        for row in matrix
        for interval in row
    )
    return MixedRetainedSafeRegion(
        region_type="certified_ordinary_fixed_time_enclosure",
        coordinate_system="ordinary_cartesian_q_then_v_12",
        physical_time_interval=enclosure.physical_time_interval,
        parameter_interval=enclosure.target_parameter_preimage_interval,
        component_intervals=components,
        provenance_checker_id=_CHECKER_ID,
    )


def _covered_interval(
    certificate: RawMixedPlanarContinuationCertificate,
    retained: tuple[MixedRetainedSafeRegion, ...],
) -> FractionInterval | tuple[()]:
    try:
        if not retained:
            return ()
        initial = Fraction.from_float(certificate.source_binding.initial_time)
        source_right = (
            initial
            + Fraction.from_float(certificate.source_chart.parameter_interval[1])
            - Fraction.from_float(certificate.source_binding.chart_parameter)
        )
        frontier_lower = retained[0].physical_time_interval[0]
        return initial, max(initial, source_right, frontier_lower)
    except Exception:
        return ()


def _nested_missing(
    result: RawGaugeAwarePlanarLCExitContainmentResult | None,
    error: str,
) -> tuple[str, ...]:
    if type(result) is RawGaugeAwarePlanarLCExitContainmentResult:
        try:
            return tuple(result.missing_obligations)
        except Exception as exception:
            return (f"nested_missing_error:{type(exception).__name__}",)
    if error:
        return (f"nested_replay_exception:{error}",)
    return ("nested_exit_result_missing",)


def _obligation(
    name: str,
    certified: bool,
    detail: str,
) -> RawMixedPlanarContinuationObligation:
    return RawMixedPlanarContinuationObligation(name, bool(certified), detail)


def _fraction_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(endpoint) is Fraction for endpoint in value)
        and value[0] <= value[1]
    )


def _optional_fraction_interval(value: object) -> bool:
    return bool(
        (type(value) is tuple and len(value) == 0)
        or _fraction_interval(value)
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


def _ordinary_enclosure_schema(value: object) -> bool:
    return bool(
        type(value) is MixedOrdinaryStateEnclosure
        and type(value.enclosure_type) is str
        and value.enclosure_type == "validated_mixed_fixed_physical_time"
        and _fraction_interval(value.physical_time_interval)
        and value.physical_time_interval[0] == value.physical_time_interval[1]
        and _fraction_interval(value.target_parameter_preimage_interval)
        and _fraction_matrix(value.position_intervals)
        and _fraction_matrix(value.velocity_intervals)
    )


def _retained_region_schema(value: object) -> bool:
    if type(value) is not MixedRetainedSafeRegion:
        return False
    expected_length = {
        "certified_ordinary_entry_slice": 12,
        "certified_lifted_lc_exit_slice": 14,
        "certified_ordinary_target_right_frontier": 12,
        "certified_ordinary_fixed_time_enclosure": 12,
    }.get(value.region_type)
    expected_coordinates = {
        "certified_ordinary_entry_slice": "ordinary_cartesian_q_then_v_12",
        "certified_lifted_lc_exit_slice": "planar_lc_lifted_14",
        "certified_ordinary_target_right_frontier": (
            "ordinary_cartesian_q_then_v_12"
        ),
        "certified_ordinary_fixed_time_enclosure": (
            "ordinary_cartesian_q_then_v_12"
        ),
    }.get(value.region_type)
    return bool(
        expected_length is not None
        and type(value.coordinate_system) is str
        and value.coordinate_system == expected_coordinates
        and _fraction_interval(value.physical_time_interval)
        and _fraction_interval(value.parameter_interval)
        and (
            value.region_type == "certified_ordinary_fixed_time_enclosure"
            or value.parameter_interval[0] == value.parameter_interval[1]
        )
        and _fraction_box(value.component_intervals, expected_length)
        and type(value.provenance_checker_id) is str
        and bool(value.provenance_checker_id)
    )


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _sha256_hex(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_evidence_sha256() -> str:
    return hashlib.sha256(_INVALID_EVIDENCE_DIGEST_PAYLOAD).hexdigest()
