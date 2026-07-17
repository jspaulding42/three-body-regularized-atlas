"""Conditional proof-carrying ordinary-to-planar-LC entry.

This private composition lemma checks one carried ``N -> LC_ij`` handoff.  It
does **not** bind a new initial-value problem and it never calls the direct
exact-IVP entry wrapper.  Its soundness statement is conditional: a parent
checker must already have established the existence of the carried ordinary
solution in ``raw_source_tube`` and must have derived the correlated clock
origin interval ``B`` for that solution.  With ordinary clock convention
``t = s + B``, this checker derives the entry-time interval

``D_in = source_right_parameter + B``.

The finite checker reconstructs the complete ordinary endpoint box, derives
the canonical one/two-patch Levi-Civita square-root atlas, appends ``D_in`` to
every thirteen-dimensional lifted patch, derives the parity graph, tests its
two coherent global deck complements, and requires one complement to put all
complete fourteen-dimensional boxes in the target LC initial ball.

The constrained-lift/deck/gauge implication is a pinned analytic kernel: for
each physical state carried by the parent, the canonical atlas contains a
constrained LC lift; the two representatives are related by
``(z,w) -> (-z,-w)``; and one globally coherent complement may therefore be
selected.  The rectangular interval boxes themselves are not claimed to be
wholly constrained.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from fractions import Fraction
import json
import math
from typing import Any

import numpy as np

from .binary_chart import (
    planar_interval_to_regularized_binary_collision_chart_atlas,
)
from .certificate_checker import (
    CertificateCheckObligation,
    CertificateCheckResult,
    OrdinaryAposterioriTubeCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    _exact_rational_chart_physical_time_at_parameter,
    _exact_rational_chart_projected_state_at_parameter,
    _exact_rational_planar_lc_lifted_state_at_parameter,
    _flatten_interval_lc_state,
    _fraction_lower_float,
    _fraction_upper_float,
    check_ordinary_aposteriori_tube,
    check_ordinary_taylor_chart,
    check_planar_lc_aposteriori_tube,
    check_planar_levi_civita_binary_chart,
)
from .certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
)
from .lc_gauge_gluing import (
    PlanarLCGaugeGluingCertificate,
    PlanarLCGaugeGluingCheckResult,
    PlanarLCGaugeGluingObligation,
    PlanarLCGaugeOverlapEdge,
    check_planar_lc_gauge_gluing,
)
from .planar_lc_mass_coefficients import (
    PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
    derive_planar_lc_mass_coefficient_witness,
)


_SCHEMA_VERSION = 1
_RECORD_TYPE = "carried_planar_lc_entry_transition"
_RECORD_SOURCE = "private_carried_planar_lc_entry_v1"
_CHECKER_ID = "carried_planar_lc_entry_checker_v2"
_ANALYTIC_KERNEL_ID = "planar_lc_constrained_lift_deck_gauge_kernel_v1"
_ORDINARY_TUBE_CHECKER_ID = "independent_ordinary_aposteriori_tube_checker_v1"
_LC_TUBE_CHECKER_ID = "independent_planar_lc_aposteriori_tube_checker_v2"
_GAUGE_CHECKER_ID = "planar_lc_z2_gauge_gluing_checker_v1"
_OBLIGATION_NAMES = (
    "carried_lc_entry_exact_raw_schemas",
    "carried_lc_entry_transition_canonical_round_trip",
    "carried_lc_entry_identifiers_match_and_are_unique",
    "carried_lc_entry_parent_clock_origin_is_exact_interval",
    "carried_lc_entry_source_ordinary_chart_freshly_certified",
    "carried_lc_entry_source_ordinary_tube_freshly_certified",
    "carried_lc_entry_target_lc_chart_freshly_certified",
    "carried_lc_entry_target_lc_tube_freshly_certified",
    "carried_lc_entry_common_planar_mass_problem",
    "carried_lc_entry_pair_is_canonical_ascending",
    "carried_lc_entry_outward_mass_arithmetic_certified",
    "carried_lc_entry_exact_source_right_to_lc_left_anchor",
    "carried_lc_entry_complete_source_endpoint_box_reconstructed",
    "carried_lc_entry_selected_pair_collision_free",
    "carried_lc_entry_canonical_square_root_atlas_reconstructed",
    "carried_lc_entry_derived_parity_graph_certified",
    "carried_lc_entry_physical_time_interval_exactly_derived",
    "carried_lc_entry_all_lift_patches_have_positive_rho",
    "carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
    "carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
    "carried_lc_entry_one_global_complement_contains_all_complete_patches",
)

FractionInterval = tuple[Fraction, Fraction]
FractionBox = tuple[FractionInterval, ...]


@dataclass(frozen=True)
class CarriedPlanarLCEntryTransitionRecord:
    """Strict private wire record for one carried ``N -> LC`` handoff."""

    transition_id: str
    source_chart_id: str
    source_tube_id: str
    target_chart_id: str
    target_tube_id: str
    source_right_parameter: float
    target_left_parameter: float
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
            "source_right_parameter": self.source_right_parameter,
            "target_left_parameter": self.target_left_parameter,
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "CarriedPlanarLCEntryTransitionRecord":
        """Parse the exact canonical record shape without scalar coercion."""

        if type(data) is not dict:
            raise TypeError("carried LC entry transition must be a dict")
        expected = {field.name for field in dataclass_fields(cls)}
        actual = set(data)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            unknown = tuple(sorted(actual - expected))
            raise ValueError(
                "carried LC entry transition fields are noncanonical; "
                f"missing={missing!r}; unknown={unknown!r}"
            )
        string_fields = (
            "transition_id",
            "source_chart_id",
            "source_tube_id",
            "target_chart_id",
            "target_tube_id",
            "record_type",
            "source",
        )
        if not (
            all(type(data[name]) is str for name in string_fields)
            and type(data["source_right_parameter"]) is float
            and math.isfinite(data["source_right_parameter"])
            and type(data["target_left_parameter"]) is float
            and math.isfinite(data["target_left_parameter"])
            and type(data["schema_version"]) is int
        ):
            raise ValueError("carried LC entry transition scalars are noncanonical")
        record = cls(
            transition_id=data["transition_id"],
            source_chart_id=data["source_chart_id"],
            source_tube_id=data["source_tube_id"],
            target_chart_id=data["target_chart_id"],
            target_tube_id=data["target_tube_id"],
            source_right_parameter=data["source_right_parameter"],
            target_left_parameter=data["target_left_parameter"],
            schema_version=data["schema_version"],
            record_type=data["record_type"],
            source=data["source"],
        )
        if _canonical_json(data) != _canonical_json(record.to_dict()):
            raise ValueError("carried LC entry transition encoding is noncanonical")
        return record


@dataclass(frozen=True)
class CarriedPlanarLCEntryResult:
    """Fresh-replay result for the conditional carried-entry theorem.

    ``certified`` means the handoff implication is proved *conditional on* a
    parent having established the carried source solution and its correlated
    clock interval.  It is deliberately not an exact-IVP binding.
    """

    transition_id: str
    checker_id: str
    analytic_kernel_id: str
    mass_arithmetic_kernel_id: str
    raw_transition: CarriedPlanarLCEntryTransitionRecord
    raw_source_chart: OrdinaryTaylorChartCertificate
    raw_source_tube: OrdinaryAposterioriTubeCertificate
    raw_target_chart: PlanarLeviCivitaBinaryChartCertificate
    raw_target_tube: PlanarLCAposterioriTubeCertificate
    parent_source_clock_origin_interval: FractionInterval
    source_chart_result: CertificateCheckResult | None
    source_tube_result: OrdinaryAposterioriTubeCheckResult | None
    target_chart_result: CertificateCheckResult | None
    target_tube_result: PlanarLCAposterioriTubeCheckResult | None
    obligations: tuple[CertificateCheckObligation, ...]
    source_endpoint_state_box: FractionBox
    relative_position_box: tuple[FractionInterval, FractionInterval]
    canonical_lift_case: str
    patch_vertex_ids: tuple[str, ...]
    lifted_patch_boxes: tuple[FractionBox, ...]
    entry_time_interval: FractionInterval | tuple[()]
    timed_lifted_patch_boxes: tuple[FractionBox, ...]
    derived_edges: tuple[PlanarLCGaugeOverlapEdge, ...]
    gauge_result: PlanarLCGaugeGluingCheckResult | None
    tested_assignments: tuple[tuple[tuple[str, int], ...], ...]
    target_anchor: tuple[Fraction, ...]
    tested_lift_max_gaps: tuple[Fraction, ...]
    tested_containments: tuple[bool, ...]
    selected_complement_index: int
    selected_assignment: tuple[tuple[str, int], ...]
    selected_transformed_timed_patch_boxes: tuple[FractionBox, ...]
    entry_rho_lower_bound: Fraction

    def _snapshot_certified(self) -> bool:
        case_suffixes = {
            "closed_upper_singleton": ("upper",),
            "closed_lower_singleton": ("lower",),
            "right_half_singleton": ("right",),
            "strict_negative_cut_two_patch": ("upper", "lower"),
        }.get(self.canonical_lift_case)
        if case_suffixes is None:
            return False
        expected_patch_ids = tuple(
            f"{self.transition_id}:patch:{index}-{suffix}"
            for index, suffix in enumerate(case_suffixes)
        )
        if len(expected_patch_ids) == 1:
            expected_edges: tuple[PlanarLCGaugeOverlapEdge, ...] = ()
        else:
            expected_edges = (
                PlanarLCGaugeOverlapEdge(
                    overlap_id=f"{self.transition_id}:negative-axis-overlap",
                    source_chart_id=expected_patch_ids[0],
                    target_chart_id=expected_patch_ids[1],
                    parity=1,
                ),
            )
        assignments_valid = bool(
            type(self.tested_assignments) is tuple
            and len(self.tested_assignments) == 2
            and all(
                _assignment(value, expected_patch_ids)
                for value in self.tested_assignments
            )
        )
        selected_assignment_map = (
            dict(self.selected_assignment)
            if _assignment(self.selected_assignment, expected_patch_ids)
            else {}
        )
        expected_transformed = (
            tuple(
                _deck_transform_box(
                    box,
                    selected_assignment_map[expected_patch_ids[index]],
                )
                for index, box in enumerate(self.timed_lifted_patch_boxes)
            )
            if (
                selected_assignment_map
                and type(self.timed_lifted_patch_boxes) is tuple
                and len(self.timed_lifted_patch_boxes) == len(expected_patch_ids)
                and all(_fraction_box(box, 14) for box in self.timed_lifted_patch_boxes)
            )
            else ()
        )
        return bool(
            type(self) is CarriedPlanarLCEntryResult
            and type(self.transition_id) is str
            and bool(self.transition_id)
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.analytic_kernel_id) is str
            and self.analytic_kernel_id == _ANALYTIC_KERNEL_ID
            and type(self.mass_arithmetic_kernel_id) is str
            and self.mass_arithmetic_kernel_id
            == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
            and type(self.raw_transition) is CarriedPlanarLCEntryTransitionRecord
            and _transition_schema(self.raw_transition)
            and self.transition_id == self.raw_transition.transition_id
            and type(self.raw_source_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_source_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.raw_target_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.raw_target_tube) is PlanarLCAposterioriTubeCertificate
            and _ordinary_chart_schema(self.raw_source_chart)
            and _ordinary_tube_schema(self.raw_source_tube)
            and _lc_chart_schema(self.raw_target_chart)
            and _lc_tube_schema(self.raw_target_tube)
            and _fraction_interval(self.parent_source_clock_origin_interval)
            and _canonical_chart_result(
                self.source_chart_result,
                self.raw_source_chart.certificate_id,
                certificate_type="ordinary_taylor",
                checker_id="independent_ordinary_taylor_checker_interval_v2",
            )
            and _canonical_ordinary_tube_result(
                self.source_tube_result,
                self.raw_source_tube,
                self.raw_source_chart,
            )
            and _canonical_chart_result(
                self.target_chart_result,
                self.raw_target_chart.certificate_id,
                certificate_type="planar_levi_civita_binary",
                checker_id="independent_planar_lc_binary_checker_interval_v2",
            )
            and _canonical_lc_tube_result(
                self.target_tube_result,
                self.raw_target_tube,
                self.raw_target_chart,
            )
            and _exact_obligation_manifest(self.obligations)
            and _fraction_box(self.source_endpoint_state_box, 12)
            and type(self.relative_position_box) is tuple
            and len(self.relative_position_box) == 2
            and all(_fraction_interval(value) for value in self.relative_position_box)
            and type(self.canonical_lift_case) is str
            and type(self.patch_vertex_ids) is tuple
            and all(
                type(patch_id) is str and bool(patch_id)
                for patch_id in self.patch_vertex_ids
            )
            and self.patch_vertex_ids == expected_patch_ids
            and type(self.lifted_patch_boxes) is tuple
            and len(self.lifted_patch_boxes) == len(expected_patch_ids)
            and all(_fraction_box(box, 13) for box in self.lifted_patch_boxes)
            and _fraction_interval(self.entry_time_interval)
            and type(self.timed_lifted_patch_boxes) is tuple
            and len(self.timed_lifted_patch_boxes) == len(expected_patch_ids)
            and all(_fraction_box(box, 14) for box in self.timed_lifted_patch_boxes)
            and self.timed_lifted_patch_boxes
            == tuple(
                box + (self.entry_time_interval,)
                for box in self.lifted_patch_boxes
            )
            and type(self.derived_edges) is tuple
            and _edge_tuple_schema(self.derived_edges)
            and self.derived_edges == expected_edges
            and _canonical_gauge_result(
                self.gauge_result,
                self.transition_id,
                expected_patch_ids,
                expected_edges,
            )
            and assignments_valid
            and self.gauge_result is not None
            and self.tested_assignments[0] == self.gauge_result.gauge_assignment
            and self.tested_assignments[1]
            == tuple(
                (chart_id, bit ^ 1)
                for chart_id, bit in self.tested_assignments[0]
            )
            and type(self.target_anchor) is tuple
            and len(self.target_anchor) == 14
            and all(type(value) is Fraction for value in self.target_anchor)
            and type(self.tested_lift_max_gaps) is tuple
            and len(self.tested_lift_max_gaps) == 2
            and all(
                type(value) is Fraction and value >= 0
                for value in self.tested_lift_max_gaps
            )
            and type(self.tested_containments) is tuple
            and len(self.tested_containments) == 2
            and all(type(value) is bool for value in self.tested_containments)
            and type(self.selected_complement_index) is int
            and self.selected_complement_index in (0, 1)
            and _assignment(self.selected_assignment, expected_patch_ids)
            and self.selected_assignment
            == self.tested_assignments[self.selected_complement_index]
            and self.tested_containments[self.selected_complement_index] is True
            and type(self.selected_transformed_timed_patch_boxes) is tuple
            and self.selected_transformed_timed_patch_boxes == expected_transformed
            and all(
                _fraction_box(box, 14)
                for box in self.selected_transformed_timed_patch_boxes
            )
            and type(self.entry_rho_lower_bound) is Fraction
            and self.entry_rho_lower_bound > 0
        )

    @property
    def certified(self) -> bool:
        """Freshly replay and exact-compare this conditional theorem result."""

        try:
            if (
                type(self) is not CarriedPlanarLCEntryResult
                or not self._snapshot_certified()
            ):
                return False
            fresh = check_carried_planar_lc_entry(
                self.raw_transition,
                self.raw_source_chart,
                self.raw_source_tube,
                self.raw_target_chart,
                self.raw_target_tube,
                self.parent_source_clock_origin_interval,
            )
            return bool(
                type(fresh) is CarriedPlanarLCEntryResult
                and fresh._snapshot_certified()
                and fresh == self
            )
        except Exception:
            return False

    @property
    def conditional_containment_certified(self) -> bool:
        """Alias emphasizing that the parent-source existence is assumed."""

        return self.certified

    @property
    def constrained_newtonian_lift_certified(self) -> bool:
        """Whether the conditional existential constrained lift is certified."""

        return self.certified

    @property
    def physical_time_strictly_monotone_certified(self) -> bool:
        """Whether the pinned kernel starts the LC branch with positive rho."""

        return bool(self.certified and self.entry_rho_lower_bound > 0)

    @property
    def exact_initial_value_problem_binding_certified(self) -> bool:
        """This composition lemma intentionally proves no new IVP binding."""

        return False

    @property
    def raw_lc_chart(self) -> PlanarLeviCivitaBinaryChartCertificate:
        """Composition-facing name for the target LC chart."""

        return self.raw_target_chart

    @property
    def raw_lc_tube(self) -> PlanarLCAposterioriTubeCertificate:
        """Composition-facing name for the target LC tube."""

        return self.raw_target_tube

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = []
        for index, obligation in enumerate(self.obligations):
            if type(obligation) is not CertificateCheckObligation:
                missing.append(f"carried_lc_entry_malformed_obligation:{index}")
            elif obligation.certified is not True:
                missing.append(obligation.obligation)
        return tuple(missing)


def check_carried_planar_lc_entry(
    transition: CarriedPlanarLCEntryTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
    parent_source_clock_origin_interval: FractionInterval,
) -> CarriedPlanarLCEntryResult:
    """Check one conditional carried ``N -> LC`` endpoint handoff.

    The caller/parent is responsible for proving that its carried solution is
    in the source tube at that tube's anchor and that the supplied ``B`` is the
    correlated clock-origin enclosure for that same solution.
    """

    expected_types = (
        (transition, CarriedPlanarLCEntryTransitionRecord),
        (source_chart, OrdinaryTaylorChartCertificate),
        (source_tube, OrdinaryAposterioriTubeCertificate),
        (target_chart, PlanarLeviCivitaBinaryChartCertificate),
        (target_tube, PlanarLCAposterioriTubeCertificate),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("all carried LC entry raw inputs must have exact classes")

    transition_canonical = _transition_schema(transition)
    raw_schemas = bool(
        transition_canonical
        and _ordinary_chart_schema(source_chart)
        and _ordinary_tube_schema(source_tube)
        and _lc_chart_schema(target_chart)
        and _lc_tube_schema(target_tube)
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
    canonical_pair = bool(
        raw_schemas and target_chart.pair in ((0, 1), (0, 2), (1, 2))
    )
    try:
        mass_witness = derive_planar_lc_mass_coefficient_witness(
            target_chart.masses,
            target_chart.pair,
        )
        outward_mass_arithmetic = bool(canonical_pair and mass_witness.certified)
    except Exception:
        outward_mass_arithmetic = False
    endpoints = _exact_endpoint_handoff(
        transition,
        source_chart,
        target_chart,
        target_tube,
    )

    source_chart_result: CertificateCheckResult | None = None
    source_result: OrdinaryAposterioriTubeCheckResult | None = None
    target_chart_result: CertificateCheckResult | None = None
    target_result: PlanarLCAposterioriTubeCheckResult | None = None
    try:
        if raw_schemas:
            source_chart_result = check_ordinary_taylor_chart(source_chart)
    except Exception:
        pass
    try:
        if raw_schemas:
            source_result = check_ordinary_aposteriori_tube(
                source_tube,
                source_chart,
            )
    except Exception:
        pass
    try:
        if raw_schemas:
            target_chart_result = check_planar_levi_civita_binary_chart(
                target_chart
            )
    except Exception:
        pass
    try:
        if raw_schemas:
            target_result = check_planar_lc_aposteriori_tube(
                target_tube,
                target_chart,
            )
    except Exception:
        pass
    source_chart_certified = _canonical_chart_result(
        source_chart_result,
        source_chart.certificate_id,
        certificate_type="ordinary_taylor",
        checker_id="independent_ordinary_taylor_checker_interval_v2",
    )
    source_certified = _canonical_ordinary_tube_result(
        source_result,
        source_tube,
        source_chart,
    )
    target_chart_certified = _canonical_chart_result(
        target_chart_result,
        target_chart.certificate_id,
        certificate_type="planar_levi_civita_binary",
        checker_id="independent_planar_lc_binary_checker_interval_v2",
    )
    target_certified = _canonical_lc_tube_result(
        target_result,
        target_tube,
        target_chart,
    )

    source_box: FractionBox = ()
    relative_box = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    canonical_lift_case = ""
    patch_ids: tuple[str, ...] = ()
    lifted_boxes: tuple[FractionBox, ...] = ()
    entry_time: FractionInterval | tuple[()] = ()
    timed_boxes: tuple[FractionBox, ...] = ()
    edges: tuple[PlanarLCGaugeOverlapEdge, ...] = ()
    gauge_result: PlanarLCGaugeGluingCheckResult | None = None
    tested_assignments: tuple[tuple[tuple[str, int], ...], ...] = ()
    target_anchor: tuple[Fraction, ...] = ()
    tested_gaps: tuple[Fraction, ...] = ()
    tested_containments: tuple[bool, ...] = ()
    selected_index = -1
    selected_assignment: tuple[tuple[str, int], ...] = ()
    selected_boxes: tuple[FractionBox, ...] = ()
    rho_lower = Fraction(0)

    source_box_reconstructed = False
    collision_free = False
    atlas_reconstructed = False
    graph_certified = False
    time_derived = False
    rho_positive = False
    target_anchor_reconstructed = False
    analytic_kernel = False
    global_complement_contains = False

    prerequisites = (
        raw_schemas,
        identifiers,
        clock_valid,
        source_chart_certified,
        source_certified,
        target_chart_certified,
        target_certified,
        common_problem,
        canonical_pair,
        outward_mass_arithmetic,
        endpoints,
    )
    if all(prerequisites):
        try:
            source_parameter = Fraction.from_float(
                transition.source_right_parameter
            )
            target_parameter = Fraction.from_float(
                transition.target_left_parameter
            )
            source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            source_centers = _flatten_fraction_matrices(source_q, source_v)
            if source_result is None:
                raise ValueError("fresh source tube replay is missing")
            source_radius = Fraction.from_float(
                source_result.gronwall_error_bound
            )
            source_box = tuple(
                (
                    Fraction.from_float(
                        _fraction_lower_float(center - source_radius)
                    ),
                    Fraction.from_float(
                        _fraction_upper_float(center + source_radius)
                    ),
                )
                for center in source_centers
            )
            source_box_reconstructed = _fraction_box(source_box, 12)

            first, second = target_chart.pair
            relative_box = tuple(
                (
                    source_box[2 * second + axis][0]
                    - source_box[2 * first + axis][1],
                    source_box[2 * second + axis][1]
                    - source_box[2 * first + axis][0],
                )
                for axis in range(2)
            )  # type: ignore[assignment]
            collision_free = bool(
                sum(_square_lower(bounds) for bounds in relative_box) > 0
            )

            expected_count = 0
            x_low, x_high = relative_box[0]
            y_low, y_high = relative_box[1]
            if y_low >= 0:
                canonical_lift_case = "closed_upper_singleton"
                expected_count = 1
            elif y_high <= 0:
                canonical_lift_case = "closed_lower_singleton"
                expected_count = 1
            elif x_low > 0:
                canonical_lift_case = "right_half_singleton"
                expected_count = 1
            elif y_low < 0 < y_high and x_high < 0:
                canonical_lift_case = "strict_negative_cut_two_patch"
                expected_count = 2

            if collision_free and expected_count in (1, 2):
                atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
                    tuple(
                        (float(lower), float(upper))
                        for lower, upper in source_box
                    ),
                    np.asarray(source_chart.masses, dtype=float),
                    pair=target_chart.pair,
                )
                lifted_boxes = tuple(
                    tuple(
                        (
                            Fraction.from_float(float(interval.lower)),
                            Fraction.from_float(float(interval.upper)),
                        )
                        for interval in _flatten_interval_lc_state(branch)
                    )
                    for branch in atlas
                )
                atlas_reconstructed = bool(
                    len(lifted_boxes) == expected_count
                    and all(_fraction_box(box, 13) for box in lifted_boxes)
                )

            if atlas_reconstructed:
                if canonical_lift_case == "strict_negative_cut_two_patch":
                    patch_ids = (
                        f"{transition.transition_id}:patch:0-upper",
                        f"{transition.transition_id}:patch:1-lower",
                    )
                    edges = (
                        PlanarLCGaugeOverlapEdge(
                            overlap_id=(
                                f"{transition.transition_id}:negative-axis-overlap"
                            ),
                            source_chart_id=patch_ids[0],
                            target_chart_id=patch_ids[1],
                            parity=1,
                        ),
                    )
                else:
                    suffix = {
                        "closed_upper_singleton": "upper",
                        "closed_lower_singleton": "lower",
                        "right_half_singleton": "right",
                    }[canonical_lift_case]
                    patch_ids = (
                        f"{transition.transition_id}:patch:0-{suffix}",
                    )
                    edges = ()
                gauge_result = check_planar_lc_gauge_gluing(
                    PlanarLCGaugeGluingCertificate(
                        certificate_id=(
                            f"{transition.transition_id}:derived-gauge-cover"
                        ),
                        chart_ids=patch_ids,
                        overlaps=edges,
                        source="derived_carried_ordinary_to_lc_lift_cover",
                    )
                )
                graph_certified = _canonical_gauge_result(
                    gauge_result,
                    transition.transition_id,
                    patch_ids,
                    edges,
                )

                entry_time = (
                    parent_source_clock_origin_interval[0] + source_parameter,
                    parent_source_clock_origin_interval[1] + source_parameter,
                )
                time_derived = _fraction_interval(entry_time)
                timed_boxes = tuple(
                    box + (entry_time,) for box in lifted_boxes
                )
                rho_lower = min(
                    sum(_square_lower(box[index]) for index in (0, 1))
                    for box in lifted_boxes
                )
                rho_positive = rho_lower > 0

            if graph_certified and time_derived and rho_positive:
                lifted_anchor = (
                    _exact_rational_planar_lc_lifted_state_at_parameter(
                        target_chart,
                        target_parameter,
                    )
                )
                target_time = _exact_rational_chart_physical_time_at_parameter(
                    target_chart,
                    target_parameter,
                )
                target_anchor = lifted_anchor + (target_time,)
                target_anchor_reconstructed = bool(
                    len(target_anchor) == 14
                    and all(type(value) is Fraction for value in target_anchor)
                )

            analytic_kernel = bool(
                source_box_reconstructed
                and collision_free
                and atlas_reconstructed
                and graph_certified
                and rho_positive
                and outward_mass_arithmetic
            )
            if analytic_kernel and target_anchor_reconstructed:
                if gauge_result is None:
                    raise ValueError("derived gauge replay is missing")
                base = gauge_result.gauge_assignment
                complement = tuple(
                    (chart_id, bit ^ 1) for chart_id, bit in base
                )
                tested_assignments = (base, complement)
                target_radius = Fraction.from_float(
                    target_tube.initial_error_bound
                )
                transformed_by_assignment = []
                gaps = []
                containments = []
                for assignment in tested_assignments:
                    by_id = dict(assignment)
                    transformed = tuple(
                        _deck_transform_box(
                            box,
                            by_id[patch_ids[index]],
                        )
                        for index, box in enumerate(timed_boxes)
                    )
                    gap = max(
                        (
                            max(abs(lower - center), abs(upper - center))
                            for box in transformed
                            for (lower, upper), center in zip(box, target_anchor)
                        ),
                        default=Fraction(0),
                    )
                    contained = bool(
                        transformed
                        and all(
                            center - target_radius <= lower
                            and upper <= center + target_radius
                            for box in transformed
                            for (lower, upper), center in zip(box, target_anchor)
                        )
                    )
                    transformed_by_assignment.append(transformed)
                    gaps.append(gap)
                    containments.append(contained)
                tested_gaps = tuple(gaps)
                tested_containments = tuple(containments)
                for index, contained in enumerate(tested_containments):
                    if contained:
                        selected_index = index
                        selected_assignment = tested_assignments[index]
                        selected_boxes = transformed_by_assignment[index]
                        break
                global_complement_contains = selected_index in (0, 1)
        except Exception:
            # Exact-class inputs still contain untrusted serialized fields.
            # Arithmetic, indexing, interval, and equality exceptions reject.
            pass

    obligations = (
        _obligation(
            "carried_lc_entry_exact_raw_schemas",
            raw_schemas,
            "exact built-in chart/tube fields and private transition types",
        ),
        _obligation(
            "carried_lc_entry_transition_canonical_round_trip",
            transition_canonical,
            "strict versioned dict round trip with no coercion or extra fields",
        ),
        _obligation(
            "carried_lc_entry_identifiers_match_and_are_unique",
            identifiers,
            (
                "transition IDs bind two distinct chart/tube vertices and do "
                "not collide with any deterministic patch, edge, or gauge ID"
            ),
        ),
        _obligation(
            "carried_lc_entry_parent_clock_origin_is_exact_interval",
            clock_valid,
            (
                "B_source is correlated parent state, not independent transition "
                f"evidence: {parent_source_clock_origin_interval!s}"
            ),
        ),
        _obligation(
            "carried_lc_entry_source_ordinary_chart_freshly_certified",
            source_chart_certified,
            _nested_detail(source_chart_result),
        ),
        _obligation(
            "carried_lc_entry_source_ordinary_tube_freshly_certified",
            source_certified,
            (
                f"{_nested_detail(source_result)}; the parent must separately "
                "prove that its carried solution lies in this tube's initial "
                f"ball at anchor={source_tube.anchor_parameter!r}"
            ),
        ),
        _obligation(
            "carried_lc_entry_target_lc_chart_freshly_certified",
            target_chart_certified,
            _nested_detail(target_chart_result),
        ),
        _obligation(
            "carried_lc_entry_target_lc_tube_freshly_certified",
            target_certified,
            _nested_detail(target_result),
        ),
        _obligation(
            "carried_lc_entry_common_planar_mass_problem",
            common_problem,
            (
                f"source_masses={source_chart.masses!r}; "
                f"target_masses={target_chart.masses!r}"
            ),
        ),
        _obligation(
            "carried_lc_entry_pair_is_canonical_ascending",
            canonical_pair,
            f"pair={target_chart.pair!r}",
        ),
        _obligation(
            "carried_lc_entry_outward_mass_arithmetic_certified",
            outward_mass_arithmetic,
            (
                "all LC mass coefficients are freshly derived as exact "
                "Fractions and tightly enclosed outward; "
                f"kernel={PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID}"
            ),
        ),
        _obligation(
            "carried_lc_entry_exact_source_right_to_lc_left_anchor",
            endpoints,
            (
                f"source_right={transition.source_right_parameter!r}; "
                f"target_left={transition.target_left_parameter!r}; "
                f"target_tube_anchor={target_tube.anchor_parameter!r}"
            ),
        ),
        _obligation(
            "carried_lc_entry_complete_source_endpoint_box_reconstructed",
            source_box_reconstructed,
            f"components={len(source_box)}",
        ),
        _obligation(
            "carried_lc_entry_selected_pair_collision_free",
            collision_free,
            f"relative_position_box={relative_box!s}",
        ),
        _obligation(
            "carried_lc_entry_canonical_square_root_atlas_reconstructed",
            atlas_reconstructed,
            (
                f"case={canonical_lift_case!r}; "
                f"patch_count={len(lifted_boxes)}"
            ),
        ),
        _obligation(
            "carried_lc_entry_derived_parity_graph_certified",
            graph_certified,
            f"edge_count={len(edges)}; gauge={_nested_detail(gauge_result)}",
        ),
        _obligation(
            "carried_lc_entry_physical_time_interval_exactly_derived",
            time_derived,
            (
                "D_in=B_source+source_right_parameter; "
                f"D_in={entry_time!s}"
            ),
        ),
        _obligation(
            "carried_lc_entry_all_lift_patches_have_positive_rho",
            rho_positive,
            f"rho_lower={rho_lower!s}",
        ),
        _obligation(
            "carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
            target_anchor_reconstructed,
            f"components={len(target_anchor)}",
        ),
        _obligation(
            "carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
            analytic_kernel,
            (
                f"kernel={_ANALYTIC_KERNEL_ID}; conditional on the parent's "
                "carried source solution, an existential constrained lift is "
                "covered; rectangular boxes are not wholly constrained"
            ),
        ),
        _obligation(
            "carried_lc_entry_one_global_complement_contains_all_complete_patches",
            global_complement_contains,
            (
                f"tested_max_gaps={tested_gaps!s}; "
                f"tested_containments={tested_containments!s}; "
                f"selected={selected_index}"
            ),
        ),
    )
    transition_id = (
        transition.transition_id if type(transition.transition_id) is str else ""
    )
    return CarriedPlanarLCEntryResult(
        transition_id=transition_id,
        checker_id=_CHECKER_ID,
        analytic_kernel_id=_ANALYTIC_KERNEL_ID,
        mass_arithmetic_kernel_id=PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
        raw_transition=transition,
        raw_source_chart=source_chart,
        raw_source_tube=source_tube,
        raw_target_chart=target_chart,
        raw_target_tube=target_tube,
        parent_source_clock_origin_interval=parent_source_clock_origin_interval,
        source_chart_result=source_chart_result,
        source_tube_result=source_result,
        target_chart_result=target_chart_result,
        target_tube_result=target_result,
        obligations=obligations,
        source_endpoint_state_box=source_box,
        relative_position_box=relative_box,
        canonical_lift_case=canonical_lift_case,
        patch_vertex_ids=patch_ids,
        lifted_patch_boxes=lifted_boxes,
        entry_time_interval=entry_time,
        timed_lifted_patch_boxes=timed_boxes,
        derived_edges=edges,
        gauge_result=gauge_result,
        tested_assignments=tested_assignments,
        target_anchor=target_anchor,
        tested_lift_max_gaps=tested_gaps,
        tested_containments=tested_containments,
        selected_complement_index=selected_index,
        selected_assignment=selected_assignment,
        selected_transformed_timed_patch_boxes=selected_boxes,
        entry_rho_lower_bound=rho_lower,
    )


# Composition-facing aliases use the longer ordinary-to-LC spelling expected
# by the repeated-chain layer.  They are aliases, not subclasses or alternate
# checker paths, so exact-type replay remains single-valued.
CarriedOrdinaryToPlanarLCEntryResult = CarriedPlanarLCEntryResult
check_carried_ordinary_to_planar_lc_entry = check_carried_planar_lc_entry


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _transition_schema(value: object) -> bool:
    try:
        return bool(
            type(value) is CarriedPlanarLCEntryTransitionRecord
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
            and _finite_float(value.source_right_parameter)
            and _finite_float(value.target_left_parameter)
            and CarriedPlanarLCEntryTransitionRecord.from_dict(value.to_dict())
            == value
        )
    except Exception:
        return False


def _ordinary_chart_schema(value: object) -> bool:
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
        and _float_tuple(value.masses, length=3, positive=True)
        and _ordinary_series(value.position_coefficients)
        and _ordinary_series(value.velocity_coefficients)
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


def _lc_chart_schema(value: object) -> bool:
    if type(value) is not PlanarLeviCivitaBinaryChartCertificate:
        return False
    vector_series = (
        value.z_coefficients,
        value.z_velocity_coefficients,
        value.binary_center_coefficients,
        value.binary_center_velocity_coefficients,
        value.third_offset_coefficients,
        value.third_offset_velocity_coefficients,
    )
    return bool(
        all(
            type(item) is str and bool(item)
            for item in (
                value.certificate_id,
                value.chart_id,
                value.chart_type,
                value.source,
            )
        )
        and value.chart_type == "planar_levi_civita_binary"
        and _float_tuple(value.masses, length=3, positive=True)
        and type(value.pair) is tuple
        and len(value.pair) == 2
        and all(type(index) is int for index in value.pair)
        and all(_vector_series(series) for series in vector_series)
        and len({len(series) for series in vector_series}) == 1
        and _float_tuple(
            value.pair_energy_coefficients,
            length=len(value.z_coefficients),
        )
        and _float_tuple(
            value.physical_time_coefficients,
            length=len(value.z_coefficients),
        )
        and _float_interval(value.parameter_interval)
        and _float_interval(value.physical_time_interval)
        and all(
            _finite_float(item)
            for item in (
                value.coefficient_tolerance,
                value.regularized_residual_tolerance,
                value.projected_residual_tolerance,
                value.tail_bound,
                value.projection_rho_lower_bound,
            )
        )
        and type(value.sample_count) is int
        and value.sample_count >= 1
    )


def _lc_tube_schema(value: object) -> bool:
    return bool(
        type(value) is PlanarLCAposterioriTubeCertificate
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
        and type(value.require_pair_energy_constraint) is bool
    )


def _identifiers_match_and_unique(
    transition: CarriedPlanarLCEntryTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
) -> bool:
    try:
        raw_identifiers = (
            transition.transition_id,
            source_chart.certificate_id,
            source_chart.chart_id,
            source_tube.tube_id,
            target_chart.certificate_id,
            target_chart.chart_id,
            target_tube.tube_id,
        )
        prefix = transition.transition_id
        derived_identifiers = (
            f"{prefix}:derived-gauge-cover",
            f"{prefix}:patch:0-upper",
            f"{prefix}:patch:0-lower",
            f"{prefix}:patch:0-right",
            f"{prefix}:patch:1-lower",
            f"{prefix}:negative-axis-overlap",
        )
        identifiers = raw_identifiers + derived_identifiers
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
    target: PlanarLeviCivitaBinaryChartCertificate,
) -> bool:
    try:
        return bool(
            _ordinary_chart_schema(source)
            and _lc_chart_schema(target)
            and source.masses == target.masses
            and source.dimension == 2
        )
    except Exception:
        return False


def _exact_endpoint_handoff(
    transition: CarriedPlanarLCEntryTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
) -> bool:
    try:
        source_domain = tuple(
            Fraction.from_float(value) for value in source_chart.parameter_interval
        )
        target_domain = tuple(
            Fraction.from_float(value) for value in target_chart.parameter_interval
        )
        return bool(
            source_domain[0] < source_domain[1]
            and target_domain[0] < target_domain[1]
            and Fraction.from_float(transition.source_right_parameter)
            == source_domain[1]
            and Fraction.from_float(transition.target_left_parameter)
            == target_domain[0]
            and Fraction.from_float(target_tube.anchor_parameter)
            == target_domain[0]
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
        raise ValueError("carried ordinary endpoint must have 12 exact components")
    return values


def _canonical_ordinary_tube_result(
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
        and result.chart_id == raw_chart.chart_id == raw_tube.chart_id
        and type(result.checker_id) is str
        and result.checker_id == _ORDINARY_TUBE_CHECKER_ID
        and _canonical_obligations(result.obligations)
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


def _canonical_chart_result(
    result: object,
    certificate_id: str,
    *,
    certificate_type: str,
    checker_id: str,
) -> bool:
    return bool(
        type(result) is CertificateCheckResult
        and type(result.certificate_id) is str
        and result.certificate_id == certificate_id
        and type(result.certificate_type) is str
        and result.certificate_type == certificate_type
        and type(result.checker_id) is str
        and result.checker_id == checker_id
        and _canonical_obligations(result.obligations)
        and _finite_float(result.max_coefficient_residual)
        and _finite_float(result.max_sampled_newton_residual)
        and result.certified
    )


def _canonical_lc_tube_result(
    result: object,
    raw_tube: PlanarLCAposterioriTubeCertificate,
    raw_chart: PlanarLeviCivitaBinaryChartCertificate,
) -> bool:
    return bool(
        type(result) is PlanarLCAposterioriTubeCheckResult
        and type(result.tube_id) is str
        and bool(result.tube_id)
        and result.tube_id == raw_tube.tube_id
        and type(result.chart_id) is str
        and result.chart_id == raw_chart.chart_id == raw_tube.chart_id
        and type(result.checker_id) is str
        and result.checker_id == _LC_TUBE_CHECKER_ID
        and type(result.mass_arithmetic_kernel_id) is str
        and result.mass_arithmetic_kernel_id
        == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
        and _canonical_obligations(result.obligations)
        and all(
            _finite_float(value)
            for value in (
                result.defect_bound,
                result.lipschitz_bound,
                result.gronwall_error_bound,
                result.third_body_distance_floor,
                result.anchor_pair_energy_constraint_residual,
            )
        )
        and result.gronwall_error_bound >= 0
        and type(result.pair_energy_constraint_anchor_certified) is bool
        and type(result.anchor_is_polynomial_center) is bool
        and result.certified
    )


def _canonical_gauge_result(
    result: object,
    transition_id: str,
    patch_ids: tuple[str, ...],
    edges: tuple[PlanarLCGaugeOverlapEdge, ...],
) -> bool:
    return bool(
        type(result) is PlanarLCGaugeGluingCheckResult
        and type(result.certificate_id) is str
        and result.certificate_id == f"{transition_id}:derived-gauge-cover"
        and type(result.checker_id) is str
        and result.checker_id == _GAUGE_CHECKER_ID
        and type(result.obligations) is tuple
        and bool(result.obligations)
        and all(
            type(item) is PlanarLCGaugeGluingObligation
            and type(item.obligation) is str
            and bool(item.obligation)
            and type(item.certified) is bool
            and item.certified is True
            and type(item.detail) is str
            for item in result.obligations
        )
        and type(result.chart_ids) is tuple
        and all(
            type(chart_id) is str and bool(chart_id)
            for chart_id in result.chart_ids
        )
        and result.chart_ids == patch_ids
        and type(result.checked_overlaps) is tuple
        and _edge_tuple_schema(result.checked_overlaps)
        and result.checked_overlaps == edges
        and _assignment(result.gauge_assignment, patch_ids)
        and type(result.component_count) is int
        and result.component_count == 1
        and type(result.obstruction_cycle_chart_ids) is tuple
        and not result.obstruction_cycle_chart_ids
        and type(result.obstruction_cycle_edges) is tuple
        and not result.obstruction_cycle_edges
        and result.certified
    )


def _edge_tuple_schema(value: object) -> bool:
    return bool(
        type(value) is tuple
        and all(
            type(edge) is PlanarLCGaugeOverlapEdge
            and type(edge.overlap_id) is str
            and bool(edge.overlap_id)
            and type(edge.source_chart_id) is str
            and bool(edge.source_chart_id)
            and type(edge.target_chart_id) is str
            and bool(edge.target_chart_id)
            and type(edge.parity) is int
            and edge.parity in (0, 1)
            for edge in value
        )
    )


def _canonical_obligations(value: object) -> bool:
    return bool(
        type(value) is tuple
        and bool(value)
        and all(
            type(item) is CertificateCheckObligation
            and type(item.obligation) is str
            and bool(item.obligation)
            and type(item.certified) is bool
            and item.certified is True
            and type(item.detail) is str
            for item in value
        )
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


def _assignment(value: object, patch_ids: tuple[str, ...]) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(patch_ids)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is int
            and item[1] in (0, 1)
            for item in value
        )
        and tuple(item[0] for item in value) == patch_ids
    )


def _deck_transform_box(box: FractionBox, bit: int) -> FractionBox:
    if type(bit) is not int or bit not in (0, 1):
        raise ValueError("deck bit must be 0 or 1")
    return tuple(
        (-upper, -lower) if bit == 1 and index < 4 else (lower, upper)
        for index, (lower, upper) in enumerate(box)
    )


def _square_lower(bounds: FractionInterval) -> Fraction:
    lower, upper = bounds
    if lower <= 0 <= upper:
        return Fraction(0)
    return min(lower * lower, upper * upper)


def _ordinary_series(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) >= 2
        and all(
            type(coefficient) is tuple
            and len(coefficient) == 3
            and all(_float_tuple(row, length=2) for row in coefficient)
            for coefficient in value
        )
    )


def _vector_series(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) >= 2
        and all(_float_tuple(coefficient, length=2) for coefficient in value)
    )


def _float_tuple(
    value: object,
    *,
    length: int,
    positive: bool = False,
) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == length
        and all(
            _finite_float(item) and (not positive or item > 0)
            for item in value
        )
    )


def _float_interval(value: object) -> bool:
    return bool(
        _float_tuple(value, length=2)
        and value[0] < value[1]
    )


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


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
