"""Private carried planar LC passage exit and ordinary re-entry.

This checker is a *conditional composition lemma*.  Its parent must already
have proved that the one root-IVP Newtonian branch is enclosed by the supplied
source ordinary tube with clock ``t = s + b`` for one
``b in parent_source_clock_origin_interval``.  The parent invariant is not
wire evidence and this module never fabricates a new point-IVP binding.

The checker freshly replays the raw carried ordinary-to-LC entry, propagates
its one constrained lift through the freshly checked LC tube, reconstructs the
complete fourteen-dimensional right slice, projects only after proving
``rho > 0``, contains the complete Cartesian projection in a fresh target
ordinary tube, and derives the target clock interval ``B = D - a``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from .certificate_checker import (
    CertificateCheckObligation,
    OrdinaryAposterioriTubeCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    _exact_rational_chart_projected_state_at_parameter,
    _planar_lc_mass_ratio_arithmetic_exact,
    _planar_lc_state_intervals,
    _project_interval_planar_lc_state,
    _regularized_binary_solution_from_certificate,
    check_ordinary_aposteriori_tube,
    check_planar_lc_aposteriori_tube,
)
from .certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
)
from .intervals import FloatInterval
from .proof_carrying_carried_planar_lc_entry import (
    CarriedPlanarLCEntryResult,
    CarriedPlanarLCEntryTransitionRecord,
    check_carried_planar_lc_entry,
)


_CHECKER_ID = "carried_planar_lc_exit_checker_v1"
_ANALYTIC_KERNEL_ID = "planar_lc_analytic_kernel_v1"
_PARENT_INVARIANT_ID = "parent_carried_ordinary_solution_invariant_v1"
_OBLIGATION_NAMES = (
    "carried_lc_exit_exact_raw_schemas",
    "carried_lc_exit_identifiers_match_and_are_unique",
    "carried_lc_exit_parent_source_invariant_is_explicit_condition",
    "carried_lc_exit_entry_freshly_replayed_and_certified",
    "carried_lc_exit_common_planar_mass_problem",
    "carried_lc_exit_pair_is_canonical_ascending",
    "carried_lc_exit_mass_ratio_arithmetic_exact",
    "carried_lc_exit_exact_right_to_left_endpoint_handoff",
    "carried_lc_exit_constrained_entry_branch_carried",
    "carried_lc_exit_constraint_invariance_kernel",
    "carried_lc_exit_lc_tube_freshly_certified",
    "carried_lc_exit_third_body_separated",
    "carried_lc_exit_target_ordinary_tube_freshly_certified",
    "carried_lc_exit_strict_physical_clock_kernel",
    "carried_lc_exit_complete_inflated_slice_reconstructed",
    "carried_lc_exit_complete_slice_rho_positive",
    "carried_lc_exit_complete_cartesian_projection_reconstructed",
    "carried_lc_exit_deck_equivariant_newton_projection_kernel",
    "carried_lc_exit_target_initial_ball_contains_complete_projection",
    "carried_lc_exit_time_interval_derived_from_component_fourteen",
    "carried_lc_exit_target_clock_origin_exactly_derived",
)

FractionInterval = tuple[Fraction, Fraction]
FractionMatrix = tuple[tuple[FractionInterval, ...], ...]


@dataclass(frozen=True)
class CarriedPlanarLCExitResult:
    """Immutable conditional carried-passage replay result."""

    transition_id: str
    checker_id: str
    analytic_kernel_id: str
    parent_source_invariant_id: str
    raw_entry_transition: CarriedPlanarLCEntryTransitionRecord
    raw_source_chart: OrdinaryTaylorChartCertificate
    raw_source_tube: OrdinaryAposterioriTubeCertificate
    parent_source_clock_origin_interval: FractionInterval
    raw_lc_chart: PlanarLeviCivitaBinaryChartCertificate
    raw_lc_tube: PlanarLCAposterioriTubeCertificate
    raw_exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate
    raw_target_chart: OrdinaryTaylorChartCertificate
    raw_target_tube: OrdinaryAposterioriTubeCertificate
    entry_result: CarriedPlanarLCEntryResult | None
    lc_tube_result: PlanarLCAposterioriTubeCheckResult | None
    target_tube_result: OrdinaryAposterioriTubeCheckResult | None
    obligations: tuple[CertificateCheckObligation, ...]
    lifted_exit_slice: tuple[FractionInterval, ...]
    exit_rho_interval: FractionInterval | tuple[()]
    projected_position_intervals: FractionMatrix
    projected_velocity_intervals: FractionMatrix
    maximum_projected_anchor_gap: Fraction | None
    exit_time_interval: FractionInterval | tuple[()]
    target_clock_origin_interval: FractionInterval | tuple[()]

    def _snapshot_certified(self) -> bool:
        return bool(
            type(self) is CarriedPlanarLCExitResult
            and type(self.transition_id) is str
            and bool(self.transition_id)
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.analytic_kernel_id) is str
            and self.analytic_kernel_id == _ANALYTIC_KERNEL_ID
            and type(self.parent_source_invariant_id) is str
            and self.parent_source_invariant_id == _PARENT_INVARIANT_ID
            and type(self.raw_entry_transition)
            is CarriedPlanarLCEntryTransitionRecord
            and type(self.raw_source_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_source_tube) is OrdinaryAposterioriTubeCertificate
            and _fraction_interval(self.parent_source_clock_origin_interval)
            and type(self.raw_lc_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.raw_lc_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_exit_transition)
            is PlanarLCToOrdinaryEnclosureTransitionCertificate
            and type(self.raw_target_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_target_tube) is OrdinaryAposterioriTubeCertificate
            and _canonical_entry_result(
                self.entry_result,
                self.raw_entry_transition,
                self.raw_source_chart,
                self.raw_source_tube,
                self.raw_lc_chart,
                self.raw_lc_tube,
            )
            and _canonical_lc_tube_result(
                self.lc_tube_result,
                self.raw_lc_tube,
                self.raw_lc_chart,
            )
            and _canonical_ordinary_tube_result(
                self.target_tube_result,
                self.raw_target_tube,
                self.raw_target_chart,
            )
            and _exact_obligation_manifest(self.obligations)
            and _fraction_box(self.lifted_exit_slice, 14)
            and _fraction_interval(self.exit_rho_interval, positive=True)
            and _fraction_matrix(self.projected_position_intervals)
            and _fraction_matrix(self.projected_velocity_intervals)
            and type(self.maximum_projected_anchor_gap) is Fraction
            and self.maximum_projected_anchor_gap >= 0
            and _fraction_interval(self.exit_time_interval)
            and _fraction_interval(self.target_clock_origin_interval)
        )

    @property
    def certified(self) -> bool:
        """Freshly replay all raw inputs and exact-compare the snapshot."""

        try:
            if (
                type(self) is not CarriedPlanarLCExitResult
                or not self._snapshot_certified()
            ):
                return False
            fresh = check_carried_planar_lc_exit(
                self.raw_entry_transition,
                self.raw_source_chart,
                self.raw_source_tube,
                self.raw_lc_chart,
                self.raw_lc_tube,
                self.raw_exit_transition,
                self.raw_target_chart,
                self.raw_target_tube,
                self.parent_source_clock_origin_interval,
            )
            return bool(
                type(fresh) is CarriedPlanarLCExitResult
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
                missing.append(f"carried_lc_exit_malformed_obligation:{index}")
            elif obligation.certified is not True:
                missing.append(obligation.obligation)
        return tuple(missing)


def check_carried_planar_lc_exit(
    entry_transition: CarriedPlanarLCEntryTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    lc_chart: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
    parent_source_clock_origin_interval: FractionInterval,
) -> CarriedPlanarLCExitResult:
    """Check one conditional carried ``N -> LC -> N`` passage exit.

    ``parent_source_clock_origin_interval`` is parent-derived state.  Supplying
    it directly proves only this conditional lemma; a top-level chain checker
    must discharge ``parent_carried_ordinary_solution_invariant_v1`` from the
    replayed root prefix.
    """

    expected_types = (
        (entry_transition, CarriedPlanarLCEntryTransitionRecord),
        (source_chart, OrdinaryTaylorChartCertificate),
        (source_tube, OrdinaryAposterioriTubeCertificate),
        (lc_chart, PlanarLeviCivitaBinaryChartCertificate),
        (lc_tube, PlanarLCAposterioriTubeCertificate),
        (exit_transition, PlanarLCToOrdinaryEnclosureTransitionCertificate),
        (target_chart, OrdinaryTaylorChartCertificate),
        (target_tube, OrdinaryAposterioriTubeCertificate),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("all carried LC exit raw inputs must have exact classes")

    parent_clock_valid = _fraction_interval(parent_source_clock_origin_interval)
    raw_schemas = bool(
        _entry_schema(entry_transition)
        and _ordinary_chart_schema(source_chart)
        and _ordinary_tube_schema(source_tube)
        and _lc_chart_schema(lc_chart)
        and _lc_tube_schema(lc_tube)
        and _exit_schema(exit_transition)
        and _ordinary_chart_schema(target_chart)
        and _ordinary_tube_schema(target_tube)
    )
    identifiers = _identifiers_match_and_unique(
        entry_transition,
        source_chart,
        source_tube,
        lc_chart,
        lc_tube,
        exit_transition,
        target_chart,
        target_tube,
    )
    common_problem = _common_problem(source_chart, lc_chart, target_chart)
    canonical_pair = bool(
        type(lc_chart.pair) is tuple
        and lc_chart.pair in ((0, 1), (0, 2), (1, 2))
    )
    try:
        exact_ratio = bool(
            canonical_pair
            and _planar_lc_mass_ratio_arithmetic_exact(
                lc_chart.masses,
                lc_chart.pair,
            )
        )
    except Exception:
        exact_ratio = False
    endpoint_handoff = _exact_endpoint_handoff(
        entry_transition,
        source_chart,
        source_tube,
        lc_chart,
        lc_tube,
        exit_transition,
        target_chart,
        target_tube,
    )

    entry_result: CarriedPlanarLCEntryResult | None = None
    lc_result: PlanarLCAposterioriTubeCheckResult | None = None
    target_result: OrdinaryAposterioriTubeCheckResult | None = None
    try:
        entry_result = check_carried_planar_lc_entry(
            entry_transition,
            source_chart,
            source_tube,
            lc_chart,
            lc_tube,
            parent_source_clock_origin_interval,
        )
    except Exception:
        pass
    try:
        if raw_schemas:
            lc_result = check_planar_lc_aposteriori_tube(lc_tube, lc_chart)
    except Exception:
        pass
    try:
        if raw_schemas:
            target_result = check_ordinary_aposteriori_tube(
                target_tube,
                target_chart,
            )
    except Exception:
        pass

    entry_certified = _canonical_entry_result(
        entry_result,
        entry_transition,
        source_chart,
        source_tube,
        lc_chart,
        lc_tube,
    )
    lc_certified = _canonical_lc_tube_result(lc_result, lc_tube, lc_chart)
    target_certified = _canonical_ordinary_tube_result(
        target_result,
        target_tube,
        target_chart,
    )
    try:
        constrained_entry = bool(
            entry_certified
            and entry_result is not None
            and entry_result.constrained_newtonian_lift_certified
        )
    except Exception:
        constrained_entry = False
    constraint_kernel = bool(constrained_entry and lc_certified)
    third_body_separated = bool(
        lc_certified
        and lc_result is not None
        and type(lc_result.third_body_distance_floor) is float
        and math.isfinite(lc_result.third_body_distance_floor)
        and lc_result.third_body_distance_floor > 0
    )
    try:
        strict_clock_kernel = bool(
            entry_certified
            and entry_result is not None
            and entry_result.physical_time_strictly_monotone_certified
            and endpoint_handoff
            and lc_certified
            and third_body_separated
        )
    except Exception:
        strict_clock_kernel = False

    lifted: tuple[FractionInterval, ...] = ()
    rho: FractionInterval | tuple[()] = ()
    positions: FractionMatrix = ()
    velocities: FractionMatrix = ()
    maximum_gap: Fraction | None = None
    exit_time: FractionInterval | tuple[()] = ()
    target_clock: FractionInterval | tuple[()] = ()
    slice_reconstructed = False
    rho_positive = False
    projection_reconstructed = False
    deck_projection_kernel = False
    target_contains = False
    time_derived = False
    clock_derived = False

    if all(
        (
            raw_schemas,
            identifiers,
            parent_clock_valid,
            entry_certified,
            common_problem,
            canonical_pair,
            exact_ratio,
            endpoint_handoff,
            constrained_entry,
            constraint_kernel,
            lc_certified,
            third_body_separated,
            target_certified,
            strict_clock_kernel,
        )
    ):
        try:
            if lc_result is None:
                raise ValueError("fresh LC tube result missing")
            solution = _regularized_binary_solution_from_certificate(lc_chart)
            exit_parameter = exit_transition.source_parameter
            state = _planar_lc_state_intervals(
                solution,
                (exit_parameter, exit_parameter),
                inflate=lc_result.gronwall_error_bound,
            )
            lifted = tuple(_fraction_bounds(item) for item in state)
            slice_reconstructed = _fraction_box(lifted, 14)
            rho_interval = state[0] * state[0] + state[1] * state[1]
            rho = _fraction_bounds(rho_interval)
            rho_positive = bool(_fraction_interval(rho) and rho[0] > 0)
            exit_time = lifted[13]
            time_derived = _fraction_interval(exit_time)
            target_anchor = Fraction.from_float(exit_transition.target_parameter)
            target_clock = (
                exit_time[0] - target_anchor,
                exit_time[1] - target_anchor,
            )
            clock_derived = _fraction_interval(target_clock)

            if rho_positive:
                projected_q, projected_v, _, projected_rho = (
                    _project_interval_planar_lc_state(
                        state,
                        np.asarray(lc_chart.masses, dtype=float),
                        lc_chart.pair,
                    )
                )
                rho = _fraction_bounds(projected_rho)
                positions = _fraction_matrix_from_intervals(projected_q)
                velocities = _fraction_matrix_from_intervals(projected_v)
                projection_reconstructed = bool(
                    _fraction_matrix(positions)
                    and _fraction_matrix(velocities)
                )
                centers_q, centers_v = (
                    _exact_rational_chart_projected_state_at_parameter(
                        target_chart,
                        target_anchor,
                    )
                )
                target_radius = Fraction.from_float(
                    target_tube.initial_error_bound
                )
                gaps: list[Fraction] = []
                contains = True
                for box, centers in (
                    (positions, centers_q),
                    (velocities, centers_v),
                ):
                    for index in np.ndindex(np.asarray(centers).shape):
                        lower, upper = box[index[0]][index[1]]
                        center = centers[index]
                        gaps.extend((abs(lower - center), abs(upper - center)))
                        contains = bool(
                            contains
                            and center - target_radius <= lower
                            and upper <= center + target_radius
                        )
                maximum_gap = max(gaps, default=Fraction(0))
                target_contains = bool(
                    projection_reconstructed and contains
                )
                deck_projection_kernel = bool(
                    constraint_kernel
                    and rho_positive
                    and projection_reconstructed
                    and exact_ratio
                )
        except Exception:
            pass

    obligations = (
        _obligation(
            "carried_lc_exit_exact_raw_schemas",
            raw_schemas,
            "exact built-in raw source, LC, exit, and target schemas",
        ),
        _obligation(
            "carried_lc_exit_identifiers_match_and_are_unique",
            identifiers,
            "all raw references bind distinct source/LC/target components",
        ),
        _obligation(
            "carried_lc_exit_parent_source_invariant_is_explicit_condition",
            parent_clock_valid,
            (
                f"conditional kernel={_PARENT_INVARIANT_ID!r}; the parent must "
                "prove one carried source branch and supplies only its derived "
                f"clock enclosure B={parent_source_clock_origin_interval!s}"
            ),
        ),
        _obligation(
            "carried_lc_exit_entry_freshly_replayed_and_certified",
            entry_certified,
            _nested_detail(entry_result),
        ),
        _obligation(
            "carried_lc_exit_common_planar_mass_problem",
            common_problem,
            f"masses={lc_chart.masses!r}",
        ),
        _obligation(
            "carried_lc_exit_pair_is_canonical_ascending",
            canonical_pair,
            f"pair={lc_chart.pair!r}",
        ),
        _obligation(
            "carried_lc_exit_mass_ratio_arithmetic_exact",
            exact_ratio,
            "all point mass ratios used by LC projection equal exact rationals",
        ),
        _obligation(
            "carried_lc_exit_exact_right_to_left_endpoint_handoff",
            endpoint_handoff,
            "N-right -> LC-left < LC-right -> N-left, all by exact Fraction equality",
        ),
        _obligation(
            "carried_lc_exit_constrained_entry_branch_carried",
            constrained_entry,
            "fresh carried entry supplies one selected constrained lift",
        ),
        _obligation(
            "carried_lc_exit_constraint_invariance_kernel",
            constraint_kernel,
            "pinned C'=0 lemma carries that selected lift; the rectangle need not be constrained",
        ),
        _obligation(
            "carried_lc_exit_lc_tube_freshly_certified",
            lc_certified,
            _nested_detail(lc_result),
        ),
        _obligation(
            "carried_lc_exit_third_body_separated",
            third_body_separated,
            (
                "fresh LC tube third-body floor="
                f"{getattr(lc_result, 'third_body_distance_floor', None)!r}"
            ),
        ),
        _obligation(
            "carried_lc_exit_target_ordinary_tube_freshly_certified",
            target_certified,
            _nested_detail(target_result),
        ),
        _obligation(
            "carried_lc_exit_strict_physical_clock_kernel",
            strict_clock_kernel,
            "pinned t'=rho and nontrivial analytic-z lemma gives strict forward physical time",
        ),
        _obligation(
            "carried_lc_exit_complete_inflated_slice_reconstructed",
            slice_reconstructed,
            f"components={len(lifted)}",
        ),
        _obligation(
            "carried_lc_exit_complete_slice_rho_positive",
            rho_positive,
            f"rho={rho!s}",
        ),
        _obligation(
            "carried_lc_exit_complete_cartesian_projection_reconstructed",
            projection_reconstructed,
            "all 12 projected position/velocity intervals",
        ),
        _obligation(
            "carried_lc_exit_deck_equivariant_newton_projection_kernel",
            deck_projection_kernel,
            "punctured constrained LC projection is Newtonian and invariant under (z,w)->(-z,-w)",
        ),
        _obligation(
            "carried_lc_exit_target_initial_ball_contains_complete_projection",
            target_contains,
            (
                f"maximum_gap={maximum_gap!s}; "
                f"target_radius={target_tube.initial_error_bound!r}"
            ),
        ),
        _obligation(
            "carried_lc_exit_time_interval_derived_from_component_fourteen",
            time_derived,
            f"D_out={exit_time!s}",
        ),
        _obligation(
            "carried_lc_exit_target_clock_origin_exactly_derived",
            clock_derived,
            f"B_target=D_out-a={target_clock!s}",
        ),
    )
    transition_id = (
        exit_transition.transition_id
        if type(exit_transition.transition_id) is str
        else ""
    )
    return CarriedPlanarLCExitResult(
        transition_id=transition_id,
        checker_id=_CHECKER_ID,
        analytic_kernel_id=_ANALYTIC_KERNEL_ID,
        parent_source_invariant_id=_PARENT_INVARIANT_ID,
        raw_entry_transition=entry_transition,
        raw_source_chart=source_chart,
        raw_source_tube=source_tube,
        parent_source_clock_origin_interval=parent_source_clock_origin_interval,
        raw_lc_chart=lc_chart,
        raw_lc_tube=lc_tube,
        raw_exit_transition=exit_transition,
        raw_target_chart=target_chart,
        raw_target_tube=target_tube,
        entry_result=entry_result,
        lc_tube_result=lc_result,
        target_tube_result=target_result,
        obligations=obligations,
        lifted_exit_slice=lifted,
        exit_rho_interval=rho,
        projected_position_intervals=positions,
        projected_velocity_intervals=velocities,
        maximum_projected_anchor_gap=maximum_gap,
        exit_time_interval=exit_time,
        target_clock_origin_interval=target_clock,
    )


def _obligation(
    name: str,
    certified: bool,
    detail: str,
) -> CertificateCheckObligation:
    return CertificateCheckObligation(name, bool(certified), detail)


def _fraction_bounds(value: FloatInterval) -> FractionInterval:
    return Fraction.from_float(value.lower), Fraction.from_float(value.upper)


def _fraction_matrix_from_intervals(values: np.ndarray) -> FractionMatrix:
    array = np.asarray(values, dtype=object)
    if array.shape != (3, 2):
        raise ValueError("projected state must have shape (3, 2)")
    return tuple(
        tuple(_fraction_bounds(array[row, axis]) for axis in range(2))
        for row in range(3)
    )


def _fraction_interval(value: object, *, positive: bool = False) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(endpoint) is Fraction for endpoint in value)
        and value[0] <= value[1]
        and (not positive or value[0] > 0)
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


def _canonical_entry_result(
    value: object,
    raw_transition: CarriedPlanarLCEntryTransitionRecord,
    raw_source_chart: OrdinaryTaylorChartCertificate,
    raw_source_tube: OrdinaryAposterioriTubeCertificate,
    raw_lc_chart: PlanarLeviCivitaBinaryChartCertificate,
    raw_lc_tube: PlanarLCAposterioriTubeCertificate,
) -> bool:
    try:
        return bool(
            type(value) is CarriedPlanarLCEntryResult
            and type(value.checker_id) is str
            and value.checker_id == "carried_planar_lc_entry_checker_v1"
            and type(value.analytic_kernel_id) is str
            and value.analytic_kernel_id
            == "planar_lc_constrained_lift_deck_gauge_kernel_v1"
            and type(value.raw_transition)
            is CarriedPlanarLCEntryTransitionRecord
            and value.raw_transition == raw_transition
            and type(value.raw_source_chart) is OrdinaryTaylorChartCertificate
            and value.raw_source_chart == raw_source_chart
            and type(value.raw_source_tube)
            is OrdinaryAposterioriTubeCertificate
            and value.raw_source_tube == raw_source_tube
            and type(value.raw_lc_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and value.raw_lc_chart == raw_lc_chart
            and type(value.raw_lc_tube) is PlanarLCAposterioriTubeCertificate
            and value.raw_lc_tube == raw_lc_tube
            and value.certified
        )
    except Exception:
        return False


def _canonical_nested_obligations(value: object) -> bool:
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
        and bool(result.chart_id)
        and result.chart_id == raw_chart.chart_id == raw_tube.chart_id
        and type(result.checker_id) is str
        and result.checker_id
        == "independent_planar_lc_aposteriori_tube_checker_v1"
        and _canonical_nested_obligations(result.obligations)
        and all(
            type(value) is float and math.isfinite(value)
            for value in (
                result.defect_bound,
                result.lipschitz_bound,
                result.gronwall_error_bound,
                result.third_body_distance_floor,
                result.anchor_pair_energy_constraint_residual,
            )
        )
        and type(result.pair_energy_constraint_anchor_certified) is bool
        and type(result.anchor_is_polynomial_center) is bool
        and result.certified
    )


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
        and bool(result.chart_id)
        and result.chart_id == raw_chart.chart_id == raw_tube.chart_id
        and type(result.checker_id) is str
        and result.checker_id
        == "independent_ordinary_aposteriori_tube_checker_v1"
        and _canonical_nested_obligations(result.obligations)
        and all(
            type(value) is float and math.isfinite(value)
            for value in (
                result.defect_bound,
                result.lipschitz_bound,
                result.gronwall_error_bound,
                result.nominal_pair_distance_floor,
                result.tube_pair_distance_floor,
            )
        )
        and result.certified
    )


def _exact_obligation_manifest(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_OBLIGATION_NAMES)
        and all(
            type(item) is CertificateCheckObligation
            and type(item.obligation) is str
            and item.obligation == expected
            and type(item.certified) is bool
            and item.certified is True
            and type(item.detail) is str
            for item, expected in zip(value, _OBLIGATION_NAMES)
        )
    )


def _entry_schema(value: object) -> bool:
    return bool(
        type(value) is CarriedPlanarLCEntryTransitionRecord
        and all(
            type(item) is str and bool(item)
            for item in (
                value.transition_id,
                value.source_chart_id,
                value.target_chart_id,
                value.source,
            )
        )
        and all(
            _finite_float(item)
            for item in (
                value.source_right_parameter,
                value.target_left_parameter,
            )
        )
        and type(value.schema_version) is int
        and value.schema_version == 1
        and type(value.record_type) is str
        and value.record_type == "carried_planar_lc_entry_transition"
        and type(value.source) is str
        and value.source == "private_carried_planar_lc_entry_v1"
    )


def _exit_schema(value: object) -> bool:
    return bool(
        type(value) is PlanarLCToOrdinaryEnclosureTransitionCertificate
        and all(
            type(item) is str and bool(item)
            for item in (
                value.transition_id,
                value.source_chart_id,
                value.target_chart_id,
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


def _lc_chart_schema(value: object) -> bool:
    def vector_series(item: object) -> bool:
        return bool(
            type(item) is tuple
            and len(item) >= 2
            and all(
                type(coefficient) is tuple
                and len(coefficient) == 2
                and all(_finite_float(component) for component in coefficient)
                for coefficient in item
            )
        )

    try:
        series = (
            value.z_coefficients,
            value.z_velocity_coefficients,
            value.binary_center_coefficients,
            value.binary_center_velocity_coefficients,
            value.third_offset_coefficients,
            value.third_offset_velocity_coefficients,
        )
        order = len(value.z_coefficients)
        return bool(
            type(value) is PlanarLeviCivitaBinaryChartCertificate
            and all(
                type(item) is str and bool(item)
                for item in (
                    value.certificate_id,
                    value.chart_id,
                    value.chart_type,
                    value.source,
                )
            )
            and type(value.masses) is tuple
            and len(value.masses) == 3
            and all(_finite_float(item) and item > 0 for item in value.masses)
            and type(value.pair) is tuple
            and len(value.pair) == 2
            and all(type(index) is int for index in value.pair)
            and all(vector_series(item) for item in series)
            and all(len(item) == order for item in series)
            and type(value.pair_energy_coefficients) is tuple
            and len(value.pair_energy_coefficients) == order
            and all(_finite_float(item) for item in value.pair_energy_coefficients)
            and type(value.physical_time_coefficients) is tuple
            and len(value.physical_time_coefficients) == order
            and all(_finite_float(item) for item in value.physical_time_coefficients)
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
    except Exception:
        return False


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
    entry: CarriedPlanarLCEntryTransitionRecord,
    source_chart: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    lc_chart: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    try:
        identifiers = (
            source_chart.certificate_id,
            source_chart.chart_id,
            source_tube.tube_id,
            entry.transition_id,
            lc_chart.certificate_id,
            lc_chart.chart_id,
            lc_tube.tube_id,
            exit_transition.transition_id,
            target_chart.certificate_id,
            target_chart.chart_id,
            target_tube.tube_id,
        )
        return bool(
            all(type(item) is str and bool(item) for item in identifiers)
            and len(identifiers) == len(set(identifiers))
            and source_tube.chart_id == source_chart.chart_id
            and entry.source_tube_id == source_tube.tube_id
            and entry.source_chart_id == source_chart.chart_id
            and entry.target_chart_id == lc_chart.chart_id == lc_tube.chart_id
            and entry.target_tube_id == lc_tube.tube_id
            and exit_transition.source_chart_id == lc_chart.chart_id
            and exit_transition.target_chart_id
            == target_chart.chart_id
            == target_tube.chart_id
        )
    except Exception:
        return False


def _common_problem(
    source: OrdinaryTaylorChartCertificate,
    lc: PlanarLeviCivitaBinaryChartCertificate,
    target: OrdinaryTaylorChartCertificate,
) -> bool:
    try:
        return bool(
            source.masses == lc.masses == target.masses
            and source.dimension == target.dimension == 2
            and len(source.masses) == 3
            and all(
                type(mass) is float and math.isfinite(mass) and mass > 0
                for mass in source.masses
            )
        )
    except Exception:
        return False


def _exact_endpoint_handoff(
    entry: CarriedPlanarLCEntryTransitionRecord,
    source: OrdinaryTaylorChartCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    lc: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    try:
        source_domain = tuple(
            Fraction.from_float(item) for item in source.parameter_interval
        )
        lc_domain = tuple(
            Fraction.from_float(item) for item in lc.parameter_interval
        )
        target_domain = tuple(
            Fraction.from_float(item) for item in target.parameter_interval
        )
        return bool(
            source_domain[0] < source_domain[1]
            and lc_domain[0] < lc_domain[1]
            and target_domain[0] < target_domain[1]
            and Fraction.from_float(source_tube.anchor_parameter)
            == source_domain[0]
            and Fraction.from_float(entry.source_right_parameter)
            == source_domain[1]
            and Fraction.from_float(entry.target_left_parameter) == lc_domain[0]
            and Fraction.from_float(lc_tube.anchor_parameter) == lc_domain[0]
            and Fraction.from_float(exit_transition.source_parameter)
            == lc_domain[1]
            and Fraction.from_float(exit_transition.target_parameter)
            == target_domain[0]
            and Fraction.from_float(target_tube.anchor_parameter)
            == target_domain[0]
        )
    except Exception:
        return False


def _float_interval(value: object) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(_finite_float(item) for item in value)
        and value[0] < value[1]
    )


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _nested_detail(result: object | None) -> str:
    if result is None:
        return "fresh replay did not produce a result"
    try:
        return (
            f"checker={result.checker_id!r}; "
            f"missing={result.missing_obligations!r}"
        )
    except Exception as error:
        return f"malformed nested result:{type(error).__name__}"
