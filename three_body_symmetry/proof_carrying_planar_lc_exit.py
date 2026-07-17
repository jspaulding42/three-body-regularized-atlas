"""Raw-replaying gauge-aware planar LC-to-ordinary exit containment.

This private milestone checks one complete exit slice.  The rectangular
fourteen-dimensional slice encloses the one constrained solution carried by
the gauge-aware entry; it is *not* a claim that every point of the rectangle
satisfies the pair-energy constraint.  The target ordinary clock convention
is ``t = s + b``, with the derived interval ``B = D - a`` enclosing its clock
origin.  This module deliberately does not evaluate a final requested time,
claim a collision event, or claim a full mixed-chain ``CERTIFIED_TO_T``
theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from .certificate_checker import (
    CertificateCheckObligation,
    GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult,
    OrdinaryAposterioriTubeCheckResult,
    PlanarLCAposterioriTubeCheckResult,
    _exact_certified_obligation_manifest,
    _exact_rational_chart_projected_state_at_parameter,
    _gauge_aware_raw_primitive_schema_valid,
    _planar_lc_mass_ratio_arithmetic_exact,
    _planar_lc_state_intervals,
    _project_interval_planar_lc_state,
    _regularized_binary_solution_from_certificate,
    check_gauge_aware_ordinary_to_planar_lc_enclosure_transition,
    check_ordinary_aposteriori_tube,
    check_planar_lc_aposteriori_tube,
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
from .intervals import FloatInterval
from .planar_lc_mass_coefficients import PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID


_CHECKER_ID = "raw_gauge_aware_planar_lc_exit_containment_checker_v1"
_ANALYTIC_KERNEL_ID = "planar_lc_analytic_kernel_v1"
_OBLIGATION_NAMES = (
    "raw_lc_exit_exact_primitive_schemas",
    "raw_lc_exit_identifiers_match_and_are_unique",
    "raw_lc_exit_source_binding_is_exact_point_ivp",
    "raw_lc_exit_stage3_entry_freshly_certified",
    "raw_lc_exit_same_lc_chart_and_tube_bound",
    "raw_lc_exit_common_planar_mass_problem",
    "raw_lc_exit_pair_canonical_ascending",
    "raw_lc_exit_mass_ratio_arithmetic_exact",
    "raw_lc_exit_endpoint_order_and_target_anchor_exact",
    "raw_lc_exit_constrained_entry_branch_carried",
    "raw_lc_exit_trusted_constraint_invariance_kernel",
    "raw_lc_exit_lc_tube_freshly_certified",
    "raw_lc_exit_target_ordinary_tube_freshly_certified",
    "raw_lc_exit_trusted_strict_clock_increase_kernel",
    "raw_lc_exit_complete_inflated_slice_reconstructed",
    "raw_lc_exit_complete_slice_rho_positive",
    "raw_lc_exit_complete_cartesian_projection_reconstructed",
    "raw_lc_exit_trusted_deck_equivariant_newton_projection_kernel",
    "raw_lc_exit_target_initial_ball_contains_complete_projection",
    "raw_lc_exit_time_interval_derived_from_fourteenth_component",
    "raw_lc_exit_target_clock_origin_interval_derived",
)

FractionInterval = tuple[Fraction, Fraction]
FractionMatrix = tuple[tuple[FractionInterval, ...], ...]


@dataclass(frozen=True)
class RawGaugeAwarePlanarLCExitContainmentResult:
    """Immutable raw replay, derived boxes, and pinned analytic kernel."""

    transition_id: str
    checker_id: str
    analytic_kernel_id: str
    raw_entry_transition: OrdinaryToPlanarLCEnclosureTransitionCertificate
    raw_source_binding: InitialValueProblemBindingCertificate
    raw_source_tube: OrdinaryAposterioriTubeCertificate
    raw_source_chart: OrdinaryTaylorChartCertificate
    raw_lc_chart: PlanarLeviCivitaBinaryChartCertificate
    raw_lc_tube: PlanarLCAposterioriTubeCertificate
    raw_exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate
    raw_target_chart: OrdinaryTaylorChartCertificate
    raw_target_tube: OrdinaryAposterioriTubeCertificate
    entry_result: GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult | None
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
            type(self) is RawGaugeAwarePlanarLCExitContainmentResult
            and type(self.checker_id) is str
            and self.checker_id == _CHECKER_ID
            and type(self.analytic_kernel_id) is str
            and self.analytic_kernel_id == _ANALYTIC_KERNEL_ID
            and type(self.transition_id) is str
            and bool(self.transition_id)
            and type(self.raw_entry_transition)
            is OrdinaryToPlanarLCEnclosureTransitionCertificate
            and type(self.raw_source_binding)
            is InitialValueProblemBindingCertificate
            and type(self.raw_source_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.raw_source_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_lc_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.raw_lc_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_exit_transition)
            is PlanarLCToOrdinaryEnclosureTransitionCertificate
            and type(self.raw_target_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_target_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.entry_result)
            is GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult
            and self.entry_result.certified
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
            and _exact_certified_obligation_manifest(
                self.obligations, _OBLIGATION_NAMES
            )
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
        """Freshly replay all retained raw evidence and exact-compare results."""

        try:
            if (
                type(self) is not RawGaugeAwarePlanarLCExitContainmentResult
                or not self._snapshot_certified()
            ):
                return False
            fresh = check_raw_gauge_aware_planar_lc_exit_containment(
                self.raw_entry_transition,
                self.raw_source_binding,
                self.raw_source_tube,
                self.raw_source_chart,
                self.raw_lc_chart,
                self.raw_lc_tube,
                self.raw_exit_transition,
                self.raw_target_chart,
                self.raw_target_tube,
            )
            return bool(
                type(fresh) is RawGaugeAwarePlanarLCExitContainmentResult
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
                missing.append(f"raw_lc_exit_malformed_obligation:{index}")
            elif obligation.certified is not True:
                missing.append(obligation.obligation)
        return tuple(missing)


def check_raw_gauge_aware_planar_lc_exit_containment(
    entry_transition: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    source_binding: InitialValueProblemBindingCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    lc_chart: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_transition: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> RawGaugeAwarePlanarLCExitContainmentResult:
    """Recompute a complete rho-positive LC exit containment from raw inputs."""

    expected_types = (
        (entry_transition, OrdinaryToPlanarLCEnclosureTransitionCertificate),
        (source_binding, InitialValueProblemBindingCertificate),
        (source_tube, OrdinaryAposterioriTubeCertificate),
        (source_chart, OrdinaryTaylorChartCertificate),
        (lc_chart, PlanarLeviCivitaBinaryChartCertificate),
        (lc_tube, PlanarLCAposterioriTubeCertificate),
        (exit_transition, PlanarLCToOrdinaryEnclosureTransitionCertificate),
        (target_chart, OrdinaryTaylorChartCertificate),
        (target_tube, OrdinaryAposterioriTubeCertificate),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("all raw LC exit inputs must have their exact certificate class")

    try:
        raw_schema = bool(
            _gauge_aware_raw_primitive_schema_valid(
                entry_transition,
                source_binding,
                source_tube,
                source_chart,
                lc_chart,
                lc_tube,
            )
            and _exit_schema(exit_transition)
            and _ordinary_chart_schema(target_chart)
            and _ordinary_tube_schema(target_tube)
        )
    except Exception:
        raw_schema = False

    entry_result = None
    lc_result = None
    target_result = None
    try:
        entry_result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
            entry_transition,
            source_binding,
            source_tube,
            source_chart,
            lc_chart,
            lc_tube,
        )
    except Exception:
        pass
    try:
        lc_result = check_planar_lc_aposteriori_tube(lc_tube, lc_chart)
    except Exception:
        pass
    try:
        target_result = check_ordinary_aposteriori_tube(target_tube, target_chart)
    except Exception:
        pass

    try:
        identifiers = _identifiers_match_and_unique(
            entry_transition,
            source_binding,
            source_tube,
            source_chart,
            lc_chart,
            lc_tube,
            exit_transition,
            target_chart,
            target_tube,
        )
    except Exception:
        identifiers = False
    exact_point_ivp = _exact_point_ivp(source_binding)
    try:
        entry_certified = bool(
            type(entry_result)
            is GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult
            and entry_result.certified
        )
    except Exception:
        entry_certified = False
    try:
        same_lc = bool(
            entry_certified
            and entry_result is not None
            and entry_result.raw_target_chart == lc_chart
            and entry_result.raw_target_tube == lc_tube
            and entry_result.target_tube_result == lc_result
        )
    except Exception:
        same_lc = False
    common_problem = _common_problem(source_binding, source_chart, lc_chart, target_chart)
    try:
        canonical_pair = bool(
            type(lc_chart.pair) is tuple
            and lc_chart.pair in ((0, 1), (0, 2), (1, 2))
        )
    except Exception:
        canonical_pair = False
    try:
        exact_ratio = bool(
            canonical_pair
            and _planar_lc_mass_ratio_arithmetic_exact(
                lc_chart.masses, lc_chart.pair
            )
        )
    except Exception:
        exact_ratio = False
    endpoint_order = _endpoint_order(
        entry_transition,
        source_chart,
        lc_chart,
        lc_tube,
        exit_transition,
        target_chart,
        target_tube,
    )
    try:
        constrained_entry = bool(
            entry_certified
            and entry_result is not None
            and entry_result.constrained_newtonian_lift_certified
        )
    except Exception:
        constrained_entry = False
    try:
        lc_certified = bool(
            type(lc_result) is PlanarLCAposterioriTubeCheckResult
            and lc_result.certified
        )
    except Exception:
        lc_certified = False
    try:
        target_certified = bool(
            type(target_result) is OrdinaryAposterioriTubeCheckResult
            and target_result.certified
        )
    except Exception:
        target_certified = False
    constraint_invariance_kernel = bool(constrained_entry and lc_certified)
    try:
        strict_clock_kernel = bool(
            entry_certified
            and entry_result is not None
            and entry_result.physical_time_strictly_monotone_certified
            and endpoint_order
            and lc_certified
            and lc_result is not None
            and lc_result.third_body_distance_floor > 0.0
        )
    except Exception:
        strict_clock_kernel = False

    lifted: tuple[FractionInterval, ...] = ()
    rho: FractionInterval | tuple[()] = ()
    positions: FractionMatrix = ()
    velocities: FractionMatrix = ()
    max_gap: Fraction | None = None
    exit_time: FractionInterval | tuple[()] = ()
    clock_origin: FractionInterval | tuple[()] = ()
    slice_reconstructed = False
    rho_positive = False
    projection_reconstructed = False
    target_contains = False
    time_derived = False
    clock_derived = False
    deck_projection_kernel = False
    if all(
        (
            raw_schema,
            identifiers,
            exact_point_ivp,
            entry_certified,
            same_lc,
            common_problem,
            canonical_pair,
            exact_ratio,
            endpoint_order,
            constrained_entry,
            constraint_invariance_kernel,
            lc_certified,
            target_certified,
            strict_clock_kernel,
        )
    ):
        try:
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
            rho_positive = rho[0] > 0
            exit_time = lifted[13]
            time_derived = _fraction_interval(exit_time)
            anchor_q = Fraction.from_float(exit_transition.target_parameter)
            clock_origin = (
                exit_time[0] - anchor_q,
                exit_time[1] - anchor_q,
            )
            clock_derived = _fraction_interval(clock_origin)
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
                    _fraction_matrix(positions) and _fraction_matrix(velocities)
                )
                centers_q, centers_v = (
                    _exact_rational_chart_projected_state_at_parameter(
                        target_chart,
                        Fraction.from_float(exit_transition.target_parameter),
                    )
                )
                radius_q = Fraction.from_float(target_tube.initial_error_bound)
                gaps = []
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
                            and center - radius_q <= lower
                            and upper <= center + radius_q
                        )
                max_gap = max(gaps, default=Fraction(0))
                target_contains = bool(projection_reconstructed and contains)
                deck_projection_kernel = bool(
                    constraint_invariance_kernel
                    and rho_positive
                    and projection_reconstructed
                    and exact_ratio
                )
        except Exception:
            pass

    obligations = (
        _obligation("raw_lc_exit_exact_primitive_schemas", raw_schema, "exact built-in raw schemas"),
        _obligation("raw_lc_exit_identifiers_match_and_are_unique", identifiers, "all raw references and primary ids"),
        _obligation("raw_lc_exit_source_binding_is_exact_point_ivp", exact_point_ivp, "all three serialized binding tolerances are exactly zero"),
        _obligation("raw_lc_exit_stage3_entry_freshly_certified", entry_certified, _nested_detail(entry_result)),
        _obligation("raw_lc_exit_same_lc_chart_and_tube_bound", same_lc, "Stage-3 target is this exact LC chart/tube"),
        _obligation("raw_lc_exit_common_planar_mass_problem", common_problem, f"masses={lc_chart.masses!r}"),
        _obligation("raw_lc_exit_pair_canonical_ascending", canonical_pair, f"pair={lc_chart.pair!r}"),
        _obligation("raw_lc_exit_mass_ratio_arithmetic_exact", exact_ratio, "all LC projection ratios equal exact serialized rationals"),
        _obligation("raw_lc_exit_endpoint_order_and_target_anchor_exact", endpoint_order, "N-right -> LC-left < LC-right -> N-left"),
        _obligation("raw_lc_exit_constrained_entry_branch_carried", constrained_entry, "Stage-3 existential selected lift is constrained"),
        _obligation("raw_lc_exit_trusted_constraint_invariance_kernel", constraint_invariance_kernel, "pinned analytic lemma C'=0 carries the one selected constrained solution; the rectangular tube is not wholly constrained"),
        _obligation("raw_lc_exit_lc_tube_freshly_certified", lc_certified, _nested_detail(lc_result)),
        _obligation("raw_lc_exit_target_ordinary_tube_freshly_certified", target_certified, _nested_detail(target_result)),
        _obligation("raw_lc_exit_trusted_strict_clock_increase_kernel", strict_clock_kernel, "pinned analytic lemma t'=rho and entry rho>0 make analytic z nontrivial, hence t strictly increases on the forward LC interval"),
        _obligation("raw_lc_exit_complete_inflated_slice_reconstructed", slice_reconstructed, f"components={len(lifted)}"),
        _obligation("raw_lc_exit_complete_slice_rho_positive", rho_positive, f"rho={rho!s}"),
        _obligation("raw_lc_exit_complete_cartesian_projection_reconstructed", projection_reconstructed, "all 12 position/velocity intervals"),
        _obligation("raw_lc_exit_trusted_deck_equivariant_newton_projection_kernel", deck_projection_kernel, "pinned analytic lemma identifies the punctured constrained LC projection with Newton and makes it invariant under z,w -> -z,-w"),
        _obligation("raw_lc_exit_target_initial_ball_contains_complete_projection", target_contains, f"max_gap={max_gap!s}; radius={target_tube.initial_error_bound!r}"),
        _obligation("raw_lc_exit_time_interval_derived_from_fourteenth_component", time_derived, f"D={exit_time!s}"),
        _obligation("raw_lc_exit_target_clock_origin_interval_derived", clock_derived, f"target clock t=s+b; B=D-a={clock_origin!s}"),
    )
    transition_id = exit_transition.transition_id if type(exit_transition.transition_id) is str else ""
    return RawGaugeAwarePlanarLCExitContainmentResult(
        transition_id=transition_id,
        checker_id=_CHECKER_ID,
        analytic_kernel_id=_ANALYTIC_KERNEL_ID,
        raw_entry_transition=entry_transition,
        raw_source_binding=source_binding,
        raw_source_tube=source_tube,
        raw_source_chart=source_chart,
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
        maximum_projected_anchor_gap=max_gap,
        exit_time_interval=exit_time,
        target_clock_origin_interval=clock_origin,
    )


def _obligation(name: str, certified: bool, detail: str) -> CertificateCheckObligation:
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
        == "independent_planar_lc_aposteriori_tube_checker_v2"
        and type(result.mass_arithmetic_kernel_id) is str
        and result.mass_arithmetic_kernel_id
        == PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
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


def _exit_schema(value: PlanarLCToOrdinaryEnclosureTransitionCertificate) -> bool:
    return bool(
        all(
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


def _ordinary_chart_schema(value: OrdinaryTaylorChartCertificate) -> bool:
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
        all(
            type(item) is str and bool(item)
            for item in (value.certificate_id, value.chart_id, value.chart_type, value.source)
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
        and not isinstance(value.sample_count, bool)
        and value.sample_count >= 1
    )


def _ordinary_tube_schema(value: OrdinaryAposterioriTubeCertificate) -> bool:
    return bool(
        all(type(item) is str and bool(item) for item in (value.tube_id, value.chart_id, value.source))
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
    entry: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    binding: InitialValueProblemBindingCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    lc_chart: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_record: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    identifiers = (
        binding.binding_id,
        source_chart.certificate_id,
        source_chart.chart_id,
        source_tube.tube_id,
        entry.transition_id,
        lc_chart.certificate_id,
        lc_chart.chart_id,
        lc_tube.tube_id,
        exit_record.transition_id,
        target_chart.certificate_id,
        target_chart.chart_id,
        target_tube.tube_id,
    )
    return bool(
        all(type(item) is str and bool(item) for item in identifiers)
        and len(identifiers) == len(set(identifiers))
        and binding.chart_id == source_chart.chart_id == source_tube.chart_id
        and entry.source_chart_id == source_chart.chart_id
        and entry.target_chart_id == lc_chart.chart_id == lc_tube.chart_id
        and exit_record.source_chart_id == lc_chart.chart_id
        and exit_record.target_chart_id == target_chart.chart_id == target_tube.chart_id
    )


def _common_problem(
    binding: InitialValueProblemBindingCertificate,
    source: OrdinaryTaylorChartCertificate,
    lc: PlanarLeviCivitaBinaryChartCertificate,
    target: OrdinaryTaylorChartCertificate,
) -> bool:
    try:
        return bool(
            binding.masses == source.masses == lc.masses == target.masses
            and source.dimension == target.dimension == 2
            and len(binding.masses) == 3
            and all(type(mass) is float and math.isfinite(mass) and mass > 0 for mass in binding.masses)
        )
    except Exception:
        return False


def _exact_point_ivp(binding: InitialValueProblemBindingCertificate) -> bool:
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


def _endpoint_order(
    entry: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    source: OrdinaryTaylorChartCertificate,
    lc: PlanarLeviCivitaBinaryChartCertificate,
    lc_tube: PlanarLCAposterioriTubeCertificate,
    exit_record: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    target: OrdinaryTaylorChartCertificate,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> bool:
    try:
        source_domain = tuple(Fraction.from_float(item) for item in source.parameter_interval)
        lc_domain = tuple(Fraction.from_float(item) for item in lc.parameter_interval)
        target_domain = tuple(Fraction.from_float(item) for item in target.parameter_interval)
        return bool(
            source_domain[0] < source_domain[1]
            and lc_domain[0] < lc_domain[1]
            and target_domain[0] < target_domain[1]
            and Fraction.from_float(entry.source_parameter) == source_domain[1]
            and Fraction.from_float(entry.target_parameter) == lc_domain[0]
            and Fraction.from_float(lc_tube.anchor_parameter) == lc_domain[0]
            and Fraction.from_float(exit_record.source_parameter) == lc_domain[1]
            and Fraction.from_float(exit_record.target_parameter) == target_domain[0]
            and Fraction.from_float(target_tube.anchor_parameter) == target_domain[0]
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
        return f"checker={result.checker_id!r}; missing={result.missing_obligations!r}"
    except Exception as error:
        return f"malformed nested result:{type(error).__name__}"
