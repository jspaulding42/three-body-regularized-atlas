from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
from math import comb
import math

import numpy as np
import pytest

from three_body_symmetry.binary_chart import (
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryToPlanarLCEnclosureTransitionCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
)
from three_body_symmetry.proof_carrying_mixed_continuation import (
    RawMixedPlanarContinuationCertificate,
    RawMixedPlanarContinuationReplayResult,
    canonical_mixed_evidence_json,
    check_raw_mixed_planar_continuation,
    raw_mixed_continuation_evidence_sha256,
)
from three_body_symmetry.proof_carrying_planar_lc_exit import (
    check_raw_gauge_aware_planar_lc_exit_containment,
)
from three_body_symmetry.series import construct_taylor_solution


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


class _RawMixedResultEqualitySpoof(RawMixedPlanarContinuationReplayResult):
    def __eq__(self, other: object) -> bool:
        return True


def _translate_ordinary_coefficients(
    coefficients: tuple[tuple[tuple[float, ...], ...], ...],
    anchor: float,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Translate ``p(s)`` to the equal polynomial ``p(s-anchor)``."""

    source = np.asarray(coefficients, dtype=float)
    translated = np.zeros_like(source)
    for degree in range(source.shape[0]):
        for original_degree in range(degree, source.shape[0]):
            translated[degree] += (
                source[original_degree]
                * comb(original_degree, degree)
                * (-anchor) ** (original_degree - degree)
            )
    return tuple(
        tuple(tuple(float(value) for value in row) for row in coefficient)
        for coefficient in translated
    )


@lru_cache(maxsize=3)
def _certificate(
    pair: tuple[int, int] = (0, 1),
) -> RawMixedPlanarContinuationCertificate:
    small_step = 2.0**-20
    target_anchor = 2.0**-10
    target_width = 2.0**-6
    masses = np.asarray((1.0, 1.0, 1.0))
    positions = np.asarray(((0.0, 0.0), (1.0, 0.0), (3.0, 1.0)))
    velocities = np.zeros((3, 2))

    source_solution = construct_taylor_solution(
        positions, velocities, masses, order=10
    )
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        source_solution,
        certificate_id="raw-mixed-source-certificate",
        chart_id="raw-mixed-source-chart",
        parameter_interval=(0.0, small_step),
        physical_time_interval=(0.0, small_step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    binding = InitialValueProblemBindingCertificate(
        binding_id="raw-mixed-binding",
        chart_id=source_chart.chart_id,
        masses=source_chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=source_chart.position_coefficients[0],
        velocities=source_chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="raw-mixed-source-tube",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-6,
        tube_radius=1.0e-4,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )

    lifted_initial = planar_to_regularized_binary_collision_chart(
        positions, velocities, masses, pair=pair
    )
    lc_solution = construct_regularized_binary_taylor_solution(
        lifted_initial, order=10
    )
    lc_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        lc_solution,
        certificate_id="raw-mixed-lc-certificate",
        chart_id="raw-mixed-lc-chart",
        parameter_interval=(0.0, small_step),
        coefficient_tolerance=1.0e-8,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-10,
        sample_count=5,
        projection_rho_lower_bound=1.0e-8,
        physical_time_shift=small_step,
    )
    lc_tube = PlanarLCAposterioriTubeCertificate(
        tube_id="raw-mixed-lc-tube",
        chart_id=lc_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=1.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    entry = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        transition_id="raw-mixed-entry",
        source_chart_id=source_chart.chart_id,
        target_chart_id=lc_chart.chart_id,
        source_parameter=small_step,
        target_parameter=0.0,
        handoff_time=small_step,
        max_time_gap=0.0,
    )

    exit_positions, exit_velocities = regularized_binary_collision_chart_to_planar(
        lc_solution.state_at(small_step)
    )
    target_solution = construct_taylor_solution(
        exit_positions, exit_velocities, masses, order=10
    )
    unshifted_target = ordinary_taylor_chart_certificate_from_solution(
        target_solution,
        certificate_id="raw-mixed-target-certificate",
        chart_id="raw-mixed-target-chart",
        parameter_interval=(0.0, target_width),
        physical_time_interval=(0.0, target_width),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    target_chart = replace(
        unshifted_target,
        position_coefficients=_translate_ordinary_coefficients(
            unshifted_target.position_coefficients,
            target_anchor,
        ),
        velocity_coefficients=_translate_ordinary_coefficients(
            unshifted_target.velocity_coefficients,
            target_anchor,
        ),
        parameter_interval=(target_anchor, target_anchor + target_width),
        # Deliberately unrelated metadata: fixed-time replay must use D-a.
        physical_time_interval=(100.0, 100.0 + target_width),
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="raw-mixed-target-tube",
        chart_id=target_chart.chart_id,
        anchor_parameter=target_anchor,
        initial_error_bound=2.0e-2,
        tube_radius=5.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    exit_transition = PlanarLCToOrdinaryEnclosureTransitionCertificate(
        transition_id="raw-mixed-exit",
        source_chart_id=lc_chart.chart_id,
        target_chart_id=target_chart.chart_id,
        source_parameter=small_step,
        target_parameter=target_anchor,
    )
    exit_result = check_raw_gauge_aware_planar_lc_exit_containment(
        entry,
        binding,
        source_tube,
        source_chart,
        lc_chart,
        lc_tube,
        exit_transition,
        target_chart,
        target_tube,
    )
    assert exit_result.certified
    target_time = float(exit_result.exit_time_interval[1]) + 2.0**-8
    return RawMixedPlanarContinuationCertificate(
        certificate_id="raw-mixed-certificate",
        entry_transition=entry,
        source_binding=binding,
        source_tube=source_tube,
        source_chart=source_chart,
        lc_chart=lc_chart,
        lc_tube=lc_tube,
        exit_transition=exit_transition,
        target_chart=target_chart,
        target_tube=target_tube,
        requested_target_time=target_time,
        requested_maximum_component_width=0.2,
    )


@lru_cache(maxsize=1)
def _positive_result():
    return check_raw_mixed_planar_continuation(_certificate())


def _fraction_upper_float(value: Fraction) -> float:
    candidate = float(value)
    if Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def test_nonzero_anchor_fixed_time_certificate_replays_exact_clock_and_all_j():
    certificate = _certificate()
    result = _positive_result()

    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.replay_consistent
    assert result.first_failed_obligation is None
    assert result.nested_exit_result is not None
    assert result.nested_exit_result.certified
    assert result.nested_missing_obligations == ()
    assert result.nested_exit_checker_id == (
        "raw_gauge_aware_planar_lc_exit_containment_checker_v1"
    )
    assert result.analytic_kernel_id == "planar_newton_lc_mixed_analytic_kernel_v1"
    assert result.autonomous_clock_kernel_id == (
        "autonomous_target_clock_cocycle_kernel_v1"
    )
    assert result.fixed_time_kernel_id == (
        "exact_rational_horner_fixed_time_kernel_v1"
    )

    anchor = Fraction.from_float(certificate.exit_transition.target_parameter)
    target = Fraction.from_float(certificate.requested_target_time)
    exit_time = result.exit_time_interval
    assert result.target_clock_origin_interval == (
        exit_time[0] - anchor,
        exit_time[1] - anchor,
    )
    assert result.target_parameter_preimage_interval == (
        anchor + target - exit_time[1],
        anchor + target - exit_time[0],
    )
    assert result.final_enclosure is not None
    assert result.final_enclosure.target_parameter_preimage_interval == (
        result.target_parameter_preimage_interval
    )
    assert result.final_enclosure.physical_time_interval == (target, target)
    assert result.maximum_final_component_width is not None
    assert result.maximum_final_component_width <= Fraction.from_float(
        certificate.requested_maximum_component_width
    )
    assert result.retained_safe_regions == ()


@pytest.mark.parametrize("pair", ((0, 1), (0, 2), (1, 2)))
def test_one_passage_checker_supports_each_canonical_binary_pair(pair):
    certificate = _certificate(pair)
    result = check_raw_mixed_planar_continuation(certificate)

    assert certificate.lc_chart.pair == pair
    assert result.status == CERTIFIED_TO_T
    assert result.certified


def test_canonical_wire_round_trip_digest_and_strict_fields():
    certificate = _certificate()
    wire = certificate.to_dict()
    rebuilt = RawMixedPlanarContinuationCertificate.from_dict(wire)

    assert rebuilt == certificate
    assert canonical_mixed_evidence_json(rebuilt) == canonical_mixed_evidence_json(
        certificate
    )
    digest = raw_mixed_continuation_evidence_sha256(certificate)
    assert raw_mixed_continuation_evidence_sha256(rebuilt) == digest
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    assert raw_mixed_continuation_evidence_sha256(
        replace(certificate, requested_maximum_component_width=0.25)
    ) != digest

    with pytest.raises(ValueError, match="unknown"):
        RawMixedPlanarContinuationCertificate.from_dict(
            {**wire, "supplied_clock_origin": [0.0, 0.0]}
        )
    missing = dict(wire)
    del missing["target_tube"]
    with pytest.raises(ValueError, match="missing"):
        RawMixedPlanarContinuationCertificate.from_dict(missing)
    numeric_text = dict(wire)
    numeric_text["requested_target_time"] = str(
        certificate.requested_target_time
    )
    with pytest.raises(ValueError, match="noncanonical"):
        RawMixedPlanarContinuationCertificate.from_dict(numeric_text)


def test_exact_exit_time_and_width_boundaries_are_inclusive_then_fail_below():
    positive = _positive_result()
    certificate = _certificate()
    assert positive.maximum_final_component_width is not None

    width_ceiling = _fraction_upper_float(positive.maximum_final_component_width)
    at_width = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            requested_maximum_component_width=width_ceiling,
        )
    )
    assert at_width.status == CERTIFIED_TO_T
    below_width = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            requested_maximum_component_width=math.nextafter(
                width_ceiling, -math.inf
            ),
        )
    )
    assert below_width.status == UNRESOLVED
    assert below_width.first_failed_obligation == (
        "raw_mixed_final_component_width_within_requested_bound"
    )
    assert below_width.replay_consistent
    assert len(below_width.retained_safe_regions) == 1
    assert below_width.retained_safe_regions[0].region_type == (
        "certified_ordinary_fixed_time_enclosure"
    )
    assert len(below_width.retained_safe_regions[0].component_intervals) == 12
    assert below_width.final_enclosure is not None
    assert below_width.covered_physical_time_interval[1] == Fraction.from_float(
        certificate.requested_target_time
    )

    exit_upper = float(positive.exit_time_interval[1])
    at_exit_upper = check_raw_mixed_planar_continuation(
        replace(certificate, requested_target_time=exit_upper)
    )
    assert at_exit_upper.status == CERTIFIED_TO_T
    before_exit = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            requested_target_time=math.nextafter(exit_upper, -math.inf),
        )
    )
    assert before_exit.status == UNRESOLVED
    assert before_exit.first_failed_obligation == (
        "raw_mixed_target_not_before_complete_exit_time_interval"
    )

    anchor = Fraction.from_float(certificate.exit_transition.target_parameter)
    target_right = Fraction.from_float(
        certificate.target_chart.parameter_interval[1]
    )
    right_boundary_target = (
        target_right - anchor + positive.exit_time_interval[0]
    )
    right_boundary_float = float(right_boundary_target)
    assert Fraction.from_float(right_boundary_float) == right_boundary_target
    at_right_boundary = check_raw_mixed_planar_continuation(
        replace(certificate, requested_target_time=right_boundary_float)
    )
    assert at_right_boundary.status == CERTIFIED_TO_T
    assert at_right_boundary.target_parameter_preimage_interval[1] == target_right
    beyond_right_boundary = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            requested_target_time=math.nextafter(
                right_boundary_float, math.inf
            ),
        )
    )
    assert beyond_right_boundary.status == UNRESOLVED
    assert beyond_right_boundary.first_failed_obligation == (
        "raw_mixed_fixed_time_preimage_inside_forward_target_slab"
    )


def test_each_of_nine_raw_inputs_is_replayed_and_mutation_fails_closed():
    certificate = _certificate()
    mutations = (
        (replace(
            certificate,
            entry_transition=replace(
                certificate.entry_transition,
                target_chart_id="wrong-lc-chart",
            ),
        ), "certified_ordinary_entry_slice"),
        (replace(
            certificate,
            source_binding=replace(
                certificate.source_binding,
                time_tolerance=1.0e-6,
            ),
        ), None),
        (replace(
            certificate,
            source_tube=replace(
                certificate.source_tube,
                chart_id="wrong-source-chart",
            ),
        ), None),
        (replace(
            certificate,
            source_chart=replace(
                certificate.source_chart,
                chart_id="wrong-source-chart",
            ),
        ), None),
        (replace(
            certificate,
            lc_chart=replace(certificate.lc_chart, pair=(1, 0)),
        ), "certified_ordinary_entry_slice"),
        (replace(
            certificate,
            lc_tube=replace(certificate.lc_tube, chart_id="wrong-lc-chart"),
        ), "certified_ordinary_entry_slice"),
        (replace(
            certificate,
            exit_transition=replace(
                certificate.exit_transition,
                target_chart_id="wrong-target-chart",
            ),
        ), "certified_lifted_lc_exit_slice"),
        (replace(
            certificate,
            target_chart=replace(
                certificate.target_chart,
                chart_id="wrong-target-chart",
            ),
        ), "certified_lifted_lc_exit_slice"),
        (replace(
            certificate,
            target_tube=replace(
                certificate.target_tube,
                chart_id="wrong-target-chart",
            ),
        ), "certified_lifted_lc_exit_slice"),
    )
    for mutation, retained_type in mutations:
        result = check_raw_mixed_planar_continuation(mutation)
        assert result.status == UNRESOLVED
        assert not result.certified
        assert result.replay_consistent
        assert result.first_failed_obligation.startswith("nested_exit:")
        if retained_type is not None:
            assert len(result.retained_safe_regions) == 1
            assert result.retained_safe_regions[0].region_type == retained_type
        else:
            assert result.retained_safe_regions == ()


def test_target_domain_requests_and_malformed_outer_fields_fail_closed():
    certificate = _certificate()
    outside = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            requested_target_time=(
                certificate.requested_target_time
                + 2.0 * (
                    certificate.target_chart.parameter_interval[1]
                    - certificate.target_chart.parameter_interval[0]
                )
            ),
        )
    )
    assert outside.status == UNRESOLVED
    assert outside.first_failed_obligation == (
        "raw_mixed_fixed_time_preimage_inside_forward_target_slab"
    )
    assert len(outside.retained_safe_regions) == 1
    assert outside.retained_safe_regions[0].region_type == (
        "certified_ordinary_target_right_frontier"
    )
    exact_entry_time = (
        Fraction.from_float(certificate.source_binding.initial_time)
        + Fraction.from_float(certificate.source_chart.parameter_interval[1])
        - Fraction.from_float(certificate.source_binding.chart_parameter)
    )
    assert outside.covered_physical_time_interval[1] >= exact_entry_time

    malformed_lc = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            lc_chart=replace(certificate.lc_chart, z_coefficients=()),
        )
    )
    assert malformed_lc.status == UNRESOLVED
    assert malformed_lc.replay_consistent
    assert malformed_lc.retained_safe_regions[0].region_type == (
        "certified_ordinary_entry_slice"
    )

    malformed_target = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            target_chart=replace(
                certificate.target_chart,
                position_coefficients=(),
            ),
        )
    )
    assert malformed_target.status == UNRESOLVED
    assert malformed_target.replay_consistent
    assert malformed_target.retained_safe_regions[0].region_type == (
        "certified_lifted_lc_exit_slice"
    )
    assert malformed_target.covered_physical_time_interval[1] >= exact_entry_time

    narrow_target_anchor = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            target_tube=replace(
                certificate.target_tube,
                initial_error_bound=1.0e-6,
            ),
        )
    )
    assert narrow_target_anchor.status == UNRESOLVED
    assert narrow_target_anchor.first_failed_obligation.startswith("nested_exit:")
    assert narrow_target_anchor.nested_exit_result is None
    assert len(narrow_target_anchor.retained_safe_regions) == 1
    assert narrow_target_anchor.retained_safe_regions[0].region_type == (
        "certified_lifted_lc_exit_slice"
    )

    for bad_target in (math.nan, math.inf, -math.inf):
        result = check_raw_mixed_planar_continuation(
            replace(certificate, requested_target_time=bad_target)
        )
        assert result.status == UNRESOLVED
        assert result.first_failed_obligation == "raw_mixed_requested_target_finite"
        assert result.requested_target_time is None
    for bad_width in (-1.0, math.nan, math.inf):
        result = check_raw_mixed_planar_continuation(
            replace(certificate, requested_maximum_component_width=bad_width)
        )
        assert result.status == UNRESOLVED
        assert result.first_failed_obligation == (
            "raw_mixed_requested_component_width_admissible"
        )
        assert result.requested_maximum_component_width is None

    hostile = check_raw_mixed_planar_continuation(
        replace(certificate, requested_target_time=_AlwaysEqual())
    )
    assert hostile.status == UNRESOLVED
    assert not hostile.certified
    assert hostile.first_failed_obligation == (
        "raw_mixed_canonical_evidence_serializable"
    )

    duplicate_outer_id = check_raw_mixed_planar_continuation(
        replace(
            certificate,
            certificate_id=certificate.source_chart.chart_id,
        )
    )
    assert duplicate_outer_id.status == UNRESOLVED
    assert duplicate_outer_id.first_failed_obligation == (
        "raw_mixed_global_identifiers_unique_and_manifest_exact"
    )


def test_copied_ledgers_kernel_ids_and_derived_snapshots_cannot_spoof_replay():
    result = _positive_result()
    assert result.certified

    assert not replace(result, checker_id=_AlwaysEqual()).certified
    assert not replace(result, analytic_kernel_id=_AlwaysEqual()).certified
    assert not replace(result, autonomous_clock_kernel_id="unreviewed").certified
    assert not replace(result, fixed_time_kernel_id=_AlwaysEqual()).certified
    assert not replace(result, evidence_sha256=_AlwaysEqual()).certified
    for field in (
        "covered_physical_time_interval",
        "exit_time_interval",
        "target_clock_origin_interval",
        "target_parameter_preimage_interval",
    ):
        assert not replace(result, **{field: _AlwaysEqual()}).certified

    forged_obligation = replace(result.obligations[0], detail=_AlwaysEqual())
    assert not replace(
        result,
        obligations=(forged_obligation,) + result.obligations[1:],
    ).certified
    assert result.final_enclosure is not None
    forged_enclosure = replace(
        result.final_enclosure,
        target_parameter_preimage_interval=_AlwaysEqual(),
    )
    assert not replace(result, final_enclosure=forged_enclosure).certified
    assert result.nested_exit_result is not None
    forged_nested = replace(result.nested_exit_result, checker_id=_AlwaysEqual())
    assert not replace(result, nested_exit_result=forged_nested).certified

    assert result.maximum_final_component_width is not None
    hostile = _RawMixedResultEqualitySpoof(
        **{
            **result.__dict__,
            "maximum_final_component_width": (
                result.maximum_final_component_width + Fraction(1)
            ),
        }
    )
    assert not hostile._snapshot_well_formed()
    assert not hostile.replay_consistent
    assert not hostile.certified

    outside = check_raw_mixed_planar_continuation(
        replace(
            _certificate(),
            requested_target_time=_certificate().requested_target_time + 1.0,
        )
    )
    copied_all_true_ledger = replace(
        outside,
        status=CERTIFIED_TO_T,
        obligations=result.obligations,
        first_failed_obligation=None,
    )
    assert not copied_all_true_ledger.certified

    assert result.maximum_final_component_width is not None
    unresolved = check_raw_mixed_planar_continuation(
        replace(
            _certificate(),
            requested_maximum_component_width=math.nextafter(
                _fraction_upper_float(result.maximum_final_component_width),
                -math.inf,
            ),
        )
    )
    assert unresolved.status == UNRESOLVED
    assert unresolved.replay_consistent
    for field in (
        "covered_physical_time_interval",
        "exit_time_interval",
        "target_clock_origin_interval",
        "target_parameter_preimage_interval",
    ):
        assert not replace(
            unresolved,
            **{field: _AlwaysEqual()},
        ).replay_consistent
