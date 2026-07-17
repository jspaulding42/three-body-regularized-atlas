from dataclasses import replace
import json
import re

import numpy as np
import pytest

from three_body_symmetry.certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryEnclosureTransitionCertificate,
    ordinary_taylor_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
    RawOrdinaryContinuationCertificate,
    RawOrdinaryContinuationObligation,
    RawOrdinaryContinuationReplayResult,
    canonical_evidence_json,
    check_raw_ordinary_continuation,
)
from three_body_symmetry.series import construct_taylor_solution


def _raw_ordinary_continuation_certificate(*, step: float = 0.02):
    masses = np.array([1.0, 0.8, 1.2])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.01],
            [-0.02, 0.04],
            [0.01, -0.03],
        ]
    )
    first_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
    )
    second_solution = construct_taylor_solution(
        first_solution.positions_at(step),
        first_solution.velocities_at(step),
        masses,
        order=10,
    )
    first = ordinary_taylor_chart_certificate_from_solution(
        first_solution,
        certificate_id="raw-ordinary-chart-certificate-0",
        chart_id="raw-ordinary-chart-0",
        parameter_interval=(0.0, step),
        physical_time_interval=(0.0, step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    second = ordinary_taylor_chart_certificate_from_solution(
        second_solution,
        certificate_id="raw-ordinary-chart-certificate-1",
        chart_id="raw-ordinary-chart-1",
        parameter_interval=(0.0, step),
        physical_time_interval=(step, 2.0 * step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    binding = InitialValueProblemBindingCertificate(
        binding_id="raw-ordinary-ivp-binding",
        chart_id=first.chart_id,
        masses=first.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=first.position_coefficients[0],
        velocities=first.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    tubes = (
        OrdinaryAposterioriTubeCertificate(
            tube_id="raw-ordinary-tube-0",
            chart_id=first.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=0.0,
            tube_radius=2.0e-2,
            max_defect_bound=1.0e-1,
            max_lipschitz_bound=70.0,
        ),
        OrdinaryAposterioriTubeCertificate(
            tube_id="raw-ordinary-tube-1",
            chart_id=second.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=2.0e-3,
            tube_radius=3.0e-2,
            max_defect_bound=2.0e-1,
            max_lipschitz_bound=100.0,
        ),
    )
    transitions = (
        OrdinaryEnclosureTransitionCertificate(
            transition_id="raw-ordinary-transition-0-1",
            source_chart_id=first.chart_id,
            target_chart_id=second.chart_id,
            source_parameter=step,
            target_parameter=0.0,
            handoff_time=step,
            max_time_gap=0.0,
        ),
    )
    return RawOrdinaryContinuationCertificate(
        certificate_id="raw-ordinary-continuation-certificate",
        binding=binding,
        charts=(first, second),
        tubes=tubes,
        transitions=transitions,
        requested_target_time=2.0 * step,
        requested_maximum_component_width=1.0,
    )


def test_raw_ordinary_continuation_replays_to_requested_target():
    certificate = _raw_ordinary_continuation_certificate()

    result = check_raw_ordinary_continuation(certificate)

    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.first_failed_obligation is None
    assert all(obligation.certified for obligation in result.obligations)
    assert re.fullmatch(r"[0-9a-f]{64}", result.evidence_sha256)
    assert result.covered_physical_time_interval == (0.0, 0.04)
    assert result.requested_target_time == 0.04
    assert result.final_enclosure is not None
    assert result.final_enclosure.physical_time_interval == (0.04, 0.04)
    assert len(result.final_enclosure.position_intervals) == 3
    assert len(result.final_enclosure.velocity_intervals) == 3
    assert len(result.retained_safe_regions) == 1
    assert result.retained_safe_regions == (result.final_enclosure,)
    assert result.requested_maximum_component_width == 1.0
    assert 0.0 <= result.maximum_final_component_width <= 1.0
    assert not replace(result, obligations=()).certified
    assert not replace(
        result,
        obligations=(
            RawOrdinaryContinuationObligation("spoofed_success", True, ""),
        ),
    ).certified
    assert not replace(result, evidence_sha256="0" * 63).certified
    assert not replace(
        result,
        nested_missing_obligations=("spoofed_nested_success",),
    ).certified
    assert not replace(result, final_enclosure=None).certified


def test_success_result_is_bound_to_fresh_replay_in_every_field():
    result = check_raw_ordinary_continuation(
        _raw_ordinary_continuation_certificate()
    )
    assert result.final_enclosure is not None

    assert not replace(result, evidence_sha256="0" * 64).certified
    assert not replace(
        result,
        covered_physical_time_interval=(0.0, 0.039),
    ).certified
    tampered_positions = (
        (
            (
                np.nextafter(
                    result.final_enclosure.position_intervals[0][0][0],
                    -np.inf,
                ),
                result.final_enclosure.position_intervals[0][0][1],
            ),
            result.final_enclosure.position_intervals[0][1],
        ),
        *result.final_enclosure.position_intervals[1:],
    )
    tampered_final = replace(
        result.final_enclosure,
        position_intervals=tampered_positions,
    )
    assert not replace(result, final_enclosure=tampered_final).certified
    assert not replace(result, checker_id="spoofed_checker").certified
    assert not replace(
        result,
        obligations=tuple(
            replace(obligation, detail="spoofed detail")
            for obligation in result.obligations
        ),
    ).certified


def test_all_true_result_built_from_scratch_cannot_spoof_certification():
    fresh = check_raw_ordinary_continuation(
        _raw_ordinary_continuation_certificate()
    )
    assert fresh.final_enclosure is not None
    forged = RawOrdinaryContinuationReplayResult(
        certificate_id=fresh.certificate_id,
        raw_certificate=fresh.raw_certificate,
        status=CERTIFIED_TO_T,
        evidence_sha256=fresh.evidence_sha256,
        obligations=tuple(
            RawOrdinaryContinuationObligation(
                obligation.obligation,
                True,
                "claimed without replay",
            )
            for obligation in fresh.obligations
        ),
        first_failed_obligation=None,
        covered_physical_time_interval=(0.0, 0.04),
        requested_target_time=0.04,
        requested_maximum_component_width=1.0,
        maximum_final_component_width=fresh.maximum_final_component_width,
        final_enclosure=fresh.final_enclosure,
        retained_safe_regions=(fresh.final_enclosure,),
        nested_checker_id=fresh.nested_checker_id,
        nested_missing_obligations=(),
    )

    assert not forged.certified


def test_raw_ordinary_continuation_replays_at_nonzero_absolute_time():
    certificate = _raw_ordinary_continuation_certificate(step=0.015625)
    shift = 7.25
    translated_charts = tuple(
        replace(
            chart,
            physical_time_interval=tuple(
                endpoint + shift for endpoint in chart.physical_time_interval
            ),
        )
        for chart in certificate.charts
    )
    translated = replace(
        certificate,
        binding=replace(
            certificate.binding,
            initial_time=certificate.binding.initial_time + shift,
        ),
        charts=translated_charts,
        transitions=tuple(
            replace(
                transition,
                handoff_time=transition.handoff_time + shift,
            )
            for transition in certificate.transitions
        ),
        requested_target_time=certificate.requested_target_time + shift,
    )

    result = check_raw_ordinary_continuation(translated)

    assert tuple(chart.parameter_interval for chart in translated.charts) == tuple(
        chart.parameter_interval for chart in certificate.charts
    )
    assert tuple(chart.position_coefficients for chart in translated.charts) == tuple(
        chart.position_coefficients for chart in certificate.charts
    )
    assert tuple(chart.velocity_coefficients for chart in translated.charts) == tuple(
        chart.velocity_coefficients for chart in certificate.charts
    )
    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.covered_physical_time_interval == (7.25, 7.28125)
    assert result.final_enclosure is not None
    assert result.final_enclosure.physical_time_interval == (7.28125, 7.28125)


def test_raw_ordinary_roundtrip_and_digest_are_stable():
    certificate = _raw_ordinary_continuation_certificate()
    canonical = canonical_evidence_json(certificate)
    restored = RawOrdinaryContinuationCertificate.from_dict(json.loads(canonical))

    assert restored == certificate
    assert restored.to_dict() == certificate.to_dict()
    assert canonical_evidence_json(restored) == canonical
    assert (
        check_raw_ordinary_continuation(restored).evidence_sha256
        == check_raw_ordinary_continuation(certificate).evidence_sha256
    )


def test_raw_ordinary_from_dict_rejects_noncanonical_wire_objects():
    payload = _raw_ordinary_continuation_certificate().to_dict()

    with_unknown = dict(payload)
    with_unknown["unknown_field"] = True
    with pytest.raises(ValueError, match="unknown"):
        RawOrdinaryContinuationCertificate.from_dict(with_unknown)

    with_missing = dict(payload)
    del with_missing["source"]
    with pytest.raises(ValueError, match="missing"):
        RawOrdinaryContinuationCertificate.from_dict(with_missing)

    with_numeric_string = dict(payload)
    with_numeric_string["requested_target_time"] = "0.04"
    with pytest.raises(ValueError, match="noncanonical"):
        RawOrdinaryContinuationCertificate.from_dict(with_numeric_string)

    nested_unknown = json.loads(json.dumps(payload))
    nested_unknown["charts"][0]["unknown_field"] = True
    with pytest.raises(ValueError, match="unknown"):
        RawOrdinaryContinuationCertificate.from_dict(nested_unknown)


def test_raw_ordinary_checker_rejects_malformed_python_object_type():
    with pytest.raises(
        TypeError,
        match="certificate must be RawOrdinaryContinuationCertificate",
    ):
        check_raw_ordinary_continuation(object())


def test_raw_ordinary_replay_fails_closed_on_tampered_mass():
    certificate = _raw_ordinary_continuation_certificate()
    tampered_chart = replace(
        certificate.charts[1],
        masses=(1.0, 0.8, 1.25),
    )
    tampered = replace(
        certificate,
        charts=(certificate.charts[0], tampered_chart),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert not result.certified
    assert (
        result.first_failed_obligation
        == "raw_ordinary_planar_common_mass_problem"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (0.0, 0.02)
    assert len(result.retained_safe_regions) == 1
    assert result.retained_safe_regions[0].enclosure_type == (
        "validated_ordinary_prefix_right_endpoint"
    )
    assert result.retained_safe_regions[0].physical_time_interval == (0.02, 0.02)


def test_raw_ordinary_replay_fails_closed_on_tampered_clock():
    certificate = _raw_ordinary_continuation_certificate()
    tampered_chart = replace(
        certificate.charts[1],
        physical_time_interval=(0.02, 0.041),
    )
    tampered = replace(
        certificate,
        charts=(certificate.charts[0], tampered_chart),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert (
        result.first_failed_obligation
        == "raw_ordinary_exact_affine_physical_clocks"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (0.0, 0.02)
    assert result.retained_safe_regions[0].physical_time_interval == (0.02, 0.02)


def test_raw_ordinary_replay_fails_closed_when_target_is_uncovered():
    certificate = replace(
        _raw_ordinary_continuation_certificate(),
        requested_target_time=0.05,
    )

    result = check_raw_ordinary_continuation(certificate)

    assert result.status == UNRESOLVED
    assert (
        result.first_failed_obligation
        == "raw_ordinary_requested_target_finite_and_covered"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (0.0, 0.04)
    assert result.retained_safe_regions[0].physical_time_interval == (0.04, 0.04)


def test_raw_ordinary_replay_retains_nothing_when_initial_binding_fails():
    certificate = _raw_ordinary_continuation_certificate()
    positions = certificate.binding.positions
    bad_positions = (
        ((positions[0][0] + 0.125, positions[0][1])),
        *positions[1:],
    )
    tampered = replace(
        certificate,
        binding=replace(certificate.binding, positions=bad_positions),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert result.first_failed_obligation == (
        f"nested:{result.nested_missing_obligations[0]}"
    )
    assert result.covered_physical_time_interval == (np.inf, -np.inf)
    assert result.retained_safe_regions == ()


def test_raw_ordinary_replay_retains_nothing_for_collided_first_chart():
    certificate = _raw_ordinary_continuation_certificate()
    first = certificate.charts[0]
    constant_positions = first.position_coefficients[0]
    collided_constant_positions = (
        constant_positions[0],
        constant_positions[0],
        constant_positions[2],
    )
    collided_first = replace(
        first,
        position_coefficients=(
            collided_constant_positions,
            *first.position_coefficients[1:],
        ),
    )
    tampered = replace(
        certificate,
        binding=replace(
            certificate.binding,
            positions=collided_constant_positions,
        ),
        charts=(collided_first, certificate.charts[1]),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert result.covered_physical_time_interval == (np.inf, -np.inf)
    assert result.retained_safe_regions == ()


def test_raw_ordinary_replay_freshly_rejects_nested_tube_failure():
    certificate = _raw_ordinary_continuation_certificate()
    bad_source_tube = replace(certificate.tubes[0], tube_radius=0.0)
    tampered = replace(
        certificate,
        tubes=(bad_source_tube, certificate.tubes[1]),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert result.nested_missing_obligations
    assert result.first_failed_obligation == (
        f"nested:{result.nested_missing_obligations[0]}"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (np.inf, -np.inf)
    assert result.retained_safe_regions == ()


def test_raw_ordinary_replay_freshly_rejects_nested_transition_failure():
    certificate = _raw_ordinary_continuation_certificate()
    uncovered_target_tube = replace(
        certificate.tubes[1],
        initial_error_bound=0.0,
    )
    tampered = replace(
        certificate,
        tubes=(certificate.tubes[0], uncovered_target_tube),
    )

    result = check_raw_ordinary_continuation(tampered)

    assert result.status == UNRESOLVED
    assert "ordinary_enclosure_transition_target_error_covers_handoff" in (
        result.nested_missing_obligations
    )
    assert result.first_failed_obligation == (
        f"nested:{result.nested_missing_obligations[0]}"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (0.0, 0.02)
    assert result.retained_safe_regions[0].physical_time_interval == (0.02, 0.02)


def test_requested_component_width_uses_exact_serialized_tolerance():
    certificate = _raw_ordinary_continuation_certificate()
    baseline = check_raw_ordinary_continuation(certificate)
    maximum_width = baseline.maximum_final_component_width

    equal = check_raw_ordinary_continuation(
        replace(
            certificate,
            requested_maximum_component_width=maximum_width,
        )
    )
    above = check_raw_ordinary_continuation(
        replace(
            certificate,
            requested_maximum_component_width=float(
                np.nextafter(maximum_width, np.inf)
            ),
        )
    )
    below = check_raw_ordinary_continuation(
        replace(
            certificate,
            requested_maximum_component_width=float(
                np.nextafter(maximum_width, -np.inf)
            ),
        )
    )

    assert baseline.status == CERTIFIED_TO_T
    assert equal.status == CERTIFIED_TO_T
    assert equal.maximum_final_component_width == maximum_width
    assert above.status == CERTIFIED_TO_T
    assert below.status == UNRESOLVED
    assert below.first_failed_obligation == (
        "raw_ordinary_final_component_width_within_requested_bound"
    )
    assert below.final_enclosure is None
    assert below.covered_physical_time_interval == (0.0, 0.04)
    assert below.retained_safe_regions[0].physical_time_interval == (0.04, 0.04)


@pytest.mark.parametrize("invalid_width", [-1.0, np.inf, -np.inf, np.nan])
def test_requested_component_width_rejects_negative_or_nonfinite(
    invalid_width: float,
):
    certificate = replace(
        _raw_ordinary_continuation_certificate(),
        requested_maximum_component_width=invalid_width,
    )

    result = check_raw_ordinary_continuation(certificate)

    assert result.status == UNRESOLVED
    assert result.first_failed_obligation == (
        "raw_ordinary_requested_component_width_admissible"
    )
    assert result.final_enclosure is None
    assert result.covered_physical_time_interval == (0.0, 0.04)
    assert result.retained_safe_regions[0].physical_time_interval == (0.04, 0.04)
