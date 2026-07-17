from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import pytest

from three_body_symmetry.certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    ordinary_taylor_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_ordinary_bridge import (
    CarriedOrdinaryBridgeResult,
    OrdinaryBridgeTransitionRecord,
    check_carried_ordinary_bridge,
)
from three_body_symmetry.series import construct_taylor_solution


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


class _OrdinaryBridgeResultEqualitySpoof(CarriedOrdinaryBridgeResult):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _Fixture:
    transition: OrdinaryBridgeTransitionRecord
    source_chart: OrdinaryTaylorChartCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    target_chart: OrdinaryTaylorChartCertificate
    target_tube: OrdinaryAposterioriTubeCertificate
    source_clock: tuple[Fraction, Fraction]


@lru_cache(maxsize=1)
def _fixture() -> _Fixture:
    step = 0.02
    masses = np.asarray((1.0, 0.8, 1.2))
    positions = np.asarray(
        ((0.8, -0.2), (-0.4, 0.6), (0.1, -0.5))
    )
    velocities = np.asarray(
        ((0.03, 0.01), (-0.02, 0.04), (0.01, -0.03))
    )
    source_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
    )
    target_solution = construct_taylor_solution(
        source_solution.positions_at(step),
        source_solution.velocities_at(step),
        masses,
        order=10,
    )
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        source_solution,
        certificate_id="ordinary-bridge-source-certificate",
        chart_id="ordinary-bridge-source-chart",
        parameter_interval=(0.0, step),
        physical_time_interval=(50.0, 50.0 + step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    target_chart = ordinary_taylor_chart_certificate_from_solution(
        target_solution,
        certificate_id="ordinary-bridge-target-certificate",
        chart_id="ordinary-bridge-target-chart",
        parameter_interval=(0.0, step),
        # Unrelated metadata: the bridge must transport the parent's B.
        physical_time_interval=(-100.0, -100.0 + step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-bridge-source-tube",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="ordinary-bridge-target-tube",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=3.0e-2,
        max_defect_bound=2.0e-1,
        max_lipschitz_bound=100.0,
    )
    transition = OrdinaryBridgeTransitionRecord(
        transition_id="ordinary-bridge-transition",
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=target_chart.chart_id,
        target_tube_id=target_tube.tube_id,
        source_parameter=step,
        target_parameter=0.0,
    )
    return _Fixture(
        transition=transition,
        source_chart=source_chart,
        source_tube=source_tube,
        target_chart=target_chart,
        target_tube=target_tube,
        source_clock=(Fraction(3, 7), Fraction(4, 7)),
    )


def _check(fixture: _Fixture) -> CarriedOrdinaryBridgeResult:
    return check_carried_ordinary_bridge(
        fixture.transition,
        fixture.source_chart,
        fixture.source_tube,
        fixture.target_chart,
        fixture.target_tube,
        fixture.source_clock,
    )


def test_carried_ordinary_bridge_replays_containment_and_clock_cocycle():
    fixture = _fixture()
    result = _check(fixture)

    assert result.certified
    assert result.missing_obligations == ()
    assert result.analytic_kernel_id == (
        "ordinary_autonomous_uniqueness_bridge_kernel_v1"
    )
    assert result.source_tube_result is not None
    assert result.source_tube_result.certified
    assert result.target_tube_result is not None
    assert result.target_tube_result.certified
    assert len(result.source_endpoint_state_intervals) == 12
    assert len(result.target_anchor_centers) == 12
    assert result.maximum_target_anchor_gap is not None
    assert result.maximum_target_anchor_gap <= Fraction.from_float(
        fixture.target_tube.initial_error_bound
    )
    delta = (
        Fraction.from_float(fixture.transition.source_parameter)
        - Fraction.from_float(fixture.transition.target_parameter)
    )
    assert result.target_clock_origin_interval == (
        fixture.source_clock[0] + delta,
        fixture.source_clock[1] + delta,
    )


def test_private_transition_wire_round_trip_and_exact_fields():
    transition = _fixture().transition
    wire = transition.to_dict()

    assert OrdinaryBridgeTransitionRecord.from_dict(wire) == transition

    with pytest.raises(ValueError, match="unknown"):
        OrdinaryBridgeTransitionRecord.from_dict({**wire, "target_clock": []})
    missing = dict(wire)
    del missing["target_tube_id"]
    with pytest.raises(ValueError, match="missing"):
        OrdinaryBridgeTransitionRecord.from_dict(missing)
    with pytest.raises(ValueError, match="scalars"):
        OrdinaryBridgeTransitionRecord.from_dict(
            {**wire, "source_parameter": str(transition.source_parameter)}
        )
    with pytest.raises(ValueError, match="scalars"):
        OrdinaryBridgeTransitionRecord.from_dict(
            {**wire, "source_parameter": math.nan}
        )
    with pytest.raises(ValueError, match="scalars"):
        OrdinaryBridgeTransitionRecord.from_dict(
            {**wire, "schema_version": True}
        )

    noncanonical_source = OrdinaryBridgeTransitionRecord.from_dict(
        {**wire, "source": "unreviewed-private-bridge"}
    )
    result = check_carried_ordinary_bridge(
        noncanonical_source,
        _fixture().source_chart,
        _fixture().source_tube,
        _fixture().target_chart,
        _fixture().target_tube,
        _fixture().source_clock,
    )
    assert not result.certified
    assert "ordinary_bridge_exact_raw_schemas" in result.missing_obligations


def test_insufficient_target_initial_radius_rejects_only_containment():
    fixture = _fixture()
    too_small = replace(
        fixture,
        target_tube=replace(
            fixture.target_tube,
            initial_error_bound=0.0,
        ),
    )
    result = _check(too_small)

    assert result.target_tube_result is not None
    assert result.target_tube_result.certified
    assert not result.certified
    assert (
        "ordinary_bridge_target_initial_ball_contains_complete_source_endpoint"
        in result.missing_obligations
    )


def test_wrong_endpoints_and_malformed_clock_or_raw_types_fail_closed():
    fixture = _fixture()
    wrong_source = replace(
        fixture,
        transition=replace(
            fixture.transition,
            source_parameter=math.nextafter(
                fixture.transition.source_parameter,
                -math.inf,
            ),
        ),
    )
    source_result = _check(wrong_source)
    assert not source_result.certified
    assert (
        "ordinary_bridge_exact_right_to_left_endpoint_handoff"
        in source_result.missing_obligations
    )

    wrong_target = replace(
        fixture,
        transition=replace(fixture.transition, target_parameter=2.0**-20),
    )
    assert not _check(wrong_target).certified

    wrong_source_anchor = replace(
        fixture,
        source_tube=replace(
            fixture.source_tube,
            anchor_parameter=2.0**-20,
        ),
    )
    anchor_result = _check(wrong_source_anchor)
    assert anchor_result.source_tube_result is not None
    assert anchor_result.source_tube_result.certified
    assert not anchor_result.certified
    assert (
        "ordinary_bridge_exact_right_to_left_endpoint_handoff"
        in anchor_result.missing_obligations
    )

    malformed_transition = replace(
        fixture,
        transition=replace(fixture.transition, source_parameter=True),
    )
    assert not _check(malformed_transition).certified

    malformed_clock = replace(
        fixture,
        source_clock=(Fraction(2), Fraction(1)),
    )
    clock_result = _check(malformed_clock)
    assert not clock_result.certified
    assert (
        "ordinary_bridge_parent_clock_origin_is_exact_interval"
        in clock_result.missing_obligations
    )

    with pytest.raises(TypeError, match="exact classes"):
        check_carried_ordinary_bridge(
            object(),
            fixture.source_chart,
            fixture.source_tube,
            fixture.target_chart,
            fixture.target_tube,
            fixture.source_clock,
        )


def test_problem_identity_and_global_identifier_manifest_fail_closed():
    fixture = _fixture()
    wrong_masses = replace(
        fixture,
        target_chart=replace(
            fixture.target_chart,
            masses=(1.0, 0.8, math.nextafter(1.2, math.inf)),
        ),
    )
    mass_result = _check(wrong_masses)
    assert not mass_result.certified
    assert (
        "ordinary_bridge_common_planar_mass_problem"
        in mass_result.missing_obligations
    )

    duplicate_identifier = replace(
        fixture,
        transition=replace(
            fixture.transition,
            transition_id=fixture.source_chart.chart_id,
        ),
    )
    identifier_result = _check(duplicate_identifier)
    assert not identifier_result.certified
    assert (
        "ordinary_bridge_identifiers_match_and_are_unique"
        in identifier_result.missing_obligations
    )


def test_changed_result_fields_and_python_equality_spoofs_do_not_certify():
    result = _check(_fixture())
    assert result.certified

    assert not replace(result, checker_id=_AlwaysEqual()).certified
    assert not replace(result, analytic_kernel_id=_AlwaysEqual()).certified
    assert not replace(result, analytic_kernel_id="unreviewed").certified
    assert not replace(result, transition_id=_AlwaysEqual()).certified
    forged_raw_transition = replace(
        result.raw_transition,
        source_parameter=math.nextafter(
            result.raw_transition.source_parameter,
            -math.inf,
        ),
    )
    assert not replace(
        result,
        raw_transition=forged_raw_transition,
    ).certified
    assert not replace(
        result,
        target_clock_origin_interval=_AlwaysEqual(),
    ).certified
    assert result.source_tube_result is not None
    forged_nested = replace(
        result.source_tube_result,
        checker_id=_AlwaysEqual(),
    )
    assert not replace(result, source_tube_result=forged_nested).certified
    forged_obligation = replace(result.obligations[0], detail=_AlwaysEqual())
    assert not replace(
        result,
        obligations=(forged_obligation,) + result.obligations[1:],
    ).certified

    assert not replace(
        result,
        parent_source_clock_origin_interval=(Fraction(0), Fraction(0)),
    ).certified
    assert not replace(
        result,
        source_endpoint_state_intervals=result.source_endpoint_state_intervals[
            :-1
        ],
    ).certified
    assert not replace(
        result,
        target_anchor_centers=result.target_anchor_centers[:-1],
    ).certified
    assert result.maximum_target_anchor_gap is not None
    assert not replace(
        result,
        maximum_target_anchor_gap=result.maximum_target_anchor_gap + Fraction(1),
    ).certified
    assert result.target_tube_result is not None
    forged_target = replace(
        result.target_tube_result,
        gronwall_error_bound=math.nextafter(
            result.target_tube_result.gronwall_error_bound,
            math.inf,
        ),
    )
    assert not replace(result, target_tube_result=forged_target).certified

    hostile = _OrdinaryBridgeResultEqualitySpoof(
        **{
            **result.__dict__,
            "maximum_target_anchor_gap": (
                result.maximum_target_anchor_gap + Fraction(1)
            ),
        }
    )
    assert not hostile._snapshot_certified()
    assert not hostile.certified
