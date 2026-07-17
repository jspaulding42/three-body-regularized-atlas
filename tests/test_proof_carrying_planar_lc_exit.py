from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from fractions import Fraction
from math import comb

import numpy as np

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
from three_body_symmetry.proof_carrying_planar_lc_exit import (
    RawGaugeAwarePlanarLCExitContainmentResult,
    check_raw_gauge_aware_planar_lc_exit_containment,
)
from three_body_symmetry.series import construct_taylor_solution


@dataclass(frozen=True)
class _Fixture:
    entry: OrdinaryToPlanarLCEnclosureTransitionCertificate
    binding: InitialValueProblemBindingCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    source_chart: object
    lc_chart: object
    lc_tube: PlanarLCAposterioriTubeCertificate
    exit: PlanarLCToOrdinaryEnclosureTransitionCertificate
    target_chart: object
    target_tube: OrdinaryAposterioriTubeCertificate


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


@lru_cache(maxsize=1)
def _fixture() -> _Fixture:
    step = 2.0**-20
    masses = np.asarray((1.0, 1.0, 1.0))
    positions = np.asarray(((0.0, 0.0), (1.0, 0.0), (3.0, 1.0)))
    velocities = np.zeros((3, 2))

    source_solution = construct_taylor_solution(
        positions, velocities, masses, order=10
    )
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        source_solution,
        certificate_id="raw-exit-source-certificate",
        chart_id="raw-exit-source-chart",
        parameter_interval=(0.0, step),
        physical_time_interval=(0.0, step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    binding = InitialValueProblemBindingCertificate(
        binding_id="raw-exit-binding",
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
        tube_id="raw-exit-source-tube",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-6,
        tube_radius=1.0e-4,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )

    lifted_initial = planar_to_regularized_binary_collision_chart(
        positions, velocities, masses, pair=(0, 1)
    )
    lc_solution = construct_regularized_binary_taylor_solution(
        lifted_initial, order=10
    )
    lc_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        lc_solution,
        certificate_id="raw-exit-lc-certificate",
        chart_id="raw-exit-lc-chart",
        parameter_interval=(0.0, step),
        coefficient_tolerance=1.0e-8,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-10,
        sample_count=5,
        projection_rho_lower_bound=1.0e-8,
        physical_time_shift=step,
    )
    lc_tube = PlanarLCAposterioriTubeCertificate(
        tube_id="raw-exit-lc-tube",
        chart_id=lc_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=1.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    entry = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        transition_id="raw-exit-entry",
        source_chart_id=source_chart.chart_id,
        target_chart_id=lc_chart.chart_id,
        source_parameter=step,
        target_parameter=0.0,
        handoff_time=step,
        max_time_gap=0.0,
    )

    exit_positions, exit_velocities = regularized_binary_collision_chart_to_planar(
        lc_solution.state_at(step)
    )
    target_solution = construct_taylor_solution(
        exit_positions, exit_velocities, masses, order=10
    )
    target_chart = ordinary_taylor_chart_certificate_from_solution(
        target_solution,
        certificate_id="raw-exit-target-certificate",
        chart_id="raw-exit-target-chart",
        parameter_interval=(0.0, step),
        physical_time_interval=(0.0, step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="raw-exit-target-tube",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-2,
        tube_radius=5.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    exit_record = PlanarLCToOrdinaryEnclosureTransitionCertificate(
        transition_id="raw-exit-transition",
        source_chart_id=lc_chart.chart_id,
        target_chart_id=target_chart.chart_id,
        source_parameter=step,
        target_parameter=0.0,
    )
    return _Fixture(
        entry,
        binding,
        source_tube,
        source_chart,
        lc_chart,
        lc_tube,
        exit_record,
        target_chart,
        target_tube,
    )


def _check(fixture: _Fixture) -> RawGaugeAwarePlanarLCExitContainmentResult:
    return check_raw_gauge_aware_planar_lc_exit_containment(
        fixture.entry,
        fixture.binding,
        fixture.source_tube,
        fixture.source_chart,
        fixture.lc_chart,
        fixture.lc_tube,
        fixture.exit,
        fixture.target_chart,
        fixture.target_tube,
    )


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
        tuple(
            tuple(float(value) for value in row)
            for row in coefficient
        )
        for coefficient in translated
    )


def test_positive_raw_exit_replays_complete_slice_projection_and_clock():
    result = _check(_fixture())

    assert result.certified
    assert len(result.lifted_exit_slice) == 14
    assert len(result.projected_position_intervals) == 3
    assert len(result.projected_velocity_intervals) == 3
    assert result.exit_rho_interval[0] > 0
    assert result.exit_time_interval == result.lifted_exit_slice[13]
    assert result.target_clock_origin_interval == result.exit_time_interval
    assert result.maximum_projected_anchor_gap is not None
    assert result.maximum_projected_anchor_gap <= Fraction.from_float(
        _fixture().target_tube.initial_error_bound
    )

    anchor = 2.0**-20
    shifted_target = replace(
        _fixture().target_chart,
        position_coefficients=_translate_ordinary_coefficients(
            _fixture().target_chart.position_coefficients,
            anchor,
        ),
        velocity_coefficients=_translate_ordinary_coefficients(
            _fixture().target_chart.velocity_coefficients,
            anchor,
        ),
        parameter_interval=(anchor, 2.0 * anchor),
    )
    shifted_fixture = replace(
        _fixture(),
        exit=replace(_fixture().exit, target_parameter=anchor),
        target_chart=shifted_target,
        target_tube=replace(_fixture().target_tube, anchor_parameter=anchor),
    )
    shifted_result = _check(shifted_fixture)
    anchor_q = Fraction.from_float(anchor)
    assert shifted_result.certified
    assert shifted_result.target_clock_origin_interval == (
        shifted_result.exit_time_interval[0] - anchor_q,
        shifted_result.exit_time_interval[1] - anchor_q,
    )

    nominal_time_mutation = replace(
        _fixture(),
        target_chart=replace(
            _fixture().target_chart,
            physical_time_interval=(100.0, 100.0 + 2.0**-20),
        ),
    )
    assert _check(nominal_time_mutation).certified


def test_endpoint_and_complete_containment_fail_closed():
    fixture = _fixture()
    interior_exit = replace(
        fixture,
        exit=replace(
            fixture.exit,
            source_parameter=0.5 * fixture.exit.source_parameter,
        ),
    )
    endpoint_result = _check(interior_exit)
    assert not endpoint_result.certified
    assert "raw_lc_exit_endpoint_order_and_target_anchor_exact" in (
        endpoint_result.missing_obligations
    )
    assert endpoint_result.lifted_exit_slice == ()

    narrow = replace(
        fixture,
        target_tube=replace(
            fixture.target_tube,
            initial_error_bound=1.0e-6,
        ),
    )
    containment_result = _check(narrow)
    assert containment_result.target_tube_result is not None
    assert containment_result.target_tube_result.certified
    assert not containment_result.certified
    assert (
        "raw_lc_exit_target_initial_ball_contains_complete_projection"
        in containment_result.missing_obligations
    )


def test_pair_mass_malformed_and_result_tampering_cannot_certify():
    fixture = _fixture()
    reversed_pair = replace(
        fixture,
        lc_chart=replace(fixture.lc_chart, pair=(1, 0)),
    )
    pair_result = _check(reversed_pair)
    assert not pair_result.certified
    assert "raw_lc_exit_pair_canonical_ascending" in pair_result.missing_obligations

    inexact_masses = (0.1, 0.2, 0.3)
    inexact_ratio = replace(
        fixture,
        binding=replace(fixture.binding, masses=inexact_masses),
        source_chart=replace(fixture.source_chart, masses=inexact_masses),
        lc_chart=replace(fixture.lc_chart, masses=inexact_masses),
        target_chart=replace(fixture.target_chart, masses=inexact_masses),
    )
    ratio_result = _check(inexact_ratio)
    assert not ratio_result.certified
    assert "raw_lc_exit_mass_ratio_arithmetic_exact" in (
        ratio_result.missing_obligations
    )

    malformed_exit = replace(
        fixture,
        exit=replace(fixture.exit, source_parameter="not-a-float"),
    )
    malformed_result = _check(malformed_exit)
    assert not malformed_result.certified
    assert "raw_lc_exit_exact_primitive_schemas" in (
        malformed_result.missing_obligations
    )

    positive_binding_tolerance = replace(
        fixture,
        binding=replace(fixture.binding, time_tolerance=1.0e-6),
    )
    tolerance_result = _check(positive_binding_tolerance)
    assert not tolerance_result.certified
    assert "raw_lc_exit_source_binding_is_exact_point_ivp" in (
        tolerance_result.missing_obligations
    )

    hostile_exact_class_payloads = (
        replace(
            fixture,
            lc_chart=replace(
                fixture.lc_chart,
                pair=np.asarray((0, 1), dtype=object),
            ),
        ),
        replace(
            fixture,
            binding=replace(
                fixture.binding,
                masses=np.asarray((1.0, 1.0, 1.0)),
            ),
        ),
        replace(
            fixture,
            target_chart=replace(fixture.target_chart, chart_id=[]),
        ),
    )
    for hostile in hostile_exact_class_payloads:
        hostile_result = _check(hostile)
        assert not hostile_result.certified
        assert "raw_lc_exit_exact_primitive_schemas" in (
            hostile_result.missing_obligations
        )

    result = _check(fixture)
    assert result.certified
    shifted_time = (
        result.exit_time_interval[0],
        result.exit_time_interval[1] + Fraction(1, 2**80),
    )
    assert not replace(result, exit_time_interval=shifted_time).certified
    false_obligation = replace(result.obligations[0], certified=False)
    assert not replace(
        result,
        obligations=(false_obligation,) + result.obligations[1:],
    ).certified
    malformed_ledger = replace(
        result,
        obligations=(object(),) + result.obligations[1:],
    )
    assert not malformed_ledger.certified
    assert malformed_ledger.missing_obligations[0] == (
        "raw_lc_exit_malformed_obligation:0"
    )


def test_always_equal_nested_snapshot_fields_cannot_spoof_fresh_replay():
    result = _check(_fixture())
    assert result.certified

    assert not replace(result, checker_id=_AlwaysEqual()).certified
    assert not replace(result, analytic_kernel_id="unreviewed-kernel").certified
    assert not replace(result, analytic_kernel_id=_AlwaysEqual()).certified
    assert result.lc_tube_result is not None
    assert result.target_tube_result is not None

    for field in ("checker_id", "defect_bound"):
        forged_lc = replace(
            result.lc_tube_result,
            **{field: _AlwaysEqual()},
        )
        assert not replace(result, lc_tube_result=forged_lc).certified

        forged_target = replace(
            result.target_tube_result,
            **{field: _AlwaysEqual()},
        )
        assert not replace(result, target_tube_result=forged_target).certified

    forged_lc_obligation = replace(
        result.lc_tube_result.obligations[0],
        detail=_AlwaysEqual(),
    )
    assert not replace(
        result,
        lc_tube_result=replace(
            result.lc_tube_result,
            obligations=(forged_lc_obligation,)
            + result.lc_tube_result.obligations[1:],
        ),
    ).certified

    forged_target_obligation = replace(
        result.target_tube_result.obligations[0],
        obligation=_AlwaysEqual(),
    )
    assert not replace(
        result,
        target_tube_result=replace(
            result.target_tube_result,
            obligations=(forged_target_obligation,)
            + result.target_tube_result.obligations[1:],
        ),
    ).certified
