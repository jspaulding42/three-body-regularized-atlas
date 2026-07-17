from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import pytest

from three_body_symmetry.certificate_checker import (
    _planar_lc_mass_ratio_arithmetic_exact,
)
from three_body_symmetry.binary_chart import (
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_entry import (
    CarriedPlanarLCEntryTransitionRecord,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_exit import (
    CarriedPlanarLCExitResult,
    check_carried_planar_lc_exit,
)
from three_body_symmetry.planar_lc_mass_coefficients import (
    PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID,
)
from three_body_symmetry.series import construct_taylor_solution


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


class _ForgedExitResult(CarriedPlanarLCExitResult):
    """Hostile exact-field subclass that spoofs the final equality check."""

    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _Fixture:
    entry: CarriedPlanarLCEntryTransitionRecord
    source_chart: OrdinaryTaylorChartCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    lc_chart: PlanarLeviCivitaBinaryChartCertificate
    lc_tube: PlanarLCAposterioriTubeCertificate
    exit: PlanarLCToOrdinaryEnclosureTransitionCertificate
    target_chart: OrdinaryTaylorChartCertificate
    target_tube: OrdinaryAposterioriTubeCertificate
    parent_clock: tuple[Fraction, Fraction]


@lru_cache(maxsize=None)
def _fixture(
    pair: tuple[int, int] = (0, 1),
    serialized_masses: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> _Fixture:
    step = 2.0**-20
    masses = np.asarray(serialized_masses)
    positions = np.asarray(((0.0, 0.0), (-1.0, 0.0), (3.0, 1.0)))
    velocities = np.zeros((3, 2))
    source_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
    )
    mass_tag = "decimal" if serialized_masses == (0.1, 0.2, 0.3) else "unit"
    tag = f"{pair[0]}{pair[1]}-{mass_tag}"
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        source_solution,
        certificate_id=f"carried-exit-source-certificate:{tag}",
        chart_id=f"carried-exit-source-chart:{tag}",
        parameter_interval=(0.0, step),
        physical_time_interval=(50.0, 50.0 + step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id=f"carried-exit-source-tube:{tag}",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-6,
        tube_radius=1.0e-4,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )

    lifted = planar_to_regularized_binary_collision_chart(
        source_solution.positions_at(step),
        source_solution.velocities_at(step),
        masses,
        pair=pair,
    )
    lc_solution = construct_regularized_binary_taylor_solution(
        lifted,
        order=10,
    )
    lc_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        lc_solution,
        certificate_id=f"carried-exit-lc-certificate:{tag}",
        chart_id=f"carried-exit-lc-chart:{tag}",
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
        tube_id=f"carried-exit-lc-tube:{tag}",
        chart_id=lc_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=1.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    entry = CarriedPlanarLCEntryTransitionRecord(
        transition_id=f"carried-exit-entry:{tag}",
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=lc_chart.chart_id,
        target_tube_id=lc_tube.tube_id,
        source_right_parameter=step,
        target_left_parameter=0.0,
    )

    exit_positions, exit_velocities = regularized_binary_collision_chart_to_planar(
        lc_solution.state_at(step)
    )
    target_solution = construct_taylor_solution(
        exit_positions,
        exit_velocities,
        masses,
        order=10,
    )
    target_chart = ordinary_taylor_chart_certificate_from_solution(
        target_solution,
        certificate_id=f"carried-exit-target-certificate:{tag}",
        chart_id=f"carried-exit-target-chart:{tag}",
        parameter_interval=(0.0, step),
        # Deliberately unrelated metadata: B_target comes from component 14.
        physical_time_interval=(-100.0, -100.0 + step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id=f"carried-exit-target-tube:{tag}",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-2,
        tube_radius=5.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    exit_record = PlanarLCToOrdinaryEnclosureTransitionCertificate(
        transition_id=f"carried-exit-transition:{tag}",
        source_chart_id=lc_chart.chart_id,
        target_chart_id=target_chart.chart_id,
        source_parameter=step,
        target_parameter=0.0,
    )
    return _Fixture(
        entry=entry,
        source_chart=source_chart,
        source_tube=source_tube,
        lc_chart=lc_chart,
        lc_tube=lc_tube,
        exit=exit_record,
        target_chart=target_chart,
        target_tube=target_tube,
        parent_clock=(Fraction(-1, 10_000), Fraction(1, 10_000)),
    )


def _check(fixture: _Fixture) -> CarriedPlanarLCExitResult:
    return check_carried_planar_lc_exit(
        fixture.entry,
        fixture.source_chart,
        fixture.source_tube,
        fixture.lc_chart,
        fixture.lc_tube,
        fixture.exit,
        fixture.target_chart,
        fixture.target_tube,
        fixture.parent_clock,
    )


@pytest.mark.parametrize("pair", ((0, 1), (0, 2), (1, 2)))
def test_carried_exit_supports_all_pairs_and_replays_clock(pair):
    fixture = _fixture(pair)
    result = _check(fixture)

    assert result.certified
    assert result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
    assert result.missing_obligations == ()
    assert result.entry_result is not None
    assert result.entry_result.certified
    assert not result.entry_result.exact_initial_value_problem_binding_certified
    assert result.parent_source_invariant_id == (
        "parent_carried_ordinary_solution_invariant_v1"
    )
    assert len(result.lifted_exit_slice) == 14
    assert result.exit_time_interval == result.lifted_exit_slice[13]
    assert result.exit_rho_interval[0] > 0
    assert len(result.projected_position_intervals) == 3
    assert len(result.projected_velocity_intervals) == 3
    anchor = Fraction.from_float(fixture.exit.target_parameter)
    assert result.target_clock_origin_interval == (
        result.exit_time_interval[0] - anchor,
        result.exit_time_interval[1] - anchor,
    )
    assert result.maximum_projected_anchor_gap is not None
    assert result.maximum_projected_anchor_gap <= Fraction.from_float(
        fixture.target_tube.initial_error_bound
    )


def test_carried_exit_accepts_decimal_masses_rejected_by_legacy_ratio_gate():
    masses = (0.1, 0.2, 0.3)
    fixture = _fixture((0, 1), serialized_masses=masses)
    result = _check(fixture)

    assert not _planar_lc_mass_ratio_arithmetic_exact(masses, (0, 1))
    assert result.certified
    assert result.entry_result is not None
    assert result.entry_result.certified
    assert result.lc_tube_result is not None
    assert result.lc_tube_result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )


def test_target_containment_and_endpoint_fail_closed():
    fixture = _fixture()
    narrow = replace(
        fixture,
        target_tube=replace(fixture.target_tube, initial_error_bound=1.0e-8),
    )
    containment = _check(narrow)
    assert containment.target_tube_result is not None
    assert containment.target_tube_result.certified
    assert not containment.certified
    assert (
        "carried_lc_exit_target_initial_ball_contains_complete_projection"
        in containment.missing_obligations
    )

    interior = replace(
        fixture,
        exit=replace(
            fixture.exit,
            source_parameter=0.5 * fixture.exit.source_parameter,
        ),
    )
    endpoint = _check(interior)
    assert not endpoint.certified
    assert (
        "carried_lc_exit_exact_right_to_left_endpoint_handoff"
        in endpoint.missing_obligations
    )
    assert endpoint.lifted_exit_slice == ()


def test_parent_clock_raw_types_pair_and_extreme_mass_arithmetic_fail_closed():
    fixture = _fixture()
    reversed_clock = replace(
        fixture,
        parent_clock=(Fraction(1), Fraction(0)),
    )
    clock_result = _check(reversed_clock)
    assert not clock_result.certified
    assert (
        "carried_lc_exit_parent_source_invariant_is_explicit_condition"
        in clock_result.missing_obligations
    )

    reversed_pair = replace(
        fixture,
        lc_chart=replace(fixture.lc_chart, pair=(1, 0)),
    )
    pair_result = _check(reversed_pair)
    assert not pair_result.certified
    assert (
        "carried_lc_exit_pair_is_canonical_ascending"
        in pair_result.missing_obligations
    )

    extreme_masses = (
        float.fromhex("0x1.fffffffffffffp+1023"),
        1.0,
        1.0,
    )
    ratio_fixture = replace(
        fixture,
        source_chart=replace(fixture.source_chart, masses=extreme_masses),
        lc_chart=replace(fixture.lc_chart, masses=extreme_masses),
        target_chart=replace(fixture.target_chart, masses=extreme_masses),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        ratio_result = _check(ratio_fixture)
    assert not ratio_result.certified
    assert (
        "carried_lc_exit_outward_mass_arithmetic_certified"
        in ratio_result.missing_obligations
    )

    malformed_exit = replace(
        fixture,
        exit=replace(fixture.exit, source_parameter=True),
    )
    schema_result = _check(malformed_exit)
    assert not schema_result.certified
    assert "carried_lc_exit_exact_raw_schemas" in schema_result.missing_obligations

    with pytest.raises(TypeError, match="exact classes"):
        check_carried_planar_lc_exit(
            object(),
            fixture.source_chart,
            fixture.source_tube,
            fixture.lc_chart,
            fixture.lc_tube,
            fixture.exit,
            fixture.target_chart,
            fixture.target_tube,
            fixture.parent_clock,
        )


def test_rho_and_full_result_snapshot_mutations_cannot_certify():
    result = _check(_fixture())
    assert result.certified

    assert not replace(
        result,
        exit_rho_interval=(Fraction(0), result.exit_rho_interval[1]),
    ).certified
    assert not replace(
        result,
        exit_time_interval=(
            result.exit_time_interval[0],
            result.exit_time_interval[1] + Fraction(1, 2**80),
        ),
    ).certified
    assert not replace(result, checker_id=_AlwaysEqual()).certified
    assert not replace(result, analytic_kernel_id="unreviewed-kernel").certified
    assert not replace(
        result,
        mass_arithmetic_kernel_id="wrong-mass-kernel",
    ).certified
    assert not replace(result, parent_source_invariant_id=_AlwaysEqual()).certified

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
        "carried_lc_exit_malformed_obligation:0"
    )

    assert result.entry_result is not None
    forged_entry = replace(result.entry_result, checker_id=_AlwaysEqual())
    assert not replace(result, entry_result=forged_entry).certified
    forged_entry_kernel = replace(
        result.entry_result,
        analytic_kernel_id="unreviewed-entry-kernel",
    )
    assert not replace(result, entry_result=forged_entry_kernel).certified
    forged_entry_mass_kernel = replace(
        result.entry_result,
        mass_arithmetic_kernel_id="wrong-mass-kernel",
    )
    assert not replace(result, entry_result=forged_entry_mass_kernel).certified
    nonmonotone_entry = replace(
        result.entry_result,
        entry_rho_lower_bound=Fraction(0),
    )
    assert not nonmonotone_entry.physical_time_strictly_monotone_certified
    assert not replace(result, entry_result=nonmonotone_entry).certified
    wrong_entry_id = replace(
        result.entry_result,
        raw_transition=replace(
            result.entry_result.raw_transition,
            transition_id="different-entry-transition",
        ),
    )
    assert not replace(result, entry_result=wrong_entry_id).certified
    assert result.lc_tube_result is not None
    forged_lc = replace(result.lc_tube_result, defect_bound=_AlwaysEqual())
    assert not replace(result, lc_tube_result=forged_lc).certified
    forged_lc_kernel = replace(
        result.lc_tube_result,
        mass_arithmetic_kernel_id=_AlwaysEqual(),
    )
    assert not replace(result, lc_tube_result=forged_lc_kernel).certified
    assert result.target_tube_result is not None
    forged_target = replace(result.target_tube_result, checker_id=_AlwaysEqual())
    assert not replace(result, target_tube_result=forged_target).certified

    forged_clock = (
        result.target_clock_origin_interval[0] + Fraction(1, 2**80),
        result.target_clock_origin_interval[1] + Fraction(1, 2**80),
    )
    hostile_subclass = _ForgedExitResult(
        **{
            **result.__dict__,
            "target_clock_origin_interval": forged_clock,
        }
    )
    assert hostile_subclass == result
    assert not hostile_subclass._snapshot_certified()
    assert not hostile_subclass.certified


def test_nonfinite_and_hostile_exact_class_payloads_reject():
    fixture = _fixture()
    hostile = (
        replace(
            fixture,
            exit=replace(fixture.exit, target_parameter=math.nan),
        ),
        replace(
            fixture,
            target_chart=replace(fixture.target_chart, chart_id=[]),
        ),
        replace(
            fixture,
            lc_chart=replace(
                fixture.lc_chart,
                pair=np.asarray((0, 1), dtype=object),
            ),
        ),
    )
    for case in hostile:
        result = _check(case)
        assert not result.certified
        assert "carried_lc_exit_exact_raw_schemas" in result.missing_obligations
