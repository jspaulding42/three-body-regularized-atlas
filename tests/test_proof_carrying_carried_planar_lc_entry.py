from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
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
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_language import (
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_entry import (
    CarriedOrdinaryToPlanarLCEntryResult,
    CarriedPlanarLCEntryTransitionRecord,
    check_carried_ordinary_to_planar_lc_entry,
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


class _ResultEqualitySpoof(CarriedOrdinaryToPlanarLCEntryResult):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _Fixture:
    transition: CarriedPlanarLCEntryTransitionRecord
    source_chart: OrdinaryTaylorChartCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    target_chart: PlanarLeviCivitaBinaryChartCertificate
    target_tube: PlanarLCAposterioriTubeCertificate
    source_clock: tuple[Fraction, Fraction]


def _negate_vector_coefficients(
    coefficients: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(-value for value in row) for row in coefficients)


@lru_cache(maxsize=None)
def _fixture(
    pair: tuple[int, int] = (0, 1),
    antipodal_target: bool = False,
    serialized_masses: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> _Fixture:
    step = 2.0**-20
    masses = np.asarray(serialized_masses)
    # Pair 01 lies on the negative branch cut and exercises the two-patch
    # atlas.  The other two canonical pairs exercise singleton cases.
    positions = np.asarray(((0.0, 0.0), (-1.0, 0.0), (3.0, 1.0)))
    velocities = np.zeros((3, 2))
    source_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=10,
    )
    mass_tag = "decimal" if serialized_masses == (0.1, 0.2, 0.3) else "unit"
    tag = (
        f"{pair[0]}{pair[1]}-"
        f"{'anti' if antipodal_target else 'base'}-{mass_tag}"
    )
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        source_solution,
        certificate_id=f"carried-entry-source-certificate:{tag}",
        chart_id=f"carried-entry-source-chart:{tag}",
        parameter_interval=(0.0, step),
        # This metadata is intentionally not the parent clock invariant.
        physical_time_interval=(50.0, 50.0 + step),
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-10,
        sample_count=5,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id=f"carried-entry-source-tube:{tag}",
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
    target_solution = construct_regularized_binary_taylor_solution(
        lifted,
        order=10,
    )
    target_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        target_solution,
        certificate_id=f"carried-entry-target-certificate:{tag}",
        chart_id=f"carried-entry-target-chart:{tag}",
        parameter_interval=(0.0, step),
        coefficient_tolerance=1.0e-8,
        regularized_residual_tolerance=1.0e-5,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-10,
        sample_count=5,
        projection_rho_lower_bound=1.0e-8,
        physical_time_shift=step,
    )
    if antipodal_target:
        target_chart = replace(
            target_chart,
            z_coefficients=_negate_vector_coefficients(
                target_chart.z_coefficients
            ),
            z_velocity_coefficients=_negate_vector_coefficients(
                target_chart.z_velocity_coefficients
            ),
        )
    target_tube = PlanarLCAposterioriTubeCertificate(
        tube_id=f"carried-entry-target-tube:{tag}",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=1.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    transition = CarriedPlanarLCEntryTransitionRecord(
        transition_id=f"carried-entry-transition:{tag}",
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=target_chart.chart_id,
        target_tube_id=target_tube.tube_id,
        source_right_parameter=step,
        target_left_parameter=0.0,
    )
    return _Fixture(
        transition=transition,
        source_chart=source_chart,
        source_tube=source_tube,
        target_chart=target_chart,
        target_tube=target_tube,
        source_clock=(Fraction(-1, 10_000), Fraction(1, 10_000)),
    )


def _check(fixture: _Fixture) -> CarriedOrdinaryToPlanarLCEntryResult:
    return check_carried_ordinary_to_planar_lc_entry(
        fixture.transition,
        fixture.source_chart,
        fixture.source_tube,
        fixture.target_chart,
        fixture.target_tube,
        fixture.source_clock,
    )


@pytest.mark.parametrize("pair", ((0, 1), (0, 2), (1, 2)))
def test_conditional_entry_supports_all_canonical_pairs_and_carries_clock(pair):
    fixture = _fixture(pair)
    result = _check(fixture)

    assert result.certified
    assert result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
    assert result.target_tube_result is not None
    assert result.target_tube_result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
    assert result.conditional_containment_certified
    assert result.constrained_newtonian_lift_certified
    assert result.physical_time_strictly_monotone_certified
    assert not result.exact_initial_value_problem_binding_certified
    assert result.missing_obligations == ()
    assert result.raw_target_chart.pair == pair
    assert len(result.source_endpoint_state_box) == 12
    assert len(result.target_anchor) == 14
    assert result.entry_rho_lower_bound > 0
    source_parameter = Fraction.from_float(
        fixture.transition.source_right_parameter
    )
    assert result.entry_time_interval == (
        fixture.source_clock[0] + source_parameter,
        fixture.source_clock[1] + source_parameter,
    )
    assert len(result.timed_lifted_patch_boxes) in (1, 2)
    assert all(
        len(box) == 14 and box[-1] == result.entry_time_interval
        for box in result.timed_lifted_patch_boxes
    )
    assert result.tested_containments[result.selected_complement_index]
    assert len(result.tested_assignments) == 2
    assert result.tested_assignments[1] == tuple(
        (chart_id, bit ^ 1)
        for chart_id, bit in result.tested_assignments[0]
    )


def test_conditional_entry_accepts_decimal_masses_rejected_by_legacy_ratio_gate():
    masses = (0.1, 0.2, 0.3)
    fixture = _fixture((0, 1), serialized_masses=masses)
    result = _check(fixture)

    assert not _planar_lc_mass_ratio_arithmetic_exact(masses, (0, 1))
    assert result.certified
    assert (
        "carried_lc_entry_outward_mass_arithmetic_certified"
        not in result.missing_obligations
    )


def test_negative_cut_atlas_and_antipodal_target_select_opposite_complements():
    principal = _check(_fixture((0, 1), antipodal_target=False))
    antipodal = _check(_fixture((0, 1), antipodal_target=True))

    assert principal.certified and antipodal.certified
    assert principal.canonical_lift_case == "strict_negative_cut_two_patch"
    assert len(principal.lifted_patch_boxes) == 2
    assert len(principal.derived_edges) == 1
    assert principal.derived_edges[0].parity == 1
    assert principal.selected_complement_index != antipodal.selected_complement_index
    assert principal.tested_containments == tuple(
        not value for value in antipodal.tested_containments
    )


def test_private_transition_wire_is_strict_and_round_trips_canonically():
    transition = _fixture().transition
    wire = transition.to_dict()

    assert CarriedPlanarLCEntryTransitionRecord.from_dict(wire) == transition

    with pytest.raises(ValueError, match="unknown"):
        CarriedPlanarLCEntryTransitionRecord.from_dict(
            {**wire, "supplied_gauge": 0}
        )
    missing = dict(wire)
    del missing["target_tube_id"]
    with pytest.raises(ValueError, match="missing"):
        CarriedPlanarLCEntryTransitionRecord.from_dict(missing)
    with pytest.raises(ValueError, match="scalars"):
        CarriedPlanarLCEntryTransitionRecord.from_dict(
            {
                **wire,
                "source_right_parameter": str(
                    transition.source_right_parameter
                ),
            }
        )
    with pytest.raises(ValueError, match="scalars"):
        CarriedPlanarLCEntryTransitionRecord.from_dict(
            {**wire, "target_left_parameter": math.nan}
        )
    with pytest.raises(ValueError, match="scalars"):
        CarriedPlanarLCEntryTransitionRecord.from_dict(
            {**wire, "schema_version": True}
        )

    alternate_source = replace(
        _fixture(),
        transition=replace(
            transition,
            source="alternate_private_source",
        ),
    )
    source_result = _check(alternate_source)
    assert not source_result.certified
    assert "carried_lc_entry_transition_canonical_round_trip" in (
        source_result.missing_obligations
    )


def test_clock_is_parent_state_and_complete_time_containment_fails_closed():
    fixture = _fixture()
    shifted_clock = replace(
        fixture,
        source_clock=(Fraction(1, 10), Fraction(1, 10)),
    )
    shifted = _check(shifted_clock)

    assert shifted.source_tube_result is not None
    assert shifted.source_tube_result.certified
    assert shifted.target_tube_result is not None
    assert shifted.target_tube_result.certified
    assert shifted.entry_time_interval == (
        Fraction(1, 10)
        + Fraction.from_float(fixture.transition.source_right_parameter),
    ) * 2
    assert not shifted.certified
    assert shifted.tested_containments == (False, False)
    assert (
        "carried_lc_entry_one_global_complement_contains_all_complete_patches"
        in shifted.missing_obligations
    )

    malformed = _check(
        replace(
            fixture,
            source_clock=(Fraction(2), Fraction(1)),
        )
    )
    assert not malformed.certified
    assert (
        "carried_lc_entry_parent_clock_origin_is_exact_interval"
        in malformed.missing_obligations
    )


def test_insufficient_target_initial_radius_rejects_complete_containment():
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
    assert result.tested_containments == (False, False)
    assert (
        "carried_lc_entry_one_global_complement_contains_all_complete_patches"
        in result.missing_obligations
    )


def test_exact_right_left_anchor_endpoints_are_mandatory():
    fixture = _fixture()
    wrong_source = replace(
        fixture,
        transition=replace(
            fixture.transition,
            source_right_parameter=math.nextafter(
                fixture.transition.source_right_parameter,
                -math.inf,
            ),
        ),
    )
    wrong_target = replace(
        fixture,
        transition=replace(
            fixture.transition,
            target_left_parameter=2.0**-30,
        ),
    )
    wrong_anchor = replace(
        fixture,
        target_tube=replace(
            fixture.target_tube,
            anchor_parameter=2.0**-30,
        ),
    )

    for candidate in (wrong_source, wrong_target, wrong_anchor):
        result = _check(candidate)
        assert not result.certified
        assert (
            "carried_lc_entry_exact_source_right_to_lc_left_anchor"
            in result.missing_obligations
        )


def test_reversed_pair_and_malformed_raw_types_reject_before_lifting():
    fixture = _fixture()
    reversed_pair = replace(
        fixture,
        target_chart=replace(fixture.target_chart, pair=(1, 0)),
    )
    reversed_result = _check(reversed_pair)
    assert not reversed_result.certified
    assert (
        "carried_lc_entry_pair_is_canonical_ascending"
        in reversed_result.missing_obligations
    )

    malformed_transition = replace(
        fixture,
        transition=replace(
            fixture.transition,
            source_right_parameter=True,
        ),
    )
    malformed_result = _check(malformed_transition)
    assert not malformed_result.certified
    assert "carried_lc_entry_exact_raw_schemas" in (
        malformed_result.missing_obligations
    )

    malformed_source = replace(
        fixture,
        source_chart=replace(fixture.source_chart, source=_AlwaysEqual()),
    )
    assert not _check(malformed_source).certified

    with pytest.raises(TypeError, match="exact classes"):
        check_carried_ordinary_to_planar_lc_entry(
            object(),
            fixture.source_chart,
            fixture.source_tube,
            fixture.target_chart,
            fixture.target_tube,
            fixture.source_clock,
        )


def test_raw_identifiers_cannot_collide_with_deterministic_gauge_namespace():
    fixture = _fixture()
    collision_id = f"{fixture.transition.transition_id}:patch:0-upper"
    collided = replace(
        fixture,
        transition=replace(
            fixture.transition,
            target_chart_id=collision_id,
        ),
        target_chart=replace(
            fixture.target_chart,
            chart_id=collision_id,
        ),
        target_tube=replace(
            fixture.target_tube,
            chart_id=collision_id,
        ),
    )
    result = _check(collided)

    assert not result.certified
    assert (
        "carried_lc_entry_identifiers_match_and_are_unique"
        in result.missing_obligations
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "obligations",
        "source_endpoint_state_box",
        "relative_position_box",
        "canonical_lift_case",
        "patch_vertex_ids",
        "lifted_patch_boxes",
        "entry_time_interval",
        "timed_lifted_patch_boxes",
        "derived_edges",
        "gauge_result",
        "tested_assignments",
        "target_anchor",
        "tested_lift_max_gaps",
        "tested_containments",
        "selected_complement_index",
        "selected_assignment",
        "selected_transformed_timed_patch_boxes",
        "entry_rho_lower_bound",
        "source_chart_result",
        "target_chart_result",
        "mass_arithmetic_kernel_id",
    ),
)
def test_mutated_derived_snapshot_fields_never_certify(field_name: str):
    result = _check(_fixture())
    assert result.certified

    assert not replace(result, **{field_name: _AlwaysEqual()}).certified


def test_mutated_raw_and_nested_results_never_certify_via_python_equality():
    result = _check(_fixture())
    assert result.certified
    assert result.source_chart_result is not None
    assert result.source_tube_result is not None
    assert result.target_chart_result is not None
    assert result.target_tube_result is not None

    forged_transition = replace(
        result.raw_transition,
        transition_id=_AlwaysEqual(),
    )
    forged_source_result = replace(
        result.source_tube_result,
        checker_id=_AlwaysEqual(),
    )
    forged_source_chart_result = replace(
        result.source_chart_result,
        checker_id=_AlwaysEqual(),
    )
    forged_target_result = replace(
        result.target_tube_result,
        gronwall_error_bound=np.float64(
            result.target_tube_result.gronwall_error_bound
        ),
    )
    forged_obligation = replace(
        result.obligations[0],
        detail=_AlwaysEqual(),
    )
    embedded_patch_id = (
        _AlwaysEqual(),
    ) + result.patch_vertex_ids[1:]
    assert result.gauge_result is not None
    forged_edge = replace(
        result.derived_edges[0],
        overlap_id=_AlwaysEqual(),
    )
    forged_gauge = replace(
        result.gauge_result,
        checked_overlaps=(forged_edge,),
    )
    subclass_spoof = _ResultEqualitySpoof(
        **{
            field.name: getattr(result, field.name)
            for field in dataclass_fields(CarriedOrdinaryToPlanarLCEntryResult)
        }
    )

    assert not replace(result, raw_transition=forged_transition).certified
    assert not replace(
        result,
        source_tube_result=forged_source_result,
    ).certified
    assert not replace(
        result,
        source_chart_result=forged_source_chart_result,
    ).certified
    assert not replace(
        result,
        target_tube_result=forged_target_result,
    ).certified
    forged_target_kernel = replace(
        result.target_tube_result,
        mass_arithmetic_kernel_id="wrong-mass-kernel",
    )
    assert not replace(
        result,
        target_tube_result=forged_target_kernel,
    ).certified
    assert not replace(
        result,
        obligations=(forged_obligation,) + result.obligations[1:],
    ).certified
    assert not replace(result, patch_vertex_ids=embedded_patch_id).certified
    assert not replace(result, gauge_result=forged_gauge).certified
    assert not subclass_spoof.certified


def test_extreme_or_nonbinary64_mass_payload_fails_outward_obligation():
    fixture = _fixture()
    cases = (
        (float.fromhex("0x1.fffffffffffffp+1023"), 1.0, 1.0),
        (1, 1.0, 1.0),
    )
    for masses in cases:
        with np.errstate(over="ignore", invalid="ignore"):
            result = _check(
                replace(
                    fixture,
                    source_chart=replace(fixture.source_chart, masses=masses),
                    target_chart=replace(fixture.target_chart, masses=masses),
                )
            )
        assert not result.certified
        assert (
            "carried_lc_entry_outward_mass_arithmetic_certified"
            in result.missing_obligations
        )
