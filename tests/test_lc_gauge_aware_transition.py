from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.binary_chart import (
    planar_to_regularized_binary_collision_chart,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_checker import (
    GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult,
    ValidatedOrdinaryIVPChartCheckResult,
    check_gauge_aware_ordinary_to_planar_lc_enclosure_transition,
    check_ordinary_to_planar_lc_enclosure_transition,
    check_validated_ordinary_ivp_chart,
)
from three_body_symmetry.certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
    OrdinaryToPlanarLCEnclosureTransitionCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.series import construct_taylor_solution


class _AlwaysEqual:
    """Adversarial stand-in whose equality cannot establish trusted schema."""

    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class _TransitionFixture:
    source_binding: InitialValueProblemBindingCertificate
    source_tube: OrdinaryAposterioriTubeCertificate
    source_chart: OrdinaryTaylorChartCertificate
    target_chart: PlanarLeviCivitaBinaryChartCertificate
    source_validation: ValidatedOrdinaryIVPChartCheckResult
    target_tube: PlanarLCAposterioriTubeCertificate
    transition: OrdinaryToPlanarLCEnclosureTransitionCertificate


def _negate_vector_coefficients(
    coefficients: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(-value for value in row) for row in coefficients)


@lru_cache(maxsize=None)
def _fixture(
    relative_x: float = -1.0,
    antipodal_target: bool = False,
    target_initial_error: float = 2.0e-2,
) -> _TransitionFixture:
    """Build a small exact entry problem with a nonzero source tube width."""

    case = "negative" if relative_x < 0.0 else "positive"
    gauge = "antipodal" if antipodal_target else "principal"
    tag = f"{case}-{gauge}-{target_initial_error:.3g}"
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [0.0, 0.0],
            [relative_x, 0.0],
            [3.0, 1.0],
        ]
    )
    velocities = np.zeros((3, 2))
    parameter_interval = (0.0, 1.0e-8)

    ordinary_solution = construct_taylor_solution(
        positions, velocities, masses, order=10
    )
    source_chart = ordinary_taylor_chart_certificate_from_solution(
        ordinary_solution,
        certificate_id=f"gauge-aware-source-certificate:{tag}",
        chart_id=f"gauge-aware-source:{tag}",
        parameter_interval=parameter_interval,
        physical_time_interval=parameter_interval,
        coefficient_tolerance=1.0e-8,
        residual_tolerance=10.0,
        tail_bound=1.0e-9,
        sample_count=5,
    )
    binding = InitialValueProblemBindingCertificate(
        binding_id=f"gauge-aware-binding:{tag}",
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
        tube_id=f"gauge-aware-source-tube:{tag}",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-4,
        tube_radius=5.0e-4,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    source_validation = check_validated_ordinary_ivp_chart(
        binding, source_tube, source_chart
    )

    initial_lift = planar_to_regularized_binary_collision_chart(
        positions, velocities, masses, pair=(0, 1)
    )
    lifted_solution = construct_regularized_binary_taylor_solution(
        initial_lift, order=10
    )
    target_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        lifted_solution,
        certificate_id=f"gauge-aware-target-certificate:{tag}",
        chart_id=f"gauge-aware-target:{tag}",
        parameter_interval=parameter_interval,
        coefficient_tolerance=1.0e-8,
        regularized_residual_tolerance=1.0e-6,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-9,
        sample_count=5,
        projection_rho_lower_bound=1.0e-8,
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
        tube_id=f"gauge-aware-target-tube:{tag}",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=target_initial_error,
        tube_radius=5.0e-2,
        max_defect_bound=10.0,
        max_lipschitz_bound=1.0e6,
    )
    transition = OrdinaryToPlanarLCEnclosureTransitionCertificate(
        transition_id=f"gauge-aware-transition:{tag}",
        source_chart_id=source_chart.chart_id,
        target_chart_id=target_chart.chart_id,
        source_parameter=0.0,
        target_parameter=0.0,
        handoff_time=0.0,
        max_time_gap=0.0,
    )
    return _TransitionFixture(
        source_binding=binding,
        source_tube=source_tube,
        source_chart=source_chart,
        target_chart=target_chart,
        source_validation=source_validation,
        target_tube=target_tube,
        transition=transition,
    )


def _gauge_check(
    fixture: _TransitionFixture,
) -> GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult:
    return check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        fixture.transition,
        fixture.source_binding,
        fixture.source_tube,
        fixture.source_chart,
        fixture.target_chart,
        fixture.target_tube,
    )


def _raw_gauge_check(
    fixture: _TransitionFixture,
    *,
    transition: OrdinaryToPlanarLCEnclosureTransitionCertificate | None = None,
    source_binding: InitialValueProblemBindingCertificate | None = None,
    source_tube: OrdinaryAposterioriTubeCertificate | None = None,
    source_chart: OrdinaryTaylorChartCertificate | None = None,
    target_chart: PlanarLeviCivitaBinaryChartCertificate | None = None,
    target_tube: PlanarLCAposterioriTubeCertificate | None = None,
) -> GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult:
    return check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        fixture.transition if transition is None else transition,
        fixture.source_binding if source_binding is None else source_binding,
        fixture.source_tube if source_tube is None else source_tube,
        fixture.source_chart if source_chart is None else source_chart,
        fixture.target_chart if target_chart is None else target_chart,
        fixture.target_tube if target_tube is None else target_tube,
    )


def test_negative_axis_two_patch_entry_requires_global_gauge_alignment():
    fixture = _fixture()
    legacy = check_ordinary_to_planar_lc_enclosure_transition(
        fixture.transition,
        fixture.source_chart,
        fixture.target_chart,
        fixture.source_validation,
        fixture.target_tube,
    )
    result = _gauge_check(fixture)

    assert fixture.source_validation.certified
    assert legacy.lift_branch_count == 2
    assert legacy.max_lift_box_gap > fixture.target_tube.initial_error_bound
    assert not legacy.certified
    assert "ordinary_to_lc_target_initial_error_contains_lift_atlas" in (
        legacy.missing_obligations
    )

    assert type(result) is GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult
    assert result.certified
    assert result.canonical_lift_case == "strict_negative_cut_two_patch"
    assert len(result.patch_vertex_ids) == 2
    assert len(result.raw_patch_boxes) == 2
    assert len(result.derived_edges) == 1
    assert result.derived_edges[0].parity == 1
    assert result.derived_edges[0].source_chart_id == result.patch_vertex_ids[0]
    assert result.derived_edges[0].target_chart_id == result.patch_vertex_ids[1]
    assert result.gauge_result is not None and result.gauge_result.certified
    assert result.tested_containments == (True, False)
    assert result.selected_complement_index == 0
    assert result.tested_lift_max_gaps[0] <= fixture.target_tube.initial_error_bound
    assert result.tested_lift_max_gaps[1] > fixture.target_tube.initial_error_bound


def test_antipodal_target_selects_the_other_global_complement():
    result = _gauge_check(_fixture(antipodal_target=True))

    assert result.certified
    assert result.canonical_lift_case == "strict_negative_cut_two_patch"
    assert result.tested_containments == (False, True)
    assert result.selected_complement_index == 1
    assert result.selected_assignment == result.tested_assignments[1]


def test_entry_rejects_when_neither_global_complement_fits_target_ball():
    fixture = _fixture(target_initial_error=1.0e-2)
    result = _gauge_check(fixture)

    assert result.target_tube_result.certified
    assert result.tested_containments == (False, False)
    assert result.selected_complement_index == -1
    assert not result.certified
    assert (
        "gauge_aware_ordinary_to_lc_one_global_complement_contains_all_lifts"
        in result.missing_obligations
    )


@pytest.mark.parametrize(
    ("antipodal_target", "expected_containments", "expected_complement"),
    [
        (False, (True, False), 0),
        (True, (False, True), 1),
    ],
)
def test_right_half_singleton_and_antipodal_singleton_select_one_gauge(
    antipodal_target: bool,
    expected_containments: tuple[bool, bool],
    expected_complement: int,
):
    result = _gauge_check(
        _fixture(relative_x=1.0, antipodal_target=antipodal_target)
    )

    assert result.certified
    assert result.canonical_lift_case == "right_half_singleton"
    assert len(result.raw_patch_boxes) == 1
    assert result.derived_edges == ()
    assert result.tested_containments == expected_containments
    assert result.selected_complement_index == expected_complement


def test_cross_chart_source_validation_substitution_rejects():
    fixture = _fixture()
    substituted_source = replace(
        fixture.source_chart,
        certificate_id="foreign-source-certificate",
        chart_id="foreign-source-chart",
    )
    substituted_transition = replace(
        fixture.transition,
        source_chart_id=substituted_source.chart_id,
    )

    result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        substituted_transition,
        fixture.source_binding,
        fixture.source_tube,
        substituted_source,
        fixture.target_chart,
        fixture.target_tube,
    )

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_source_exact_enclosure_certified" in (
        result.missing_obligations
    )
    assert result.source_state_box == ()


def test_physical_time_must_fit_inside_the_14d_target_radius():
    fixture = _fixture()
    time_shift = 3.0e-2
    shifted_target = replace(
        fixture.target_chart,
        physical_time_coefficients=(
            fixture.target_chart.physical_time_coefficients[0] + time_shift,
        )
        + fixture.target_chart.physical_time_coefficients[1:],
    )
    shifted_transition = replace(
        fixture.transition,
        handoff_time=time_shift,
        max_time_gap=time_shift,
    )

    result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        shifted_transition,
        fixture.source_binding,
        fixture.source_tube,
        fixture.source_chart,
        shifted_target,
        fixture.target_tube,
    )

    assert result.target_tube_result.certified
    assert result.exact_declared_time_gap <= time_shift
    assert result.exact_source_target_time_gap > fixture.target_tube.initial_error_bound
    assert not result.certified
    assert (
        "gauge_aware_ordinary_to_lc_target_initial_error_contains_physical_time"
        in result.missing_obligations
    )


def test_replace_tampering_cannot_promote_or_preserve_a_certified_snapshot():
    result = _gauge_check(_fixture())
    assert result.certified

    tampered_obligation = replace(result.obligations[0], certified=False)
    assert not replace(
        result,
        obligations=(tampered_obligation,) + result.obligations[1:],
    ).certified
    assert not replace(
        result,
        selected_assignment=result.tested_assignments[1],
    ).certified
    assert not replace(result, selected_complement_index=1).certified


def test_tampered_retained_source_validation_is_recomputed_and_rejected():
    result = _gauge_check(_fixture())
    assert result.certified

    tampered_tube_result = replace(
        result.source_validation.tube_result,
        gronwall_error_bound=(
            result.source_validation.tube_result.gronwall_error_bound + 1.0e-6
        ),
    )
    tampered_validation = replace(
        result.source_validation,
        tube_result=tampered_tube_result,
    )

    assert not replace(result, source_validation=tampered_validation).certified


def test_ndarray_raw_identifier_and_selected_index_fail_closed():
    result = _gauge_check(_fixture())
    assert result.certified

    ndarray_identifier = replace(
        result.raw_transition_certificate,
        transition_id=np.array([result.transition_id], dtype=object),
    )
    tampered_results = (
        replace(
            result,
            raw_transition_certificate=ndarray_identifier,
        ),
        replace(
            result,
            selected_complement_index=np.array(
                [result.selected_complement_index], dtype=int
            ),
        ),
    )

    assert all(candidate.certified is False for candidate in tampered_results)


@pytest.mark.parametrize(
    "field_name",
    (
        "obligations",
        "source_state_box",
        "relative_position_box",
        "canonical_lift_case",
        "patch_vertex_ids",
        "raw_patch_boxes",
        "derived_edges",
        "gauge_result",
        "tested_assignments",
        "tested_lift_max_gaps",
        "tested_containments",
        "selected_complement_index",
        "selected_assignment",
        "selected_transformed_patch_boxes",
        "exact_declared_time_gap",
        "exact_source_target_time_gap",
        "entry_lift_rho_lower_bound",
    ),
)
def test_always_equal_cannot_replace_any_derived_snapshot_field(
    field_name: str,
):
    result = _gauge_check(_fixture())
    assert result.certified

    assert not replace(result, **{field_name: _AlwaysEqual()}).certified


def test_fake_nested_ledger_and_nonbuiltin_result_number_reject():
    result = _gauge_check(_fixture())
    assert result.certified

    fake_ledger = (
        SimpleNamespace(
            obligation="attribute-compatible-fake",
            certified=True,
            detail="not an exact obligation row",
        ),
    )
    fake_validation = replace(
        result.source_validation,
        obligations=fake_ledger,
    )
    nonbuiltin_numeric_result = replace(
        result.target_tube_result,
        defect_bound=np.float64(result.target_tube_result.defect_bound),
    )

    assert not replace(result, source_validation=fake_validation).certified
    assert not replace(
        result,
        target_tube_result=nonbuiltin_numeric_result,
    ).certified


@pytest.mark.parametrize(
    "raw_input_name",
    (
        "transition",
        "source_binding",
        "source_tube",
        "source_chart",
        "target_chart",
        "target_tube",
    ),
)
def test_nonstring_source_provenance_rejects_before_normalization(
    raw_input_name: str,
):
    fixture = _fixture()
    raw_input = getattr(fixture, raw_input_name)

    result = _raw_gauge_check(
        fixture,
        **{raw_input_name: replace(raw_input, source=_AlwaysEqual())},
    )

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )
    assert result.source_state_box == ()


@pytest.mark.parametrize("field_name", ("positions", "velocities"))
def test_ndarray_binding_state_rejects_before_normalization(field_name: str):
    fixture = _fixture()
    ndarray_binding = replace(
        fixture.source_binding,
        **{
            field_name: np.asarray(
                getattr(fixture.source_binding, field_name), dtype=float
            )
        },
    )

    result = _raw_gauge_check(fixture, source_binding=ndarray_binding)

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "anchor_parameter",
        "initial_error_bound",
        "tube_radius",
        "max_defect_bound",
        "max_lipschitz_bound",
    ),
)
def test_numpy_source_tube_scalars_reject_before_normalization(field_name: str):
    fixture = _fixture()
    numpy_scalar_tube = replace(
        fixture.source_tube,
        **{field_name: np.float64(getattr(fixture.source_tube, field_name))},
    )

    result = _raw_gauge_check(fixture, source_tube=numpy_scalar_tube)

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "field_name",
    ("parameter_interval", "physical_time_interval"),
)
def test_list_source_intervals_reject_before_normalization(field_name: str):
    fixture = _fixture()
    list_interval_chart = replace(
        fixture.source_chart,
        **{field_name: list(getattr(fixture.source_chart, field_name))},
    )

    result = _raw_gauge_check(fixture, source_chart=list_interval_chart)

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "chart_type",
        "physical_time_interval",
        "coefficient_tolerance",
        "regularized_residual_tolerance",
        "projected_residual_tolerance",
        "tail_bound",
        "sample_count",
        "projection_rho_lower_bound",
    ),
)
def test_always_equal_target_chart_metadata_rejects_before_normalization(
    field_name: str,
):
    fixture = _fixture()
    malformed_target = replace(
        fixture.target_chart,
        **{field_name: _AlwaysEqual()},
    )

    result = _raw_gauge_check(fixture, target_chart=malformed_target)

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "masses",
        "pair",
        "z_coefficients",
        "z_velocity_coefficients",
        "pair_energy_coefficients",
        "binary_center_coefficients",
        "binary_center_velocity_coefficients",
        "third_offset_coefficients",
        "third_offset_velocity_coefficients",
        "physical_time_coefficients",
        "parameter_interval",
    ),
)
def test_list_target_chart_tuples_reject_before_normalization(field_name: str):
    fixture = _fixture()
    list_field_target = replace(
        fixture.target_chart,
        **{field_name: list(getattr(fixture.target_chart, field_name))},
    )

    result = _raw_gauge_check(fixture, target_chart=list_field_target)

    assert not result.certified
    assert "gauge_aware_ordinary_to_lc_exact_input_types" in (
        result.missing_obligations
    )


def test_always_equal_outer_checker_identifier_rejects():
    result = _gauge_check(_fixture())
    assert result.certified

    assert not replace(result, checker_id=_AlwaysEqual()).certified


@pytest.mark.parametrize("field_name", ("obligation", "detail"))
def test_always_equal_nested_gauge_obligation_strings_reject(field_name: str):
    result = _gauge_check(_fixture())
    assert result.certified
    assert result.gauge_result is not None

    tampered_obligation = replace(
        result.gauge_result.obligations[0],
        **{field_name: _AlwaysEqual()},
    )
    tampered_gauge_result = replace(
        result.gauge_result,
        obligations=(tampered_obligation,) + result.gauge_result.obligations[1:],
    )

    assert not replace(result, gauge_result=tampered_gauge_result).certified


def test_source_time_uses_binding_clock_not_metadata_interval_slope():
    fixture = _fixture()
    source_parameter = fixture.source_chart.parameter_interval[1]
    warped_source = replace(
        fixture.source_chart,
        physical_time_interval=(0.0, 1.0),
        coefficient_tolerance=2.0,
    )
    tolerant_binding = replace(
        fixture.source_binding,
        time_tolerance=2.0,
    )
    endpoint_transition = replace(
        fixture.transition,
        source_parameter=source_parameter,
        max_time_gap=2.0e-8,
    )

    result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        endpoint_transition,
        tolerant_binding,
        fixture.source_tube,
        warped_source,
        fixture.target_chart,
        fixture.target_tube,
    )
    binding_time = (
        Fraction.from_float(tolerant_binding.initial_time)
        + Fraction.from_float(source_parameter)
        - Fraction.from_float(tolerant_binding.chart_parameter)
    )
    target_time = Fraction.from_float(
        fixture.target_chart.physical_time_coefficients[0]
    )

    assert result.source_validation.certified
    assert result.certified
    assert result.exact_source_target_time_gap == abs(binding_time - target_time)
    assert result.exact_source_target_time_gap < Fraction.from_float(1.0)
    assert warped_source.physical_time_interval[1] == 1.0


def test_collision_containing_candidate_source_tube_rejects_before_grammar():
    fixture = _fixture()
    collision_containing_tube = replace(
        fixture.source_tube,
        initial_error_bound=6.0e-1,
        tube_radius=7.0e-1,
    )

    result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        fixture.transition,
        fixture.source_binding,
        collision_containing_tube,
        fixture.source_chart,
        fixture.target_chart,
        fixture.target_tube,
    )

    assert 2.0 * collision_containing_tube.initial_error_bound > 1.0
    assert not result.source_validation.tube_result.certified
    assert not result.certified
    assert result.source_state_box == ()
    assert result.canonical_lift_case == ""


def test_malformed_exact_class_evidence_rejects_without_branch_inference():
    fixture = _fixture()
    malformed_transition = replace(fixture.transition, source_parameter=True)

    result = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
        malformed_transition,
        fixture.source_binding,
        fixture.source_tube,
        fixture.source_chart,
        fixture.target_chart,
        fixture.target_tube,
    )

    assert not result.certified
    assert result.canonical_lift_case == ""
    assert "gauge_aware_ordinary_to_lc_parameters_inside_and_anchor_matches" in (
        result.missing_obligations
    )
    assert "gauge_aware_ordinary_to_lc_canonical_branch_grammar" in (
        result.missing_obligations
    )
