from dataclasses import replace
from math import isinf

from three_body_symmetry.certificate_checker import (
    CertificateCheckObligation,
    CertificateCheckResult,
    TransitionCheckResult,
    check_chart_chain,
    check_ordinary_chart_transition,
    check_planar_levi_civita_transition,
)
from three_body_symmetry.certificate_language import (
    ChartChainCertificate,
    OrdinaryChartTransitionCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    PlanarLeviCivitaTransitionCertificate,
)


MASS_TRIPLE = (1.0, 1.0, 1.2)
PLANAR_POSITIONS = ((-0.5, 0.0), (0.5, 0.0), (2.0, 0.0))
PLANAR_VELOCITIES = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def _ordinary_chart(
    chart_id: str,
    physical_time_interval: tuple[float, float],
    *,
    masses: tuple[float, float, float] = MASS_TRIPLE,
    dimension: int = 2,
) -> OrdinaryTaylorChartCertificate:
    positions = tuple(
        tuple((*position, *([0.0] * (dimension - 2))))
        for position in PLANAR_POSITIONS
    )
    velocities = tuple(
        tuple((*velocity, *([0.0] * (dimension - 2))))
        for velocity in PLANAR_VELOCITIES
    )
    return OrdinaryTaylorChartCertificate(
        certificate_id=f"certificate:{chart_id}",
        chart_id=chart_id,
        chart_type="ordinary_taylor",
        masses=masses,
        position_coefficients=(positions,),
        velocity_coefficients=(velocities,),
        parameter_interval=(0.0, 1.0),
        physical_time_interval=physical_time_interval,
        coefficient_tolerance=0.0,
        residual_tolerance=0.0,
        tail_bound=0.0,
    )


def _ordinary_transition() -> tuple[
    OrdinaryTaylorChartCertificate,
    OrdinaryTaylorChartCertificate,
    OrdinaryChartTransitionCertificate,
]:
    source = _ordinary_chart("ordinary:source", (0.0, 1.0))
    target = _ordinary_chart("ordinary:target", (1.0, 2.0))
    transition = OrdinaryChartTransitionCertificate(
        transition_id="transition:ordinary",
        source_chart_id=source.chart_id,
        target_chart_id=target.chart_id,
        transition_type="ordinary_overlap_handoff",
        handoff_time=1.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    return source, target, transition


def _lc_chart(
    chart_id: str = "lc:target",
    *,
    pair: tuple[int, int] = (0, 1),
    masses: tuple[float, float, float] = MASS_TRIPLE,
    physical_time_interval: tuple[float, float] = (0.0, 1.0),
) -> PlanarLeviCivitaBinaryChartCertificate:
    return PlanarLeviCivitaBinaryChartCertificate(
        certificate_id=f"certificate:{chart_id}",
        chart_id=chart_id,
        chart_type="planar_levi_civita_binary",
        masses=masses,
        pair=pair,
        z_coefficients=((1.0, 0.0),),
        z_velocity_coefficients=((0.0, 0.0),),
        pair_energy_coefficients=(0.0,),
        binary_center_coefficients=((0.0, 0.0),),
        binary_center_velocity_coefficients=((0.0, 0.0),),
        third_offset_coefficients=((2.0, 0.0),),
        third_offset_velocity_coefficients=((0.0, 0.0),),
        physical_time_coefficients=(0.0, 1.0),
        parameter_interval=(0.0, 1.0),
        physical_time_interval=physical_time_interval,
        coefficient_tolerance=0.0,
        regularized_residual_tolerance=0.0,
        projected_residual_tolerance=0.0,
        tail_bound=0.0,
    )


def _ordinary_to_lc_transition() -> tuple[
    OrdinaryTaylorChartCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    PlanarLeviCivitaTransitionCertificate,
]:
    ordinary = _ordinary_chart("ordinary:lc-source", (0.0, 1.0))
    lc = _lc_chart()
    transition = PlanarLeviCivitaTransitionCertificate(
        transition_id="transition:ordinary-to-lc",
        source_chart_id=ordinary.chart_id,
        target_chart_id=lc.chart_id,
        transition_type="ordinary_to_binary_event_handoff",
        handoff_time=0.0,
        source_parameter=0.0,
        target_parameter=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
        physical_time_tolerance=0.0,
    )
    return ordinary, lc, transition


def _checked_chart_result(chart: object) -> CertificateCheckResult:
    return CertificateCheckResult(
        certificate_id=chart.certificate_id,
        certificate_type=chart.chart_type,
        checker_id="test_checked_chart",
        obligations=(
            CertificateCheckObligation("test_chart_checked", True, "fixture"),
        ),
        max_coefficient_residual=0.0,
        max_sampled_newton_residual=0.0,
    )


def _checked_transition_result(transition_id: str) -> TransitionCheckResult:
    return TransitionCheckResult(
        transition_id=transition_id,
        transition_type="test_transition",
        checker_id="test_checked_transition",
        obligations=(
            CertificateCheckObligation("test_transition_checked", True, "fixture"),
        ),
        max_position_gap=0.0,
        max_velocity_gap=0.0,
    )


def test_ordinary_transition_rejects_tampered_mass_before_state_evaluation():
    source, target, transition = _ordinary_transition()
    assert check_ordinary_chart_transition(transition, (source, target)).certified

    tampered_target = replace(target, masses=(1.0, 1.0, 9.0))
    assert tampered_target.position_coefficients == target.position_coefficients
    assert tampered_target.velocity_coefficients == target.velocity_coefficients

    result = check_ordinary_chart_transition(
        transition,
        (source, tampered_target),
    )

    assert not result.certified
    assert "transition_common_problem_identity" in result.missing_obligations
    assert "transition_state_continuity" in result.missing_obligations
    assert "transition_exact_rational_state_continuity" in result.missing_obligations
    assert isinf(result.max_position_gap)
    assert isinf(result.max_velocity_gap)


def test_planar_lc_transition_rejects_unused_tampered_mass_before_evaluation():
    ordinary, lc, transition = _ordinary_to_lc_transition()
    assert check_planar_levi_civita_transition(transition, (ordinary, lc)).certified

    # The third mass is unused by this pair's point projection, so the same
    # serialized polynomials still project to exactly the same handoff state.
    tampered_lc = replace(lc, masses=(1.0, 1.0, 9.0))
    assert tampered_lc.z_coefficients == lc.z_coefficients
    assert tampered_lc.third_offset_coefficients == lc.third_offset_coefficients

    result = check_planar_levi_civita_transition(
        transition,
        (ordinary, tampered_lc),
    )

    assert not result.certified
    assert "transition_common_problem_identity" in result.missing_obligations
    assert "transition_physical_time_match" in result.missing_obligations
    assert "transition_state_continuity" in result.missing_obligations
    assert "transition_exact_rational_state_continuity" in result.missing_obligations
    assert isinf(result.max_position_gap)
    assert isinf(result.max_velocity_gap)


def test_planar_lc_transition_rejects_incompatible_ordinary_dimension():
    ordinary, lc, transition = _ordinary_to_lc_transition()
    spatial_ordinary = _ordinary_chart(
        ordinary.chart_id,
        ordinary.physical_time_interval,
        dimension=3,
    )

    result = check_planar_levi_civita_transition(
        transition,
        (spatial_ordinary, lc),
    )

    assert not result.certified
    assert "transition_common_problem_identity" in result.missing_obligations
    assert "transition_state_continuity" in result.missing_obligations
    assert isinf(result.max_position_gap)
    assert isinf(result.max_velocity_gap)


def test_chart_chain_rejects_tampered_mass_despite_prechecked_ids():
    source, target, transition = _ordinary_transition()
    checked_charts = tuple(_checked_chart_result(chart) for chart in (source, target))
    checked_transition = check_ordinary_chart_transition(
        transition,
        (source, target),
    )
    chain = ChartChainCertificate(
        certificate_id="certificate:chain",
        chain_id="chain:ordinary",
        chain_type="finite_time_chart_chain",
        chart_ids=(source.chart_id, target.chart_id),
        transition_ids=(transition.transition_id,),
        target_physical_time_interval=(0.0, 2.0),
    )
    assert check_chart_chain(
        chain,
        (source, target),
        (transition,),
        checked_charts,
        (checked_transition,),
    ).certified

    tampered_target = replace(target, masses=(1.0, 1.0, 9.0))
    result = check_chart_chain(
        chain,
        (source, tampered_target),
        (transition,),
        checked_charts,
        (checked_transition,),
    )

    assert not result.certified
    assert "chart_chain_common_problem_identity" in result.missing_obligations
    details = {obligation.obligation: obligation for obligation in result.obligations}
    assert details["chart_chain_charts_independently_checked"].certified
    assert details["chart_chain_transitions_independently_checked"].certified


def test_chart_chain_problem_identity_does_not_bind_lc_pair_across_ordinary_bridge():
    left_lc = _lc_chart("lc:left", pair=(0, 1), physical_time_interval=(0.0, 1.0))
    ordinary = _ordinary_chart("ordinary:bridge", (1.0, 2.0))
    right_lc = _lc_chart("lc:right", pair=(1, 2), physical_time_interval=(2.0, 3.0))
    transitions = (
        PlanarLeviCivitaTransitionCertificate(
            transition_id="transition:left-to-ordinary",
            source_chart_id=left_lc.chart_id,
            target_chart_id=ordinary.chart_id,
            transition_type="binary_to_ordinary_event_handoff",
            handoff_time=1.0,
            source_parameter=1.0,
            target_parameter=0.0,
            position_tolerance=0.0,
            velocity_tolerance=0.0,
            physical_time_tolerance=0.0,
        ),
        PlanarLeviCivitaTransitionCertificate(
            transition_id="transition:ordinary-to-right",
            source_chart_id=ordinary.chart_id,
            target_chart_id=right_lc.chart_id,
            transition_type="ordinary_to_binary_event_handoff",
            handoff_time=2.0,
            source_parameter=1.0,
            target_parameter=0.0,
            position_tolerance=0.0,
            velocity_tolerance=0.0,
            physical_time_tolerance=0.0,
        ),
    )
    charts = (left_lc, ordinary, right_lc)
    chain = ChartChainCertificate(
        certificate_id="certificate:pair-switch-chain",
        chain_id="chain:pair-switch",
        chain_type="regularized_atlas_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(item.transition_id for item in transitions),
        target_physical_time_interval=(0.0, 3.0),
    )

    result = check_chart_chain(
        chain,
        charts,
        transitions,
        tuple(_checked_chart_result(chart) for chart in charts),
        tuple(_checked_transition_result(item.transition_id) for item in transitions),
    )
    details = {obligation.obligation: obligation for obligation in result.obligations}

    assert result.certified
    assert details["chart_chain_common_problem_identity"].certified
