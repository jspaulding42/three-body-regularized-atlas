from dataclasses import replace

import numpy as np
import pytest

from three_body_symmetry.binary_chart import RegularizedBinaryCollisionChartState
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_checker import (
    CertificateCheckObligation,
    PlanarLCExactGaugeAtlasCheckResult,
    check_planar_lc_exact_gauge_atlas,
    check_planar_lc_exact_overlap_anchor,
)
from three_body_symmetry.certificate_language import (
    PlanarLCAposterioriTubeCertificate,
    PlanarLCExactGaugeAtlasCertificate,
    PlanarLCExactOverlapAnchorCertificate,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.lc_gauge_gluing import (
    PlanarLCGaugeGluingCertificate,
    PlanarLCGaugeOverlapEdge,
    check_planar_lc_gauge_gluing,
)


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


def _base_collision_chart():
    masses = np.array([1.0, 1.0, 1.2])
    initial = RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([1.0, 0.0]),
        pair_energy=0.0,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.01, -0.02]),
        third_offset=np.array([2.0, 1.0]),
        third_offset_velocity=np.array([-0.01, 0.015]),
    )
    solution = construct_regularized_binary_taylor_solution(initial, order=12)
    return planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="exact-gauge-base-certificate",
        chart_id="exact-gauge-base-chart",
        parameter_interval=(-1.0e-4, 1.0e-4),
        coefficient_tolerance=1.0e-10,
        regularized_residual_tolerance=1.0e-7,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-9,
        sample_count=8,
        projection_rho_lower_bound=1.0e-12,
    )


def _signed_chart(label: str, parity: int):
    base = _base_collision_chart()

    def signed_vectors(values):
        if parity == 0:
            return values
        return tuple(tuple(-component for component in row) for row in values)

    return replace(
        base,
        certificate_id=f"exact-gauge-chart-certificate:{label}",
        chart_id=f"chart:{label}",
        z_coefficients=signed_vectors(base.z_coefficients),
        z_velocity_coefficients=signed_vectors(base.z_velocity_coefficients),
    )


def _tube(chart, label: str, *, initial_error: float = 0.0):
    return PlanarLCAposterioriTubeCertificate(
        tube_id=f"tube:{label}",
        chart_id=chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=initial_error,
        tube_radius=1.0e-6,
        max_defect_bound=1.0e-7,
        max_lipschitz_bound=2.0,
        require_pair_energy_constraint=True,
    )


def _overlap(overlap_id: str, source_chart, source_tube, target_chart, target_tube):
    return PlanarLCExactOverlapAnchorCertificate(
        overlap_id=overlap_id,
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=target_chart.chart_id,
        target_tube_id=target_tube.tube_id,
        source_anchor_parameter=source_tube.anchor_parameter,
        target_anchor_parameter=target_tube.anchor_parameter,
    )


def _triangle_bundle():
    charts = (
        _signed_chart("a", 0),
        _signed_chart("b", 1),
        _signed_chart("c", 0),
    )
    tubes = tuple(_tube(chart, label) for chart, label in zip(charts, "abc"))
    overlaps = (
        _overlap("overlap:ab", charts[0], tubes[0], charts[1], tubes[1]),
        _overlap("overlap:bc", charts[1], tubes[1], charts[2], tubes[2]),
        _overlap("overlap:ac", charts[0], tubes[0], charts[2], tubes[2]),
    )
    certificate = PlanarLCExactGaugeAtlasCertificate(
        atlas_id="exact-gauge-atlas:triangle",
        chart_ids=tuple(chart.chart_id for chart in charts),
        tube_ids=tuple(tube.tube_id for tube in tubes),
        overlap_ids=tuple(overlap.overlap_id for overlap in overlaps),
    )
    return certificate, charts, tubes, overlaps


def test_exact_gauge_atlas_recomputes_triangle_parities_and_primal_assignment():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    round_trip = PlanarLCExactGaugeAtlasCertificate.from_dict(
        certificate.to_dict()
    )

    result = check_planar_lc_exact_gauge_atlas(
        round_trip, charts, tubes, overlaps
    )

    assert type(result) is PlanarLCExactGaugeAtlasCheckResult
    assert result.certified
    assert result.lifted_atlas_certified
    assert result.constrained_newtonian_atlas_certified
    assert result.all_overlaps_two_sided
    assert tuple(edge.parity for edge in result.derived_edges) == (1, 1, 0)
    assert dict(result.gauge_assignment) == {
        "chart:a": 0,
        "chart:b": 1,
        "chart:c": 0,
    }
    assert result.gauge_result is not None
    assert result.gauge_result.checked_overlaps == result.derived_edges
    assert result.raw_atlas_certificate == round_trip
    assert result.checked_charts == charts
    assert result.checked_tubes == tubes
    assert result.checked_overlap_certificates == overlaps
    assert all(translation == (0, 0, 0) for translation in result.anchor_parameter_translations)


def test_exact_gauge_atlas_accepts_trivial_constrained_vertex():
    chart = _signed_chart("only", 0)
    tube = _tube(chart, "only")
    certificate = PlanarLCExactGaugeAtlasCertificate(
        "exact-gauge-atlas:single",
        (chart.chart_id,),
        (tube.tube_id,),
        (),
    )

    result = check_planar_lc_exact_gauge_atlas(
        certificate, (chart,), (tube,), ()
    )

    assert result.certified
    assert result.constrained_newtonian_atlas_certified
    assert result.all_overlaps_two_sided  # vacuous diagnostic
    assert result.gauge_assignment == ((chart.chart_id, 0),)


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "tube-order"))
def test_exact_gauge_atlas_rejects_duplicate_missing_or_misordered_manifest_ids(
    mutation,
):
    certificate, charts, tubes, overlaps = _triangle_bundle()
    if mutation == "duplicate":
        certificate = replace(
            certificate,
            chart_ids=(certificate.chart_ids[0],) * len(certificate.chart_ids),
        )
    elif mutation == "missing":
        certificate = replace(certificate, overlap_ids=certificate.overlap_ids[:-1])
    else:
        certificate = replace(
            certificate,
            tube_ids=(certificate.tube_ids[1], certificate.tube_ids[0], certificate.tube_ids[2]),
        )

    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )

    assert not result.certified
    assert result.gauge_result is None


def test_exact_gauge_atlas_rejects_positive_error_vertex_before_graph():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    tubes = (tubes[0], replace(tubes[1], initial_error_bound=1.0e-12), tubes[2])

    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )

    assert not result.certified
    assert result.gauge_result is None
    assert "planar_lc_exact_gauge_atlas_vertex_tubes_zero_error_certified" in (
        result.missing_obligations
    )


def test_exact_gauge_atlas_never_consumes_failed_overlap_candidate_edge():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    # The exact anchor components still have an antipodal relation, but the
    # declared source parameter no longer binds the raw source tube anchor.
    overlaps = (
        replace(overlaps[0], source_anchor_parameter=1.0e-5),
        overlaps[1],
        overlaps[2],
    )

    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )

    assert not result.certified
    assert result.gauge_result is None
    assert "overlap:ab" not in {
        edge.overlap_id for edge in result.derived_edges
    }
    assert result.overlap_results[0].derived_edge is None
    assert {edge.overlap_id for edge in result.derived_edges} == {
        "overlap:bc",
        "overlap:ac",
    }


def test_exact_gauge_atlas_rejects_prebuilt_results_and_misbound_raw_overlap():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    prebuilt = check_planar_lc_exact_overlap_anchor(
        overlaps[0], tubes[0], charts[0], tubes[1], charts[1]
    )
    with pytest.raises(TypeError):
        check_planar_lc_exact_gauge_atlas(
            certificate,
            charts,
            tubes,
            (prebuilt, overlaps[1], overlaps[2]),  # type: ignore[arg-type]
        )

    misbound = (
        replace(overlaps[0], source_tube_id=tubes[2].tube_id),
        overlaps[1],
        overlaps[2],
    )
    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, misbound
    )
    assert not result.certified
    assert result.gauge_result is None
    assert result.overlap_results == ()


def test_exact_gauge_atlas_result_tampering_cannot_certify():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )
    assert result.certified

    assert not replace(
        result,
        obligations=(CertificateCheckObligation("fake", True, "forged"),),
    ).certified
    assert not replace(
        result,
        raw_atlas_certificate=replace(
            certificate,
            atlas_id="forged-exact-gauge-atlas",
        ),
    ).certified

    bad_edge = replace(result.derived_edges[0], parity=0)
    assert not replace(
        result,
        derived_edges=(bad_edge,) + result.derived_edges[1:],
    ).certified
    assert not replace(result, checked_tubes=tuple(reversed(result.checked_tubes))).certified
    assert not replace(
        result,
        anchor_parameter_translations=((0, 0, 1),)
        + result.anchor_parameter_translations[1:],
    ).certified
    assert not replace(result, gauge_result=None).certified

    forged_overlap = replace(
        result.overlap_results[0],
        derived_edge=PlanarLCGaugeOverlapEdge(
            "overlap:ab", "chart:a", "chart:b", 0
        ),
    )
    assert not replace(
        result,
        overlap_results=(forged_overlap,) + result.overlap_results[1:],
    ).certified


def test_exact_gauge_atlas_snapshot_rejects_equality_spoofed_metadata():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )
    assert result.certified

    assert not replace(result, checker_id=_AlwaysEqual()).certified
    for field_name in ("obligation", "detail"):
        tampered_obligation = replace(
            result.obligations[0],
            **{field_name: _AlwaysEqual()},
        )
        assert not replace(
            result,
            obligations=(tampered_obligation,) + result.obligations[1:],
        ).certified


def test_exact_gauge_atlas_rejects_coordinated_anchor_edge_and_graph_forgery():
    certificate, charts, tubes, overlaps = _triangle_bundle()
    result = check_planar_lc_exact_gauge_atlas(
        certificate, charts, tubes, overlaps
    )
    assert result.certified

    forged_ab_edge = replace(result.derived_edges[0], parity=0)
    forged_ab = replace(
        result.overlap_results[0],
        target_exact_anchor=result.overlap_results[0].source_exact_anchor,
        derived_edge=forged_ab_edge,
    )
    ac = result.overlap_results[2]
    forged_ac_edge = replace(result.derived_edges[2], parity=1)
    forged_ac_target = tuple(
        -value if index < 4 else value
        for index, value in enumerate(ac.source_exact_anchor)
    )
    forged_ac = replace(
        ac,
        target_exact_anchor=forged_ac_target,
        derived_edge=forged_ac_edge,
    )
    forged_overlap_results = (
        forged_ab,
        result.overlap_results[1],
        forged_ac,
    )
    forged_edges = (
        forged_ab_edge,
        result.derived_edges[1],
        forged_ac_edge,
    )
    forged_graph = check_planar_lc_gauge_gluing(
        PlanarLCGaugeGluingCertificate(
            certificate_id=f"{certificate.atlas_id}:derived-gauge-graph",
            chart_ids=result.chart_ids,
            overlaps=forged_edges,
            source="coordinated_adversarial_forgery",
        )
    )
    assert forged_graph.certified

    forged_result = replace(
        result,
        overlap_results=forged_overlap_results,
        derived_edges=forged_edges,
        gauge_result=forged_graph,
    )
    assert not forged_result.certified


def test_exact_gauge_atlas_parser_preserves_malformed_identifier_values():
    malformed = PlanarLCExactGaugeAtlasCertificate.from_dict(
        {
            "atlas_id": True,
            "chart_ids": [True],
            "tube_ids": "tube:not-an-array",
            "overlap_ids": [1],
            "source": 7,
        }
    )

    assert malformed.atlas_id is True
    assert malformed.chart_ids == (True,)
    assert malformed.tube_ids == "tube:not-an-array"
    assert malformed.overlap_ids == (1,)
    assert malformed.source == 7
    assert malformed.to_dict()["tube_ids"] == "tube:not-an-array"
