from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from three_body_symmetry.binary_chart import RegularizedBinaryCollisionChartState
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_checker import (
    CertificateCheckObligation,
    PlanarLCExactOverlapAnchorCheckResult,
    check_planar_lc_exact_overlap_anchor,
)
from three_body_symmetry.certificate_language import (
    PlanarLCAposterioriTubeCertificate,
    PlanarLCExactOverlapAnchorCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.lc_gauge_gluing import PlanarLCGaugeOverlapEdge


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


def _exact_collision_chart(
    *,
    certificate_id: str = "lc-exact-overlap-source-certificate",
    chart_id: str = "lc-exact-overlap-source",
) -> PlanarLeviCivitaBinaryChartCertificate:
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
        certificate_id=certificate_id,
        chart_id=chart_id,
        parameter_interval=(-1.0e-4, 1.0e-4),
        coefficient_tolerance=1.0e-10,
        regularized_residual_tolerance=1.0e-7,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-9,
        sample_count=8,
        projection_rho_lower_bound=1.0e-12,
    )


def _renamed_chart(
    chart: PlanarLeviCivitaBinaryChartCertificate,
    *,
    chart_id: str = "lc-exact-overlap-target",
    negate_z: bool = False,
    negate_w: bool = False,
) -> PlanarLeviCivitaBinaryChartCertificate:
    def maybe_negate(
        coefficients: tuple[tuple[float, ...], ...], negate: bool
    ) -> tuple[tuple[float, ...], ...]:
        if not negate:
            return coefficients
        return tuple(
            tuple(-value for value in coefficient)
            for coefficient in coefficients
        )

    return replace(
        chart,
        certificate_id=f"{chart_id}-certificate",
        chart_id=chart_id,
        z_coefficients=maybe_negate(chart.z_coefficients, negate_z),
        z_velocity_coefficients=maybe_negate(
            chart.z_velocity_coefficients, negate_w
        ),
    )


def _tube(
    chart: PlanarLeviCivitaBinaryChartCertificate,
    tube_id: str,
    *,
    anchor: float = 0.0,
    initial_error: float = 0.0,
) -> PlanarLCAposterioriTubeCertificate:
    return PlanarLCAposterioriTubeCertificate(
        tube_id=tube_id,
        chart_id=chart.chart_id,
        anchor_parameter=anchor,
        initial_error_bound=initial_error,
        tube_radius=1.0e-6,
        max_defect_bound=1.0e-7,
        max_lipschitz_bound=2.0,
        require_pair_energy_constraint=True,
    )


def _certificate(
    source_chart: PlanarLeviCivitaBinaryChartCertificate,
    source_tube: PlanarLCAposterioriTubeCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
    *,
    source_anchor: float = 0.0,
    target_anchor: float = 0.0,
) -> PlanarLCExactOverlapAnchorCertificate:
    return PlanarLCExactOverlapAnchorCertificate(
        overlap_id="lc-exact-overlap:source-target",
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=target_chart.chart_id,
        target_tube_id=target_tube.tube_id,
        source_anchor_parameter=source_anchor,
        target_anchor_parameter=target_anchor,
    )


def _check(
    source_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    *,
    source_tube: PlanarLCAposterioriTubeCertificate | None = None,
    target_tube: PlanarLCAposterioriTubeCertificate | None = None,
    certificate: PlanarLCExactOverlapAnchorCertificate | None = None,
) -> PlanarLCExactOverlapAnchorCheckResult:
    source_tube = source_tube or _tube(source_chart, "lc-exact-overlap-source-tube")
    target_tube = target_tube or _tube(target_chart, "lc-exact-overlap-target-tube")
    certificate = certificate or _certificate(
        source_chart, source_tube, target_chart, target_tube
    )
    return check_planar_lc_exact_overlap_anchor(
        certificate,
        source_tube,
        source_chart,
        target_tube,
        target_chart,
    )


@pytest.fixture(scope="module")
def base_overlap():
    source = _exact_collision_chart()
    target = _renamed_chart(source)
    source_tube = _tube(source, "lc-exact-overlap-source-tube")
    target_tube = _tube(target, "lc-exact-overlap-target-tube")
    certificate = _certificate(source, source_tube, target, target_tube)
    result = _check(
        source,
        target,
        source_tube=source_tube,
        target_tube=target_tube,
        certificate=certificate,
    )
    return source, target, source_tube, target_tube, certificate, result


def test_renamed_identical_collision_chart_derives_identity_edge(base_overlap) -> None:
    source, target, source_tube, target_tube, certificate, result = base_overlap

    assert type(result) is PlanarLCExactOverlapAnchorCheckResult
    assert result.certified
    assert result.lifted_overlap_certified
    assert result.constrained_newtonian_overlap_certified
    assert result.two_sided_overlap
    assert result.two_sided_overlap_certified
    assert result.missing_obligations == ()
    assert type(result.derived_edge) is PlanarLCGaugeOverlapEdge
    assert result.derived_edge.parity == 0
    assert result.raw_overlap_certificate == certificate
    assert result.raw_source_chart == source
    assert result.raw_target_chart == target
    assert result.raw_source_tube == source_tube
    assert result.raw_target_tube == target_tube
    assert result.source_exact_anchor[0:2] == (0, 0)
    assert result.source_exact_anchor[2:4] == (1, 0)


def test_antipodal_collision_chart_derives_parity_from_nonzero_w(base_overlap) -> None:
    source = base_overlap[0]
    target = _renamed_chart(source, negate_z=True, negate_w=True)
    result = _check(source, target)

    assert result.certified
    assert result.constrained_newtonian_overlap_certified
    assert result.two_sided_overlap
    assert result.source_exact_anchor[0:2] == result.target_exact_anchor[0:2] == (0, 0)
    assert result.source_exact_anchor[2:4] == (1, 0)
    assert result.target_exact_anchor[2:4] == (-1, 0)
    assert result.derived_edge is not None
    assert result.derived_edge.parity == 1


def test_exact_overlap_certificate_json_round_trip(base_overlap) -> None:
    certificate = base_overlap[4]
    restored = PlanarLCExactOverlapAnchorCertificate.from_dict(
        json.loads(json.dumps(certificate.to_dict()))
    )

    assert restored == certificate
    assert _check(
        base_overlap[0],
        base_overlap[1],
        source_tube=base_overlap[2],
        target_tube=base_overlap[3],
        certificate=restored,
    ).certified


@pytest.mark.parametrize(
    ("negate_z", "negate_w"),
    ((True, False), (False, True)),
)
def test_mixed_z_w_sign_chart_is_not_accepted_as_an_overlap(
    base_overlap, negate_z: bool, negate_w: bool
) -> None:
    source = base_overlap[0]
    target = _renamed_chart(source, negate_z=negate_z, negate_w=negate_w)
    result = _check(source, target)

    # At the collision anchor z=0, the anchor alone cannot expose a z-only
    # coefficient flip.  The independently recomputed tube equations reject
    # both mixed-sign polynomials.
    assert not result.certified
    assert result.derived_edge is None
    assert not result.target_tube_result.certified
    assert "planar_lc_exact_overlap_target_tube_certified" in (
        result.missing_obligations
    )


@pytest.mark.parametrize("field", ("binary_center", "physical_time"))
def test_gauge_fixed_or_physical_time_mismatch_is_rejected(
    base_overlap, field: str
) -> None:
    source = base_overlap[0]
    target = _renamed_chart(source)
    if field == "binary_center":
        coefficients = [list(row) for row in target.binary_center_coefficients]
        coefficients[0][0] += 1.0
        target = replace(
            target,
            binary_center_coefficients=tuple(
                tuple(row) for row in coefficients
            ),
        )
    else:
        coefficients = list(target.physical_time_coefficients)
        coefficients[0] += 1.0
        target = replace(target, physical_time_coefficients=tuple(coefficients))

    result = _check(source, target)

    assert not result.certified
    assert result.derived_edge is None
    assert "planar_lc_exact_overlap_fixed_components_equal" in (
        result.missing_obligations
    )


@pytest.mark.parametrize(
    "replacement",
    (
        {"masses": (1.0, 1.0, 1.25)},
        {"pair": (1, 0)},
    ),
)
def test_mass_or_ordered_pair_mismatch_is_rejected(base_overlap, replacement) -> None:
    source = base_overlap[0]
    target = replace(_renamed_chart(source), **replacement)
    result = _check(source, target)

    assert not result.certified
    assert result.derived_edge is None
    assert "planar_lc_exact_overlap_masses_and_ordered_pair_match" in (
        result.missing_obligations
    )


def test_positive_initial_error_is_rejected_even_when_tube_itself_certifies(
    base_overlap,
) -> None:
    source, target = base_overlap[0:2]
    source_tube = _tube(source, "lc-exact-overlap-source-tube")
    target_tube = _tube(
        target,
        "lc-exact-overlap-target-tube",
        initial_error=1.0e-12,
    )
    certificate = _certificate(source, source_tube, target, target_tube)
    result = _check(
        source,
        target,
        source_tube=source_tube,
        target_tube=target_tube,
        certificate=certificate,
    )

    assert result.target_tube_result.certified
    assert not result.target_tube_result.anchor_is_polynomial_center
    assert not result.certified
    assert result.derived_edge is None
    assert "planar_lc_exact_overlap_zero_error_centers" in result.missing_obligations


def test_point_only_common_shifted_domain_is_rejected(base_overlap) -> None:
    source = replace(base_overlap[0], parameter_interval=(-1.0e-4, 0.0))
    target = replace(
        _renamed_chart(base_overlap[0]), parameter_interval=(0.0, 1.0e-4)
    )
    result = _check(source, target)

    assert not result.certified
    assert result.derived_edge is None
    assert result.common_parameter_offset_interval == (0, 0)
    assert "planar_lc_exact_overlap_common_interval_nondegenerate" in (
        result.missing_obligations
    )


@pytest.mark.parametrize("malformed_anchor", (0, False, "0.0", None))
def test_malformed_serialized_anchor_parameter_is_not_coerced_or_accepted(
    base_overlap, malformed_anchor
) -> None:
    source, target, source_tube, target_tube, certificate = base_overlap[0:5]
    payload = certificate.to_dict()
    payload["target_anchor_parameter"] = malformed_anchor
    malformed = PlanarLCExactOverlapAnchorCertificate.from_dict(payload)
    result = _check(
        source,
        target,
        source_tube=source_tube,
        target_tube=target_tube,
        certificate=malformed,
    )

    assert malformed.target_anchor_parameter is malformed_anchor
    assert not result.certified
    assert result.derived_edge is None
    assert "planar_lc_exact_overlap_anchor_parameters_match" in (
        result.missing_obligations
    )


@pytest.mark.parametrize("duplicate", ("chart", "tube"))
def test_duplicate_chart_or_tube_ids_are_rejected(base_overlap, duplicate: str) -> None:
    source = base_overlap[0]
    target = _renamed_chart(source)
    source_tube = _tube(source, "lc-exact-overlap-source-tube")
    target_tube = _tube(target, "lc-exact-overlap-target-tube")
    if duplicate == "chart":
        target = replace(target, chart_id=source.chart_id)
        target_tube = replace(target_tube, chart_id=source.chart_id)
    else:
        target_tube = replace(target_tube, tube_id=source_tube.tube_id)
    certificate = _certificate(source, source_tube, target, target_tube)
    result = _check(
        source,
        target,
        source_tube=source_tube,
        target_tube=target_tube,
        certificate=certificate,
    )

    assert not result.certified
    assert result.derived_edge is None
    assert "planar_lc_exact_overlap_distinct_charts_and_tubes" in (
        result.missing_obligations
    )


def test_result_and_derived_edge_tampering_cannot_certify(base_overlap) -> None:
    result = base_overlap[5]
    assert result.certified
    assert result.derived_edge is not None

    wrong_edge = replace(result.derived_edge, parity=1)
    wrong_anchor = list(result.target_exact_anchor)
    wrong_anchor[0] += 1

    assert not replace(result, checker_id="forged-checker").certified
    assert not replace(result, obligations=()).certified
    assert not replace(
        result,
        obligations=(CertificateCheckObligation("fake", True, "forged"),),
    ).certified
    assert not replace(result, derived_edge=wrong_edge).certified
    assert not replace(
        result, target_exact_anchor=tuple(wrong_anchor)
    ).certified
    hostile_anchor = replace(result, source_exact_anchor=None)  # type: ignore[arg-type]
    assert not hostile_anchor.certified
    assert not hostile_anchor.constrained_newtonian_overlap_certified

    antipodal_target = tuple(
        -value if index < 4 else value
        for index, value in enumerate(result.target_exact_anchor)
    )
    assert not replace(
        result,
        target_exact_anchor=antipodal_target,
        derived_edge=wrong_edge,
    ).certified

    assert not replace(
        result,
        source_tube_result=result.target_tube_result,
    ).certified
    assert not replace(
        result,
        source_tube_result=replace(
            result.source_tube_result,
            checker_id="forged-planar-lc-tube-checker",
        ),
    ).certified
    assert not replace(
        result,
        source_tube_result=replace(
            result.source_tube_result,
            tube_id=result.target_tube_result.tube_id,
        ),
    ).certified


def test_exact_overlap_snapshot_rejects_equality_spoofed_metadata(
    base_overlap,
) -> None:
    result = base_overlap[5]
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


@pytest.mark.parametrize(
    "malformed",
    ("chart", "tube", "chart-id", "tube-id"),
)
def test_malformed_exact_class_chart_or_tube_rejects_without_raising(
    base_overlap, malformed: str
) -> None:
    source, target, source_tube, target_tube, certificate = base_overlap[0:5]
    if malformed == "chart":
        source = replace(source, parameter_interval="not-an-interval")
    elif malformed == "tube":
        source_tube = replace(source_tube, anchor_parameter=("not", "a", "float"))
    elif malformed == "chart-id":
        source = replace(source, chart_id=np.array(["hostile-chart-id"]))
    else:
        source_tube = replace(source_tube, tube_id=np.array(["hostile-tube-id"]))

    result = _check(
        source,
        target,
        source_tube=source_tube,
        target_tube=target_tube,
        certificate=certificate,
    )

    assert not result.certified
    assert result.derived_edge is None
    if malformed != "tube-id":
        assert not result.source_tube_result.certified
    assert result.source_tube_result.checker_id == (
        "independent_planar_lc_aposteriori_tube_checker_v2"
    )
