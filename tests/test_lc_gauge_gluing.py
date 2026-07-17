from __future__ import annotations

from dataclasses import replace
from itertools import product
import random

import numpy as np
import pytest

from three_body_symmetry.binary_chart import (
    pair_energy_constraint,
    planar_interval_to_regularized_binary_collision_chart_atlas,
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_rhs,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.lc_gauge_gluing import (
    PlanarLCGaugeGluingCertificate,
    PlanarLCGaugeGluingObligation,
    PlanarLCGaugeOverlapEdge,
    check_planar_lc_gauge_gluing,
)


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


def _edge(overlap_id: str, source: str, target: str, parity: int):
    return PlanarLCGaugeOverlapEdge(overlap_id, source, target, parity)


def _certificate(
    chart_ids: tuple[str, ...],
    overlaps: tuple[PlanarLCGaugeOverlapEdge, ...],
    *,
    certificate_id: str = "lc-gauge:test",
) -> PlanarLCGaugeGluingCertificate:
    return PlanarLCGaugeGluingCertificate(certificate_id, chart_ids, overlaps)


def _assert_assignment_satisfies(
    certificate: PlanarLCGaugeGluingCertificate,
    assignment_items: tuple[tuple[str, int], ...],
) -> None:
    assignment = dict(assignment_items)
    assert set(assignment) == set(certificate.chart_ids)
    for edge in certificate.overlaps:
        assert assignment[edge.source_chart_id] ^ assignment[edge.target_chart_id] == edge.parity


def _assert_obstruction_is_cycle(
    certificate: PlanarLCGaugeGluingCertificate,
    result,
) -> None:
    edge_by_id = {edge.overlap_id: edge for edge in certificate.overlaps}
    vertices = result.obstruction_cycle_chart_ids
    overlap_ids = result.obstruction_cycle_overlap_ids
    assert vertices[0] == vertices[-1]
    assert len(vertices) == len(overlap_ids) + 1
    parity = 0
    for left, right, overlap_id in zip(vertices, vertices[1:], overlap_ids):
        edge = edge_by_id[overlap_id]
        assert {left, right} == {edge.source_chart_id, edge.target_chart_id}
        parity ^= edge.parity
    assert parity == 1
    assert result.obstruction_parity == 1


def test_single_chart_has_the_normalized_zero_gauge() -> None:
    result = check_planar_lc_gauge_gluing(_certificate(("only",), ()))

    assert result.certified
    assert result.compatible
    assert result.connected
    assert result.component_count == 1
    assert result.gauge_assignment == (("only", 0),)
    assert not result.obstruction_certified


def test_mixed_parity_tree_returns_a_primal_assignment() -> None:
    certificate = _certificate(
        ("d", "b", "a", "c"),
        (
            _edge("bc", "b", "c", 0),
            _edge("ab", "a", "b", 1),
            _edge("cd", "c", "d", 1),
        ),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert result.certified
    assert result.gauge_assignment == (("a", 0), ("b", 1), ("c", 1), ("d", 0))
    _assert_assignment_satisfies(certificate, result.gauge_assignment)


def test_even_parity_cycle_is_compatible() -> None:
    certificate = _certificate(
        ("a", "b", "c"),
        (
            _edge("ab", "a", "b", 1),
            _edge("bc", "b", "c", 0),
            _edge("ca", "c", "a", 1),
        ),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert result.certified
    _assert_assignment_satisfies(certificate, result.gauge_assignment)


def test_odd_parity_cycle_returns_a_dual_obstruction() -> None:
    certificate = _certificate(
        ("a", "b", "c"),
        (
            _edge("ab", "a", "b", 0),
            _edge("bc", "b", "c", 0),
            _edge("ca", "c", "a", 1),
        ),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert result.well_formed
    assert result.connected
    assert not result.compatible
    assert not result.certified
    assert result.obstruction_certified
    _assert_obstruction_is_cycle(certificate, result)


def test_contradictory_parallel_edges_form_a_two_edge_obstruction() -> None:
    certificate = _certificate(
        ("a", "b"),
        (
            _edge("parallel-0", "a", "b", 0),
            _edge("parallel-1", "a", "b", 1),
        ),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert result.obstruction_certified
    assert len(result.obstruction_cycle_edges) == 2
    assert result.obstruction_cycle_chart_ids == ("a", "b", "a")
    _assert_obstruction_is_cycle(certificate, result)


def test_disconnected_consistent_graph_is_not_a_single_atlas_certificate() -> None:
    certificate = _certificate(
        ("a", "b", "c"),
        (_edge("ab", "a", "b", 1),),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert result.well_formed
    assert result.compatible
    assert not result.connected
    assert not result.certified
    assert result.component_count == 2
    _assert_assignment_satisfies(certificate, result.gauge_assignment)


def test_odd_cycle_witness_remains_valid_inside_a_disconnected_graph() -> None:
    certificate = _certificate(
        ("a", "b", "c", "isolated"),
        (
            _edge("ab", "a", "b", 0),
            _edge("bc", "b", "c", 0),
            _edge("ca", "c", "a", 1),
        ),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert not result.connected
    assert not result.compatible
    assert not result.certified
    assert result.obstruction_certified
    _assert_obstruction_is_cycle(certificate, result)


@pytest.mark.parametrize("parity", [True, -1, 2, "1"])
def test_non_exact_gf2_parities_are_rejected(parity) -> None:
    certificate = _certificate(
        ("a", "b"),
        (_edge("ab", "a", "b", parity),),
    )

    result = check_planar_lc_gauge_gluing(certificate)

    assert not result.well_formed
    assert not result.certified
    assert "lc_gauge_overlap_schema" in result.missing_obligations


@pytest.mark.parametrize(
    "certificate",
    [
        _certificate(("a", "a"), ()),
        _certificate(("a", "b"), (_edge("aa", "a", "a", 0),)),
        _certificate(("a", "b"), (_edge("ac", "a", "c", 0),)),
        _certificate(
            ("a", "b"),
            (_edge("duplicate", "a", "b", 0), _edge("duplicate", "a", "b", 0)),
        ),
        _certificate(("a",), (), certificate_id=""),
    ],
)
def test_malformed_ids_and_endpoints_are_rejected(certificate) -> None:
    result = check_planar_lc_gauge_gluing(certificate)

    assert not result.certified
    assert not result.obstruction_certified


def test_serialization_round_trip_preserves_exact_solution() -> None:
    certificate = _certificate(
        ("a", "b", "c"),
        (_edge("ab", "a", "b", 1), _edge("bc", "b", "c", 0)),
    )
    restored = PlanarLCGaugeGluingCertificate.from_dict(certificate.to_dict())

    assert restored == certificate
    assert check_planar_lc_gauge_gluing(restored) == check_planar_lc_gauge_gluing(certificate)


def test_solver_is_independent_of_input_order() -> None:
    first = _certificate(
        ("c", "a", "b"),
        (_edge("bc", "b", "c", 0), _edge("ab", "a", "b", 1)),
    )
    second = _certificate(
        ("b", "c", "a"),
        (_edge("ab", "a", "b", 1), _edge("bc", "b", "c", 0)),
    )

    first_result = check_planar_lc_gauge_gluing(first)
    second_result = check_planar_lc_gauge_gluing(second)

    assert first_result.certified and second_result.certified
    assert first_result.gauge_assignment == second_result.gauge_assignment


def test_result_certification_rechecks_named_obligations_and_primal_shape() -> None:
    result = check_planar_lc_gauge_gluing(
        _certificate(("a", "b"), (_edge("ab", "a", "b", 1),))
    )
    assert result.certified

    assert not replace(result, gauge_assignment=()).certified
    assert not replace(
        result,
        obligations=(
            PlanarLCGaugeGluingObligation("unrelated_true_flag", True, ""),
        ),
    ).certified

    disconnected = check_planar_lc_gauge_gluing(
        _certificate(("a", "b", "c"), (_edge("ab", "a", "b", 0),))
    )
    forged_connected_obligations = tuple(
        replace(obligation, certified=True)
        if obligation.obligation == "lc_gauge_graph_connected"
        else obligation
        for obligation in disconnected.obligations
    )
    forged_connected = replace(
        disconnected,
        obligations=forged_connected_obligations,
        component_count=1,
    )
    assert not forged_connected.connected
    assert not forged_connected.certified


def test_checker_identifier_requires_an_exact_string_for_primal_and_obstruction() -> None:
    primal = check_planar_lc_gauge_gluing(
        _certificate(("a", "b"), (_edge("ab", "a", "b", 1),))
    )
    obstruction = check_planar_lc_gauge_gluing(
        _certificate(
            ("a", "b", "c", "isolated"),
            (
                _edge("ab", "a", "b", 0),
                _edge("bc", "b", "c", 0),
                _edge("ca", "c", "a", 1),
            ),
        )
    )
    assert primal.certified
    assert not obstruction.connected
    assert not obstruction.compatible
    assert obstruction.obstruction_certified

    for result in (primal, obstruction):
        tampered = replace(result, checker_id=_AlwaysEqual())
        assert not tampered.well_formed
        assert not tampered.certified
        assert not tampered.obstruction_certified


@pytest.mark.parametrize("field_name", ("obligation", "detail"))
def test_obligation_ledger_strings_require_exact_primitives_for_both_witnesses(
    field_name: str,
) -> None:
    primal = check_planar_lc_gauge_gluing(
        _certificate(("a", "b"), (_edge("ab", "a", "b", 1),))
    )
    obstruction = check_planar_lc_gauge_gluing(
        _certificate(
            ("a", "b", "c", "isolated"),
            (
                _edge("ab", "a", "b", 0),
                _edge("bc", "b", "c", 0),
                _edge("ca", "c", "a", 1),
            ),
        )
    )
    assert primal.certified
    assert obstruction.obstruction_certified

    for result in (primal, obstruction):
        tampered_row = replace(
            result.obligations[0],
            **{field_name: _AlwaysEqual()},
        )
        tampered = replace(
            result,
            obligations=(tampered_row,) + result.obligations[1:],
        )
        assert not tampered.well_formed
        assert not tampered.certified
        assert not tampered.obstruction_certified
        assert tampered.missing_obligations == (
            "lc_gauge_gluing_obligation_schema",
        )


@pytest.mark.parametrize("source", (_AlwaysEqual(), 7, ""))
def test_raw_source_provenance_requires_a_nonempty_exact_string(source) -> None:
    primal_certificate = _certificate(
        ("a", "b"),
        (_edge("ab", "a", "b", 1),),
    )
    obstruction_certificate = _certificate(
        ("a", "b", "c", "isolated"),
        (
            _edge("ab", "a", "b", 0),
            _edge("bc", "b", "c", 0),
            _edge("ca", "c", "a", 1),
        ),
    )

    for certificate in (primal_certificate, obstruction_certificate):
        result = check_planar_lc_gauge_gluing(
            replace(certificate, source=source)
        )
        assert not result.well_formed
        assert not result.certified
        assert not result.obstruction_certified
        assert "lc_gauge_certificate_identity" in result.missing_obligations


def test_result_schema_rejects_unhashable_edge_endpoint_without_raising() -> None:
    result = check_planar_lc_gauge_gluing(_certificate(("a",), ()))
    malformed_edge = PlanarLCGaugeOverlapEdge("bad", [], "a", 0)  # type: ignore[arg-type]
    malformed_result = replace(result, checked_overlaps=(malformed_edge,))

    assert not malformed_result.well_formed
    assert not malformed_result.certified
    assert not malformed_result.obstruction_certified

    malformed_chart_ids = replace(result, chart_ids=([],))  # type: ignore[arg-type]
    assert not malformed_chart_ids.well_formed
    assert not malformed_chart_ids.compatible
    assert not malformed_chart_ids.certified

    malformed_obstruction = replace(
        result,
        obstruction_cycle_chart_ids=("a", "a", "a"),
        obstruction_cycle_edges=(
            PlanarLCGaugeOverlapEdge([], "a", "a", 1),  # type: ignore[arg-type]
            PlanarLCGaugeOverlapEdge("other", "a", "a", 0),
        ),
    )
    assert malformed_obstruction.obstruction_cycle_overlap_ids == ()
    assert not malformed_obstruction.obstruction_certified

    missing_obstruction_edges = replace(  # type: ignore[arg-type]
        result, obstruction_cycle_edges=None
    )
    assert missing_obstruction_edges.obstruction_parity == -1
    assert not missing_obstruction_edges.obstruction_certified


def test_checker_rejects_non_iterable_certificate_fields_without_raising() -> None:
    malformed = PlanarLCGaugeGluingCertificate(  # type: ignore[arg-type]
        None,
        None,
        None,
    )

    result = check_planar_lc_gauge_gluing(malformed)

    assert not result.well_formed
    assert not result.certified
    assert not result.obstruction_certified


def test_random_small_multigraphs_match_brute_force() -> None:
    generator = random.Random(20260716)
    for chart_count in range(1, 6):
        chart_ids = tuple(chr(ord("a") + index) for index in range(chart_count))
        for trial in range(100):
            edge_count = (
                generator.randrange(0, min(9, chart_count * (chart_count - 1) + 1))
                if chart_count > 1
                else 0
            )
            overlaps = tuple(
                _edge(
                    f"edge-{index}",
                    *generator.sample(chart_ids, 2),
                    generator.randrange(2),
                )
                for index in range(edge_count)
            )
            certificate = _certificate(
                chart_ids,
                overlaps,
                certificate_id=f"lc-gauge:fuzz:{chart_count}:{trial}",
            )
            result = check_planar_lc_gauge_gluing(certificate)
            brute_force_compatible = any(
                all(
                    bits[chart_ids.index(edge.source_chart_id)]
                    ^ bits[chart_ids.index(edge.target_chart_id)]
                    == edge.parity
                    for edge in overlaps
                )
                for bits in product((0, 1), repeat=chart_count)
            )

            assert result.compatible is brute_force_compatible
            if result.compatible:
                _assert_assignment_satisfies(certificate, result.gauge_assignment)
            else:
                _assert_obstruction_is_cycle(certificate, result)


def test_production_lc_state_projection_constraint_and_rhs_are_gauge_invariant() -> None:
    masses = np.array([1.0, 0.7, 1.3])
    positions = np.array([[0.0, 0.0], [0.8, 0.3], [3.0, -0.4]])
    velocities = np.array([[0.1, -0.2], [-0.3, 0.4], [0.05, 0.1]])
    state = planar_to_regularized_binary_collision_chart(
        positions, velocities, masses, pair=(0, 1)
    )
    antipodal = replace(
        state,
        z=-state.z,
        z_velocity=-state.z_velocity,
    )

    projected = regularized_binary_collision_chart_to_planar(state)
    antipodal_projected = regularized_binary_collision_chart_to_planar(antipodal)
    np.testing.assert_allclose(antipodal_projected[0], projected[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(antipodal_projected[1], projected[1], rtol=0.0, atol=1.0e-15)
    assert pair_energy_constraint(antipodal) == pytest.approx(
        pair_energy_constraint(state), rel=0.0, abs=1.0e-15
    )

    derivative = regularized_binary_collision_chart_rhs(state)
    antipodal_derivative = regularized_binary_collision_chart_rhs(antipodal)
    np.testing.assert_allclose(antipodal_derivative.z, -derivative.z, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        antipodal_derivative.z_velocity,
        -derivative.z_velocity,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert antipodal_derivative.pair_energy == pytest.approx(
        derivative.pair_energy, rel=0.0, abs=1.0e-15
    )
    for name in (
        "binary_center",
        "binary_center_velocity",
        "third_offset",
        "third_offset_velocity",
    ):
        np.testing.assert_allclose(
            getattr(antipodal_derivative, name),
            getattr(derivative, name),
            rtol=0.0,
            atol=1.0e-15,
        )
    assert antipodal_derivative.physical_time == pytest.approx(
        derivative.physical_time, rel=0.0, abs=1.0e-15
    )


def test_existing_branch_cut_atlas_has_a_standalone_parity_diagnostic() -> None:
    masses = np.ones(3)
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
        state_interval, masses, pair=(0, 1)
    )
    chart_ids = tuple(
        chart.branch_certificate.branch
        for chart in atlas
        if chart.branch_certificate is not None
    )
    assert set(chart_ids) == {
        "atlas_upper_closed_half",
        "atlas_lower_closed_half",
    }

    certificate = _certificate(
        chart_ids,
        (
            _edge(
                "negative-axis-overlap",
                "atlas_upper_closed_half",
                "atlas_lower_closed_half",
                1,
            ),
        ),
        certificate_id="lc-gauge:existing-branch-cut-diagnostic",
    )
    result = check_planar_lc_gauge_gluing(certificate)

    assert result.certified
    _assert_assignment_satisfies(certificate, result.gauge_assignment)
    # This remains a supplied diagnostic: it is not consulted by the existing
    # ordinary-to-LC transition acceptance path.
