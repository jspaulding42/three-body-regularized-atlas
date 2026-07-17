from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import pytest

from three_body_symmetry.certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    ordinary_taylor_chart_certificate_from_solution,
)
from three_body_symmetry.proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
)
from three_body_symmetry.proof_carrying_ordinary_bridge import (
    OrdinaryBridgeTransitionRecord,
)
from three_body_symmetry.proof_carrying_planar_chain import (
    OrdinaryBridgeV1Segment,
    RawPlanarChainCertificate,
    RawPlanarChainReplayResult,
    canonical_planar_chain_evidence_json,
    check_raw_planar_chain,
    raw_planar_chain_evidence_sha256,
)
from three_body_symmetry.series import construct_taylor_solution


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __bool__(self) -> bool:
        return True


class _ForgedReplayResult(RawPlanarChainReplayResult):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _Fixture:
    certificate: RawPlanarChainCertificate


@lru_cache(maxsize=1)
def _fixture() -> _Fixture:
    step = 0.02
    masses = np.asarray((1.0, 0.8, 1.2))
    positions = np.asarray(((0.8, -0.2), (-0.4, 0.6), (0.1, -0.5)))
    velocities = np.asarray(((0.03, 0.01), (-0.02, 0.04), (0.01, -0.03)))
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
        certificate_id="planar-chain-root-chart-certificate",
        chart_id="planar-chain-root-chart",
        parameter_interval=(0.0, step),
        physical_time_interval=(50.0, 50.0 + step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    target_chart = ordinary_taylor_chart_certificate_from_solution(
        target_solution,
        certificate_id="planar-chain-target-chart-certificate",
        chart_id="planar-chain-target-chart",
        parameter_interval=(0.0, step),
        # Deliberately unrelated: only the carried B ledger is authoritative.
        physical_time_interval=(-100.0, -100.0 + step),
        coefficient_tolerance=1.0e-11,
        residual_tolerance=1.0e-8,
        tail_bound=1.0e-9,
        sample_count=7,
    )
    source_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="planar-chain-root-tube",
        chart_id=source_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=2.0e-2,
        max_defect_bound=1.0e-1,
        max_lipschitz_bound=70.0,
    )
    target_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="planar-chain-target-tube",
        chart_id=target_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=2.0e-3,
        tube_radius=3.0e-2,
        max_defect_bound=2.0e-1,
        max_lipschitz_bound=100.0,
    )
    transition = OrdinaryBridgeTransitionRecord(
        transition_id="planar-chain-ordinary-transition",
        source_chart_id=source_chart.chart_id,
        source_tube_id=source_tube.tube_id,
        target_chart_id=target_chart.chart_id,
        target_tube_id=target_tube.tube_id,
        source_parameter=step,
        target_parameter=0.0,
    )
    binding = InitialValueProblemBindingCertificate(
        binding_id="planar-chain-root-binding",
        chart_id=source_chart.chart_id,
        masses=source_chart.masses,
        initial_time=50.0,
        chart_parameter=0.0,
        positions=source_chart.position_coefficients[0],
        velocities=source_chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    certificate = RawPlanarChainCertificate(
        certificate_id="planar-chain-certificate",
        root_binding=binding,
        initial_chart=source_chart,
        initial_tube=source_tube,
        segments=(
            OrdinaryBridgeV1Segment(
                transition=transition,
                target_chart=target_chart,
                target_tube=target_tube,
            ),
        ),
        requested_target_time=50.03,
        requested_maximum_component_width=1.0,
    )
    return _Fixture(certificate=certificate)


def test_n_to_n_chain_replays_root_clock_fold_and_fixed_time_enclosure():
    certificate = _fixture().certificate
    result = check_raw_planar_chain(certificate)

    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.certified_segment_count == 1
    assert result.first_failed_obligation is None
    assert result.failed_segment_index is None
    assert len(result.clock_origin_ledger) == 2
    assert len(result.cocycle_records) == 1
    root_b = Fraction.from_float(certificate.root_binding.initial_time) - (
        Fraction.from_float(certificate.root_binding.chart_parameter)
    )
    assert result.clock_origin_ledger[0].clock_origin_interval == (root_b, root_b)
    cocycle = result.cocycle_records[0]
    delta = Fraction.from_float(
        certificate.segments[0].transition.source_parameter
    ) - Fraction.from_float(
        certificate.segments[0].transition.target_parameter
    )
    assert cocycle.exact_parameter_translation == delta
    assert result.current_clock_origin_interval == (root_b + delta,) * 2
    target = Fraction.from_float(certificate.requested_target_time)
    assert result.target_parameter_preimage_interval == (
        target - root_b - delta,
        target - root_b - delta,
    )
    assert result.final_enclosure is not None
    assert result.maximum_final_component_width is not None
    assert result.retained_regions == ()


def test_wire_round_trip_digest_and_strict_union_fields():
    certificate = _fixture().certificate
    wire = certificate.to_dict()
    rebuilt = RawPlanarChainCertificate.from_dict(wire)

    assert rebuilt == certificate
    assert canonical_planar_chain_evidence_json(rebuilt) == (
        canonical_planar_chain_evidence_json(certificate)
    )
    assert raw_planar_chain_evidence_sha256(rebuilt) == (
        raw_planar_chain_evidence_sha256(certificate)
    )

    unknown = dict(wire)
    unknown["later_binding"] = wire["root_binding"]
    with pytest.raises(ValueError, match="unknown"):
        RawPlanarChainCertificate.from_dict(unknown)
    segment_unknown = dict(wire)
    segment_unknown["segments"] = [
        {**wire["segments"][0], "supplied_clock_origin": [0, 0]}
    ]
    with pytest.raises(ValueError, match="unknown"):
        RawPlanarChainCertificate.from_dict(segment_unknown)
    bad_tag = dict(wire)
    bad_tag["segments"] = [
        {**wire["segments"][0], "segment_type": "future_union_arm"}
    ]
    with pytest.raises(ValueError, match="unknown tag"):
        RawPlanarChainCertificate.from_dict(bad_tag)


def test_width_failure_retains_rigorous_fixed_time_enclosure():
    narrow = replace(
        _fixture().certificate,
        requested_maximum_component_width=0.0,
    )
    result = check_raw_planar_chain(narrow)

    assert result.status == UNRESOLVED
    assert not result.certified
    assert result.first_failed_obligation == (
        "raw_planar_chain_final_component_width_within_requested_bound"
    )
    assert result.final_enclosure is not None
    assert result.maximum_final_component_width is not None
    assert len(result.retained_regions) == 1
    assert result.retained_regions[0].region_type == (
        "certified_ordinary_fixed_time_enclosure"
    )
    assert result.replay_consistent


def test_clock_reset_endpoint_mass_and_global_namespace_mutations_fail_closed():
    certificate = _fixture().certificate
    segment = certificate.segments[0]

    reset = replace(
        certificate,
        segments=(
            replace(
                segment,
                transition=replace(segment.transition, source_parameter=0.01),
            ),
        ),
    )
    reset_result = check_raw_planar_chain(reset)
    assert reset_result.status == UNRESOLVED
    assert reset_result.failed_segment_index == 0
    assert reset_result.first_failed_obligation.startswith("segment[0]:")
    assert len(reset_result.retained_regions) == 1
    assert reset_result.retained_regions[0].region_type == (
        "certified_current_ordinary_right_frontier"
    )

    wrong_mass_chart = replace(
        segment.target_chart,
        masses=(1.0, 1.0, 1.0),
    )
    wrong_mass = replace(
        certificate,
        segments=(replace(segment, target_chart=wrong_mass_chart),),
    )
    assert check_raw_planar_chain(wrong_mass).failed_segment_index == 0

    collision = replace(
        certificate,
        certificate_id=segment.target_tube.tube_id,
    )
    namespace = check_raw_planar_chain(collision)
    assert namespace.status == UNRESOLVED
    assert namespace.first_failed_obligation == (
        "raw_planar_chain_global_identifier_namespace_unique"
    )


def test_unresolved_exact_self_replay_rejects_always_equal_result_fields():
    unresolved = check_raw_planar_chain(
        replace(_fixture().certificate, requested_maximum_component_width=0.0)
    )
    assert unresolved.replay_consistent

    for name in (
        "current_clock_origin_interval",
        "covered_physical_time_interval",
        "target_parameter_preimage_interval",
    ):
        forged = replace(unresolved, **{name: _AlwaysEqual()})
        assert not forged._snapshot_well_formed()
        assert not forged.replay_consistent
        assert not forged.certified
    subclass = _ForgedReplayResult(**unresolved.__dict__)
    assert not subclass._snapshot_well_formed()
    assert not subclass.replay_consistent


def test_nan_request_is_deterministic_replay_consistent_unresolved():
    certificate = replace(
        _fixture().certificate,
        requested_target_time=math.nan,
    )
    first = check_raw_planar_chain(certificate)
    second = check_raw_planar_chain(certificate)

    assert first == second
    assert first.status == UNRESOLVED
    assert first.first_failed_obligation == (
        "raw_planar_chain_requested_target_finite"
    )
    assert first.requested_target_time is None
    assert first.replay_consistent


def test_raw_mutation_changes_digest_and_exact_result_snapshot():
    certificate = _fixture().certificate
    result = check_raw_planar_chain(certificate)
    mutated = replace(certificate, requested_target_time=50.031)

    assert raw_planar_chain_evidence_sha256(mutated) != result.evidence_sha256
    forged = replace(result, raw_certificate=mutated)
    assert not forged._snapshot_well_formed()
    assert not forged.replay_consistent
