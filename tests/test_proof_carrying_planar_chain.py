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
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.binary_chart import (
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_checker import (
    _planar_lc_mass_ratio_arithmetic_exact,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_entry import (
    CarriedPlanarLCEntryTransitionRecord,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_exit import (
    check_carried_planar_lc_exit,
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
    PlanarLCPassageCocycleRecord,
    PlanarLCPassageV1Segment,
    RawPlanarChainCertificate,
    RawPlanarChainReplayResult,
    canonical_planar_chain_evidence_json,
    check_raw_planar_chain,
    raw_planar_chain_evidence_sha256,
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


class _ForgedReplayResult(RawPlanarChainReplayResult):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _Fixture:
    certificate: RawPlanarChainCertificate


@dataclass(frozen=True)
class _AllPairFixture:
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


@lru_cache(maxsize=1)
def _all_pair_fixture() -> _AllPairFixture:
    """A decimal-mass prefix visiting LC_01, LC_02, LC_12, then LC_01."""

    masses = np.asarray((0.1, 0.2, 0.3))
    step = 2.0**-20
    positions = np.asarray(((0.0, 0.0), (1.0, 0.5), (-0.5, 1.5)))
    velocities = np.asarray(((0.01, 0.0), (-0.005, 0.01), (0.0, -0.01)))
    solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
    )
    current_chart = ordinary_taylor_chart_certificate_from_solution(
        solution,
        certificate_id="all-pair-n-certificate:0",
        chart_id="all-pair-n-chart:0",
        parameter_interval=(0.0, step),
        physical_time_interval=(0.0, step),
        coefficient_tolerance=1.0e-9,
        residual_tolerance=1.0,
        tail_bound=1.0e-11,
        sample_count=5,
    )
    current_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="all-pair-n-tube:0",
        chart_id=current_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-4,
        max_defect_bound=1.0,
        max_lipschitz_bound=1.0e5,
    )
    initial_chart = current_chart
    initial_tube = current_tube
    clock = (Fraction(0), Fraction(0))
    segments: list[PlanarLCPassageV1Segment] = []
    settings = (
        ((0, 1), 1.0e-7, 1.0e-6, step),
        ((0, 2), 3.0e-6, 2.0e-5, step),
        ((1, 2), 8.0e-5, 3.0e-4, 1.0e-3),
        ((0, 1), 2.0e-3, 1.0e-2, 1.0e-2),
    )
    for index, (pair, lc_initial, target_initial, target_right) in enumerate(
        settings
    ):
        source_right = float(current_chart.parameter_interval[1])
        lifted = planar_to_regularized_binary_collision_chart(
            solution.positions_at(source_right),
            solution.velocities_at(source_right),
            masses,
            pair=pair,
        )
        lc_solution = construct_regularized_binary_taylor_solution(
            lifted,
            order=12,
        )
        physical_shift = 0.5 * float(clock[0] + clock[1]) + source_right
        lc_chart = planar_levi_civita_binary_chart_certificate_from_solution(
            lc_solution,
            certificate_id=f"all-pair-lc-certificate:{index}",
            chart_id=f"all-pair-lc-chart:{index}",
            parameter_interval=(0.0, step),
            coefficient_tolerance=1.0e-9,
            regularized_residual_tolerance=1.0e-6,
            projected_residual_tolerance=0.1,
            tail_bound=1.0e-11,
            sample_count=5,
            projection_rho_lower_bound=1.0e-8,
            physical_time_shift=physical_shift,
        )
        lc_tube = PlanarLCAposterioriTubeCertificate(
            tube_id=f"all-pair-lc-tube:{index}",
            chart_id=lc_chart.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=lc_initial,
            tube_radius=max(1.5 * lc_initial, 1.0e-4),
            max_defect_bound=1.0,
            max_lipschitz_bound=1.0e5,
        )
        entry = CarriedPlanarLCEntryTransitionRecord(
            transition_id=f"all-pair-entry:{index}",
            source_chart_id=current_chart.chart_id,
            source_tube_id=current_tube.tube_id,
            target_chart_id=lc_chart.chart_id,
            target_tube_id=lc_tube.tube_id,
            source_right_parameter=source_right,
            target_left_parameter=0.0,
        )
        target_positions, target_velocities = (
            regularized_binary_collision_chart_to_planar(
                lc_solution.state_at(step)
            )
        )
        target_solution = construct_taylor_solution(
            target_positions,
            target_velocities,
            masses,
            order=12,
        )
        target_chart = ordinary_taylor_chart_certificate_from_solution(
            target_solution,
            certificate_id=f"all-pair-n-certificate:{index + 1}",
            chart_id=f"all-pair-n-chart:{index + 1}",
            parameter_interval=(0.0, target_right),
            # No later ordinary physical-time metadata is authoritative.
            physical_time_interval=(-10.0, -10.0 + target_right),
            coefficient_tolerance=1.0e-9,
            residual_tolerance=1.0,
            tail_bound=1.0e-11,
            sample_count=5,
        )
        target_tube = OrdinaryAposterioriTubeCertificate(
            tube_id=f"all-pair-n-tube:{index + 1}",
            chart_id=target_chart.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=target_initial,
            tube_radius=max(1.5 * target_initial, 1.0e-4),
            max_defect_bound=1.0,
            max_lipschitz_bound=1.0e5,
        )
        exit_transition = PlanarLCToOrdinaryEnclosureTransitionCertificate(
            transition_id=f"all-pair-exit:{index}",
            source_chart_id=lc_chart.chart_id,
            target_chart_id=target_chart.chart_id,
            source_parameter=step,
            target_parameter=0.0,
        )
        # This is producer-side fixture construction only.  The raw chain
        # below carries no supplied result, clock, gauge, or enclosure.
        passage = check_carried_planar_lc_exit(
            entry,
            current_chart,
            current_tube,
            lc_chart,
            lc_tube,
            exit_transition,
            target_chart,
            target_tube,
            clock,
        )
        assert passage.certified, (
            passage.missing_obligations,
            (
                passage.entry_result.missing_obligations,
                passage.entry_result.tested_lift_max_gaps,
                lc_initial,
            )
            if passage.entry_result is not None
            else None,
            passage.maximum_projected_anchor_gap,
            target_initial,
        )
        segments.append(
            PlanarLCPassageV1Segment(
                entry_transition=entry,
                lc_chart=lc_chart,
                lc_tube=lc_tube,
                exit_transition=exit_transition,
                target_chart=target_chart,
                target_tube=target_tube,
            )
        )
        current_chart = target_chart
        current_tube = target_tube
        solution = target_solution
        clock = passage.target_clock_origin_interval

    root_binding = InitialValueProblemBindingCertificate(
        binding_id="all-pair-root-binding",
        chart_id=initial_chart.chart_id,
        masses=initial_chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=initial_chart.position_coefficients[0],
        velocities=initial_chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    return _AllPairFixture(
        certificate=RawPlanarChainCertificate(
            certificate_id="all-pair-planar-chain",
            root_binding=root_binding,
            initial_chart=initial_chart,
            initial_tube=initial_tube,
            segments=tuple(segments),
            requested_target_time=float(clock[1]),
            requested_maximum_component_width=1.0e-1,
        )
    )


def test_n_to_n_chain_replays_root_clock_fold_and_fixed_time_enclosure():
    certificate = _fixture().certificate
    result = check_raw_planar_chain(certificate)

    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
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


def test_raw_chain_rejects_uncertified_initial_ordinary_chart():
    certificate = _fixture().certificate
    invalid_initial_chart = replace(
        certificate.initial_chart,
        coefficient_tolerance=-1.0,
    )
    invalid = replace(certificate, initial_chart=invalid_initial_chart)

    result = check_raw_planar_chain(invalid)

    assert result.status == UNRESOLVED
    assert not result.certified
    assert result.certified_segment_count == 0
    assert result.first_failed_obligation == "root:finite_checker_tolerances"
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


def test_decimal_chain_visits_all_pairs_revisits_01_and_keeps_each_gauge():
    certificate = _all_pair_fixture().certificate
    rebuilt = RawPlanarChainCertificate.from_dict(certificate.to_dict())
    result = check_raw_planar_chain(rebuilt)

    assert rebuilt == certificate
    assert result.status == CERTIFIED_TO_T
    assert result.certified
    assert result.mass_arithmetic_kernel_id == (
        PLANAR_LC_MASS_COEFFICIENT_KERNEL_ID
    )
    assert not _planar_lc_mass_ratio_arithmetic_exact(
        certificate.initial_chart.masses,
        (0, 1),
    )
    assert not _planar_lc_mass_ratio_arithmetic_exact(
        certificate.initial_chart.masses,
        (0, 2),
    )
    assert result.certified_segment_count == 4
    assert len(result.clock_origin_ledger) == 5
    assert result.certified_transition_checker_ids == (
        "carried_planar_lc_exit_checker_v2",
    ) * 4
    assert all(
        type(record) is PlanarLCPassageCocycleRecord
        for record in result.cocycle_records
    )
    assert tuple(record.canonical_pair for record in result.cocycle_records) == (
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1),
    )
    assignments = tuple(
        record.gauge_assignment for record in result.cocycle_records
    )
    assert all(assignments)
    assert assignments[0] != assignments[3]
    assert len(
        {
            patch_id
            for assignment in assignments
            for patch_id, _bit in assignment
        }
    ) == sum(len(assignment) for assignment in assignments)
    for index, (segment, record) in enumerate(
        zip(certificate.segments, result.cocycle_records)
    ):
        source_clock = result.clock_origin_ledger[index].clock_origin_interval
        source_right = Fraction.from_float(
            segment.entry_transition.source_right_parameter
        )
        assert record.entry_time_interval == (
            source_right + source_clock[0],
            source_right + source_clock[1],
        )
        target_anchor = Fraction.from_float(
            segment.exit_transition.target_parameter
        )
        assert record.target_clock_origin_interval == (
            record.exit_time_interval[0] - target_anchor,
            record.exit_time_interval[1] - target_anchor,
        )
    assert result.target_parameter_preimage_interval[0] == 0
    assert result.target_parameter_preimage_interval[1] > 0


def test_failed_revisit_exit_retains_typed_lc_right_maximal_prefix():
    certificate = _all_pair_fixture().certificate
    final_segment = certificate.segments[-1]
    broken_final = replace(
        final_segment,
        target_tube=replace(
            final_segment.target_tube,
            initial_error_bound=0.0,
        ),
    )
    broken = replace(
        certificate,
        segments=certificate.segments[:-1] + (broken_final,),
    )
    result = check_raw_planar_chain(broken)

    assert result.status == UNRESOLVED
    assert result.certified_segment_count == 3
    assert result.failed_segment_index == 3
    assert result.first_failed_obligation.startswith("segment[3]:")
    assert (
        "carried_lc_exit_target_initial_ball_contains_complete_projection"
        in result.failed_segment_missing_obligations
    )
    assert len(result.retained_regions) == 1
    retained = result.retained_regions[0]
    assert retained.region_type == "certified_lifted_lc_right_frontier"
    assert retained.coordinate_system == "planar_lc_lifted_14"
    assert len(retained.component_intervals) == 14
    assert retained.physical_time_interval == retained.component_intervals[13]
    assert retained.certified_segment_count == 3
    current_right = Fraction.from_float(
        broken.segments[2].target_chart.parameter_interval[1]
    )
    expected_covered_right = max(
        Fraction.from_float(broken.root_binding.initial_time),
        retained.physical_time_interval[0],
        current_right + result.current_clock_origin_interval[0],
    )
    assert result.covered_physical_time_interval[1] == expected_covered_right
    assert result.covered_physical_time_interval[1] < (
        retained.physical_time_interval[1]
    )
    assert result.replay_consistent


def test_reserved_derived_gauge_identifier_collision_is_global_failure():
    certificate = _all_pair_fixture().certificate
    collision_id = (
        f"{certificate.segments[0].entry_transition.transition_id}"
        ":patch:0-upper"
    )
    final = certificate.segments[-1]
    collided_final = replace(
        final,
        target_tube=replace(final.target_tube, tube_id=collision_id),
    )
    collided = replace(
        certificate,
        segments=certificate.segments[:-1] + (collided_final,),
    )
    result = check_raw_planar_chain(collided)

    assert result.status == UNRESOLVED
    assert result.first_failed_obligation == (
        "raw_planar_chain_global_identifier_namespace_unique"
    )
    assert not result.obligations[1].certified
    assert result.replay_consistent


def test_lc_cocycle_and_clock_ledger_mutations_cannot_certify():
    result = check_raw_planar_chain(_all_pair_fixture().certificate)
    first = result.cocycle_records[0]
    assignment = first.gauge_assignment
    flipped = tuple((patch_id, 1 - bit) for patch_id, bit in assignment)
    forged_cocycle = replace(first, gauge_assignment=flipped)
    forged_result = replace(
        result,
        cocycle_records=(forged_cocycle,) + result.cocycle_records[1:],
    )

    assert forged_result._snapshot_well_formed()
    assert not forged_result.replay_consistent
    assert not forged_result.certified

    wrong_kernel = replace(
        result,
        mass_arithmetic_kernel_id="wrong-mass-kernel",
    )
    assert not wrong_kernel._snapshot_well_formed()
    assert not wrong_kernel.replay_consistent
    assert not wrong_kernel.certified

    ledger_entry = result.clock_origin_ledger[1]
    shifted_clock = (
        ledger_entry.clock_origin_interval[0] + 1,
        ledger_entry.clock_origin_interval[1] + 1,
    )
    forged_ledger = replace(
        ledger_entry,
        clock_origin_interval=shifted_clock,
    )
    ledger_result = replace(
        result,
        clock_origin_ledger=(
            result.clock_origin_ledger[0],
            forged_ledger,
        )
        + result.clock_origin_ledger[2:],
    )
    assert not ledger_result._snapshot_well_formed()
    assert not ledger_result.replay_consistent
