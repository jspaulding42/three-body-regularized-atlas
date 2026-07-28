from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path

from three_body_symmetry.planar_chain_review_artifact import (
    strict_load_raw_planar_chain,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_exit import (
    ProofGradeCarriedPlanarLCExitResult,
    check_carried_planar_lc_exit,
    check_proof_grade_carried_planar_lc_exit,
)
from three_body_symmetry.proof_carrying_planar_chain import (
    OrdinaryBridgeV1Segment,
    PlanarLCPassageCocycleRecord,
    PlanarLCPassageV1Segment,
    check_raw_planar_chain,
)


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_RAW = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain" / "success.raw.json"

_PROOF_GRADE_OBLIGATION_IDS = (
    "proof_grade_carried_lc_exit_exact_raw_schemas",
    "proof_grade_carried_lc_exit_identifiers_match_and_are_unique",
    "proof_grade_carried_lc_exit_parent_source_invariant_is_explicit_condition",
    "proof_grade_carried_lc_exit_entry_freshly_replayed_and_certified",
    "proof_grade_carried_lc_exit_common_planar_mass_problem",
    "proof_grade_carried_lc_exit_pair_is_canonical_ascending",
    "proof_grade_carried_lc_exit_outward_mass_arithmetic_certified",
    "proof_grade_carried_lc_exit_exact_right_to_left_endpoint_handoff",
    "proof_grade_carried_lc_exit_constrained_entry_branch_carried",
    "proof_grade_carried_lc_exit_constraint_invariance_kernel",
    "proof_grade_carried_lc_exit_lc_tube_freshly_certified",
    "proof_grade_carried_lc_exit_third_body_separated",
    "proof_grade_carried_lc_exit_target_ordinary_tube_freshly_certified",
    "proof_grade_carried_lc_exit_strict_physical_clock_kernel",
    "proof_grade_carried_lc_exit_complete_inflated_slice_reconstructed",
    "proof_grade_carried_lc_exit_complete_slice_rho_positive",
    "proof_grade_carried_lc_exit_complete_cartesian_projection_reconstructed",
    "proof_grade_carried_lc_exit_deck_equivariant_newton_projection_kernel",
    "proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection",
    "proof_grade_carried_lc_exit_time_interval_derived_from_component_fourteen",
    "proof_grade_carried_lc_exit_target_clock_origin_exactly_derived",
)


class _HostileString(str):
    def __eq__(self, other: object) -> bool:
        return True


class _ForgedProofExitResult(ProofGradeCarriedPlanarLCExitResult):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _LCExitLeg:
    entry_transition: object
    source_chart: object
    source_tube: object
    lc_chart: object
    lc_tube: object
    exit_transition: object
    target_chart: object
    target_tube: object
    source_clock: tuple[Fraction, Fraction]

    @property
    def args(self) -> tuple[object, ...]:
        return (
            self.entry_transition,
            self.source_chart,
            self.source_tube,
            self.lc_chart,
            self.lc_tube,
            self.exit_transition,
            self.target_chart,
            self.target_tube,
            self.source_clock,
        )


@lru_cache(maxsize=1)
def _archived_lc_legs() -> tuple[_LCExitLeg, ...]:
    """Strictly load every archived LC leg and its replayed source clock."""

    certificate = strict_load_raw_planar_chain(SUCCESS_RAW)
    replay = check_raw_planar_chain(certificate)
    assert replay.certified
    current_chart = certificate.initial_chart
    current_tube = certificate.initial_tube
    legs = []
    for index, segment in enumerate(certificate.segments):
        if type(segment) is OrdinaryBridgeV1Segment:
            current_chart = segment.target_chart
            current_tube = segment.target_tube
            continue
        assert type(segment) is PlanarLCPassageV1Segment
        cocycle = next(
            record
            for record in replay.cocycle_records
            if type(record) is PlanarLCPassageCocycleRecord
            and record.segment_index == index
        )
        legs.append(
            _LCExitLeg(
                entry_transition=segment.entry_transition,
                source_chart=current_chart,
                source_tube=current_tube,
                lc_chart=segment.lc_chart,
                lc_tube=segment.lc_tube,
                exit_transition=segment.exit_transition,
                target_chart=segment.target_chart,
                target_tube=segment.target_tube,
                source_clock=cocycle.source_clock_origin_interval,
            )
        )
        current_chart = segment.target_chart
        current_tube = segment.target_tube
    return tuple(legs)


def _first_leg() -> _LCExitLeg:
    return _archived_lc_legs()[0]


def _proof(leg: _LCExitLeg) -> ProofGradeCarriedPlanarLCExitResult:
    result = check_proof_grade_carried_planar_lc_exit(*leg.args)
    assert type(result) is ProofGradeCarriedPlanarLCExitResult
    return result


def test_all_archived_lc_legs_certify_the_exact_proof_grade_ledger():
    legs = _archived_lc_legs()

    assert tuple(leg.lc_chart.pair for leg in legs) == (
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1),
    )
    for leg in legs:
        result = _proof(leg)
        assert result.certified
        assert result.profile_id == (
            "binary64_outward_proof_grade_carried_planar_lc_exit_v04"
        )
        assert tuple(item.obligation for item in result.obligations) == (
            _PROOF_GRADE_OBLIGATION_IDS
        )
        assert all(item.certified is True for item in result.obligations)
        assert result.missing_obligations == ()


def test_source_and_lc_claimed_tail_failures_stay_nested_diagnostics_only():
    leg = _first_leg()
    baseline = _proof(leg)
    mutated = replace(
        leg,
        source_chart=replace(leg.source_chart, tail_bound=-1.0),
        lc_chart=replace(leg.lc_chart, tail_bound=-1.0),
    )
    result = _proof(mutated)
    compatibility = check_carried_planar_lc_exit(*mutated.args)

    assert result.certified
    assert result.obligations == baseline.obligations
    assert result.lifted_exit_slice == baseline.lifted_exit_slice
    assert result.exit_rho_interval == baseline.exit_rho_interval
    assert result.projected_position_intervals == baseline.projected_position_intervals
    assert result.projected_velocity_intervals == baseline.projected_velocity_intervals
    assert result.maximum_projected_anchor_gap == baseline.maximum_projected_anchor_gap
    assert result.exit_time_interval == baseline.exit_time_interval
    assert result.target_clock_origin_interval == baseline.target_clock_origin_interval
    assert result.entry_result is not None
    assert result.entry_result.source_chart_result is not None
    assert result.entry_result.target_chart_result is not None
    assert not result.entry_result.source_chart_result.certified
    assert not result.entry_result.target_chart_result.certified
    assert not compatibility.certified


def test_target_direct_tube_failure_retains_independent_right_slice_evidence():
    baseline = _proof(_first_leg())
    leg = replace(
        _first_leg(),
        target_tube=replace(_first_leg().target_tube, max_defect_bound=0.0),
    )
    result = _proof(leg)

    assert not result.certified
    assert tuple(item.certified for item in result.obligations) == (
        (True,) * 12
        + (False, True, True, True, True, True, False, True, True)
    )
    assert result.lifted_exit_slice == baseline.lifted_exit_slice
    assert result.exit_rho_interval == baseline.exit_rho_interval
    assert result.projected_position_intervals == baseline.projected_position_intervals
    assert result.projected_velocity_intervals == baseline.projected_velocity_intervals
    assert result.maximum_projected_anchor_gap == baseline.maximum_projected_anchor_gap
    assert result.exit_time_interval == baseline.exit_time_interval
    assert result.target_clock_origin_interval == baseline.target_clock_origin_interval


def test_tiny_target_radius_fails_only_complete_projection_containment():
    leg = replace(
        _first_leg(),
        target_tube=replace(_first_leg().target_tube, initial_error_bound=0.0),
    )
    result = _proof(leg)

    assert not result.certified
    assert tuple(item.certified for item in result.obligations) == (
        (True,) * 18 + (False, True, True)
    )
    assert result.missing_obligations == (
        "proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection",
    )


def test_target_domain_failure_retains_right_slice_but_not_target_clock():
    baseline = _proof(_first_leg())
    step = _first_leg().target_chart.parameter_interval[1]
    leg = replace(
        _first_leg(),
        target_chart=replace(
            _first_leg().target_chart,
            parameter_interval=(-step, step),
        ),
    )
    result = _proof(leg)

    assert not result.certified
    assert tuple(item.certified for item in result.obligations) == (
        (True,) * 7
        + (False,)
        + (True,) * 5
        + (False,)
        + (True,) * 4
        + (False, True, False)
    )
    assert result.lifted_exit_slice == baseline.lifted_exit_slice
    assert result.exit_rho_interval == baseline.exit_rho_interval
    assert result.projected_position_intervals == baseline.projected_position_intervals
    assert result.projected_velocity_intervals == baseline.projected_velocity_intervals
    assert result.exit_time_interval == baseline.exit_time_interval
    assert result.target_clock_origin_interval == ()


def test_source_anchor_and_target_left_endpoint_mutations_split_dependencies():
    leg = _first_leg()
    step = leg.target_chart.parameter_interval[1]
    source_anchor = _proof(
        replace(
            leg,
            source_tube=replace(leg.source_tube, anchor_parameter=step / 2),
        )
    )
    target_left = _proof(
        replace(
            leg,
            exit_transition=replace(
                leg.exit_transition,
                target_parameter=step / 2,
            ),
        )
    )

    assert source_anchor.entry_result is not None
    assert source_anchor.entry_result.certified
    assert not source_anchor.obligations[7].certified
    assert source_anchor.lifted_exit_slice == ()
    assert not target_left.obligations[7].certified
    assert all(item.certified for item in target_left.obligations[14:18])
    assert target_left.obligations[19].certified
    assert not target_left.obligations[20].certified
    assert len(target_left.lifted_exit_slice) == 14
    assert target_left.target_clock_origin_interval == ()


def test_lc_right_entry_and_direct_tube_mutations_suppress_the_slice():
    leg = _first_leg()
    mutations = (
        replace(
            leg,
            exit_transition=replace(
                leg.exit_transition,
                source_parameter=math.nextafter(
                    leg.exit_transition.source_parameter,
                    -math.inf,
                ),
            ),
        ),
        replace(
            leg,
            entry_transition=replace(
                leg.entry_transition,
                source_right_parameter=math.nextafter(
                    leg.entry_transition.source_right_parameter,
                    -math.inf,
                ),
            ),
        ),
        replace(
            leg,
            source_tube=replace(leg.source_tube, max_defect_bound=0.0),
        ),
        replace(
            leg,
            lc_tube=replace(leg.lc_tube, max_defect_bound=0.0),
        ),
    )

    for mutated in mutations:
        result = _proof(mutated)
        assert not result.certified
        assert result.lifted_exit_slice == ()
        assert result.exit_rho_interval == ()
        assert result.projected_position_intervals == ()
        assert result.projected_velocity_intervals == ()
        assert result.exit_time_interval == ()
        assert result.target_clock_origin_interval == ()


def test_exact_self_replay_rejects_nested_diagnostics_artifacts_and_hostile_equality():
    result = _proof(_first_leg())
    assert result.entry_result is not None
    assert result.entry_result.source_chart_result is not None
    forged_diagnostic = replace(
        result.entry_result.source_chart_result,
        max_coefficient_residual=(
            result.entry_result.source_chart_result.max_coefficient_residual + 1.0
        ),
    )
    forged_entry = replace(
        result.entry_result,
        source_chart_result=forged_diagnostic,
    )
    forged_nested_entry_artifact = replace(
        result.entry_result,
        tested_lift_max_gaps=(
            result.entry_result.tested_lift_max_gaps[0] + Fraction(1),
            result.entry_result.tested_lift_max_gaps[1],
        ),
    )
    forged_artifact = replace(
        result,
        maximum_projected_anchor_gap=(
            result.maximum_projected_anchor_gap + Fraction(1)
        ),
    )
    hostile_identifier = replace(
        result,
        transition_id=_HostileString(result.transition_id),
    )
    hostile_subclass = _ForgedProofExitResult(**result.__dict__)

    assert not replace(result, entry_result=forged_entry).certified
    assert not replace(result, entry_result=forged_nested_entry_artifact).certified
    assert not forged_artifact.certified
    assert not hostile_identifier.certified
    assert hostile_subclass == result
    assert not hostile_subclass.certified
