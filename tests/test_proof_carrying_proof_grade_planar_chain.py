from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

import three_body_symmetry.proof_carrying_proof_grade_planar_chain as proof_chain
from three_body_symmetry.planar_chain_review_artifact import (
    strict_load_raw_planar_chain,
)
from three_body_symmetry.proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
)
from three_body_symmetry.proof_carrying_planar_chain import (
    OrdinaryBridgeV1Segment,
    PlanarLCPassageV1Segment,
    RawPlanarChainCertificate,
    check_raw_planar_chain,
)
from three_body_symmetry.proof_carrying_proof_grade_planar_chain import (
    HARD_MAX_PROOF_GRADE_RAW_PLANAR_CHAIN_SEGMENTS,
    PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS,
    ProofGradeRawPlanarChainReplayResult,
    check_proof_grade_raw_planar_chain,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain"


class _HostileString(str):
    def __eq__(self, other: object) -> bool:
        return True


class _ForgedReplay(ProofGradeRawPlanarChainReplayResult):
    def __eq__(self, other: object) -> bool:
        return True


@lru_cache(maxsize=1)
def _success() -> RawPlanarChainCertificate:
    return strict_load_raw_planar_chain(ARCHIVE / "success.raw.json")


@lru_cache(maxsize=1)
def _failed_revisit() -> RawPlanarChainCertificate:
    return strict_load_raw_planar_chain(ARCHIVE / "failed-revisit.raw.json")


def _negative_claimed_tails(
    certificate: RawPlanarChainCertificate,
) -> RawPlanarChainCertificate:
    segments = []
    for segment in certificate.segments:
        if type(segment) is OrdinaryBridgeV1Segment:
            segments.append(replace(
                segment,
                target_chart=replace(segment.target_chart, tail_bound=-1.0),
            ))
        else:
            assert type(segment) is PlanarLCPassageV1Segment
            segments.append(replace(
                segment,
                lc_chart=replace(segment.lc_chart, tail_bound=-1.0),
                target_chart=replace(segment.target_chart, tail_bound=-1.0),
            ))
    return replace(
        certificate,
        initial_chart=replace(certificate.initial_chart, tail_bound=-1.0),
        segments=tuple(segments),
    )


def test_archived_success_has_the_exact_13_row_proof_grade_ledger():
    result = check_proof_grade_raw_planar_chain(_success())

    assert result.profile_id == "binary64_outward_proof_grade_raw_planar_chain_v04"
    assert result.checker_id == "proof_grade_raw_planar_chain_checker_v04"
    assert tuple(row.obligation for row in result.obligations) == (
        PROOF_GRADE_RAW_PLANAR_CHAIN_OBLIGATION_IDS
    )
    assert all(row.certified is True for row in result.obligations)
    assert result.status == CERTIFIED_TO_T
    assert result.certified_segment_count == 5
    assert result.failed_segment_index is None
    assert result.first_failed_obligation is None
    assert result.retained_regions == ()
    assert result.replay_consistent
    assert result.certified


def test_archived_failed_revisit_retains_the_authenticated_lifted_lc_right():
    result = check_proof_grade_raw_planar_chain(_failed_revisit())

    assert [row.certified for row in result.obligations] == [True] * 7 + [False] * 6
    assert result.status == UNRESOLVED
    assert result.certified_segment_count == 4
    assert result.failed_segment_index == 4
    assert result.first_failed_obligation == (
        "segment[4]:"
        "proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection"
    )
    assert result.failed_segment_missing_obligations == (
        "proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection",
    )
    assert len(result.local_segment_results) == 5
    assert len(result.retained_regions) == 1
    retained = result.retained_regions[0]
    assert retained.region_type == "certified_lifted_lc_right_frontier"
    assert retained.coordinate_system == "planar_lc_lifted_14"
    assert retained.chart_id == _failed_revisit().segments[4].lc_chart.chart_id
    assert retained.pair == (0, 1)
    assert retained.failed_segment_index == 4
    assert len(retained.component_intervals) == 14
    assert retained.parameter_interval == (
        retained.parameter_interval[0], retained.parameter_interval[0]
    )
    assert result.replay_consistent
    assert not result.certified


def test_all_negative_finite_claimed_tails_remain_nondecisive_diagnostics():
    baseline = check_proof_grade_raw_planar_chain(_success())
    mutated_certificate = _negative_claimed_tails(_success())
    proof = check_proof_grade_raw_planar_chain(mutated_certificate)
    compatibility = check_raw_planar_chain(mutated_certificate)

    assert proof.certified
    assert tuple(row.certified for row in proof.obligations) == tuple(
        row.certified for row in baseline.obligations
    )
    assert proof.final_enclosure == baseline.final_enclosure
    assert proof.maximum_final_component_width == baseline.maximum_final_component_width
    assert not compatibility.certified


def test_root_exact_gate_suppresses_direct_root_replay_when_anchor_is_not_exact():
    certificate = _success()
    mismatched = replace(
        certificate,
        root_binding=replace(certificate.root_binding, time_tolerance=1.0),
    )
    result = check_proof_grade_raw_planar_chain(mismatched)

    assert result.root_result is None
    assert result.obligations[5].certified is False
    assert result.obligations[6].certified is False
    assert result.status == UNRESOLVED
    assert result.replay_consistent


def test_direct_ordinary_segment_mutation_fails_transactionally_before_commit():
    certificate = _success()
    first = certificate.segments[0]
    assert type(first) is OrdinaryBridgeV1Segment
    mutated = replace(
        certificate,
        segments=(replace(
            first,
            target_tube=replace(first.target_tube, initial_error_bound=0.0),
        ),) + certificate.segments[1:],
    )
    result = check_proof_grade_raw_planar_chain(mutated)

    assert result.status == UNRESOLVED
    assert result.certified_segment_count == 0
    assert result.failed_segment_index == 0
    assert result.first_failed_obligation == (
        "segment[0]:"
        "ordinary_bridge_target_initial_ball_contains_complete_source_endpoint"
    )
    assert result.replay_consistent


def test_width_only_failure_retains_neutral_fixed_time_artifact():
    certificate = replace(_success(), requested_maximum_component_width=0.0)
    result = check_proof_grade_raw_planar_chain(certificate)

    assert result.status == UNRESOLVED
    assert all(row.certified for row in result.obligations[:-1])
    assert result.obligations[-1].certified is False
    assert result.final_enclosure is not None
    assert len(result.retained_regions) == 1
    retained = result.retained_regions[0]
    assert retained.region_type == "certified_ordinary_fixed_time_enclosure"
    assert retained.coordinate_system == "planar_cartesian_12"
    assert retained.chart_id == result.current_chart_id
    assert retained.failed_segment_index is None
    assert retained.pair is None
    assert result.replay_consistent


def test_negative_width_is_retained_as_raw_finite_evidence_but_not_admissible():
    result = check_proof_grade_raw_planar_chain(replace(
        _success(), requested_maximum_component_width=-1.0,
    ))

    assert result.requested_maximum_component_width == -1.0
    assert result.obligations[4].certified is False
    assert result.final_enclosure is not None
    assert result.retained_regions[0].region_type == (
        "certified_ordinary_fixed_time_enclosure"
    )
    assert result.replay_consistent


def test_nonserializable_exact_outer_string_uses_replay_stable_digest_fallback():
    result = check_proof_grade_raw_planar_chain(replace(
        # The raw exact-type validator admits strings, while the canonical
        # JSON bytes cannot UTF-8 encode an unpaired surrogate.
        _success(), certificate_id="nonserializable-\ud800",
    ))

    assert result.obligations[2].certified is False
    assert result.status == UNRESOLVED
    assert result.replay_consistent


def test_deterministic_local_exception_is_a_structured_authentic_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_bridge(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("deterministic bridge failure")

    monkeypatch.setattr(
        proof_chain,
        "check_carried_ordinary_bridge",
        fail_bridge,
    )
    result = proof_chain.check_proof_grade_raw_planar_chain(_success())

    assert result.status == UNRESOLVED
    assert result.certified_segment_count == 0
    assert result.failed_segment_index == 0
    assert result.local_segment_results == ()
    assert result.failed_segment_missing_obligations == (
        "ordinary_bridge_result_missing",
    )
    assert result.replay_failure == "ordinary_bridge_exception:RuntimeError"
    assert result.replay_consistent


def test_outer_only_failure_keeps_a_local_fixed_time_proof_without_retention():
    result = check_proof_grade_raw_planar_chain(replace(
        _success(), source="noncanonical-but-exact-string-source",
    ))

    assert result.status == UNRESOLVED
    assert result.obligations[0].certified is False
    assert all(row.certified for row in result.obligations[5:12])
    assert result.obligations[12].certified is True
    assert result.final_enclosure is not None
    assert result.retained_regions == ()
    assert result.replay_consistent


def test_outer_and_width_failure_retains_the_local_fixed_time_enclosure():
    result = check_proof_grade_raw_planar_chain(replace(
        _success(),
        source="noncanonical-but-exact-string-source",
        requested_maximum_component_width=0.0,
    ))

    assert result.status == UNRESOLVED
    assert result.obligations[0].certified is False
    assert all(row.certified for row in result.obligations[5:12])
    assert result.obligations[12].certified is False
    assert result.final_enclosure is not None
    assert len(result.retained_regions) == 1
    assert result.retained_regions[0].region_type == (
        "certified_ordinary_fixed_time_enclosure"
    )
    assert result.replay_consistent


def test_over_cap_chain_stops_before_any_segment_commit_or_namespace_scan():
    certificate = _success()
    capped = replace(
        certificate,
        segments=(certificate.segments[0],) * (
            HARD_MAX_PROOF_GRADE_RAW_PLANAR_CHAIN_SEGMENTS + 1
        ),
    )
    result = check_proof_grade_raw_planar_chain(capped)

    assert result.status == UNRESOLVED
    assert result.certified_segment_count == 0
    assert result.local_segment_results == ()
    assert result.failed_segment_index is None
    assert result.replay_failure == "segment_word_limit_exceeded:actual=257;limit=256"
    assert result.obligations[7].certified is False
    assert result.replay_consistent


def test_exact_class_and_whole_replay_defenses_reject_hostile_or_forged_values():
    certificate = _success()
    with pytest.raises(TypeError):
        check_proof_grade_raw_planar_chain(replace(
            certificate, certificate_id=_HostileString(certificate.certificate_id),
        ))

    result = check_proof_grade_raw_planar_chain(certificate)
    assert not replace(result, current_chart_id="forged-chart").replay_consistent
    assert not replace(result, schema_version=True).replay_consistent
    assert not replace(result, replay_failure=0).replay_consistent
    forged = _ForgedReplay(**result.__dict__)
    assert not forged.replay_consistent
    assert not forged.certified
