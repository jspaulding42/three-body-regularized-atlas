use num_traits::Zero;

use three_body_planar_chain_verifier::raw_schema::{
    decode_raw_chain, RawPlanarChainWire, SchemaProfile, SegmentWire,
};
use three_body_planar_chain_verifier::{
    checked_real_binary64_from_json, parse_wire_json,
    replay_raw_ordinary_only_chain_exact_rational_v04, OrdinaryChainReplayFailure,
    DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    EXACT_RATIONAL_RAW_ORDINARY_ONLY_CHAIN_V04_PROFILE_ID, HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
    RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

fn decode(bytes: &[u8]) -> RawPlanarChainWire {
    let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
    decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
}

fn real(lexeme: &str) -> three_body_planar_chain_verifier::CheckedJsonBinary64 {
    checked_real_binary64_from_json(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
}

fn one_bridge_chain(bytes: &[u8]) -> RawPlanarChainWire {
    let mut chain = decode(bytes);
    let SegmentWire::OrdinaryBridge(segment) = &chain.segments[0] else {
        panic!("first canonical segment is not ordinary");
    };
    chain.requested_target_time = segment.target_chart.physical_time_interval[0].clone();
    chain.segments.truncate(1);
    chain
}

fn two_bridge_chain() -> RawPlanarChainWire {
    let mut chain = one_bridge_chain(SUCCESS_RAW);
    let SegmentWire::OrdinaryBridge(first) = &chain.segments[0] else {
        unreachable!();
    };
    let mut second = (**first).clone();
    second.transition.transition_id = "review-v04:ordinary:transition:1".to_owned();
    second.transition.source_chart_id = first.target_chart.chart_id.clone();
    second.transition.source_tube_id = first.target_tube.tube_id.clone();
    second.transition.target_chart_id = "review-v04:n:chart:2".to_owned();
    second.transition.target_tube_id = "review-v04:n:tube:2".to_owned();
    second.transition.source_parameter = real("9.5367431640625e-7");
    second.transition.target_parameter = real("9.5367431640625e-7");
    second.target_chart.certificate_id = "review-v04:n:certificate:2".to_owned();
    second.target_chart.chart_id = second.transition.target_chart_id.clone();
    second.target_chart.parameter_interval =
        [real("9.5367431640625e-7"), real("1.9073486328125e-6")];
    second.target_chart.physical_time_interval =
        [real("9.536752259009518e-7"), real("1.9073495423072018e-6")];
    second.target_tube.tube_id = second.transition.target_tube_id.clone();
    second.target_tube.chart_id = second.transition.target_chart_id.clone();
    second.target_tube.anchor_parameter = real("9.5367431640625e-7");
    second.target_tube.initial_error_bound = real("0.0002");
    second.target_tube.tube_radius = real("0.001");
    chain.requested_target_time = second.target_chart.physical_time_interval[0].clone();
    chain
        .segments
        .push(SegmentWire::OrdinaryBridge(Box::new(second)));
    chain
}

#[test]
fn canonical_chains_commit_one_bridge_then_return_the_committed_right_frontier() {
    for (case_id, bytes) in [
        ("success", SUCCESS_RAW),
        ("failed-revisit", FAILED_REVISIT_RAW),
    ] {
        let chain = decode(bytes);
        let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain)
            .unwrap_or_else(|error| panic!("{case_id}: {error}"));

        assert_eq!(replay.certified_segment_count(), 1, "{case_id}");
        assert_eq!(replay.failed_segment_index(), None, "{case_id}");
        assert_eq!(replay.unsupported_segment_index(), Some(1), "{case_id}");
        assert_eq!(replay.clock_ledger().len(), 2, "{case_id}");
        assert_eq!(replay.bridge_replays().len(), 1, "{case_id}");
        assert!(replay.obligations()[0].satisfied(), "{case_id}");
        assert!(replay.obligations()[1].satisfied(), "{case_id}");
        assert!(!replay.obligations()[2].satisfied(), "{case_id}");
        let frontier = replay
            .right_frontier()
            .expect("certified ordinary frontier");
        assert_eq!(frontier.committed_segment_count(), 1, "{case_id}");
        assert_eq!(frontier.chart_id(), replay.clock_ledger()[1].chart_id());
        let SegmentWire::OrdinaryBridge(first_segment) = &chain.segments[0] else {
            unreachable!();
        };
        let expected_right = first_segment.target_chart.parameter_interval[1].binary64_rational();
        assert_eq!(frontier.right_parameter(), expected_right);
        assert_eq!(
            frontier.physical_time_interval().lower(),
            &(expected_right + replay.clock_ledger()[1].clock_origin().lower())
        );
        assert_eq!(
            frontier.physical_time_interval().upper(),
            &(expected_right + replay.clock_ledger()[1].clock_origin().upper())
        );
        assert_eq!(frontier.position_intervals().len(), 6);
        assert_eq!(frontier.velocity_intervals().len(), 6);
        assert!(frontier.tube_replay().certified());
        assert!(replay.final_enclosure().is_none());
    }
}

#[test]
fn one_bridge_ordinary_chain_certifies_the_requested_target() {
    let chain = one_bridge_chain(SUCCESS_RAW);
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert_eq!(
        replay.profile_id(),
        EXACT_RATIONAL_RAW_ORDINARY_ONLY_CHAIN_V04_PROFILE_ID
    );
    assert_eq!(
        replay
            .obligations()
            .iter()
            .map(|obligation| obligation.id())
            .collect::<Vec<_>>(),
        RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS
    );
    assert!(replay.profile_satisfied(), "{replay:#?}");
    assert!(replay.mathematical_to_target());
    assert_eq!(replay.certified_segment_count(), 1);
    assert!(replay.final_enclosure().is_some());
    assert!(replay.right_frontier().is_none());
}

#[test]
fn width_failure_retains_the_stronger_fixed_time_enclosure() {
    let mut chain = one_bridge_chain(SUCCESS_RAW);
    chain.requested_maximum_component_width = real("0.0");
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert!(replay.mathematical_to_target());
    assert!(!replay.profile_satisfied());
    assert!(replay.final_enclosure().is_some());
    assert!(replay.maximum_component_width().is_some());
    assert!(replay.right_frontier().is_none());
    assert!(!replay.obligations()[7].satisfied());
}

#[test]
fn failed_bridge_does_not_commit_its_clock_or_target_chart() {
    let mut chain = one_bridge_chain(SUCCESS_RAW);
    let SegmentWire::OrdinaryBridge(segment) = &mut chain.segments[0] else {
        unreachable!();
    };
    segment.target_tube.initial_error_bound = real("0.0");
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert_eq!(replay.failed_segment_index(), Some(0));
    assert_eq!(replay.certified_segment_count(), 0);
    assert_eq!(replay.clock_ledger().len(), 1);
    assert!(!replay.bridge_replays()[0].local_bridge_satisfied());
    assert!(replay.bridge_replays()[0].obligations()[8].satisfied());
    let frontier = replay.right_frontier().expect("root frontier");
    assert_eq!(frontier.committed_segment_count(), 0);
    assert_eq!(frontier.chart_id(), replay.clock_ledger()[0].chart_id());
}

#[test]
fn strict_root_gate_rejects_tolerance_identity_and_mass_mutations_before_replay() {
    for mutation in ["tolerance", "identity", "mass"] {
        let mut chain = decode(SUCCESS_RAW);
        match mutation {
            "tolerance" => chain.root_binding.time_tolerance = real("5e-324"),
            "identity" => chain.root_binding.chart_id.push_str("-wrong"),
            "mass" => chain.root_binding.masses[0] = real("2.0"),
            _ => unreachable!(),
        }
        let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();
        assert!(!replay.obligations()[0].satisfied(), "{mutation}");
        assert!(replay.initial_chart_replay().is_none(), "{mutation}");
        assert!(replay.root_replay().is_none(), "{mutation}");
        assert!(replay.clock_ledger().is_empty(), "{mutation}");
        assert!(replay.right_frontier().is_none(), "{mutation}");
    }
}

#[test]
fn false_claimed_tail_blocks_root_admission_even_when_validated_root_is_true() {
    let mut chain = decode(SUCCESS_RAW);
    chain.initial_chart.tail_bound = real("-1.0");
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert!(replay.obligations()[0].satisfied());
    assert!(!replay
        .initial_chart_replay()
        .unwrap()
        .conditional_profile_satisfied());
    assert!(replay.root_replay().unwrap().validated_root_satisfied());
    assert!(!replay.obligations()[1].satisfied());
    assert!(replay.clock_ledger().is_empty());
    assert!(replay.right_frontier().is_none());
}

#[test]
fn target_before_and_target_outside_current_domain_fail_distinct_obligations() {
    let mut before = one_bridge_chain(SUCCESS_RAW);
    before.requested_target_time = real("-5e-324");
    let before_replay = replay_raw_ordinary_only_chain_exact_rational_v04(&before).unwrap();
    assert!(!before_replay.obligations()[3].satisfied());
    assert!(before_replay.target_preimage().is_none());
    assert!(before_replay.right_frontier().is_some());

    let mut outside = one_bridge_chain(SUCCESS_RAW);
    outside.requested_target_time = real("0.001");
    let outside_replay = replay_raw_ordinary_only_chain_exact_rational_v04(&outside).unwrap();
    assert!(outside_replay.obligations()[3].satisfied());
    assert!(outside_replay.obligations()[4].satisfied());
    assert!(!outside_replay.obligations()[5].satisfied());
    assert!(outside_replay.target_preimage().is_some());
    assert!(outside_replay.final_enclosure().is_none());
    assert!(outside_replay.right_frontier().is_some());
}

#[test]
fn malformed_later_chart_is_structured_failure_and_preserves_the_root_frontier() {
    let mut chain = one_bridge_chain(SUCCESS_RAW);
    let SegmentWire::OrdinaryBridge(segment) = &mut chain.segments[0] else {
        unreachable!();
    };
    segment.target_chart.chart_type = "not-an-ordinary-chart".to_owned();
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert_eq!(replay.failed_segment_index(), Some(0));
    assert_eq!(replay.certified_segment_count(), 0);
    assert_eq!(replay.clock_ledger().len(), 1);
    assert!(matches!(
        replay.replay_failure(),
        Some(OrdinaryChainReplayFailure::TargetChartSemantic {
            segment_index: 0,
            ..
        })
    ));
    let frontier = replay.right_frontier().expect("retained root frontier");
    assert_eq!(frontier.committed_segment_count(), 0);
    assert_eq!(frontier.chart_id(), replay.clock_ledger()[0].chart_id());
}

#[test]
fn segment_preflight_cap_is_fail_closed_and_retains_the_certified_root() {
    let mut chain = one_bridge_chain(SUCCESS_RAW);
    let repeated = chain.segments[0].clone();
    chain.segments = vec![repeated; HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS + 1];
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();

    assert_eq!(replay.certified_segment_count(), 0);
    assert!(replay.bridge_replays().is_empty());
    assert!(matches!(
        replay.replay_failure(),
        Some(OrdinaryChainReplayFailure::SegmentLimitExceeded {
            actual,
            limit: HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
        }) if *actual == HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS + 1
    ));
    assert!(!replay.obligations()[2].satisfied());
    assert!(replay.right_frontier().is_some());
}

#[test]
fn two_consecutive_ordinary_bridges_carry_nonzero_clock_and_second_failure_rolls_back() {
    let chain = two_bridge_chain();
    let replay = replay_raw_ordinary_only_chain_exact_rational_v04(&chain).unwrap();
    assert_eq!(
        replay.certified_segment_count(),
        2,
        "bridge ledgers: {:?}",
        replay
            .bridge_replays()
            .iter()
            .map(|bridge| bridge
                .obligations()
                .iter()
                .map(|obligation| obligation.satisfied())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    assert!(replay.profile_satisfied());
    assert_eq!(replay.clock_ledger().len(), 3);
    assert_ne!(
        replay.clock_ledger()[2].chart_id(),
        replay.clock_ledger()[1].chart_id()
    );
    assert!(!replay.clock_ledger()[2].clock_origin().lower().is_zero());

    let mut failed = chain;
    let SegmentWire::OrdinaryBridge(second) = &mut failed.segments[1] else {
        unreachable!();
    };
    second.target_tube.initial_error_bound = real("0.0");
    let failed_replay = replay_raw_ordinary_only_chain_exact_rational_v04(&failed).unwrap();
    assert_eq!(failed_replay.failed_segment_index(), Some(1));
    assert_eq!(failed_replay.certified_segment_count(), 1);
    assert_eq!(failed_replay.clock_ledger().len(), 2);
    let frontier = failed_replay
        .right_frontier()
        .expect("first committed frontier");
    assert_eq!(frontier.committed_segment_count(), 1);
    assert_eq!(
        frontier.chart_id(),
        failed_replay.clock_ledger()[1].chart_id()
    );
}
