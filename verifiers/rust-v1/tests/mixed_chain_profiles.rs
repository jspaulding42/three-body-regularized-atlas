use num_rational::BigRational;

use three_body_planar_chain_verifier::raw_schema::SegmentWire;
use three_body_planar_chain_verifier::{
    replay_raw_mixed_planar_chain_exact_rational_v04, CanonicalRawV1Admission,
    MixedChainCocycleLedgerEntry, MixedChainReplayFailure, MixedChainRetainedRegion,
    MixedPlanarSegmentKind, MixedPlanarSegmentReplay,
    EXACT_RATIONAL_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID, RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

#[test]
fn canonical_success_commits_the_complete_mixed_word_and_reaches_target() {
    let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(
        replay.profile_id(),
        EXACT_RATIONAL_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID
    );
    assert_eq!(
        replay
            .obligations()
            .iter()
            .map(|row| row.id())
            .collect::<Vec<_>>(),
        RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS
    );
    assert!(replay.profile_satisfied(), "{replay:#?}");
    assert!(replay.mathematical_to_target());
    assert_eq!(replay.certified_segment_count(), 5);
    assert_eq!(replay.failed_segment_index(), None);
    assert!(replay.failed_local_obligation_ids().is_empty());
    assert!(replay.replay_failure().is_none());
    assert_eq!(replay.segment_replays().len(), 5);
    assert_eq!(replay.clock_ledger().len(), 6);
    assert_eq!(replay.cocycle_ledger().len(), 5);
    for (index, row) in replay.clock_ledger().iter().enumerate() {
        assert_eq!(row.vertex_index(), index);
    }

    let pairs = replay
        .cocycle_ledger()
        .iter()
        .filter_map(|record| match record {
            MixedChainCocycleLedgerEntry::PlanarLc(value) => Some(value.pair()),
            MixedChainCocycleLedgerEntry::Ordinary(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(pairs, [[0, 1], [0, 2], [1, 2], [0, 1]]);
    assert!(matches!(
        replay.cocycle_ledger()[0],
        MixedChainCocycleLedgerEntry::Ordinary(_)
    ));
    for record in replay.cocycle_ledger() {
        match record {
            MixedChainCocycleLedgerEntry::Ordinary(value) => {
                let translated = value
                    .source_clock_origin()
                    .add(
                        &three_body_planar_chain_verifier::RationalInterval::try_point(
                            value.parameter_translation().clone(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                assert_eq!(&translated, value.target_clock_origin());
            }
            MixedChainCocycleLedgerEntry::PlanarLc(value) => {
                assert!(!value.entry_transition_id().is_empty());
                assert!(!value.exit_transition_id().is_empty());
                assert!(!value.selected_gauge_assignment().is_empty());
                assert_eq!(value.exit_time_interval(), value.target_clock_origin());
            }
        }
    }

    let final_enclosure = replay.final_enclosure().expect("12D fixed-time enclosure");
    assert_eq!(final_enclosure.position_intervals().len(), 6);
    assert_eq!(final_enclosure.velocity_intervals().len(), 6);
    assert!(replay.final_tube_replay().unwrap().certified());
    assert!(replay.maximum_component_width().is_some());
    assert!(replay.retained_region().is_none());
    let covered = replay.covered_physical_interval().unwrap();
    assert_eq!(
        covered.lower(),
        admission
            .wire()
            .root_binding
            .initial_time
            .binary64_rational()
    );
    assert_eq!(
        covered.upper(),
        admission.wire().requested_target_time.binary64_rational()
    );
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        replay.clock_ledger()[5].chart_id()
    );
    assert_eq!(
        replay.current_clock_origin(),
        Some(replay.clock_ledger()[5].clock_origin())
    );
}

#[test]
fn failed_revisit_rolls_back_and_retains_the_later_lc_right_frontier() {
    let admission = CanonicalRawV1Admission::admit(FAILED_REVISIT_RAW).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert!(!replay.profile_satisfied());
    assert!(!replay.mathematical_to_target());
    assert_eq!(
        replay
            .obligations()
            .iter()
            .map(|row| row.satisfied())
            .collect::<Vec<_>>(),
        [true, true, false, false, false, false, false, false]
    );
    assert_eq!(replay.certified_segment_count(), 4);
    assert_eq!(replay.failed_segment_index(), Some(4));
    assert_eq!(
        replay.failed_local_obligation_ids(),
        &["carried_lc_exit_target_initial_ball_contains_complete_projection"]
    );
    assert!(matches!(
        replay.replay_failure(),
        Some(MixedChainReplayFailure::LocalObligations {
            segment_index: 4,
            kind: MixedPlanarSegmentKind::PlanarLcPassage,
        })
    ));
    assert_eq!(replay.segment_replays().len(), 5);
    let MixedPlanarSegmentReplay::PlanarLc(failed_local) = &replay.segment_replays()[4] else {
        panic!("failed revisit did not retain its failed LC replay")
    };
    let local = failed_local.replay().obligations();
    assert!(local[14..18].iter().all(|row| row.satisfied()));
    assert!(!local[18].satisfied());
    assert!(local[19..21].iter().all(|row| row.satisfied()));
    assert_eq!(replay.cocycle_ledger().len(), 4);
    assert_eq!(replay.clock_ledger().len(), 5);
    assert_eq!(replay.clock_ledger()[4].vertex_index(), 4);
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        replay.clock_ledger()[4].chart_id()
    );
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        "review-v03:n:chart:4"
    );
    assert_eq!(
        replay.current_clock_origin(),
        Some(replay.clock_ledger()[4].clock_origin())
    );
    assert!(replay.target_preimage().is_none());
    assert!(replay.final_tube_replay().is_none());
    assert!(replay.final_enclosure().is_none());

    let MixedChainRetainedRegion::LiftedPlanarLcRight(frontier) =
        replay.retained_region().expect("typed LC-right frontier")
    else {
        panic!("failed revisit retained the wrong coordinate system")
    };
    assert_eq!(frontier.failed_segment_index(), 4);
    assert_eq!(frontier.pair(), [0, 1]);
    assert_eq!(frontier.lifted_state().components().len(), 14);
    assert_eq!(
        frontier.physical_time_interval(),
        frontier.lifted_state().physical_time()
    );

    let current_chart = replay.current_chart().unwrap();
    let current_clock = replay.current_clock_origin().unwrap();
    let ordinary_lower = current_chart.parameter_interval().upper() + current_clock.lower();
    let initial_time = admission
        .wire()
        .root_binding
        .initial_time
        .binary64_rational();
    let expected_upper: BigRational = initial_time
        .clone()
        .max(ordinary_lower)
        .max(frontier.physical_time_interval().lower().clone());
    let covered = replay.covered_physical_interval().unwrap();
    assert_eq!(covered.lower(), initial_time);
    assert_eq!(covered.upper(), &expected_upper);
}

fn first_segment(raw: &str) -> &str {
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let bytes = raw.as_bytes();
    assert_eq!(bytes[start], b'{');
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for index in start..bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &raw[start..=index];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated first segment")
}

fn top_level_segments(raw: &str) -> Vec<&str> {
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let end = raw.rfind("],\"source\":").unwrap();
    let bytes = raw.as_bytes();
    let mut segments = Vec::new();
    let mut object_start = None;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for index in start..end {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    object_start = Some(index);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    segments.push(&raw[object_start.take().unwrap()..=index]);
                }
            }
            _ => {}
        }
    }
    assert_eq!(depth, 0);
    segments
}

fn with_segment_word(raw: &str, segments: &[&str]) -> String {
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let end = raw.rfind("],\"source\":").unwrap();
    format!("{}{}{}", &raw[..start], segments.join(","), &raw[end..])
}

fn replace_requested(mut raw: String, target: &str, width: &str) -> String {
    raw = raw.replacen(
        "\"requested_target_time\":0.008008225523335941",
        &format!("\"requested_target_time\":{target}"),
        1,
    );
    raw.replacen(
        "\"requested_maximum_component_width\":0.1",
        &format!("\"requested_maximum_component_width\":{width}"),
        1,
    )
}

fn one_bridge_raw(target: &str, width: &str) -> String {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    replace_requested(with_segment_word(raw, &[first_segment(raw)]), target, width)
}

#[test]
fn ordinary_only_width_time_domain_and_cap_boundaries_are_fail_closed() {
    let width_bytes = one_bridge_raw("9.094947017729282e-13", "0.0");
    let width_admission = CanonicalRawV1Admission::admit(width_bytes.as_bytes()).unwrap();
    let width = replay_raw_mixed_planar_chain_exact_rational_v04(&width_admission).unwrap();
    assert_eq!(
        width
            .obligations()
            .iter()
            .map(|row| row.satisfied())
            .collect::<Vec<_>>(),
        [true, true, true, true, true, true, true, false]
    );
    assert!(width.mathematical_to_target());
    assert!(matches!(
        width.retained_region(),
        Some(MixedChainRetainedRegion::OrdinaryFixedTime(_))
    ));
    assert_eq!(
        width.covered_physical_interval().unwrap().upper(),
        width_admission
            .wire()
            .requested_target_time
            .binary64_rational()
    );

    let before_bytes = one_bridge_raw("-5e-324", "0.1");
    let before_admission = CanonicalRawV1Admission::admit(before_bytes.as_bytes()).unwrap();
    let before = replay_raw_mixed_planar_chain_exact_rational_v04(&before_admission).unwrap();
    assert!(!before.obligations()[3].satisfied());
    assert!(before.target_preimage().is_none());
    assert!(matches!(
        before.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));

    let outside_bytes = one_bridge_raw("0.001", "0.1");
    let outside_admission = CanonicalRawV1Admission::admit(outside_bytes.as_bytes()).unwrap();
    let outside = replay_raw_mixed_planar_chain_exact_rational_v04(&outside_admission).unwrap();
    assert!(outside.obligations()[3].satisfied());
    assert!(outside.obligations()[4].satisfied());
    assert!(!outside.obligations()[5].satisfied());
    assert!(outside.target_preimage().is_some());
    assert!(matches!(
        outside.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));

    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let segment = first_segment(raw);
    let repeated = vec![segment; 257];
    let cap_bytes = with_segment_word(raw, &repeated);
    let cap_admission = CanonicalRawV1Admission::admit(cap_bytes.as_bytes()).unwrap();
    let cap = replay_raw_mixed_planar_chain_exact_rational_v04(&cap_admission).unwrap();
    assert_eq!(cap.certified_segment_count(), 0);
    assert!(cap.segment_replays().is_empty());
    assert!(matches!(
        cap.replay_failure(),
        Some(MixedChainReplayFailure::SegmentLimitExceeded {
            actual: 257,
            limit: 256,
        })
    ));
    assert!(matches!(
        cap.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));
}

#[test]
fn initial_chart_semantic_defect_returns_empty_false_replay() {
    let raw = one_bridge_raw("9.094947017729282e-13", "0.1");
    let malformed = raw.replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    assert_ne!(malformed, raw);
    let admission = CanonicalRawV1Admission::admit(malformed.as_bytes()).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(
        replay
            .obligations()
            .iter()
            .map(|row| row.satisfied())
            .collect::<Vec<_>>(),
        [false, false, false, false, false, false, false, false]
    );
    assert!(matches!(
        replay.replay_failure(),
        Some(MixedChainReplayFailure::InitialChartSemantic { .. })
    ));
    assert_eq!(replay.certified_segment_count(), 0);
    assert_eq!(replay.failed_segment_index(), None);
    assert!(replay.failed_local_obligation_ids().is_empty());
    assert!(replay.initial_chart_replay().is_none());
    assert!(replay.root_replay().is_none());
    assert!(replay.segment_replays().is_empty());
    assert!(replay.clock_ledger().is_empty());
    assert!(replay.cocycle_ledger().is_empty());
    assert!(replay.current_chart().is_none());
    assert!(replay.current_clock_origin().is_none());
    assert!(replay.target_preimage().is_none());
    assert!(replay.final_tube_replay().is_none());
    assert!(replay.final_enclosure().is_none());
    assert!(replay.maximum_component_width().is_none());
    assert!(replay.covered_physical_interval().is_none());
    assert!(replay.retained_region().is_none());
    assert!(!replay.mathematical_to_target());
    assert!(!replay.profile_satisfied());
}

#[test]
fn first_ordinary_false_and_malformed_targets_roll_back_without_panicking() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let one = one_bridge_raw("9.094947017729282e-13", "0.1");
    let false_bytes = one.replacen(
        "\"chart_id\":\"review-v03:n:chart:1\",\"initial_error_bound\":1e-10",
        "\"chart_id\":\"review-v03:n:chart:1\",\"initial_error_bound\":0.0",
        1,
    );
    let false_admission = CanonicalRawV1Admission::admit(false_bytes.as_bytes()).unwrap();
    let false_replay = replay_raw_mixed_planar_chain_exact_rational_v04(&false_admission).unwrap();
    assert_eq!(false_replay.certified_segment_count(), 0);
    assert_eq!(false_replay.failed_segment_index(), Some(0));
    assert_eq!(false_replay.clock_ledger().len(), 1);
    assert_eq!(false_replay.cocycle_ledger().len(), 0);
    assert_eq!(false_replay.segment_replays().len(), 1);
    assert_eq!(
        false_replay.failed_local_obligation_ids(),
        &["ordinary_bridge_target_initial_ball_contains_complete_source_endpoint"]
    );
    assert!(matches!(
        false_replay.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));

    let malformed_full = raw.replacen(
        "\"certificate_id\":\"review-v03:n:certificate:1\",\"chart_id\":\"review-v03:n:chart:1\",\"chart_type\":\"ordinary_taylor\"",
        "\"certificate_id\":\"review-v03:n:certificate:1\",\"chart_id\":\"review-v03:n:chart:1\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    let malformed = replace_requested(
        with_segment_word(&malformed_full, &[first_segment(&malformed_full)]),
        "9.094947017729282e-13",
        "0.1",
    );
    let malformed_admission = CanonicalRawV1Admission::admit(malformed.as_bytes()).unwrap();
    let malformed_replay =
        replay_raw_mixed_planar_chain_exact_rational_v04(&malformed_admission).unwrap();
    assert_eq!(malformed_replay.certified_segment_count(), 0);
    assert_eq!(malformed_replay.failed_segment_index(), Some(0));
    assert!(malformed_replay.segment_replays().is_empty());
    assert!(matches!(
        malformed_replay.replay_failure(),
        Some(MixedChainReplayFailure::OrdinaryTargetSemantic {
            segment_index: 0,
            ..
        })
    ));
    assert!(matches!(
        malformed_replay.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));
}

fn first_lc_case(mutated_full: &str) -> String {
    let segments = top_level_segments(mutated_full);
    assert!(segments.len() >= 2);
    with_segment_word(mutated_full, &segments[..2])
}

#[test]
fn malformed_first_lc_target_still_retains_independent_lc_right() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let mutated = raw.replacen(
        "\"certificate_id\":\"review-v03:n:certificate:2\",\"chart_id\":\"review-v03:n:chart:2\",\"chart_type\":\"ordinary_taylor\"",
        "\"certificate_id\":\"review-v03:n:certificate:2\",\"chart_id\":\"review-v03:n:chart:2\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    assert_ne!(mutated, raw);
    let bytes = first_lc_case(&mutated);
    let admission = CanonicalRawV1Admission::admit(bytes.as_bytes()).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(replay.certified_segment_count(), 1);
    assert_eq!(replay.failed_segment_index(), Some(1));
    assert_eq!(replay.clock_ledger().len(), 2);
    assert_eq!(replay.cocycle_ledger().len(), 1);
    assert_eq!(replay.segment_replays().len(), 1);
    assert!(replay.failed_local_obligation_ids().is_empty());
    assert!(matches!(
        replay.replay_failure(),
        Some(MixedChainReplayFailure::PlanarLcExitKernel {
            segment_index: 1,
            source: three_body_planar_chain_verifier::CarriedPlanarLcExitError::TargetSemantic(_),
        })
    ));
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        "review-v03:n:chart:1"
    );
    let MixedChainRetainedRegion::LiftedPlanarLcRight(frontier) =
        replay.retained_region().expect("independent LC right")
    else {
        panic!("malformed LC target erased the independently certified LC right")
    };
    assert_eq!(frontier.failed_segment_index(), 1);
    assert_eq!(frontier.pair(), [0, 1]);
    assert_eq!(frontier.lifted_state().components().len(), 14);
}

#[test]
fn first_lc_entry_containment_failure_rolls_back_to_ordinary_right() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let mutated = raw.replacen(
        "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":1e-07",
        "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":0.0",
        1,
    );
    assert_ne!(mutated, raw);
    let bytes = first_lc_case(&mutated);
    let admission = CanonicalRawV1Admission::admit(bytes.as_bytes()).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(replay.certified_segment_count(), 1);
    assert_eq!(replay.failed_segment_index(), Some(1));
    assert_eq!(replay.clock_ledger().len(), 2);
    assert_eq!(replay.cocycle_ledger().len(), 1);
    assert_eq!(replay.segment_replays().len(), 2);
    assert!(replay
        .failed_local_obligation_ids()
        .contains(&"carried_lc_exit_entry_freshly_replayed_and_certified"));
    assert!(matches!(
        replay.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        "review-v03:n:chart:1"
    );
}

#[test]
fn first_lc_tube_failure_before_right_rolls_back_to_ordinary_right() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let mutated = raw.replacen(
        "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":1e-07,\"max_defect_bound\":1.0",
        "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":1e-07,\"max_defect_bound\":0.0",
        1,
    );
    assert_ne!(mutated, raw);
    let bytes = first_lc_case(&mutated);
    let admission = CanonicalRawV1Admission::admit(bytes.as_bytes()).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(replay.certified_segment_count(), 1);
    assert_eq!(replay.failed_segment_index(), Some(1));
    assert_eq!(replay.clock_ledger().len(), 2);
    assert_eq!(replay.cocycle_ledger().len(), 1);
    assert_eq!(replay.segment_replays().len(), 2);
    assert!(replay
        .failed_local_obligation_ids()
        .contains(&"carried_lc_exit_lc_tube_freshly_certified"));
    assert!(matches!(
        replay.retained_region(),
        Some(MixedChainRetainedRegion::CurrentOrdinaryRight(_))
    ));
    assert_eq!(
        replay.current_chart().unwrap().chart_id(),
        "review-v03:n:chart:1"
    );
}

#[test]
fn mismatched_exit_source_still_retains_the_independent_chart_right() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let mutated = raw.replacen(
        "\"source_chart_id\":\"review-v03:lc:chart:0\",\"source_parameter\":9.5367431640625e-07",
        "\"source_chart_id\":\"review-v03:lc:chart:0\",\"source_parameter\":4.76837158203125e-07",
        1,
    );
    assert_ne!(mutated, raw);
    let bytes = first_lc_case(&mutated);
    let admission = CanonicalRawV1Admission::admit(bytes.as_bytes()).unwrap();
    let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();

    assert_eq!(replay.certified_segment_count(), 1);
    assert_eq!(replay.failed_segment_index(), Some(1));
    assert_eq!(replay.clock_ledger().len(), 2);
    assert_eq!(replay.cocycle_ledger().len(), 1);
    let MixedPlanarSegmentReplay::PlanarLc(failed) = &replay.segment_replays()[1] else {
        panic!("exit-source mismatch did not retain its failed local replay")
    };
    assert!(!failed.replay().obligations()[7].satisfied());
    assert!(failed.replay().lifted_exit_slice().is_none());
    let MixedChainRetainedRegion::LiftedPlanarLcRight(frontier) =
        replay.retained_region().expect("independent chart right")
    else {
        panic!("exit-source mismatch incorrectly erased the chart right")
    };
    assert_eq!(frontier.failed_segment_index(), 1);
    assert_eq!(frontier.pair(), [0, 1]);
    assert_eq!(frontier.lifted_state().components().len(), 14);
    let SegmentWire::PlanarLcPassage(passage) = &admission.wire().segments[1] else {
        unreachable!()
    };
    assert_eq!(
        frontier.right_parameter(),
        passage.lc_chart.parameter_interval[1].binary64_rational()
    );
}
