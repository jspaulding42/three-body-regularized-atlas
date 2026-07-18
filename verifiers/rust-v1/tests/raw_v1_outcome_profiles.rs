use serde_json::Value;

use three_body_planar_chain_verifier::{
    replay_admitted_raw_v1_outcome_exact_rational_v04, CanonicalRawV1Admission,
    MixedChainReplayFailure, RawV1OutcomeSegmentKind, RawV1OutcomeStatus,
    EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID, RAW_V1_OUTCOME_OBLIGATION_IDS,
    RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

const SUCCESS_SHA256: &str = "ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11";
const FAILED_REVISIT_SHA256: &str =
    "c6830919726c22fbca6e45d5781b2465e54876bfa89c25bc26e97284ff918727";

#[test]
fn canonical_success_has_complete_local_outcome_and_portable_projection() {
    let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert_eq!(
        outcome.profile_id(),
        EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    );
    assert_eq!(
        outcome.result_schema_id(),
        RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID
    );
    assert_eq!(outcome.evidence_sha256(), SUCCESS_SHA256);
    assert_eq!(outcome.status(), RawV1OutcomeStatus::CertifiedToT);
    assert!(outcome.certified_to_t());
    assert_eq!(outcome.first_failed_obligation(), None);
    assert_eq!(outcome.certified_segment_count(), 5);
    assert_eq!(outcome.failed_segment_index(), None);
    assert!(outcome.failed_local_obligation_ids().is_empty());
    assert_eq!(outcome.obligations().len(), 13);
    assert_eq!(
        outcome
            .obligations()
            .iter()
            .map(|row| row.id())
            .collect::<Vec<_>>(),
        RAW_V1_OUTCOME_OBLIGATION_IDS
    );
    assert!(outcome.obligations().iter().all(|row| row.satisfied()));
    assert_eq!(outcome.mixed_replay().clock_ledger().len(), 6);
    assert_eq!(
        outcome
            .mixed_replay()
            .final_enclosure()
            .unwrap()
            .position_intervals()
            .len()
            + outcome
                .mixed_replay()
                .final_enclosure()
                .unwrap()
                .velocity_intervals()
                .len(),
        12
    );
    assert!(outcome.mixed_replay().retained_region().is_none());

    let profiles = outcome.segment_profiles();
    assert_eq!(profiles.len(), 5);
    assert_eq!(profiles[0].kind(), RawV1OutcomeSegmentKind::OrdinaryBridge);
    assert_eq!(profiles[0].pair(), None);
    assert_eq!(
        profiles[1..]
            .iter()
            .map(|segment| segment.pair().unwrap())
            .collect::<Vec<_>>(),
        [[0, 1], [0, 2], [1, 2], [0, 1]]
    );

    let json = deterministic_json(&outcome);
    assert_eq!(json["schema"], RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID);
    assert_eq!(
        json["profile"],
        EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    );
    assert_eq!(json["status"], "CERTIFIED_TO_T");
    assert_eq!(json["evidence_sha256"], SUCCESS_SHA256);
    assert_eq!(json["top_level_obligations"].as_array().unwrap().len(), 13);
    assert_eq!(json["clock_ledger"].as_array().unwrap().len(), 6);
    assert_eq!(json["segment_profiles"].as_array().unwrap().len(), 5);
    assert_eq!(json["segment_profiles"][0]["kind"], "ordinary_bridge");
    assert!(json["segment_profiles"][0]["pair"].is_null());
    assert_eq!(
        json["segment_profiles"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
            .map(|segment| segment["pair"].clone())
            .collect::<Vec<_>>(),
        [
            serde_json::json!([0, 1]),
            serde_json::json!([0, 2]),
            serde_json::json!([1, 2]),
            serde_json::json!([0, 1])
        ]
    );
    assert_eq!(
        json["request"]["target_physical_time"]["numerator"],
        "2308213854898857"
    );
    assert_eq!(
        json["request"]["target_physical_time"]["denominator"],
        "288230376151711744"
    );
    let components = json["final_enclosure"]["components"].as_array().unwrap();
    assert_eq!(components.len(), 12);
    assert_eq!(
        components
            .iter()
            .map(|component| component["component"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["q1x", "q1y", "q2x", "q2y", "q3x", "q3y", "v1x", "v1y", "v2x", "v2y", "v3x", "v3y"]
    );
    assert!(json["retained_region"].is_null());
    assert_all_rational_parts_are_strings(&json);
}

#[test]
fn failed_revisit_reports_nested_row_nineteen_and_full_lc_frontier() {
    let admission = CanonicalRawV1Admission::admit(FAILED_REVISIT_RAW).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert_eq!(outcome.evidence_sha256(), FAILED_REVISIT_SHA256);
    assert_eq!(outcome.status(), RawV1OutcomeStatus::Unresolved);
    assert!(!outcome.certified_to_t());
    assert_eq!(outcome.certified_segment_count(), 4);
    assert_eq!(outcome.failed_segment_index(), Some(4));
    assert_eq!(
        outcome.failed_local_obligation_ids(),
        &["carried_lc_exit_target_initial_ball_contains_complete_projection"]
    );
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("segment[4]:carried_lc_exit_target_initial_ball_contains_complete_projection")
    );
    assert_eq!(
        outcome
            .obligations()
            .iter()
            .map(|row| row.satisfied())
            .collect::<Vec<_>>(),
        [true, true, true, true, true, true, true, false, false, false, false, false, false]
    );

    let json = deterministic_json(&outcome);
    assert_eq!(json["status"], "UNRESOLVED");
    assert_eq!(json["certified_segment_count"], 4);
    assert_eq!(json["failed_segment_index"], 4);
    assert_eq!(
        json["first_failed_obligation"],
        "segment[4]:carried_lc_exit_target_initial_ball_contains_complete_projection"
    );
    assert_eq!(
        json["retained_region"]["region_type"],
        "certified_lifted_lc_right_frontier"
    );
    assert_eq!(
        json["retained_region"]["coordinate_system"],
        "planar_lc_lifted_14"
    );
    assert_eq!(json["retained_region"]["pair"], serde_json::json!([0, 1]));
    let components = json["retained_region"]["components"].as_array().unwrap();
    assert_eq!(components.len(), 14);
    assert_eq!(
        components
            .iter()
            .map(|component| component["component"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["zx", "zy", "wx", "wy", "h", "Rx", "Ry", "Ux", "Uy", "yx", "yy", "Vx", "Vy", "t"]
    );
    assert!(json["final_enclosure"].is_null());
    assert!(json["maximum_final_component_width"].is_null());
    assert_all_rational_parts_are_strings(&json);
}

#[test]
fn outer_width_failure_has_priority_over_the_mixed_width_row() {
    let raw = one_bridge_success_raw().replacen(
        "\"requested_maximum_component_width\":0.1",
        "\"requested_maximum_component_width\":-0.1",
        1,
    );
    let admission = CanonicalRawV1Admission::admit(raw.as_bytes()).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert!(!outcome.obligations()[4].satisfied());
    assert!(!outcome.obligations()[12].satisfied());
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("raw_planar_chain_requested_width_admissible")
    );
    assert_eq!(outcome.status(), RawV1OutcomeStatus::Unresolved);

    let bytes = outcome.to_portable_json_bytes().unwrap();
    assert!(!bytes.ends_with(b"\n"));
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["retained_region"]["coordinate_system"],
        "planar_cartesian_12"
    );
    let components = json["retained_region"]["components"].as_array().unwrap();
    assert_eq!(components.len(), 12);
    assert_eq!(
        components
            .iter()
            .map(|component| component["component"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["q1x", "q1y", "q2x", "q2y", "q3x", "q3y", "v1x", "v1y", "v2x", "v2y", "v3x", "v3y"]
    );
}

#[test]
fn outer_namespace_failure_has_priority_over_nested_local_failures() {
    let raw = one_bridge_success_raw().replacen(
        "\"certificate_id\":\"review-v03:n:certificate:1\"",
        "\"certificate_id\":\"review-v03:n:certificate:0\"",
        1,
    );
    let admission = CanonicalRawV1Admission::admit(raw.as_bytes()).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert!(!outcome.obligations()[1].satisfied());
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("raw_planar_chain_global_identifier_namespace_unique")
    );
    assert_eq!(outcome.status(), RawV1OutcomeStatus::Unresolved);
}

#[test]
fn admitted_target_semantic_diagnostic_is_serializable_unresolved() {
    let raw = one_bridge_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:1\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:1\",\"chart_type\":\"ordinary_taylor_bad\"",
        1,
    );
    let admission = CanonicalRawV1Admission::admit(raw.as_bytes()).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert_eq!(outcome.status(), RawV1OutcomeStatus::Unresolved);
    assert_eq!(outcome.certified_segment_count(), 0);
    assert_eq!(outcome.failed_segment_index(), Some(0));
    assert!(outcome.failed_local_obligation_ids().is_empty());
    assert!(matches!(
        outcome.mixed_replay().replay_failure(),
        Some(MixedChainReplayFailure::OrdinaryTargetSemantic {
            segment_index: 0,
            ..
        })
    ));
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("raw_planar_chain_all_segments_freshly_folded")
    );
    let bytes = outcome.to_portable_json_bytes().unwrap();
    assert!(!bytes
        .windows("UnexpectedChartType".len())
        .any(|window| window == b"UnexpectedChartType"));
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "UNRESOLVED");
}

#[test]
fn actual_nested_root_boolean_has_stable_root_prefixed_failure_id() {
    let raw = one_bridge_success_raw().replacen("\"tail_bound\":1e-11", "\"tail_bound\":-1e-11", 1);
    let admission = CanonicalRawV1Admission::admit(raw.as_bytes()).unwrap();
    let outcome = replay_admitted_raw_v1_outcome_exact_rational_v04(&admission).unwrap();

    assert!(outcome.obligations()[5].satisfied());
    assert!(!outcome.obligations()[6].satisfied());
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("root:tail_bound_admissible")
    );
    assert_eq!(outcome.status(), RawV1OutcomeStatus::Unresolved);
}

fn deterministic_json(outcome: &three_body_planar_chain_verifier::RawV1Outcome) -> Value {
    let first = outcome.to_portable_json_bytes().unwrap();
    let second = outcome.to_portable_json_bytes().unwrap();
    assert_eq!(first, second);
    assert!(!first.ends_with(b"\n"));
    assert!(first.starts_with(
        b"{\"schema\":\"raw-v1-rust-semantic-outcome-v1\",\"profile\":\"exact_rational_admitted_raw_v1_outcome_v04\""
    ));
    serde_json::from_slice(&first).unwrap()
}

fn one_bridge_success_raw() -> String {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let bytes = raw.as_bytes();
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut first_end = None;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
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
                    first_end = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let first_end = first_end.unwrap();
    let segments_end = raw.rfind("],\"source\":").unwrap();
    format!(
        "{}{}{}",
        &raw[..start],
        &raw[start..=first_end],
        &raw[segments_end..]
    )
}

fn assert_all_rational_parts_are_strings(value: &Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                assert_all_rational_parts_are_strings(value);
            }
        }
        Value::Object(object) => {
            if let (Some(numerator), Some(denominator)) =
                (object.get("numerator"), object.get("denominator"))
            {
                assert!(numerator.is_string());
                assert!(denominator.is_string());
            }
            for value in object.values() {
                assert_all_rational_parts_are_strings(value);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}
