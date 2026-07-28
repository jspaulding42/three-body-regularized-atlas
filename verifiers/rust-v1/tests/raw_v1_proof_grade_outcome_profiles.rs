use serde_json::Value;

use three_body_planar_chain_verifier::{
    replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04, CanonicalRawV1Admission,
    MixedChainRetainedRegion, OrdinaryChartReplayError, ProofGradeRawV1Outcome,
    ProofGradeRawV1OutcomeSegmentKind, ProofGradeRawV1OutcomeStatus,
    EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
    EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID,
    EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID,
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS, PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS,
    RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID,
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

fn outcome(bytes: &[u8]) -> ProofGradeRawV1Outcome {
    let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
    replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04(&admission).unwrap()
}

#[test]
fn canonical_success_has_exact_proof_grade_contract_and_compact_projection() {
    let outcome = outcome(SUCCESS_RAW);

    assert_eq!(
        outcome.profile_id(),
        EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    );
    assert_eq!(
        outcome.result_schema_id(),
        RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID
    );
    assert_eq!(outcome.evidence_sha256(), SUCCESS_SHA256);
    assert_eq!(outcome.status(), ProofGradeRawV1OutcomeStatus::CertifiedToT);
    assert!(outcome.certified_to_t());
    assert_eq!(outcome.first_failed_obligation(), None);
    assert_eq!(outcome.certified_segment_count(), 5);
    assert_eq!(outcome.failed_segment_index(), None);
    assert_eq!(
        outcome
            .obligations()
            .iter()
            .map(|row| row.id())
            .collect::<Vec<_>>(),
        PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS
    );
    assert!(outcome.obligations().iter().all(|row| row.satisfied()));
    assert_eq!(
        &PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[5..],
        PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS
    );
    assert_eq!(outcome.mixed_replay().clock_ledger().len(), 6);
    let enclosure = outcome.mixed_replay().final_enclosure().unwrap();
    assert_eq!(
        enclosure.position_intervals().len() + enclosure.velocity_intervals().len(),
        12
    );

    let profiles = outcome.segment_profiles();
    assert_eq!(profiles.len(), 5);
    assert_eq!(
        profiles[0].kind(),
        ProofGradeRawV1OutcomeSegmentKind::OrdinaryBridge
    );
    assert_eq!(
        profiles[0].profile_id(),
        EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID
    );
    assert_eq!(profiles[0].pair(), None);
    assert!(profiles[1..].iter().all(|profile| {
        profile.kind() == ProofGradeRawV1OutcomeSegmentKind::PlanarLcPassage
            && profile.profile_id()
                == EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID
    }));
    assert_eq!(
        profiles[1..]
            .iter()
            .map(|profile| profile.pair().unwrap())
            .collect::<Vec<_>>(),
        [[0, 1], [0, 2], [1, 2], [0, 1]]
    );

    let (bytes, json) = deterministic_json(&outcome);
    assert_eq!(
        json["schema"],
        RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID
    );
    assert_eq!(
        json["profile"],
        EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    );
    assert_eq!(json["status"], "CERTIFIED_TO_T");
    assert_eq!(json["evidence_sha256"], SUCCESS_SHA256);
    assert_eq!(json["top_level_obligations"].as_array().unwrap().len(), 13);
    assert_eq!(json["clock_ledger"].as_array().unwrap().len(), 6);
    assert_eq!(
        json["final_enclosure"]["components"]
            .as_array()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(
        json["request"]["target_physical_time"]["numerator"],
        "2308213854898857"
    );
    assert_eq!(
        json["request"]["target_physical_time"]["denominator"],
        "288230376151711744"
    );
    assert_no_diagnostic_projection(&bytes);
    assert_all_rational_parts_are_strings(&json);
}

#[test]
fn all_claimed_tail_mutations_preserve_the_admitted_proof_grade_semantics() {
    let baseline = outcome(SUCCESS_RAW);
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let changed = raw.replace("\"tail_bound\":1e-11", "\"tail_bound\":-1.0");
    assert_eq!(changed.matches("\"tail_bound\":-1.0").count(), 10);
    let mutated = outcome(changed.as_bytes());

    assert_eq!(mutated.status(), ProofGradeRawV1OutcomeStatus::CertifiedToT);
    assert_ne!(mutated.evidence_sha256(), baseline.evidence_sha256());
    assert_eq!(
        outcome_bool_vector(&mutated),
        outcome_bool_vector(&baseline)
    );
    assert_eq!(
        mutated.first_failed_obligation(),
        baseline.first_failed_obligation()
    );
    assert_eq!(
        mutated.mixed_replay().clock_ledger(),
        baseline.mixed_replay().clock_ledger()
    );
    assert_eq!(
        mutated.mixed_replay().final_enclosure(),
        baseline.mixed_replay().final_enclosure()
    );
    assert_eq!(
        mutated.mixed_replay().retained_region(),
        baseline.mixed_replay().retained_region()
    );

    let (baseline_bytes, mut baseline_json) = deterministic_json(&baseline);
    let (mutated_bytes, mut mutated_json) = deterministic_json(&mutated);
    baseline_json
        .as_object_mut()
        .unwrap()
        .remove("evidence_sha256");
    mutated_json
        .as_object_mut()
        .unwrap()
        .remove("evidence_sha256");
    assert_eq!(mutated_json, baseline_json);
    assert_no_diagnostic_projection(&baseline_bytes);
    assert_no_diagnostic_projection(&mutated_bytes);
}

#[test]
fn claimed_tail_resource_diagnostic_stays_out_of_the_portable_proof_grade_outcome() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let padded = extend_ordinary_coefficients_to(raw, "review-v03:n:certificate:0", 146);
    let baseline = outcome(SUCCESS_RAW);
    let padded_outcome = outcome(padded.as_bytes());

    assert_eq!(
        padded_outcome.status(),
        ProofGradeRawV1OutcomeStatus::CertifiedToT
    );
    assert!(matches!(
        padded_outcome
            .mixed_replay()
            .initial_claimed_tail_chart_diagnostic(),
        Some(Err(OrdinaryChartReplayError::WorkLimitExceeded {
            required: 2_018_400,
            limit: 2_000_000,
        }))
    ));
    assert_eq!(
        outcome_bool_vector(&padded_outcome),
        outcome_bool_vector(&baseline)
    );
    assert_eq!(
        padded_outcome.first_failed_obligation(),
        baseline.first_failed_obligation()
    );
    assert_eq!(
        padded_outcome.mixed_replay().clock_ledger(),
        baseline.mixed_replay().clock_ledger()
    );
    assert_eq!(
        padded_outcome.mixed_replay().final_enclosure(),
        baseline.mixed_replay().final_enclosure()
    );
    assert_eq!(
        padded_outcome.mixed_replay().retained_region(),
        baseline.mixed_replay().retained_region()
    );

    let (bytes, json) = deterministic_json(&padded_outcome);
    assert_eq!(json["status"], "CERTIFIED_TO_T");
    assert_no_diagnostic_projection(&bytes);
}

#[test]
fn failed_revisit_is_unresolved_at_the_proof_grade_lc_right_frontier() {
    let outcome = outcome(FAILED_REVISIT_RAW);

    assert_eq!(outcome.status(), ProofGradeRawV1OutcomeStatus::Unresolved);
    assert_eq!(outcome.certified_segment_count(), 4);
    assert_eq!(outcome.failed_segment_index(), Some(4));
    assert_eq!(
        outcome.failed_local_obligation_ids(),
        &["proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection"]
    );
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("segment[4]:proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection")
    );
    assert_eq!(
        outcome_bool_vector(&outcome),
        vec![true, true, true, true, true, true, true, false, false, false, false, false, false]
    );
    assert!(matches!(
        outcome.mixed_replay().retained_region(),
        Some(MixedChainRetainedRegion::LiftedPlanarLcRight(frontier))
            if frontier.failed_segment_index() == 4 && frontier.lifted_state().components().len() == 14
    ));
    assert!(outcome.mixed_replay().final_enclosure().is_none());

    let (bytes, json) = deterministic_json(&outcome);
    assert_eq!(
        json["retained_region"]["coordinate_system"],
        "planar_lc_lifted_14"
    );
    assert_eq!(
        json["retained_region"]["components"]
            .as_array()
            .unwrap()
            .len(),
        14
    );
    assert!(json["final_enclosure"].is_null());
    assert_no_diagnostic_projection(&bytes);
}

#[test]
fn outer_negative_requested_width_has_priority_over_later_proof_rows() {
    let raw = one_bridge_success_raw().replacen(
        "\"requested_maximum_component_width\":0.1",
        "\"requested_maximum_component_width\":-0.1",
        1,
    );
    let outcome = outcome(raw.as_bytes());

    assert!(!outcome.obligations()[4].satisfied());
    assert!(!outcome.obligations()[12].satisfied());
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("proof_grade_raw_planar_chain_requested_width_admissible")
    );
    assert_eq!(outcome.status(), ProofGradeRawV1OutcomeStatus::Unresolved);
}

#[test]
fn duplicate_namespace_is_a_boolean_outer_failure_not_a_replay_error() {
    let raw = one_bridge_success_raw().replacen(
        "\"certificate_id\":\"review-v03:n:certificate:1\"",
        "\"certificate_id\":\"review-v03:n:certificate:0\"",
        1,
    );
    let admission = CanonicalRawV1Admission::admit(raw.as_bytes()).unwrap();
    let outcome = replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04(&admission)
        .expect("duplicate namespace is a Boolean proof outcome");

    assert!(!outcome.obligations()[1].satisfied());
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("proof_grade_raw_planar_chain_global_identifier_namespace_unique")
    );
    assert_eq!(outcome.status(), ProofGradeRawV1OutcomeStatus::Unresolved);
}

#[test]
fn malformed_initial_semantics_is_a_proof_root_failure_without_diagnostic_serialization() {
    let raw = one_bridge_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    let outcome = outcome(raw.as_bytes());

    assert_eq!(outcome.status(), ProofGradeRawV1OutcomeStatus::Unresolved);
    assert_eq!(
        outcome_bool_vector(&outcome),
        vec![true, true, true, true, true, false, false, false, false, false, false, false, false]
    );
    assert_eq!(
        outcome.first_failed_obligation(),
        Some("proof_grade_raw_planar_chain_root_exact_point_left_anchor")
    );
    assert!(outcome
        .mixed_replay()
        .initial_claimed_tail_chart_diagnostic()
        .is_none());
    let (bytes, _) = deterministic_json(&outcome);
    assert_no_diagnostic_projection(&bytes);
}

#[test]
fn admission_owns_the_evidence_bytes_used_for_hash_and_projection() {
    let mut source = SUCCESS_RAW.to_vec();
    let admission = CanonicalRawV1Admission::admit(&source).unwrap();
    let outcome =
        replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04(&admission).unwrap();
    let before = outcome.to_portable_json_bytes().unwrap();
    source[0] = b'[';

    assert_eq!(outcome.evidence_sha256(), SUCCESS_SHA256);
    assert_eq!(outcome.to_portable_json_bytes().unwrap(), before);
    assert_eq!(admission.canonical_bytes(), SUCCESS_RAW);
}

fn outcome_bool_vector(outcome: &ProofGradeRawV1Outcome) -> Vec<bool> {
    outcome
        .obligations()
        .iter()
        .map(|row| row.satisfied())
        .collect()
}

fn deterministic_json(outcome: &ProofGradeRawV1Outcome) -> (Vec<u8>, Value) {
    let first = outcome.to_portable_json_bytes().unwrap();
    let second = outcome.to_portable_json_bytes().unwrap();
    assert_eq!(first, second);
    assert!(!first.ends_with(b"\n"));
    assert!(first.starts_with(
        b"{\"schema\":\"raw-v1-rust-proof-grade-semantic-outcome-v1\",\"profile\":\"exact_rational_proof_grade_admitted_raw_v1_outcome_v04\""
    ));
    let parsed = serde_json::from_slice(&first).unwrap();
    (first, parsed)
}

fn assert_no_diagnostic_projection(bytes: &[u8]) {
    for forbidden in [
        b"diagnostic".as_slice(),
        b"tail_bound".as_slice(),
        b"WorkLimitExceeded".as_slice(),
        b"resource".as_slice(),
        b"required".as_slice(),
        b"limit".as_slice(),
    ] {
        assert!(
            !bytes
                .windows(forbidden.len())
                .any(|window| window == forbidden),
            "forbidden diagnostic fragment appeared: {:?}",
            String::from_utf8_lossy(forbidden)
        );
    }
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

fn matching_array_end(raw: &str, start: usize) -> usize {
    let bytes = raw.as_bytes();
    assert_eq!(bytes[start], b'[');
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
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
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    panic!("unterminated array")
}

fn containing_object_start(raw: &str, position: usize) -> usize {
    let mut object_stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in raw.as_bytes().iter().copied().enumerate().take(position) {
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
            b'{' => object_stack.push(index),
            b'}' => {
                object_stack.pop().expect("unbalanced object");
            }
            _ => {}
        }
    }
    *object_stack.last().expect("anchor outside object")
}

fn extend_ordinary_coefficients_to(raw: &str, certificate_id: &str, count: usize) -> String {
    let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
    let zero = "[[0.0,0.0],[0.0,0.0],[0.0,0.0]]";
    let mut result = raw.to_owned();
    for field in ["position_coefficients", "velocity_coefficients"] {
        let anchor_start = result.find(&anchor).expect("missing record anchor");
        let object_start = containing_object_start(&result, anchor_start);
        let marker = format!("\"{field}\":[");
        let relative = result[object_start..]
            .find(&marker)
            .expect("missing anchored coefficient array");
        let array_start = object_start + relative + marker.len() - 1;
        let array_end = matching_array_end(&result, array_start);
        let inner = &result[array_start + 1..array_end];
        let mut depth = 0_usize;
        let mut existing = usize::from(!inner.is_empty());
        for byte in inner.as_bytes() {
            match byte {
                b'[' => depth += 1,
                b']' => depth -= 1,
                b',' if depth == 0 => existing += 1,
                _ => {}
            }
        }
        assert!(existing <= count);
        let padding = vec![zero; count - existing].join(",");
        let replacement = if padding.is_empty() {
            format!("[{inner}]")
        } else {
            format!("[{inner},{padding}]")
        };
        result = format!(
            "{}{}{}",
            &result[..array_start],
            replacement,
            &result[array_end + 1..]
        );
    }
    result
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
