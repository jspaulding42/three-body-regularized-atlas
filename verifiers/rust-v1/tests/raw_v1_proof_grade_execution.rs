use std::{collections::BTreeSet, fs, path::Path};

use serde_json::Value;

use three_body_planar_chain_verifier::{
    execute_proof_grade_raw_v1_bytes_exact_rational_v04, ProofGradeRawV1EvaluationErrorStage,
    ProofGradeRawV1EvaluationOutcome, ProofGradeRawV1OutcomeStatus, ProofGradeRawV1ParseOutcome,
    ProofGradeRawV1RejectionStage, EXACT_RATIONAL_PROOF_GRADE_RAW_V1_EXECUTION_V04_PROFILE_ID,
    RAW_V1_RUST_PROOF_GRADE_EXECUTION_V1_SCHEMA_ID,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));

macro_rules! corpus_bytes {
    ($file_name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/raw-v1/inputs/",
            $file_name
        ))
    };
}

#[derive(Clone, Copy, Debug)]
struct RejectedCase {
    file_name: &'static str,
    bytes: &'static [u8],
    stage: ProofGradeRawV1RejectionStage,
}

const REJECTED_CASES: &[RejectedCase] = &[
    RejectedCase {
        file_name: "reject-alternate-real-spelling.json",
        bytes: corpus_bytes!("reject-alternate-real-spelling.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-decoded-duplicate-unicode-key.json",
        bytes: corpus_bytes!("reject-decoded-duplicate-unicode-key.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-duplicate-outer-key.json",
        bytes: corpus_bytes!("reject-duplicate-outer-key.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-invalid-utf8.bin",
        bytes: corpus_bytes!("reject-invalid-utf8.bin"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-leading-whitespace.json",
        bytes: corpus_bytes!("reject-leading-whitespace.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-lone-surrogate.json",
        bytes: corpus_bytes!("reject-lone-surrogate.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-missing-outer-field.json",
        bytes: corpus_bytes!("reject-missing-outer-field.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-nonfinite-token.json",
        bytes: corpus_bytes!("reject-nonfinite-token.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-overflow-real.json",
        bytes: corpus_bytes!("reject-overflow-real.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-trailing-newline.json",
        bytes: corpus_bytes!("reject-trailing-newline.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-unknown-outer-field.json",
        bytes: corpus_bytes!("reject-unknown-outer-field.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-unknown-segment-tag.json",
        bytes: corpus_bytes!("reject-unknown-segment-tag.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-unsorted-outer-keys.json",
        bytes: corpus_bytes!("reject-unsorted-outer-keys.json"),
        stage: ProofGradeRawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-wrong-interval-length.json",
        bytes: corpus_bytes!("reject-wrong-interval-length.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-wrong-outer-scalar-class.json",
        bytes: corpus_bytes!("reject-wrong-outer-scalar-class.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-wrong-pair-length.json",
        bytes: corpus_bytes!("reject-wrong-pair-length.json"),
        stage: ProofGradeRawV1RejectionStage::Schema,
    },
];

#[test]
fn rejection_table_covers_all_sixteen_checked_in_rejection_fixtures() {
    let table_names = REJECTED_CASES
        .iter()
        .map(|case| case.file_name.to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(table_names.len(), REJECTED_CASES.len());
    assert_eq!(REJECTED_CASES.len(), 16);

    let input_directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/raw-v1/inputs");
    let checked_in_names = fs::read_dir(input_directory)
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .map(|entry| entry.file_name().into_string().unwrap())
        .filter(|name| name.starts_with("reject-"))
        .collect::<BTreeSet<_>>();
    assert_eq!(checked_in_names, table_names);
}

#[test]
fn every_rejection_fixture_has_a_total_stable_not_run_execution_result() {
    for case in REJECTED_CASES {
        let execution = execute_proof_grade_raw_v1_bytes_exact_rational_v04(case.bytes);
        assert_eq!(
            execution.profile_id(),
            EXACT_RATIONAL_PROOF_GRADE_RAW_V1_EXECUTION_V04_PROFILE_ID
        );
        assert_eq!(
            execution.result_schema_id(),
            RAW_V1_RUST_PROOF_GRADE_EXECUTION_V1_SCHEMA_ID
        );
        assert_eq!(
            execution.parse_outcome(),
            ProofGradeRawV1ParseOutcome::Reject
        );
        assert_eq!(
            execution.rejection_stage(),
            Some(case.stage),
            "{}",
            case.file_name
        );
        assert_eq!(
            execution.evaluation_outcome(),
            ProofGradeRawV1EvaluationOutcome::NotRun
        );
        assert_eq!(execution.evaluation_error_stage(), None);
        assert!(execution.semantic_result().is_none());
        assert!(execution.source_error().unwrap().source().is_some());

        let first = execution.to_portable_json_bytes().unwrap();
        assert_eq!(first, execution.to_portable_json_bytes().unwrap());
        assert!(!first.ends_with(b"\n"));
        let expected = format!(
            "{{\"schema\":\"raw-v1-rust-proof-grade-execution-v1\",\"profile\":\"exact_rational_proof_grade_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"{}\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}}",
            case.stage.as_str()
        );
        assert_eq!(first, expected.as_bytes(), "{}", case.file_name);
    }
}

#[test]
fn zero_segment_result_embeds_the_proof_grade_semantic_projection_byte_for_byte() {
    let bytes = zero_segment_success_raw();
    let execution = execute_proof_grade_raw_v1_bytes_exact_rational_v04(bytes.as_bytes());
    assert_eq!(
        execution.parse_outcome(),
        ProofGradeRawV1ParseOutcome::Accept
    );
    assert_eq!(execution.rejection_stage(), None);
    assert_eq!(
        execution.evaluation_outcome(),
        ProofGradeRawV1EvaluationOutcome::Result
    );
    assert_eq!(execution.evaluation_error_stage(), None);
    let semantic = execution.semantic_result().unwrap();

    let first = execution.to_portable_json_bytes().unwrap();
    assert_eq!(first, execution.to_portable_json_bytes().unwrap());
    assert!(!first.ends_with(b"\n"));
    let nested = semantic.to_portable_json_bytes().unwrap();
    let marker = b"\"semantic_result\":";
    let marker_index = first
        .windows(marker.len())
        .position(|window| window == marker)
        .unwrap();
    assert_eq!(&first[marker_index + marker.len()..first.len() - 1], nested);
    assert_no_private_diagnostic_fields(&first);
}

#[test]
fn admitted_semantic_defect_is_an_unresolved_result_without_private_diagnostics() {
    let bytes = zero_segment_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor_bad\"",
        1,
    );
    let execution = execute_proof_grade_raw_v1_bytes_exact_rational_v04(bytes.as_bytes());
    assert_eq!(
        execution.parse_outcome(),
        ProofGradeRawV1ParseOutcome::Accept
    );
    assert_eq!(
        execution.evaluation_outcome(),
        ProofGradeRawV1EvaluationOutcome::Result
    );
    assert_eq!(execution.evaluation_error_stage(), None);
    let semantic = execution.semantic_result().unwrap();
    assert_eq!(semantic.status(), ProofGradeRawV1OutcomeStatus::Unresolved);
    assert_eq!(semantic.certified_segment_count(), 0);
    assert_eq!(
        semantic.first_failed_obligation(),
        Some("proof_grade_raw_planar_chain_root_exact_point_left_anchor")
    );

    let bytes = execution.to_portable_json_bytes().unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["evaluation_outcome"], "RESULT");
    assert_eq!(json["semantic_result"]["status"], "UNRESOLVED");
    assert_no_private_diagnostic_fields(&bytes);
}

#[test]
fn actual_initial_resource_exhaustion_is_classified_as_mixed_replay_error() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let padded = extend_ordinary_coefficients_to(raw, "review-v03:n:certificate:0", 4_097);
    let execution = execute_proof_grade_raw_v1_bytes_exact_rational_v04(padded.as_bytes());
    assert_eq!(
        execution.parse_outcome(),
        ProofGradeRawV1ParseOutcome::Accept
    );
    assert_eq!(execution.rejection_stage(), None);
    assert_eq!(
        execution.evaluation_outcome(),
        ProofGradeRawV1EvaluationOutcome::Error
    );
    assert_eq!(
        execution.evaluation_error_stage(),
        Some(ProofGradeRawV1EvaluationErrorStage::MixedReplay)
    );
    assert!(execution.semantic_result().is_none());
    let bytes = execution.to_portable_json_bytes().unwrap();
    assert_eq!(
        bytes,
        b"{\"schema\":\"raw-v1-rust-proof-grade-execution-v1\",\"profile\":\"exact_rational_proof_grade_raw_v1_execution_v04\",\"parse_outcome\":\"ACCEPT\",\"rejection_stage\":null,\"evaluation_outcome\":\"ERROR\",\"evaluation_error_stage\":\"MIXED_REPLAY\",\"semantic_result\":null}"
    );
    assert_no_private_diagnostic_fields(&bytes);
}

fn assert_no_private_diagnostic_fields(bytes: &[u8]) {
    for forbidden in [
        b"diagnostic".as_slice(),
        b"tail_bound".as_slice(),
        b"WorkLimitExceeded".as_slice(),
        b"required".as_slice(),
        b"resource".as_slice(),
    ] {
        assert!(
            !bytes
                .windows(forbidden.len())
                .any(|window| window == forbidden),
            "forbidden fragment appeared: {:?}",
            String::from_utf8_lossy(forbidden)
        );
    }
}

fn zero_segment_success_raw() -> String {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let end = raw.rfind("],\"source\":").unwrap();
    format!("{}{}", &raw[..start], &raw[end..])
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
