use std::{collections::BTreeSet, fs, path::Path};

use three_body_planar_chain_verifier::{
    execute_raw_v1_bytes_exact_rational_v04, RawV1EvaluationErrorStage, RawV1EvaluationOutcome,
    RawV1ParseOutcome, RawV1RejectionStage, EXACT_RATIONAL_RAW_V1_EXECUTION_V04_PROFILE_ID,
    RAW_V1_RUST_EXECUTION_V1_SCHEMA_ID,
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
    stage: RawV1RejectionStage,
}

const REJECTED_CASES: &[RejectedCase] = &[
    RejectedCase {
        file_name: "reject-alternate-real-spelling.json",
        bytes: corpus_bytes!("reject-alternate-real-spelling.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-decoded-duplicate-unicode-key.json",
        bytes: corpus_bytes!("reject-decoded-duplicate-unicode-key.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-duplicate-outer-key.json",
        bytes: corpus_bytes!("reject-duplicate-outer-key.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-invalid-utf8.bin",
        bytes: corpus_bytes!("reject-invalid-utf8.bin"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-leading-whitespace.json",
        bytes: corpus_bytes!("reject-leading-whitespace.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-lone-surrogate.json",
        bytes: corpus_bytes!("reject-lone-surrogate.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-missing-outer-field.json",
        bytes: corpus_bytes!("reject-missing-outer-field.json"),
        stage: RawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-nonfinite-token.json",
        bytes: corpus_bytes!("reject-nonfinite-token.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-overflow-real.json",
        bytes: corpus_bytes!("reject-overflow-real.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-trailing-newline.json",
        bytes: corpus_bytes!("reject-trailing-newline.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-unknown-outer-field.json",
        bytes: corpus_bytes!("reject-unknown-outer-field.json"),
        stage: RawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-unknown-segment-tag.json",
        bytes: corpus_bytes!("reject-unknown-segment-tag.json"),
        stage: RawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-unsorted-outer-keys.json",
        bytes: corpus_bytes!("reject-unsorted-outer-keys.json"),
        stage: RawV1RejectionStage::CanonicalWire,
    },
    RejectedCase {
        file_name: "reject-wrong-interval-length.json",
        bytes: corpus_bytes!("reject-wrong-interval-length.json"),
        stage: RawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-wrong-outer-scalar-class.json",
        bytes: corpus_bytes!("reject-wrong-outer-scalar-class.json"),
        stage: RawV1RejectionStage::Schema,
    },
    RejectedCase {
        file_name: "reject-wrong-pair-length.json",
        bytes: corpus_bytes!("reject-wrong-pair-length.json"),
        stage: RawV1RejectionStage::Schema,
    },
];

#[test]
fn explicit_execution_table_covers_exactly_the_rejected_corpus_files() {
    let table_names = REJECTED_CASES
        .iter()
        .map(|case| case.file_name.to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(table_names.len(), REJECTED_CASES.len());

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
    assert_eq!(REJECTED_CASES.len(), 16);
}

#[test]
fn every_rejected_case_has_stable_stage_and_not_run_json() {
    for case in REJECTED_CASES {
        let execution = execute_raw_v1_bytes_exact_rational_v04(case.bytes);
        assert_eq!(
            execution.profile_id(),
            EXACT_RATIONAL_RAW_V1_EXECUTION_V04_PROFILE_ID
        );
        assert_eq!(
            execution.result_schema_id(),
            RAW_V1_RUST_EXECUTION_V1_SCHEMA_ID
        );
        assert_eq!(execution.parse_outcome(), RawV1ParseOutcome::Reject);
        assert_eq!(
            execution.rejection_stage(),
            Some(case.stage),
            "{}",
            case.file_name
        );
        assert_eq!(
            execution.evaluation_outcome(),
            RawV1EvaluationOutcome::NotRun
        );
        assert_eq!(execution.evaluation_error_stage(), None);
        assert!(execution.semantic_result().is_none());
        let source = execution.source_error().unwrap();
        assert!(source.source().is_some());

        let first = execution.to_portable_json_bytes().unwrap();
        let second = execution.to_portable_json_bytes().unwrap();
        assert_eq!(first, second);
        assert!(!first.ends_with(b"\n"));
        let expected = format!(
            "{{\"schema\":\"raw-v1-rust-execution-v1\",\"profile\":\"exact_rational_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"{}\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}}",
            case.stage.as_str()
        );
        assert_eq!(first, expected.as_bytes(), "{}", case.file_name);
    }
}

#[test]
fn zero_segment_result_embeds_the_existing_projection_byte_for_byte() {
    let bytes = zero_segment_success_raw();
    let execution = execute_raw_v1_bytes_exact_rational_v04(bytes.as_bytes());
    assert_eq!(execution.parse_outcome(), RawV1ParseOutcome::Accept);
    assert_eq!(execution.rejection_stage(), None);
    assert_eq!(
        execution.evaluation_outcome(),
        RawV1EvaluationOutcome::Result
    );
    assert_eq!(execution.evaluation_error_stage(), None);
    assert!(execution.source_error().is_none());
    let semantic = execution.semantic_result().unwrap();

    let first = execution.to_portable_json_bytes().unwrap();
    let second = execution.to_portable_json_bytes().unwrap();
    let nested = semantic.to_portable_json_bytes().unwrap();
    assert_eq!(first, second);
    assert!(!first.ends_with(b"\n"));
    let marker = b"\"semantic_result\":";
    let marker_index = first
        .windows(marker.len())
        .position(|window| window == marker)
        .unwrap();
    let nested_start = marker_index + marker.len();
    assert_eq!(&first[nested_start..first.len() - 1], nested);
}

#[test]
fn admitted_initial_semantic_failure_is_evaluation_error_without_theorem_json() {
    let bytes = zero_segment_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor_bad\"",
        1,
    );
    let execution = execute_raw_v1_bytes_exact_rational_v04(bytes.as_bytes());
    assert_eq!(execution.parse_outcome(), RawV1ParseOutcome::Accept);
    assert_eq!(execution.rejection_stage(), None);
    assert_eq!(
        execution.evaluation_outcome(),
        RawV1EvaluationOutcome::Error
    );
    assert_eq!(
        execution.evaluation_error_stage(),
        Some(RawV1EvaluationErrorStage::MixedReplay)
    );
    assert!(execution.semantic_result().is_none());
    let source = execution.source_error().unwrap();
    assert!(source.source().is_some());
    assert_eq!(
        execution.to_portable_json_bytes().unwrap(),
        b"{\"schema\":\"raw-v1-rust-execution-v1\",\"profile\":\"exact_rational_raw_v1_execution_v04\",\"parse_outcome\":\"ACCEPT\",\"rejection_stage\":null,\"evaluation_outcome\":\"ERROR\",\"evaluation_error_stage\":\"MIXED_REPLAY\",\"semantic_result\":null}"
    );
}

fn zero_segment_success_raw() -> String {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let end = raw.rfind("],\"source\":").unwrap();
    format!("{}{}", &raw[..start], &raw[end..])
}
