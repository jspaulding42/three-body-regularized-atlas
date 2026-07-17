use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use three_body_planar_chain_verifier::raw_schema::{
    decode_raw_chain, SchemaDecodeErrorKind, SchemaProfile,
};
use three_body_planar_chain_verifier::{
    validate_canonical_wire_json, WireJsonError, DEFAULT_WIRE_JSON_LIMITS,
};

#[derive(Clone, Copy, Debug)]
enum ExpectedOutcome {
    Accept,
    Wire(WireFailure),
    Schema(SchemaFailure),
}

#[derive(Clone, Copy, Debug)]
enum WireFailure {
    NonCanonical,
    DuplicateDecodedKey(&'static str),
    InvalidUtf8,
    UnexpectedByte(u8),
    InvalidNumber,
    LoneUnicodeSurrogate,
}

#[derive(Clone, Copy, Debug)]
struct SchemaFailure {
    path: &'static str,
    kind: SchemaFailureKind,
}

#[derive(Clone, Copy, Debug)]
enum SchemaFailureKind {
    UnknownField,
    MissingField,
    ExpectedInteger,
    UnknownSegmentType(&'static str),
    WrongArrayLength { expected: usize, actual: usize },
}

#[derive(Clone, Copy, Debug)]
struct CorpusCase {
    file_name: &'static str,
    bytes: &'static [u8],
    expected: ExpectedOutcome,
}

macro_rules! corpus_bytes {
    ($file_name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/raw-v1/inputs/",
            $file_name
        ))
    };
}

// This table is intentionally independent of the JSON expectation records.
// A conforming verifier must not use the same parser to discover both the
// input outcome and its expected answer.
const CASES: &[CorpusCase] = &[
    CorpusCase {
        file_name: "failed-revisit.raw.json",
        bytes: corpus_bytes!("failed-revisit.raw.json"),
        expected: ExpectedOutcome::Accept,
    },
    CorpusCase {
        file_name: "reject-alternate-real-spelling.json",
        bytes: corpus_bytes!("reject-alternate-real-spelling.json"),
        expected: ExpectedOutcome::Wire(WireFailure::NonCanonical),
    },
    CorpusCase {
        file_name: "reject-decoded-duplicate-unicode-key.json",
        bytes: corpus_bytes!("reject-decoded-duplicate-unicode-key.json"),
        expected: ExpectedOutcome::Wire(WireFailure::DuplicateDecodedKey("source")),
    },
    CorpusCase {
        file_name: "reject-duplicate-outer-key.json",
        bytes: corpus_bytes!("reject-duplicate-outer-key.json"),
        expected: ExpectedOutcome::Wire(WireFailure::DuplicateDecodedKey("certificate_id")),
    },
    CorpusCase {
        file_name: "reject-invalid-utf8.bin",
        bytes: corpus_bytes!("reject-invalid-utf8.bin"),
        expected: ExpectedOutcome::Wire(WireFailure::InvalidUtf8),
    },
    CorpusCase {
        file_name: "reject-leading-whitespace.json",
        bytes: corpus_bytes!("reject-leading-whitespace.json"),
        expected: ExpectedOutcome::Wire(WireFailure::NonCanonical),
    },
    CorpusCase {
        file_name: "reject-lone-surrogate.json",
        bytes: corpus_bytes!("reject-lone-surrogate.json"),
        expected: ExpectedOutcome::Wire(WireFailure::LoneUnicodeSurrogate),
    },
    CorpusCase {
        file_name: "reject-missing-outer-field.json",
        bytes: corpus_bytes!("reject-missing-outer-field.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.source",
            kind: SchemaFailureKind::MissingField,
        }),
    },
    CorpusCase {
        file_name: "reject-nonfinite-token.json",
        bytes: corpus_bytes!("reject-nonfinite-token.json"),
        expected: ExpectedOutcome::Wire(WireFailure::UnexpectedByte(b'N')),
    },
    CorpusCase {
        file_name: "reject-overflow-real.json",
        bytes: corpus_bytes!("reject-overflow-real.json"),
        expected: ExpectedOutcome::Wire(WireFailure::InvalidNumber),
    },
    CorpusCase {
        file_name: "reject-trailing-newline.json",
        bytes: corpus_bytes!("reject-trailing-newline.json"),
        expected: ExpectedOutcome::Wire(WireFailure::NonCanonical),
    },
    CorpusCase {
        file_name: "reject-unknown-outer-field.json",
        bytes: corpus_bytes!("reject-unknown-outer-field.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.zz_unknown",
            kind: SchemaFailureKind::UnknownField,
        }),
    },
    CorpusCase {
        file_name: "reject-unknown-segment-tag.json",
        bytes: corpus_bytes!("reject-unknown-segment-tag.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.segments[0].segment_type",
            kind: SchemaFailureKind::UnknownSegmentType("ordinary_bridge_v2"),
        }),
    },
    CorpusCase {
        file_name: "reject-unsorted-outer-keys.json",
        bytes: corpus_bytes!("reject-unsorted-outer-keys.json"),
        expected: ExpectedOutcome::Wire(WireFailure::NonCanonical),
    },
    CorpusCase {
        file_name: "reject-wrong-interval-length.json",
        bytes: corpus_bytes!("reject-wrong-interval-length.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.initial_chart.parameter_interval",
            kind: SchemaFailureKind::WrongArrayLength {
                expected: 2,
                actual: 1,
            },
        }),
    },
    CorpusCase {
        file_name: "reject-wrong-outer-scalar-class.json",
        bytes: corpus_bytes!("reject-wrong-outer-scalar-class.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.schema_version",
            kind: SchemaFailureKind::ExpectedInteger,
        }),
    },
    CorpusCase {
        file_name: "reject-wrong-pair-length.json",
        bytes: corpus_bytes!("reject-wrong-pair-length.json"),
        expected: ExpectedOutcome::Schema(SchemaFailure {
            path: "$.segments[1].lc_chart.pair",
            kind: SchemaFailureKind::WrongArrayLength {
                expected: 2,
                actual: 1,
            },
        }),
    },
    CorpusCase {
        file_name: "success.raw.json",
        bytes: corpus_bytes!("success.raw.json"),
        expected: ExpectedOutcome::Accept,
    },
];

#[test]
fn explicit_case_table_covers_every_current_raw_v1_input() {
    let table_names = CASES
        .iter()
        .map(|case| case.file_name.to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        table_names.len(),
        CASES.len(),
        "the explicit corpus table contains a duplicate file name"
    );

    let input_directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/raw-v1/inputs");
    let checked_in_names = fs::read_dir(&input_directory)
        .unwrap_or_else(|error| {
            panic!("failed to enumerate {}: {error}", input_directory.display())
        })
        .map(|entry| entry.expect("failed to read a raw-v1 corpus directory entry"))
        .filter(|entry| {
            entry
                .file_type()
                .expect("failed to inspect a raw-v1 corpus directory entry")
                .is_file()
        })
        .map(|entry| {
            entry
                .file_name()
                .into_string()
                .expect("raw-v1 corpus file names must be valid UTF-8")
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(
        checked_in_names, table_names,
        "the independent Rust table must enumerate every current raw-v1 input exactly once"
    );
}

#[test]
fn raw_v1_corpus_has_the_explicit_parser_outcomes() {
    for case in CASES {
        assert!(
            case.bytes.len() <= DEFAULT_WIRE_JSON_LIMITS.max_input_bytes,
            "{} exceeds the verifier's bounded input limit",
            case.file_name
        );

        match case.expected {
            ExpectedOutcome::Accept => {
                let value = validate_canonical_wire_json(case.bytes, DEFAULT_WIRE_JSON_LIMITS)
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} should pass canonical wire validation: {error:?}",
                            case.file_name
                        )
                    });
                decode_raw_chain(value, SchemaProfile::V03Compatible).unwrap_or_else(|error| {
                    panic!(
                        "{} should pass typed raw-v1 decoding: {error:?}",
                        case.file_name
                    )
                });
            }
            ExpectedOutcome::Wire(expected) => {
                let error = validate_canonical_wire_json(case.bytes, DEFAULT_WIRE_JSON_LIMITS)
                    .expect_err("a strict-wire rejection case unexpectedly passed");
                assert_wire_failure(case.file_name, error, expected);
            }
            ExpectedOutcome::Schema(expected) => {
                let value = validate_canonical_wire_json(case.bytes, DEFAULT_WIRE_JSON_LIMITS)
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} should reach typed-schema decoding, but strict-wire validation failed: {error:?}",
                            case.file_name
                        )
                    });
                let error = decode_raw_chain(value, SchemaProfile::V03Compatible)
                    .expect_err("a typed-schema rejection case unexpectedly decoded");
                assert_schema_failure(case.file_name, &error, expected);
            }
        }
    }
}

fn assert_wire_failure(file_name: &str, actual: WireJsonError, expected: WireFailure) {
    let matches = match (&actual, expected) {
        (WireJsonError::NonCanonicalBytes { .. }, WireFailure::NonCanonical) => true,
        (
            WireJsonError::DuplicateObjectKey { key: actual_key },
            WireFailure::DuplicateDecodedKey(expected_key),
        ) => actual_key == expected_key,
        (WireJsonError::InvalidUtf8, WireFailure::InvalidUtf8) => true,
        (
            WireJsonError::UnexpectedByte {
                byte: actual_byte, ..
            },
            WireFailure::UnexpectedByte(expected_byte),
        ) => *actual_byte == expected_byte,
        (WireJsonError::InvalidNumber { .. }, WireFailure::InvalidNumber) => true,
        (WireJsonError::LoneUnicodeSurrogate { .. }, WireFailure::LoneUnicodeSurrogate) => true,
        _ => false,
    };
    assert!(
        matches,
        "{file_name} failed at the strict-wire stage with {actual:?}, expected {expected:?}"
    );
}

fn assert_schema_failure(
    file_name: &str,
    actual: &three_body_planar_chain_verifier::raw_schema::SchemaDecodeError,
    expected: SchemaFailure,
) {
    assert_eq!(
        actual.path(),
        expected.path,
        "{file_name} failed at an unexpected typed-schema path"
    );

    let matches = match (actual.kind(), expected.kind) {
        (SchemaDecodeErrorKind::UnknownField, SchemaFailureKind::UnknownField) => true,
        (SchemaDecodeErrorKind::MissingField, SchemaFailureKind::MissingField) => true,
        (SchemaDecodeErrorKind::ExpectedInteger, SchemaFailureKind::ExpectedInteger) => true,
        (
            SchemaDecodeErrorKind::UnknownSegmentType { tag: actual_tag },
            SchemaFailureKind::UnknownSegmentType(expected_tag),
        ) => actual_tag == expected_tag,
        (
            SchemaDecodeErrorKind::WrongArrayLength {
                expected: actual_expected,
                actual: actual_actual,
            },
            SchemaFailureKind::WrongArrayLength {
                expected: expected_expected,
                actual: expected_actual,
            },
        ) => *actual_expected == expected_expected && *actual_actual == expected_actual,
        _ => false,
    };
    assert!(
        matches,
        "{file_name} failed with typed-schema kind {:?}, expected {:?}",
        actual.kind(),
        expected.kind
    );
}
