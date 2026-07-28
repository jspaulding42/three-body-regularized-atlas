use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use three_body_planar_chain_verifier::{
    execute_proof_grade_raw_v1_bytes_exact_rational_v04, DEFAULT_WIRE_JSON_LIMITS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));

#[test]
fn cli_reject_writes_only_proof_grade_json_and_exits_zero() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/raw-v1/inputs/reject-trailing-newline.json");
    let output = run_cli(&[path.as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        output.stdout,
        b"{\"schema\":\"raw-v1-rust-proof-grade-execution-v1\",\"profile\":\"exact_rational_proof_grade_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"CANONICAL_WIRE\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}"
    );
    assert!(!output.stdout.ends_with(b"\n"));
}

#[test]
fn cli_result_matches_the_proof_grade_library_envelope_and_has_no_diagnostics() {
    let bytes = zero_segment_success_raw();
    let expected = execute_proof_grade_raw_v1_bytes_exact_rational_v04(bytes.as_bytes())
        .to_portable_json_bytes()
        .unwrap();
    let input = TempInput::from_bytes(bytes.as_bytes());
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, expected);
    assert!(!output.stdout.ends_with(b"\n"));
    assert_no_private_diagnostic_fields(&output.stdout);
}

#[test]
fn cli_admitted_semantic_defect_writes_an_unresolved_result_and_exits_zero() {
    let bytes = zero_segment_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor_bad\"",
        1,
    );
    let expected = execute_proof_grade_raw_v1_bytes_exact_rational_v04(bytes.as_bytes())
        .to_portable_json_bytes()
        .unwrap();
    let input = TempInput::from_bytes(bytes.as_bytes());
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, expected);
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["evaluation_outcome"], "RESULT");
    assert_eq!(json["semantic_result"]["status"], "UNRESOLVED");
    assert_eq!(
        json["semantic_result"]["first_failed_obligation"],
        "proof_grade_raw_planar_chain_root_exact_point_left_anchor"
    );
    assert_no_private_diagnostic_fields(&output.stdout);
}

#[test]
fn cli_mixed_replay_error_matches_the_library_and_uses_exit_two() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let bytes = extend_ordinary_coefficients_to(raw, "review-v03:n:certificate:0", 4_097);
    let expected = execute_proof_grade_raw_v1_bytes_exact_rational_v04(bytes.as_bytes())
        .to_portable_json_bytes()
        .unwrap();
    let input = TempInput::from_bytes(bytes.as_bytes());
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, expected);
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["evaluation_outcome"], "ERROR");
    assert_eq!(json["evaluation_error_stage"], "MIXED_REPLAY");
    assert!(json["semantic_result"].is_null());
    assert_no_private_diagnostic_fields(&output.stdout);
}

#[test]
fn cli_usage_is_non_json_stderr_and_has_the_usage_exit_code() {
    let output = run_cli(&[]);

    assert_eq!(output.status.code(), Some(64));
    assert!(output.stdout.is_empty());
    assert_eq!(
        output.stderr,
        b"usage: raw_v1_proof_grade_verify <raw-v1-path>\n"
    );
}

#[test]
fn cli_sparse_oversize_is_bounded_to_a_canonical_wire_rejection() {
    let input =
        TempInput::sparse(u64::try_from(DEFAULT_WIRE_JSON_LIMITS.max_input_bytes).unwrap() * 16);
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        output.stdout,
        b"{\"schema\":\"raw-v1-rust-proof-grade-execution-v1\",\"profile\":\"exact_rational_proof_grade_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"CANONICAL_WIRE\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}"
    );
    assert!(!output.stdout.ends_with(b"\n"));
}

fn run_cli(arguments: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_raw_v1_proof_grade_verify"))
        .args(arguments)
        .output()
        .unwrap()
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

struct TempInput {
    path: PathBuf,
}

impl TempInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let path = unique_temp_path();
        fs::write(&path, bytes).unwrap();
        Self { path }
    }

    fn sparse(length: u64) -> Self {
        let path = unique_temp_path();
        File::create(&path).unwrap().set_len(length).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempInput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn unique_temp_path() -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "three-body-proof-grade-raw-v1-cli-{}-{id}.json",
        std::process::id()
    ))
}
