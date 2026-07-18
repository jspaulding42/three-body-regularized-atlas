use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use three_body_planar_chain_verifier::{
    execute_raw_v1_bytes_exact_rational_v04, DEFAULT_WIRE_JSON_LIMITS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));

#[test]
fn cli_reject_writes_only_json_and_exits_zero() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/raw-v1/inputs/reject-trailing-newline.json");
    let output = run_cli(&[path.as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        output.stdout,
        b"{\"schema\":\"raw-v1-rust-execution-v1\",\"profile\":\"exact_rational_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"CANONICAL_WIRE\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}"
    );
    assert!(!output.stdout.ends_with(b"\n"));
}

#[test]
fn cli_result_matches_the_library_envelope_and_exits_zero() {
    let bytes = zero_segment_success_raw();
    let expected = execute_raw_v1_bytes_exact_rational_v04(bytes.as_bytes())
        .to_portable_json_bytes()
        .unwrap();
    let input = TempInput::from_bytes(bytes.as_bytes());
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, expected);
    assert!(!output.stdout.ends_with(b"\n"));
}

#[test]
fn cli_evaluation_error_writes_stage_json_and_exits_two() {
    let bytes = zero_segment_success_raw().replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor_bad\"",
        1,
    );
    let expected = execute_raw_v1_bytes_exact_rational_v04(bytes.as_bytes())
        .to_portable_json_bytes()
        .unwrap();
    let input = TempInput::from_bytes(bytes.as_bytes());
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, expected);
    assert!(!output.stdout.ends_with(b"\n"));
}

#[test]
fn cli_usage_is_non_json_stderr_and_nonzero() {
    let output = run_cli(&[]);

    assert_eq!(output.status.code(), Some(64));
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"usage: raw_v1_verify <raw-v1-path>\n");
}

#[test]
fn cli_sparse_oversize_is_bounded_canonical_wire_reject() {
    let input =
        TempInput::sparse(u64::try_from(DEFAULT_WIRE_JSON_LIMITS.max_input_bytes).unwrap() * 16);
    let output = run_cli(&[input.path().as_os_str()]);

    assert_eq!(output.status.code(), Some(0));
    assert!(output.stderr.is_empty());
    assert_eq!(
        output.stdout,
        b"{\"schema\":\"raw-v1-rust-execution-v1\",\"profile\":\"exact_rational_raw_v1_execution_v04\",\"parse_outcome\":\"REJECT\",\"rejection_stage\":\"CANONICAL_WIRE\",\"evaluation_outcome\":\"NOT_RUN\",\"evaluation_error_stage\":null,\"semantic_result\":null}"
    );
    assert!(!output.stdout.ends_with(b"\n"));
}

fn run_cli(arguments: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_raw_v1_verify"))
        .args(arguments)
        .output()
        .unwrap()
}

fn zero_segment_success_raw() -> String {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let marker = "\"segments\":[";
    let start = raw.find(marker).unwrap() + marker.len();
    let end = raw.rfind("],\"source\":").unwrap();
    format!("{}{}", &raw[..start], &raw[end..])
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
        "three-body-raw-v1-cli-{}-{id}.json",
        std::process::id()
    ))
}
