#![forbid(unsafe_code)]

use std::{
    collections::TryReserveError,
    env,
    ffi::OsString,
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    process::ExitCode,
};

use three_body_planar_chain_verifier::{
    execute_proof_grade_raw_v1_bytes_exact_rational_v04, ProofGradeRawV1EvaluationOutcome,
    ProofGradeRawV1ExecutionSerializationError, DEFAULT_WIRE_JSON_LIMITS,
};

const EVALUATION_ERROR_EXIT: u8 = 2;
const CLI_ERROR_EXIT: u8 = 1;
const USAGE_ERROR_EXIT: u8 = 64;

fn main() -> ExitCode {
    match run(env::args_os()) {
        Ok(code) => ExitCode::from(code),
        Err(source) => {
            let mut stderr = io::stderr().lock();
            let _ = writeln!(stderr, "{source}");
            ExitCode::from(source.exit_code())
        }
    }
}

fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<u8, CliError> {
    let mut arguments = arguments.into_iter();
    let _program = arguments.next();
    let input_path = arguments.next().ok_or(CliError::Usage)?;
    if arguments.next().is_some() {
        return Err(CliError::Usage);
    }

    let bytes = read_bounded(Path::new(&input_path))?;
    let execution = execute_proof_grade_raw_v1_bytes_exact_rational_v04(&bytes);
    let json = execution
        .to_portable_json_bytes()
        .map_err(CliError::Serialization)?;
    let exit_code = if execution.evaluation_outcome() == ProofGradeRawV1EvaluationOutcome::Error {
        EVALUATION_ERROR_EXIT
    } else {
        0
    };

    let mut stdout = io::stdout().lock();
    stdout.write_all(&json).map_err(|source| CliError::Io {
        operation: "write stdout",
        source,
    })?;
    stdout.flush().map_err(|source| CliError::Io {
        operation: "flush stdout",
        source,
    })?;
    Ok(exit_code)
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, CliError> {
    let file = File::open(path).map_err(|source| CliError::Io {
        operation: "open input",
        source,
    })?;
    let read_limit = DEFAULT_WIRE_JSON_LIMITS
        .max_input_bytes
        .checked_add(1)
        .ok_or(CliError::InputLimitOverflow)?;
    let read_limit_u64 = u64::try_from(read_limit).map_err(|_| CliError::InputLimitOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(read_limit)
        .map_err(CliError::Allocation)?;
    let mut reader = file.take(read_limit_u64);
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|source| CliError::Io {
            operation: "read input",
            source,
        })?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    Ok(bytes)
}

#[derive(Debug)]
enum CliError {
    Usage,
    InputLimitOverflow,
    Allocation(TryReserveError),
    Io {
        operation: &'static str,
        source: io::Error,
    },
    Serialization(ProofGradeRawV1ExecutionSerializationError),
}

impl CliError {
    const fn exit_code(&self) -> u8 {
        match self {
            Self::Usage => USAGE_ERROR_EXIT,
            Self::InputLimitOverflow
            | Self::Allocation(_)
            | Self::Io { .. }
            | Self::Serialization(_) => CLI_ERROR_EXIT,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => formatter.write_str("usage: raw_v1_proof_grade_verify <raw-v1-path>"),
            Self::InputLimitOverflow => {
                formatter.write_str("configured raw-v1 input limit cannot be represented")
            }
            Self::Allocation(source) => {
                write!(
                    formatter,
                    "cannot allocate bounded raw-v1 input buffer: {source}"
                )
            }
            Self::Io { operation, source } => write!(formatter, "{operation} failed: {source}"),
            Self::Serialization(source) => write!(formatter, "{source}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(source) => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::Serialization(source) => Some(source),
            Self::Usage | Self::InputLimitOverflow => None,
        }
    }
}
