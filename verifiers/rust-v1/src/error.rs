use core::fmt;

/// Fail-closed errors from the exact numeric kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NumericError {
    NonFiniteBinary64 {
        bits: u64,
    },
    MalformedJsonNumber,
    JsonNumberSignificandDigitLimitExceeded,
    JsonNumberExponentDigitLimitExceeded,
    JsonNumberExponentMagnitudeLimitExceeded,
    JsonNumberEffectiveExponentLimitExceeded,
    JsonRealLexemeRequired,
    HostBinary64ParseRejected,
    Binary64CandidateNonFinite {
        bits: u64,
    },
    Binary64Overflow,
    Binary64UnderflowSignMismatch,
    Binary64CandidateSignMismatch,
    Binary64NearestProofFailed,
    RationalComponentBitLimitExceeded {
        numerator_bits: u64,
        denominator_bits: u64,
        limit: u64,
    },
    InvalidRationalDenominator,
    NonCanonicalRational,
    PolynomialCoefficientCountLimitExceeded {
        coefficient_count: usize,
        limit: usize,
    },
    DualDimensionLimitExceeded {
        dimension: usize,
        limit: usize,
    },
    DualDimensionMismatch {
        left_dimension: usize,
        right_dimension: usize,
    },
    DualIndexOutOfBounds {
        index: usize,
        dimension: usize,
    },
    DualSquareRootRequiresStrictlyPositiveValue,
    InvalidIntervalBounds,
    DivisionByZeroInterval,
    NegativeSquareRoot,
    PrecisionOverflow,
    SquareRootPrecisionBitLimitExceeded {
        precision_bits: usize,
        limit: usize,
    },
    InternalSquareRootPostconditionFailure,
}

impl fmt::Display for NumericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteBinary64 { bits } => {
                write!(formatter, "non-finite binary64 bit pattern 0x{bits:016x}")
            }
            Self::MalformedJsonNumber => {
                formatter.write_str("lexeme does not match the RFC 8259 number grammar")
            }
            Self::JsonNumberSignificandDigitLimitExceeded => {
                formatter.write_str("JSON number significand digit limit exceeded")
            }
            Self::JsonNumberExponentDigitLimitExceeded => {
                formatter.write_str("JSON number exponent digit limit exceeded")
            }
            Self::JsonNumberExponentMagnitudeLimitExceeded => {
                formatter.write_str("JSON number written exponent magnitude limit exceeded")
            }
            Self::JsonNumberEffectiveExponentLimitExceeded => {
                formatter.write_str("JSON number effective decimal exponent limit exceeded")
            }
            Self::JsonRealLexemeRequired => {
                formatter.write_str("a real-field JSON number lexeme is required")
            }
            Self::HostBinary64ParseRejected => {
                formatter.write_str("host binary64 parser rejected a valid JSON number")
            }
            Self::Binary64CandidateNonFinite { bits } => {
                write!(
                    formatter,
                    "host proposed non-finite binary64 bits 0x{bits:016x}"
                )
            }
            Self::Binary64Overflow => {
                formatter.write_str("exact decimal lies in the binary64 overflow region")
            }
            Self::Binary64UnderflowSignMismatch => {
                formatter.write_str("zero binary64 candidate does not preserve underflow sign")
            }
            Self::Binary64CandidateSignMismatch => {
                formatter.write_str("nonzero binary64 candidate has the wrong sign")
            }
            Self::Binary64NearestProofFailed => formatter.write_str(
                "binary64 candidate is not the exact nearest-even rounding of the decimal",
            ),
            Self::RationalComponentBitLimitExceeded {
                numerator_bits,
                denominator_bits,
                limit,
            } => write!(
                formatter,
                "exact rational component limit exceeded: numerator={numerator_bits} bits, \
                 denominator={denominator_bits} bits, limit={limit} bits"
            ),
            Self::InvalidRationalDenominator => formatter
                .write_str("exact rational denominator must be strictly positive and nonzero"),
            Self::NonCanonicalRational => {
                formatter.write_str("exact rational must be in canonical reduced form")
            }
            Self::PolynomialCoefficientCountLimitExceeded {
                coefficient_count,
                limit,
            } => write!(
                formatter,
                "polynomial coefficient count {coefficient_count} exceeds hard limit {limit}"
            ),
            Self::DualDimensionLimitExceeded { dimension, limit } => write!(
                formatter,
                "interval-dual dimension {dimension} exceeds hard limit {limit}"
            ),
            Self::DualDimensionMismatch {
                left_dimension,
                right_dimension,
            } => write!(
                formatter,
                "interval-dual dimension mismatch: left={left_dimension}, \
                 right={right_dimension}"
            ),
            Self::DualIndexOutOfBounds { index, dimension } => write!(
                formatter,
                "interval-dual index {index} is outside dimension {dimension}"
            ),
            Self::DualSquareRootRequiresStrictlyPositiveValue => formatter.write_str(
                "interval-dual square-root derivative requires a strictly positive value interval",
            ),
            Self::InvalidIntervalBounds => {
                formatter.write_str("rational interval lower bound exceeds upper bound")
            }
            Self::DivisionByZeroInterval => {
                formatter.write_str("cannot invert a rational interval containing zero")
            }
            Self::NegativeSquareRoot => {
                formatter.write_str("cannot enclose the square root of a negative rational")
            }
            Self::PrecisionOverflow => {
                formatter.write_str("requested dyadic precision overflows its size type")
            }
            Self::SquareRootPrecisionBitLimitExceeded {
                precision_bits,
                limit,
            } => write!(
                formatter,
                "requested square-root precision {precision_bits} bits exceeds hard limit {limit}"
            ),
            Self::InternalSquareRootPostconditionFailure => formatter.write_str(
                "integer square-root construction failed an exact endpoint postcondition",
            ),
        }
    }
}

impl std::error::Error for NumericError {}
