//! Independent numeric primitives for the raw-v1 planar-chain verifier.
//!
//! This crate deliberately contains no Python bridge and no certificate
//! semantics.  The public surface is limited to exact binary64 decoding,
//! rational interval algebra, and certified dyadic square-root enclosure.

#![forbid(unsafe_code)]

mod binary64;
mod error;
mod interval;
mod json_number;
mod sqrt;

pub use binary64::{rational_from_f64, rational_from_f64_bits, ExactBinary64};
pub use error::NumericError;
pub use interval::RationalInterval;
pub use json_number::{
    checked_real_binary64_from_json, parse_json_number_lexeme, verify_binary64_candidate,
    CheckedJsonBinary64, JsonNumberKind, JsonNumberLimits, ParsedJsonNumber,
    DEFAULT_JSON_NUMBER_LIMITS,
};
pub use sqrt::{sqrt_enclosure_dyadic, DyadicSqrtEnclosure};
