//! Independent strict-wire and numeric primitives for the raw-v1 planar-chain
//! verifier.
//!
//! This crate deliberately contains no Python bridge and no certificate
//! semantics. The public surface covers strict UTF-8 JSON parsing and
//! canonicalization, exact binary64 decoding, rational interval algebra, and
//! certified dyadic square-root enclosure.

#![forbid(unsafe_code)]

mod binary64;
mod error;
mod interval;
mod json_number;
pub mod raw_schema;
mod sqrt;
mod wire_json;

pub use binary64::{rational_from_f64, rational_from_f64_bits, ExactBinary64};
pub use error::NumericError;
pub use interval::RationalInterval;
pub use json_number::{
    checked_real_binary64_from_json, parse_json_number_lexeme, verify_binary64_candidate,
    CheckedJsonBinary64, JsonNumberKind, JsonNumberLimits, ParsedJsonNumber,
    DEFAULT_JSON_NUMBER_LIMITS,
};
pub use sqrt::{sqrt_enclosure_dyadic, DyadicSqrtEnclosure};
pub use wire_json::{
    parse_wire_json, to_canonical_bytes, validate_canonical_wire_json, WireJsonError,
    WireJsonLimits, WireJsonValue, DEFAULT_WIRE_JSON_LIMITS,
};
