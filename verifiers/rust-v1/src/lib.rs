//! Independent strict-wire and numeric primitives for the raw-v1 planar-chain
//! verifier.
//!
//! This crate deliberately contains no Python bridge and does not yet replay
//! certificate semantics. The public surface covers strict UTF-8 JSON parsing
//! and canonicalization, exact binary64 decoding, rational interval algebra,
//! certified dyadic square-root and rational exponential enclosures, and exact
//! outward mass coefficients.

#![forbid(unsafe_code)]

mod binary64;
mod dual;
mod error;
mod exp;
mod interval;
mod json_number;
mod ordinary_defect;
mod ordinary_field;
pub mod outward_mass;
mod polynomial;
mod rational_input;
pub mod raw_schema;
mod sqrt;
mod wire_json;

pub use binary64::{rational_from_f64, rational_from_f64_bits, ExactBinary64};
pub use dual::{IntervalDual, HARD_MAX_DUAL_DIMENSION};
pub use error::NumericError;
pub use exp::{
    exp_enclosure_rational, ExpEnclosureError, ExpEnclosureLimits, RationalExpEnclosure,
    DEFAULT_EXP_ENCLOSURE_LIMITS, HARD_MAX_EXP_INPUT_COMPONENT_BITS,
    HARD_MAX_EXP_INTERMEDIATE_COMPONENT_BITS, HARD_MAX_EXP_RANGE_REDUCTIONS,
    HARD_MAX_EXP_TAYLOR_CUTOFF, HARD_MAX_EXP_WITNESS_STORAGE_BITS, HARD_MAX_EXP_WORK_UNITS,
};
pub use interval::{RationalInterval, HARD_MAX_INTERVAL_HORNER_COEFFICIENTS};
pub use json_number::{
    checked_real_binary64_from_json, parse_json_number_lexeme, verify_binary64_candidate,
    CheckedJsonBinary64, JsonNumberKind, JsonNumberLimits, ParsedJsonNumber,
    DEFAULT_JSON_NUMBER_LIMITS,
};
pub use ordinary_defect::{
    evaluate_planar_three_body_ordinary_polynomial_defect, OrdinaryPolynomialDefectEnclosure,
    OrdinaryPolynomialDefectError,
};
pub use ordinary_field::{
    evaluate_planar_three_body_ordinary_field, OrdinaryFieldEnclosure, OrdinaryFieldError,
};
pub use polynomial::{
    ExactRationalPolynomial, PolynomialError, HARD_MAX_POLYNOMIAL_DEGREE,
    HARD_MAX_POLYNOMIAL_DIMENSION, HARD_MAX_POLYNOMIAL_WORK_UNITS,
};
pub use rational_input::HARD_MAX_RATIONAL_COMPONENT_BITS;
pub use sqrt::{sqrt_enclosure_dyadic, DyadicSqrtEnclosure, HARD_MAX_SQRT_PRECISION_BITS};
pub use wire_json::{
    parse_wire_json, to_canonical_bytes, validate_canonical_wire_json, WireJsonError,
    WireJsonLimits, WireJsonValue, DEFAULT_WIRE_JSON_LIMITS,
};
