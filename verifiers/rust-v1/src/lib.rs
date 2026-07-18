//! Independent strict-wire and numeric primitives for the raw-v1 planar-chain
//! verifier.
//!
//! This crate deliberately contains no Python bridge and does not claim
//! complete continuation-chain replay. The public surface covers strict wire
//! decoding, fixed-limit opaque ownership of canonical bytes plus their typed
//! decode, bounded exact arithmetic, non-certifying ordinary and planar-LC
//! semantic inputs, exact 14-component LC state polynomial/interval
//! arithmetic, an independent interval-dual LC field and direct polynomial
//! defect enclosure, conditional ordinary chart and tube replay, exact planar
//! initial-value binding, proof-oriented validated-root replay, and local
//! parent-carried ordinary-bridge composition, and a bounded ordinary-only raw
//! chain checkpoint for top-level obligations 6--13 under separately named
//! exact-rational profiles. The checkpoint stops fail-closed at LC and is not
//! outer-obligation, LC-obligation, or full-chain replay. Public LC-entry
//! semantic construction requires the opaque admission plus a segment index;
//! it proves no LC chart, tube, entry, gauge, lift, projection, or exit claim.

#![forbid(unsafe_code)]

mod binary64;
mod dual;
mod error;
mod exp;
mod interval;
mod json_number;
mod ordinary_binding;
mod ordinary_bridge;
mod ordinary_chain;
mod ordinary_chart;
mod ordinary_defect;
mod ordinary_field;
mod ordinary_semantic;
mod ordinary_tube;
pub mod outward_mass;
mod planar_lc_defect;
mod planar_lc_field;
mod planar_lc_semantic;
mod planar_lc_state;
mod polynomial;
mod rational_input;
mod raw_admission;
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
pub use ordinary_binding::{
    replay_initial_value_binding_exact_rational_v04,
    replay_validated_ordinary_root_exact_rational_v04, InitialValueBindingObligation,
    InitialValueBindingReplay, InitialValueBindingReplayError, ValidatedOrdinaryRootObligation,
    ValidatedOrdinaryRootReplay, ValidatedOrdinaryRootReplayError,
    EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID,
    EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID, INITIAL_VALUE_BINDING_OBLIGATION_IDS,
    VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS,
};
pub use ordinary_bridge::{
    replay_carried_ordinary_bridge_exact_rational_v04, OrdinaryBridgeObligation,
    OrdinaryBridgeReplay, OrdinaryBridgeReplayError,
    EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
    ORDINARY_AUTONOMOUS_UNIQUENESS_BRIDGE_KERNEL_V1_ID, ORDINARY_BRIDGE_OBLIGATION_IDS,
};
pub use ordinary_chain::{
    replay_raw_ordinary_only_chain_exact_rational_v04, OrdinaryChainClockLedgerEntry,
    OrdinaryChainFinalEnclosure, OrdinaryChainReplayFailure, OrdinaryChainRightFrontier,
    RawOrdinaryOnlyChainObligation, RawOrdinaryOnlyChainReplay, RawOrdinaryOnlyChainReplayError,
    EXACT_RATIONAL_RAW_ORDINARY_ONLY_CHAIN_V04_PROFILE_ID, HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
    RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS,
};
pub use ordinary_chart::{
    replay_ordinary_chart_exact_rational_claimed_tail_v04,
    ExactRationalOrdinaryChartClaimedTailV04, OrdinaryChartObligation, OrdinaryChartReplay,
    OrdinaryChartReplayError, EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID,
    HARD_MAX_ORDINARY_CHART_SERIES_WORK_UNITS, ORDINARY_CHART_OBLIGATION_IDS,
};
pub use ordinary_defect::{
    evaluate_planar_three_body_ordinary_polynomial_defect, OrdinaryPolynomialDefectEnclosure,
    OrdinaryPolynomialDefectError,
};
pub use ordinary_field::{
    evaluate_planar_three_body_ordinary_field, OrdinaryFieldEnclosure, OrdinaryFieldError,
};
pub use ordinary_semantic::{
    ordinary_chart_input_from_wire, ordinary_tube_binding_status, ordinary_tube_input_from_wire,
    OrdinaryChartInput, OrdinaryCoefficientKind, OrdinaryIntervalKind, OrdinarySemanticError,
    OrdinarySemanticResource, OrdinaryTubeBindingStatus, OrdinaryTubeInput,
    HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT,
};
pub use ordinary_tube::{
    replay_ordinary_tube_exact_rational_v04, ExactRationalOrdinaryTubeV04, OrdinaryTubeObligation,
    OrdinaryTubeReplay, OrdinaryTubeReplayError, EXACT_RATIONAL_ORDINARY_TUBE_V04_PROFILE_ID,
    ORDINARY_TUBE_OBLIGATION_IDS,
};
pub use planar_lc_defect::{
    evaluate_planar_lc_polynomial_defect, evaluate_planar_lc_polynomial_defect_default,
    PlanarLcPolynomialDefectEnclosure, PlanarLcPolynomialDefectError,
};
pub use planar_lc_field::{
    evaluate_planar_lc_field, evaluate_planar_lc_field_default, PlanarLcFieldEnclosure,
    PlanarLcFieldError, PlanarLcThirdDenominator, PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS,
};
pub use planar_lc_semantic::{
    planar_lc_chart_input_from_wire, planar_lc_entry_input_from_admission,
    planar_lc_tube_input_from_wire, PlanarLcChartInput, PlanarLcEntryInput, PlanarLcIntervalKind,
    PlanarLcSemanticError, PlanarLcSemanticResource, PlanarLcSeriesKind, PlanarLcTubeInput,
    HARD_MAX_PLANAR_LC_SEMANTIC_COEFFICIENT_COUNT, HARD_MAX_PLANAR_LC_SEMANTIC_TOTAL_WORK_UNITS,
};
pub use planar_lc_state::{
    PlanarLcAnchorPoint, PlanarLcIntervalState, PlanarLcStateError, PlanarLcStatePolynomial, LC_H,
    LC_RX, LC_RY, LC_T, LC_UX, LC_UY, LC_VX, LC_VY, LC_WX, LC_WY, LC_YX, LC_YY, LC_ZX, LC_ZY,
    PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES, PLANAR_LC_LIFTED_STATE_DIMENSION,
    PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_COMPONENT_NAMES,
    PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION,
};
pub use polynomial::{
    ExactRationalPolynomial, PolynomialError, HARD_MAX_POLYNOMIAL_DEGREE,
    HARD_MAX_POLYNOMIAL_DIMENSION, HARD_MAX_POLYNOMIAL_WORK_UNITS,
};
pub use rational_input::HARD_MAX_RATIONAL_COMPONENT_BITS;
pub use raw_admission::{CanonicalRawV1Admission, CanonicalRawV1AdmissionError};
pub use sqrt::{sqrt_enclosure_dyadic, DyadicSqrtEnclosure, HARD_MAX_SQRT_PRECISION_BITS};
pub use wire_json::{
    parse_wire_json, to_canonical_bytes, validate_canonical_wire_json, WireJsonError,
    WireJsonLimits, WireJsonValue, DEFAULT_WIRE_JSON_LIMITS,
};
