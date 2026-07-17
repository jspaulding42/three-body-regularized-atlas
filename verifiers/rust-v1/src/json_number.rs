use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{rational_from_f64_bits, NumericError};

const SIGN_MASK: u64 = 1_u64 << 63;
const EXPONENT_MASK: u64 = 0x7ff;
const MAX_FINITE_MAGNITUDE_BITS: u64 = 0x7fef_ffff_ffff_ffff;

/// Explicit denial-of-service limits for manual JSON-number parsing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonNumberLimits {
    pub max_significand_digits: usize,
    pub max_exponent_digits: usize,
    /// Bounds both the written exponent and the effective exponent after the
    /// fractional digit count is subtracted.
    pub max_absolute_decimal_exponent: u32,
}

pub const DEFAULT_JSON_NUMBER_LIMITS: JsonNumberLimits = JsonNumberLimits {
    max_significand_digits: 4096,
    max_exponent_digits: 6,
    max_absolute_decimal_exponent: 10_000,
};

/// Whether the original JSON token belongs to an integer or real-valued field.
///
/// A decimal point or exponent marker makes the lexeme `Real` even when its
/// exact mathematical value is an integer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JsonNumberKind {
    Integer,
    Real,
}

/// A manually parsed RFC 8259 number and its exact decimal rational value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedJsonNumber {
    lexeme: String,
    kind: JsonNumberKind,
    negative_lexeme: bool,
    exact: BigRational,
}

impl ParsedJsonNumber {
    pub fn lexeme(&self) -> &str {
        &self.lexeme
    }

    pub const fn kind(&self) -> JsonNumberKind {
        self.kind
    }

    pub const fn is_negative_lexeme(&self) -> bool {
        self.negative_lexeme
    }

    pub fn is_negative_zero_lexeme(&self) -> bool {
        self.negative_lexeme && self.exact.is_zero()
    }

    pub fn exact_decimal(&self) -> &BigRational {
        &self.exact
    }
}

/// A host-proposed binary64 value whose nearest-even relation to the exact
/// decimal has been independently proved with rational arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedJsonBinary64 {
    parsed: ParsedJsonNumber,
    bits: u64,
    binary64_rational: BigRational,
}

impl CheckedJsonBinary64 {
    pub fn parsed(&self) -> &ParsedJsonNumber {
        &self.parsed
    }

    pub fn lexeme(&self) -> &str {
        self.parsed.lexeme()
    }

    /// The exact decimal value denoted by the original JSON lexeme.
    pub fn exact_decimal(&self) -> &BigRational {
        self.parsed.exact_decimal()
    }

    /// The theorem-facing exact dyadic value denoted by the retained bits.
    pub fn binary64_rational(&self) -> &BigRational {
        &self.binary64_rational
    }

    pub const fn bits(&self) -> u64 {
        self.bits
    }

    pub fn value(&self) -> f64 {
        f64::from_bits(self.bits)
    }
}

/// Parse exactly the RFC 8259 number grammar without delegating token syntax
/// to a JSON or floating-point library.
pub fn parse_json_number_lexeme(
    lexeme: &str,
    limits: JsonNumberLimits,
) -> Result<ParsedJsonNumber, NumericError> {
    let bytes = lexeme.as_bytes();
    if bytes.is_empty() {
        return Err(NumericError::MalformedJsonNumber);
    }

    let mut index = 0_usize;
    let negative_lexeme = bytes[index] == b'-';
    if negative_lexeme {
        index += 1;
        if index == bytes.len() {
            return Err(NumericError::MalformedJsonNumber);
        }
    }

    let mut significand = BigInt::zero();
    let mut significand_digits = 0_usize;
    let mut push_digit = |digit: u8| -> Result<(), NumericError> {
        significand_digits = significand_digits
            .checked_add(1)
            .ok_or(NumericError::JsonNumberSignificandDigitLimitExceeded)?;
        if significand_digits > limits.max_significand_digits {
            return Err(NumericError::JsonNumberSignificandDigitLimitExceeded);
        }
        significand *= 10_u8;
        significand += digit - b'0';
        Ok(())
    };

    match bytes.get(index).copied() {
        Some(b'0') => {
            push_digit(b'0')?;
            index += 1;
            if matches!(bytes.get(index), Some(b'0'..=b'9')) {
                return Err(NumericError::MalformedJsonNumber);
            }
        }
        Some(b'1'..=b'9') => {
            while let Some(digit @ b'0'..=b'9') = bytes.get(index).copied() {
                push_digit(digit)?;
                index += 1;
            }
        }
        _ => return Err(NumericError::MalformedJsonNumber),
    }

    let mut kind = JsonNumberKind::Integer;
    let mut fraction_digits = 0_usize;
    if bytes.get(index) == Some(&b'.') {
        kind = JsonNumberKind::Real;
        index += 1;
        let fraction_start = index;
        while let Some(digit @ b'0'..=b'9') = bytes.get(index).copied() {
            push_digit(digit)?;
            fraction_digits = fraction_digits
                .checked_add(1)
                .ok_or(NumericError::JsonNumberEffectiveExponentLimitExceeded)?;
            index += 1;
        }
        if index == fraction_start {
            return Err(NumericError::MalformedJsonNumber);
        }
    }

    let mut written_exponent = 0_i64;
    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        kind = JsonNumberKind::Real;
        index += 1;
        let exponent_negative = match bytes.get(index) {
            Some(b'+') => {
                index += 1;
                false
            }
            Some(b'-') => {
                index += 1;
                true
            }
            _ => false,
        };
        let exponent_start = index;
        let mut exponent_digits = 0_usize;
        let mut exponent_magnitude = 0_u32;
        while let Some(digit @ b'0'..=b'9') = bytes.get(index).copied() {
            exponent_digits = exponent_digits
                .checked_add(1)
                .ok_or(NumericError::JsonNumberExponentDigitLimitExceeded)?;
            if exponent_digits > limits.max_exponent_digits {
                return Err(NumericError::JsonNumberExponentDigitLimitExceeded);
            }
            exponent_magnitude = exponent_magnitude
                .checked_mul(10)
                .and_then(|value| value.checked_add(u32::from(digit - b'0')))
                .ok_or(NumericError::JsonNumberExponentMagnitudeLimitExceeded)?;
            if exponent_magnitude > limits.max_absolute_decimal_exponent {
                return Err(NumericError::JsonNumberExponentMagnitudeLimitExceeded);
            }
            index += 1;
        }
        if index == exponent_start {
            return Err(NumericError::MalformedJsonNumber);
        }
        written_exponent = i64::from(exponent_magnitude);
        if exponent_negative {
            written_exponent = -written_exponent;
        }
    }

    if index != bytes.len() {
        return Err(NumericError::MalformedJsonNumber);
    }

    let fraction_digits_i64 = i64::try_from(fraction_digits)
        .map_err(|_| NumericError::JsonNumberEffectiveExponentLimitExceeded)?;
    let effective_exponent = written_exponent
        .checked_sub(fraction_digits_i64)
        .ok_or(NumericError::JsonNumberEffectiveExponentLimitExceeded)?;
    if effective_exponent.unsigned_abs() > u64::from(limits.max_absolute_decimal_exponent) {
        return Err(NumericError::JsonNumberEffectiveExponentLimitExceeded);
    }

    if negative_lexeme {
        significand = -significand;
    }
    let exact = if significand.is_zero() {
        BigRational::zero()
    } else {
        let decimal_power = BigInt::from(10_u8).pow(effective_exponent.unsigned_abs() as u32);
        if effective_exponent >= 0 {
            BigRational::from_integer(significand * decimal_power)
        } else {
            BigRational::new(significand, decimal_power)
        }
    };

    Ok(ParsedJsonNumber {
        lexeme: lexeme.to_owned(),
        kind,
        negative_lexeme,
        exact,
    })
}

/// Parse a real-field lexeme, let the host parser propose binary64 bits, and
/// independently prove exact nearest-even rounding before retaining them.
pub fn checked_real_binary64_from_json(
    lexeme: &str,
    limits: JsonNumberLimits,
) -> Result<CheckedJsonBinary64, NumericError> {
    let parsed = parse_json_number_lexeme(lexeme, limits)?;
    if parsed.kind() != JsonNumberKind::Real {
        return Err(NumericError::JsonRealLexemeRequired);
    }
    if absolute_rational(parsed.exact_decimal()) >= overflow_midpoint() {
        return Err(NumericError::Binary64Overflow);
    }
    let candidate = lexeme
        .parse::<f64>()
        .map_err(|_| NumericError::HostBinary64ParseRejected)?;
    let bits = candidate.to_bits();
    verify_binary64_candidate(&parsed, bits)?;
    let binary64_rational = rational_from_f64_bits(bits)
        .map_err(|_| NumericError::Binary64CandidateNonFinite { bits })?;
    Ok(CheckedJsonBinary64 {
        parsed,
        bits,
        binary64_rational,
    })
}

/// Verify a proposed finite binary64 encoding against an already parsed exact
/// decimal. This function does not invoke the host floating-point parser.
pub fn verify_binary64_candidate(
    parsed: &ParsedJsonNumber,
    candidate_bits: u64,
) -> Result<(), NumericError> {
    let exponent_field = (candidate_bits >> 52) & EXPONENT_MASK;
    if exponent_field == EXPONENT_MASK {
        return Err(NumericError::Binary64CandidateNonFinite {
            bits: candidate_bits,
        });
    }

    let exact_magnitude = absolute_rational(parsed.exact_decimal());
    if exact_magnitude >= overflow_midpoint() {
        return Err(NumericError::Binary64Overflow);
    }
    let candidate_negative = candidate_bits & SIGN_MASK != 0;
    let magnitude_bits = candidate_bits & !SIGN_MASK;
    if magnitude_bits == 0 {
        if candidate_negative != parsed.is_negative_lexeme() {
            return Err(NumericError::Binary64UnderflowSignMismatch);
        }
    } else {
        let exact_negative = parsed.exact_decimal() < &BigRational::zero();
        if candidate_negative != exact_negative {
            return Err(NumericError::Binary64CandidateSignMismatch);
        }
    }

    let candidate = rational_from_f64_bits(magnitude_bits)?;
    let candidate_distance = rational_distance(&exact_magnitude, &candidate);
    let candidate_is_even = magnitude_bits & 1 == 0;

    if magnitude_bits > 0 {
        let lower = rational_from_f64_bits(magnitude_bits - 1)?;
        if !wins_nearest_even(
            &candidate_distance,
            &rational_distance(&exact_magnitude, &lower),
            candidate_is_even,
        ) {
            return Err(NumericError::Binary64NearestProofFailed);
        }
    }

    let upper = if magnitude_bits < MAX_FINITE_MAGNITUDE_BITS {
        rational_from_f64_bits(magnitude_bits + 1)?
    } else {
        virtual_overflow_neighbor()
    };
    if !wins_nearest_even(
        &candidate_distance,
        &rational_distance(&exact_magnitude, &upper),
        candidate_is_even,
    ) {
        return Err(NumericError::Binary64NearestProofFailed);
    }
    Ok(())
}

fn wins_nearest_even(
    candidate_distance: &BigRational,
    neighbor_distance: &BigRational,
    candidate_is_even: bool,
) -> bool {
    candidate_distance < neighbor_distance
        || (candidate_distance == neighbor_distance && candidate_is_even)
}

fn rational_distance(first: &BigRational, second: &BigRational) -> BigRational {
    if first >= second {
        first - second
    } else {
        second - first
    }
}

fn absolute_rational(value: &BigRational) -> BigRational {
    if value < &BigRational::zero() {
        -value
    } else {
        value.clone()
    }
}

fn virtual_overflow_neighbor() -> BigRational {
    BigRational::from_integer(BigInt::one() << 1024_usize)
}

fn overflow_midpoint() -> BigRational {
    // max_finite = 2^1024 - 2^971. The virtual next value is 2^1024,
    // whose even significand wins the midpoint tie.
    BigRational::from_integer((BigInt::one() << 1024_usize) - (BigInt::one() << 970_usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(lexeme: &str) -> ParsedJsonNumber {
        parse_json_number_lexeme(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn checked(lexeme: &str) -> CheckedJsonBinary64 {
        checked_real_binary64_from_json(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn power_of_two(exponent: usize) -> BigInt {
        BigInt::one() << exponent
    }

    #[test]
    fn decimal_point_one_retains_lexeme_exact_value_and_proved_bits() {
        let value = checked("0.1");
        assert_eq!(value.lexeme(), "0.1");
        assert_eq!(value.bits(), 0x3fb9_9999_9999_999a);
        assert_eq!(value.value(), 0.1_f64);
        assert_eq!(
            value.exact_decimal(),
            &BigRational::new(BigInt::one(), BigInt::from(10))
        );
        assert_eq!(
            value.binary64_rational(),
            &BigRational::new(
                BigInt::from(3_602_879_701_896_397_u64),
                BigInt::from(36_028_797_018_963_968_u64),
            )
        );
        assert_eq!(value.parsed().kind(), JsonNumberKind::Real);
    }

    #[test]
    fn negative_zero_preserves_original_sign_and_bits() {
        let value = checked("-0.0");
        assert_eq!(value.lexeme(), "-0.0");
        assert_eq!(value.exact_decimal(), &BigRational::zero());
        assert_eq!(value.binary64_rational(), &BigRational::zero());
        assert!(value.parsed().is_negative_zero_lexeme());
        assert_eq!(value.bits(), 0x8000_0000_0000_0000);

        let tiny = checked("-1e-999");
        assert!(!tiny.exact_decimal().is_zero());
        assert_eq!(tiny.binary64_rational(), &BigRational::zero());
        assert_eq!(tiny.bits(), 0x8000_0000_0000_0000);
        assert_eq!(
            verify_binary64_candidate(tiny.parsed(), 0),
            Err(NumericError::Binary64UnderflowSignMismatch)
        );
    }

    #[test]
    fn nearest_even_midpoint_ties_choose_even_significands() {
        let down = checked("1.00000000000000011102230246251565404236316680908203125");
        assert_eq!(
            down.exact_decimal(),
            &BigRational::new(power_of_two(53) + BigInt::one(), power_of_two(53))
        );
        assert_eq!(down.bits(), 0x3ff0_0000_0000_0000);

        let up = checked("1.00000000000000033306690738754696212708950042724609375");
        assert_eq!(
            up.exact_decimal(),
            &BigRational::new(power_of_two(53) + BigInt::from(3_u8), power_of_two(53),)
        );
        assert_eq!(up.bits(), 0x3ff0_0000_0000_0002);

        assert_eq!(
            verify_binary64_candidate(down.parsed(), 0x3ff0_0000_0000_0001),
            Err(NumericError::Binary64NearestProofFailed)
        );
    }

    #[test]
    fn exponent_forms_are_exact_and_integer_tokens_remain_distinct() {
        let integer = parse("125");
        assert_eq!(integer.kind(), JsonNumberKind::Integer);
        assert_eq!(
            integer.exact_decimal(),
            &BigRational::from_integer(BigInt::from(125))
        );
        assert_eq!(
            checked_real_binary64_from_json("125", DEFAULT_JSON_NUMBER_LIMITS),
            Err(NumericError::JsonRealLexemeRequired)
        );

        let positive = checked("1.25e2");
        assert_eq!(positive.parsed().kind(), JsonNumberKind::Real);
        assert_eq!(positive.exact_decimal(), integer.exact_decimal());
        assert_eq!(positive.bits(), 125_f64.to_bits());

        let negative = checked("-3E-2");
        assert_eq!(
            negative.exact_decimal(),
            &BigRational::new(BigInt::from(-3), BigInt::from(100))
        );
        assert_eq!(negative.bits(), (-0.03_f64).to_bits());
        assert_eq!(parse("12e0").kind(), JsonNumberKind::Real);
    }

    #[test]
    fn subnormal_normal_and_max_finite_boundaries_are_proved() {
        assert_eq!(checked("5e-324").bits(), 0x0000_0000_0000_0001);
        assert_eq!(
            checked("2.2250738585072014e-308").bits(),
            0x0010_0000_0000_0000
        );
        assert_eq!(
            checked("1.7976931348623157e308").bits(),
            MAX_FINITE_MAGNITUDE_BITS
        );

        // Exercise the virtual 2^1024 upper neighbor at the top finite cell.
        let midpoint_integer = (BigInt::one() << 1024_usize) - (BigInt::one() << 970_usize);
        let just_below = &midpoint_integer - BigInt::one();
        let just_below_lexeme = format!("{just_below}.0");
        assert_eq!(
            checked(&just_below_lexeme).bits(),
            MAX_FINITE_MAGNITUDE_BITS
        );
        assert_eq!(
            checked_real_binary64_from_json(
                &format!("{midpoint_integer}.0"),
                DEFAULT_JSON_NUMBER_LIMITS,
            ),
            Err(NumericError::Binary64Overflow)
        );
    }

    #[test]
    fn half_min_subnormal_ties_to_signed_even_zero() {
        // 2^-1075 = 5^1075 / 10^1075, so it has a finite decimal expansion.
        let numerator_digits = BigInt::from(5_u8).pow(1075).to_string();
        let decimal = format!("0.{numerator_digits:0>1075}");
        let positive = checked(&decimal);
        assert_eq!(
            positive.exact_decimal(),
            &BigRational::new(BigInt::one(), BigInt::one() << 1075_usize)
        );
        assert_eq!(positive.bits(), 0);

        let negative = checked(&format!("-{decimal}"));
        assert_eq!(negative.bits(), SIGN_MASK);
    }

    #[test]
    fn malformed_and_nonfinite_tokens_are_rejected_before_host_parsing() {
        for lexeme in [
            "",
            "+1",
            ".1",
            "01",
            "-01",
            "1.",
            "1e",
            "1e+",
            "--1",
            " 1",
            "1 ",
            "NaN",
            "Infinity",
            "-Infinity",
            "1_0",
            "0x1",
            "1é",
        ] {
            assert_eq!(
                parse_json_number_lexeme(lexeme, DEFAULT_JSON_NUMBER_LIMITS),
                Err(NumericError::MalformedJsonNumber),
                "unexpected result for {lexeme:?}"
            );
        }

        let point_one = parse("0.1");
        assert_eq!(
            verify_binary64_candidate(&point_one, f64::INFINITY.to_bits()),
            Err(NumericError::Binary64CandidateNonFinite {
                bits: f64::INFINITY.to_bits(),
            })
        );
    }

    #[test]
    fn digit_and_exponent_limits_fail_before_large_allocation() {
        let tight = JsonNumberLimits {
            max_significand_digits: 3,
            max_exponent_digits: 2,
            max_absolute_decimal_exponent: 10,
        };
        assert_eq!(
            parse_json_number_lexeme("123.4", tight),
            Err(NumericError::JsonNumberSignificandDigitLimitExceeded)
        );
        assert_eq!(
            parse_json_number_lexeme("1e100", tight),
            Err(NumericError::JsonNumberExponentDigitLimitExceeded)
        );
        assert_eq!(
            parse_json_number_lexeme("1e11", tight),
            Err(NumericError::JsonNumberExponentMagnitudeLimitExceeded)
        );
        assert_eq!(
            parse_json_number_lexeme("0.00000000001", tight),
            Err(NumericError::JsonNumberSignificandDigitLimitExceeded)
        );

        let effective_only = JsonNumberLimits {
            max_significand_digits: 32,
            max_exponent_digits: 2,
            max_absolute_decimal_exponent: 10,
        };
        assert_eq!(
            parse_json_number_lexeme("0.00000000001", effective_only),
            Err(NumericError::JsonNumberEffectiveExponentLimitExceeded)
        );
    }

    #[test]
    fn overflow_and_wrong_finite_candidates_are_rejected() {
        assert_eq!(
            checked_real_binary64_from_json("1e309", DEFAULT_JSON_NUMBER_LIMITS),
            Err(NumericError::Binary64Overflow)
        );
        let point_one = parse("0.1");
        assert_eq!(
            verify_binary64_candidate(&point_one, 0.1_f64.to_bits() - 1),
            Err(NumericError::Binary64NearestProofFailed)
        );
        assert_eq!(
            verify_binary64_candidate(&point_one, (-0.1_f64).to_bits()),
            Err(NumericError::Binary64CandidateSignMismatch)
        );
    }

    #[test]
    fn round_trip_formatter_stress_is_independently_proved_for_all_exponents() {
        const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
        const FINITE_EXPONENT_COUNT: usize = 2047;
        const SAMPLE_COUNT: usize = 4096;

        // Each finite exponent field appears once with each sign. An LCG
        // supplies deterministic, nontrivial significands without adding a
        // random-number dependency. The final two samples begin a third pass.
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let mut integer_reformats = 0_usize;
        for sample_index in 0..SAMPLE_COUNT {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let exponent_field = (sample_index % FINITE_EXPONENT_COUNT) as u64;
            let sign = ((sample_index / FINITE_EXPONENT_COUNT) as u64 & 1) << 63;
            let bits = sign | (exponent_field << 52) | (state & FRACTION_MASK);
            let rendered = f64::from_bits(bits).to_string();

            // Rust's round-trip formatter may render an exactly integral
            // float without a decimal point or exponent. Such a token belongs
            // to the raw-v1 integer class, so append an exact `.0` rather than
            // weakening the real-field checker or skipping the sample.
            let lexeme = if rendered
                .bytes()
                .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
            {
                rendered
            } else {
                integer_reformats += 1;
                format!("{rendered}.0")
            };

            let checked = checked_real_binary64_from_json(&lexeme, DEFAULT_JSON_NUMBER_LIMITS)
                .unwrap_or_else(|error| {
                    panic!(
                    "sample {sample_index} bits 0x{bits:016x} lexeme {lexeme:?} failed: {error}"
                )
                });
            assert_eq!(
                checked.bits(),
                bits,
                "sample {sample_index} lexeme {lexeme:?}"
            );
        }

        // This also proves the explicit integer-token reformat branch is live
        // for the current round-trip formatter rather than dead fallback code.
        assert!(integer_reformats > 0);
    }
}
