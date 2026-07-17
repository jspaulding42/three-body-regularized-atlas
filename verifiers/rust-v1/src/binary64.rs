use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::NumericError;

const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
const EXPONENT_MASK: u64 = 0x7ff;
const SIGN_MASK: u64 = 1_u64 << 63;

/// A finite binary64 value with both its lexical bit identity and exact value.
///
/// `+0.0` and `-0.0` have the same theorem-facing rational value, while
/// `bits()` and `is_negative_zero()` retain their distinct encodings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactBinary64 {
    bits: u64,
    rational: BigRational,
}

impl ExactBinary64 {
    pub fn from_bits(bits: u64) -> Result<Self, NumericError> {
        let exponent_field = (bits >> 52) & EXPONENT_MASK;
        let fraction_field = bits & FRACTION_MASK;
        if exponent_field == EXPONENT_MASK {
            return Err(NumericError::NonFiniteBinary64 { bits });
        }

        let rational = if exponent_field == 0 && fraction_field == 0 {
            BigRational::zero()
        } else {
            let (significand, binary_exponent) = if exponent_field == 0 {
                // A subnormal is fraction_field * 2^-1074.
                (fraction_field, -1074_i32)
            } else {
                // A normal is (2^52 + fraction_field) *
                // 2^(unbiased_exponent - 52).
                (
                    (1_u64 << 52) | fraction_field,
                    exponent_field as i32 - 1023 - 52,
                )
            };

            let mut numerator = BigInt::from(significand);
            let mut denominator = BigInt::one();
            if binary_exponent >= 0 {
                numerator <<= binary_exponent as usize;
            } else {
                denominator <<= (-binary_exponent) as usize;
            }
            if bits & SIGN_MASK != 0 {
                numerator = -numerator;
            }
            BigRational::new(numerator, denominator)
        };

        Ok(Self { bits, rational })
    }

    pub fn from_f64(value: f64) -> Result<Self, NumericError> {
        Self::from_bits(value.to_bits())
    }

    pub const fn bits(&self) -> u64 {
        self.bits
    }

    pub fn rational(&self) -> &BigRational {
        &self.rational
    }

    pub const fn is_negative_zero(&self) -> bool {
        self.bits == SIGN_MASK
    }

    pub fn into_rational(self) -> BigRational {
        self.rational
    }
}

pub fn rational_from_f64_bits(bits: u64) -> Result<BigRational, NumericError> {
    ExactBinary64::from_bits(bits).map(ExactBinary64::into_rational)
}

pub fn rational_from_f64(value: f64) -> Result<BigRational, NumericError> {
    rational_from_f64_bits(value.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn power_of_two(exponent: usize) -> BigInt {
        BigInt::one() << exponent
    }

    #[test]
    fn signed_zero_has_one_rational_value_and_two_bit_identities() {
        let positive = ExactBinary64::from_bits(0x0000_0000_0000_0000).unwrap();
        let negative = ExactBinary64::from_bits(0x8000_0000_0000_0000).unwrap();

        assert_eq!(positive.rational(), &BigRational::zero());
        assert_eq!(negative.rational(), &BigRational::zero());
        assert!(!positive.is_negative_zero());
        assert!(negative.is_negative_zero());
        assert_ne!(positive.bits(), negative.bits());
    }

    #[test]
    fn hand_derived_normal_values_decode_exactly() {
        assert_eq!(
            rational_from_f64_bits(1.0_f64.to_bits()).unwrap(),
            BigRational::from_integer(BigInt::one())
        );
        assert_eq!(
            rational_from_f64_bits((-2.5_f64).to_bits()).unwrap(),
            BigRational::new(BigInt::from(-5), BigInt::from(2))
        );
        assert_eq!(
            rational_from_f64_bits(0.1_f64.to_bits()).unwrap(),
            BigRational::new(
                BigInt::from(3_602_879_701_896_397_u64),
                BigInt::from(36_028_797_018_963_968_u64),
            )
        );
    }

    #[test]
    fn subnormal_and_normal_boundaries_decode_exactly() {
        let smallest_subnormal = rational_from_f64_bits(0x0000_0000_0000_0001).unwrap();
        assert_eq!(
            smallest_subnormal,
            BigRational::new(BigInt::one(), power_of_two(1074))
        );

        let largest_subnormal = rational_from_f64_bits(0x000f_ffff_ffff_ffff).unwrap();
        assert_eq!(
            largest_subnormal,
            BigRational::new(power_of_two(52) - BigInt::one(), power_of_two(1074),)
        );

        let smallest_normal = rational_from_f64_bits(0x0010_0000_0000_0000).unwrap();
        assert_eq!(
            smallest_normal,
            BigRational::new(BigInt::one(), power_of_two(1022))
        );

        let negative_smallest_subnormal = rational_from_f64_bits(0x8000_0000_0000_0001).unwrap();
        assert_eq!(negative_smallest_subnormal, -smallest_subnormal);
    }

    #[test]
    fn largest_finite_value_decodes_without_overflow() {
        let decoded = rational_from_f64_bits(0x7fef_ffff_ffff_ffff).unwrap();
        let expected =
            BigRational::from_integer((power_of_two(53) - BigInt::one()) * power_of_two(971));
        assert_eq!(decoded, expected);
        assert_eq!(
            rational_from_f64_bits(0xffef_ffff_ffff_ffff).unwrap(),
            -decoded
        );
    }

    #[test]
    fn infinities_and_nan_payloads_are_rejected() {
        for bits in [
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0x7ff8_0000_0000_0000,
            0xffff_ffff_ffff_ffff,
        ] {
            assert_eq!(
                rational_from_f64_bits(bits),
                Err(NumericError::NonFiniteBinary64 { bits })
            );
        }
    }
}
