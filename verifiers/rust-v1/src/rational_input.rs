use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::NumericError;

/// Non-configurable component-size ceiling for public exact-rational values.
///
/// The check is applied to the supplied numerator and denominator before any
/// normalization, greatest-common-divisor calculation, comparison, or other
/// rational arithmetic.  The ceiling also admits the exponential kernel's
/// largest configured exact intermediate.
pub const HARD_MAX_RATIONAL_COMPONENT_BITS: u64 = 65_536;

/// Admit one caller-supplied rational only when it is bounded and canonical.
///
/// `num_rational::Ratio::new_raw` can construct a value with a zero or
/// negative denominator, or an unreduced representation.  Several `Ratio`
/// operations panic on a zero denominator, so every public rational boundary
/// calls this function before comparing or operating on the value.
pub(crate) fn validate_public_rational(value: &BigRational) -> Result<(), NumericError> {
    let numerator_bits = value.numer().bits();
    let denominator_bits = value.denom().bits();
    if numerator_bits > HARD_MAX_RATIONAL_COMPONENT_BITS
        || denominator_bits > HARD_MAX_RATIONAL_COMPONENT_BITS
    {
        return Err(NumericError::RationalComponentBitLimitExceeded {
            numerator_bits,
            denominator_bits,
            limit: HARD_MAX_RATIONAL_COMPONENT_BITS,
        });
    }

    if value.denom().is_zero() || value.denom().sign() != Sign::Plus {
        return Err(NumericError::InvalidRationalDenominator);
    }

    // The potentially nontrivial gcd work happens only after the hard cap.
    let canonical = BigRational::new(value.numer().clone(), value.denom().clone());
    if canonical.numer() != value.numer() || canonical.denom() != value.denom() {
        return Err(NumericError::NonCanonicalRational);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::*;

    #[test]
    fn validation_rejects_malformed_and_unreduced_raw_ratios() {
        assert_eq!(
            validate_public_rational(&BigRational::new_raw(BigInt::from(1), BigInt::from(0))),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            validate_public_rational(&BigRational::new_raw(BigInt::from(1), BigInt::from(-2))),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            validate_public_rational(&BigRational::new_raw(BigInt::from(2), BigInt::from(4))),
            Err(NumericError::NonCanonicalRational)
        );
        assert_eq!(
            validate_public_rational(&BigRational::new_raw(BigInt::from(0), BigInt::from(2))),
            Err(NumericError::NonCanonicalRational)
        );
    }

    #[test]
    fn component_limit_has_priority_over_normalization() {
        let oversized = BigInt::from(1_u8) << HARD_MAX_RATIONAL_COMPONENT_BITS as usize;
        let raw = BigRational::new_raw(oversized.clone(), oversized);
        assert_eq!(
            validate_public_rational(&raw),
            Err(NumericError::RationalComponentBitLimitExceeded {
                numerator_bits: HARD_MAX_RATIONAL_COMPONENT_BITS + 1,
                denominator_bits: HARD_MAX_RATIONAL_COMPONENT_BITS + 1,
                limit: HARD_MAX_RATIONAL_COMPONENT_BITS,
            })
        );
    }
}
