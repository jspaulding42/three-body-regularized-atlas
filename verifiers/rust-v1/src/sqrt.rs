use num_bigint::{BigInt, BigUint, Sign};
use num_rational::BigRational;
use num_traits::One;

use crate::{rational_input::validate_public_rational, NumericError, RationalInterval};

/// Non-configurable precision ceiling checked before every integer shift.
///
/// Together with the shared 65,536-bit rational-component ceiling, this keeps
/// the scaled integer below 98,304 bits before the integer square root.
pub const HARD_MAX_SQRT_PRECISION_BITS: usize = 16_384;

/// A dyadic square-root enclosure and the grid precision used to build it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DyadicSqrtEnclosure {
    interval: RationalInterval,
    precision_bits: usize,
}

impl DyadicSqrtEnclosure {
    pub fn interval(&self) -> &RationalInterval {
        &self.interval
    }

    pub const fn precision_bits(&self) -> usize {
        self.precision_bits
    }

    pub fn lower_square(&self) -> BigRational {
        self.interval.lower() * self.interval.lower()
    }

    pub fn upper_square(&self) -> BigRational {
        self.interval.upper() * self.interval.upper()
    }

    /// Recheck every theorem-facing postcondition with exact arithmetic.
    pub fn verifies(&self, radicand: &BigRational) -> bool {
        if validate_public_rational(radicand).is_err()
            || validate_public_rational(self.interval.lower()).is_err()
            || validate_public_rational(self.interval.upper()).is_err()
            || self.precision_bits > HARD_MAX_SQRT_PRECISION_BITS
            || radicand.numer().sign() == Sign::Minus
            || self.interval.lower().numer().sign() == Sign::Minus
        {
            return false;
        }
        let grid_denominator = BigInt::one() << self.precision_bits;
        let grid_scale = BigRational::from_integer(grid_denominator.clone());
        let lower_scaled = self.interval.lower() * &grid_scale;
        let upper_scaled = self.interval.upper() * &grid_scale;
        let grid_membership =
            lower_scaled.denom() == &BigInt::one() && upper_scaled.denom() == &BigInt::one();
        let unit = BigRational::new(BigInt::one(), grid_denominator);

        grid_membership
            && self.lower_square() <= *radicand
            && *radicand <= self.upper_square()
            && self.interval.width() <= unit
    }
}

/// Enclose `sqrt(radicand)` on the dyadic grid with spacing `2^-precision_bits`.
///
/// The construction computes
/// `floor(sqrt(radicand) * 2^precision_bits)` using only integer shifts,
/// division, and an integer square root. Before returning, it rechecks exact
/// lower-square, upper-square, grid-membership, and width postconditions.
pub fn sqrt_enclosure_dyadic(
    radicand: &BigRational,
    precision_bits: usize,
) -> Result<DyadicSqrtEnclosure, NumericError> {
    validate_public_rational(radicand)?;
    if precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(NumericError::SquareRootPrecisionBitLimitExceeded {
            precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        });
    }
    if radicand.numer().sign() == Sign::Minus {
        return Err(NumericError::NegativeSquareRoot);
    }
    let doubled_precision = precision_bits
        .checked_mul(2)
        .ok_or(NumericError::PrecisionOverflow)?;

    let numerator = radicand
        .numer()
        .to_biguint()
        .ok_or(NumericError::InternalSquareRootPostconditionFailure)?;
    let denominator = radicand
        .denom()
        .to_biguint()
        .ok_or(NumericError::InternalSquareRootPostconditionFailure)?;
    let scaled_floor = (numerator << doubled_precision) / denominator;
    let lower_grid_numerator = integer_sqrt_floor(&scaled_floor);

    let grid_denominator = BigInt::one() << precision_bits;
    let lower = BigRational::new(
        BigInt::from(lower_grid_numerator.clone()),
        grid_denominator.clone(),
    );
    let lower_square = &lower * &lower;
    let upper = if lower_square == *radicand {
        lower.clone()
    } else {
        BigRational::new(
            BigInt::from(lower_grid_numerator + BigUint::one()),
            grid_denominator,
        )
    };
    let enclosure = DyadicSqrtEnclosure {
        interval: RationalInterval::new(lower, upper)?,
        precision_bits,
    };
    if !enclosure.verifies(radicand) {
        return Err(NumericError::InternalSquareRootPostconditionFailure);
    }
    Ok(enclosure)
}

/// Integer floor square root via monotone Newton iteration from a power-of-two
/// upper bound. The returned value `r` satisfies `r^2 <= n < (r+1)^2`.
fn integer_sqrt_floor(value: &BigUint) -> BigUint {
    if value <= &BigUint::one() {
        return value.clone();
    }
    let initial_shift = value.bits().div_ceil(2) as usize;
    let mut estimate = BigUint::one() << initial_shift;
    loop {
        let next = (&estimate + value / &estimate) >> 1_usize;
        if next >= estimate {
            debug_assert!(&estimate * &estimate <= *value);
            let successor = &estimate + BigUint::one();
            debug_assert!(&successor * &successor > *value);
            return estimate;
        }
        estimate = next;
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;
    use crate::rational_from_f64_bits;

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    #[test]
    fn integer_square_root_hits_and_brackets_boundaries() {
        for (value, expected) in [
            (0_u64, 0_u64),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (15, 3),
            (16, 4),
            (17, 4),
            (24, 4),
            (25, 5),
        ] {
            assert_eq!(
                integer_sqrt_floor(&BigUint::from(value)),
                BigUint::from(expected)
            );
        }

        let root = (BigUint::one() << 300_usize) + BigUint::from(123_u32);
        let square = &root * &root;
        assert_eq!(integer_sqrt_floor(&square), root);
        assert_eq!(
            integer_sqrt_floor(&(&square - BigUint::one())),
            &root - BigUint::one()
        );
        assert_eq!(integer_sqrt_floor(&(&square + &root)), root);
    }

    #[test]
    fn exact_dyadic_squares_collapse_to_points() {
        for (radicand, precision, root) in [
            (rational(0, 1), 0_usize, rational(0, 1)),
            (rational(4, 1), 0, rational(2, 1)),
            (rational(9, 16), 8, rational(3, 4)),
            (rational(1, 1024), 12, rational(1, 32)),
        ] {
            let enclosure = sqrt_enclosure_dyadic(&radicand, precision).unwrap();
            assert_eq!(
                enclosure.interval(),
                &RationalInterval::try_point(root).unwrap()
            );
            assert!(enclosure.verifies(&radicand));
            assert_eq!(enclosure.lower_square(), radicand);
            assert_eq!(enclosure.upper_square(), radicand);
        }
    }

    #[test]
    fn irrational_examples_match_hand_derived_dyadic_brackets() {
        let sqrt_two = sqrt_enclosure_dyadic(&rational(2, 1), 3).unwrap();
        assert_eq!(sqrt_two.interval().lower(), &rational(11, 8));
        assert_eq!(sqrt_two.interval().upper(), &rational(3, 2));
        assert_eq!(sqrt_two.lower_square(), rational(121, 64));
        assert_eq!(sqrt_two.upper_square(), rational(9, 4));

        let sqrt_half = sqrt_enclosure_dyadic(&rational(1, 2), 4).unwrap();
        assert_eq!(sqrt_half.interval().lower(), &rational(11, 16));
        assert_eq!(sqrt_half.interval().upper(), &rational(3, 4));
        assert!(sqrt_half.verifies(&rational(1, 2)));

        let zero_precision = sqrt_enclosure_dyadic(&rational(2, 1), 0).unwrap();
        assert_eq!(zero_precision.interval().lower(), &rational(1, 1));
        assert_eq!(zero_precision.interval().upper(), &rational(2, 1));
    }

    #[test]
    fn high_precision_postconditions_are_exact() {
        let radicand = rational(1, 3);
        let enclosure = sqrt_enclosure_dyadic(&radicand, 256).unwrap();
        assert_eq!(enclosure.precision_bits(), 256);
        assert!(enclosure.verifies(&radicand));
        assert!(enclosure.lower_square() <= radicand);
        assert!(enclosure.upper_square() >= radicand);
        assert_eq!(
            enclosure.interval().width(),
            BigRational::new(BigInt::one(), BigInt::one() << 256_usize)
        );
    }

    #[test]
    fn binary64_extremes_can_be_enclosed_without_floating_arithmetic() {
        let smallest_subnormal = rational_from_f64_bits(1).unwrap();
        let exact_small_root = sqrt_enclosure_dyadic(&smallest_subnormal, 600).unwrap();
        assert_eq!(
            exact_small_root.interval(),
            &RationalInterval::try_point(BigRational::new(
                BigInt::one(),
                BigInt::one() << 537_usize,
            ))
            .unwrap()
        );

        let largest_finite = rational_from_f64_bits(0x7fef_ffff_ffff_ffff).unwrap();
        let large_enclosure = sqrt_enclosure_dyadic(&largest_finite, 96).unwrap();
        assert!(large_enclosure.verifies(&largest_finite));
    }

    #[test]
    fn negative_radicands_are_rejected() {
        assert_eq!(
            sqrt_enclosure_dyadic(&rational(-1, 100), 64),
            Err(NumericError::NegativeSquareRoot)
        );
    }

    #[test]
    fn malformed_and_oversized_radicands_fail_without_panicking() {
        let zero_denominator = BigRational::new_raw(BigInt::from(1), BigInt::from(0));
        let negative_denominator = BigRational::new_raw(BigInt::from(1), BigInt::from(-2));
        let unreduced = BigRational::new_raw(BigInt::from(2), BigInt::from(4));
        let oversized = BigRational::from_integer(
            BigInt::one() << crate::HARD_MAX_RATIONAL_COMPONENT_BITS as usize,
        );

        let result = std::panic::catch_unwind(|| sqrt_enclosure_dyadic(&zero_denominator, 64));
        assert_eq!(
            result.unwrap(),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            sqrt_enclosure_dyadic(&negative_denominator, 64),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            sqrt_enclosure_dyadic(&unreduced, 64),
            Err(NumericError::NonCanonicalRational)
        );
        assert!(matches!(
            sqrt_enclosure_dyadic(&oversized, 64),
            Err(NumericError::RationalComponentBitLimitExceeded { .. })
        ));

        let enclosure = sqrt_enclosure_dyadic(&rational(2, 1), 64).unwrap();
        assert!(!enclosure.verifies(&zero_denominator));
        assert!(!enclosure.verifies(&negative_denominator));
        assert!(!enclosure.verifies(&unreduced));
        assert!(!enclosure.verifies(&oversized));
    }

    #[test]
    fn precision_limit_is_checked_before_any_shift() {
        assert_eq!(
            sqrt_enclosure_dyadic(&BigRational::zero(), HARD_MAX_SQRT_PRECISION_BITS + 1,),
            Err(NumericError::SquareRootPrecisionBitLimitExceeded {
                precision_bits: HARD_MAX_SQRT_PRECISION_BITS + 1,
                limit: HARD_MAX_SQRT_PRECISION_BITS,
            })
        );
        assert!(sqrt_enclosure_dyadic(&BigRational::zero(), HARD_MAX_SQRT_PRECISION_BITS,).is_ok());

        let mut enclosure = sqrt_enclosure_dyadic(&rational(2, 1), 8).unwrap();
        enclosure.precision_bits = HARD_MAX_SQRT_PRECISION_BITS + 1;
        assert!(!enclosure.verifies(&rational(2, 1)));
    }
}
