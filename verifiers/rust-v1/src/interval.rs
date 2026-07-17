use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{rational_input::validate_public_rational, NumericError};

/// Non-configurable work ceiling for public interval Horner evaluation.
pub const HARD_MAX_INTERVAL_HORNER_COEFFICIENTS: usize = 4_096;

/// A closed interval with normalized arbitrary-size rational endpoints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalInterval {
    lower: BigRational,
    upper: BigRational,
}

impl RationalInterval {
    /// Construct a bounded interval from canonical rational endpoints.
    ///
    /// Both endpoints are admitted before their first comparison, so malformed
    /// values made with `BigRational::new_raw` fail without reaching a panicking
    /// rational operation.
    pub fn new(lower: BigRational, upper: BigRational) -> Result<Self, NumericError> {
        validate_public_rational(&lower)?;
        validate_public_rational(&upper)?;
        if lower > upper {
            return Err(NumericError::InvalidIntervalBounds);
        }
        Ok(Self { lower, upper })
    }

    /// Construct a bounded point interval from a caller-supplied rational.
    pub fn try_point(value: BigRational) -> Result<Self, NumericError> {
        validate_public_rational(&value)?;
        Ok(Self::point_validated(value))
    }

    /// Private construction after this module has admitted the value.
    fn point_validated(value: BigRational) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }

    pub fn lower(&self) -> &BigRational {
        &self.lower
    }

    pub fn upper(&self) -> &BigRational {
        &self.upper
    }

    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }

    /// Check scalar containment after admitting the caller-supplied rational.
    pub fn contains(&self, value: &BigRational) -> Result<bool, NumericError> {
        validate_public_rational(value)?;
        Ok(self.lower <= *value && *value <= self.upper)
    }

    pub fn contains_interval(&self, other: &Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    pub fn width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    pub fn add(&self, other: &Self) -> Result<Self, NumericError> {
        Self::new(&self.lower + &other.lower, &self.upper + &other.upper)
    }

    pub fn subtract(&self, other: &Self) -> Result<Self, NumericError> {
        Self::new(&self.lower - &other.upper, &self.upper - &other.lower)
    }

    pub fn multiply(&self, other: &Self) -> Result<Self, NumericError> {
        let products = [
            &self.lower * &other.lower,
            &self.lower * &other.upper,
            &self.upper * &other.lower,
            &self.upper * &other.upper,
        ];
        let mut lower = products[0].clone();
        let mut upper = products[0].clone();
        for product in &products[1..] {
            if product < &lower {
                lower = product.clone();
            }
            if product > &upper {
                upper = product.clone();
            }
        }
        Self::new(lower, upper)
    }

    pub fn reciprocal(&self) -> Result<Self, NumericError> {
        if self.contains(&BigRational::zero())? {
            return Err(NumericError::DivisionByZeroInterval);
        }
        let one = BigRational::one();
        Self::new(&one / &self.upper, &one / &self.lower)
    }

    pub fn divide(&self, other: &Self) -> Result<Self, NumericError> {
        self.multiply(&other.reciprocal()?)
    }

    pub fn scale(&self, scalar: &BigRational) -> Result<Self, NumericError> {
        validate_public_rational(scalar)?;
        if scalar.numer().sign() != Sign::Minus {
            Self::new(&self.lower * scalar, &self.upper * scalar)
        } else {
            Self::new(&self.upper * scalar, &self.lower * scalar)
        }
    }

    /// Evaluate `c[0] + c[1] x + ... + c[n] x^n` by interval Horner.
    ///
    /// The empty coefficient slice denotes the zero polynomial.
    pub fn horner_ascending(
        coefficients: &[BigRational],
        argument: &Self,
    ) -> Result<Self, NumericError> {
        if coefficients.len() > HARD_MAX_INTERVAL_HORNER_COEFFICIENTS {
            return Err(NumericError::PolynomialCoefficientCountLimitExceeded {
                coefficient_count: coefficients.len(),
                limit: HARD_MAX_INTERVAL_HORNER_COEFFICIENTS,
            });
        }
        for coefficient in coefficients {
            validate_public_rational(coefficient)?;
        }
        let Some(highest) = coefficients.last() else {
            return Self::try_point(BigRational::zero());
        };
        let mut accumulator = Self::point_validated(highest.clone());
        for coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
            accumulator = accumulator
                .multiply(argument)?
                .add(&Self::point_validated(coefficient.clone()))?;
        }
        Ok(accumulator)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::*;

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn interval(
        lower_numerator: i64,
        lower_denominator: i64,
        upper_numerator: i64,
        upper_denominator: i64,
    ) -> RationalInterval {
        RationalInterval::new(
            rational(lower_numerator, lower_denominator),
            rational(upper_numerator, upper_denominator),
        )
        .unwrap()
    }

    #[test]
    fn construction_normalizes_endpoints_and_rejects_reversal() {
        let normalized = interval(2, 4, 6, 8);
        assert_eq!(normalized.lower(), &rational(1, 2));
        assert_eq!(normalized.upper(), &rational(3, 4));
        assert_eq!(normalized.width(), rational(1, 4));
        assert_eq!(
            RationalInterval::new(rational(2, 1), rational(1, 1)),
            Err(NumericError::InvalidIntervalBounds)
        );
        assert_eq!(
            RationalInterval::try_point(rational(1, 2)).unwrap(),
            interval(1, 2, 1, 2)
        );
    }

    #[test]
    fn addition_subtraction_and_containment_use_complete_endpoints() {
        let first = interval(-1, 2, 3, 2);
        let second = interval(1, 3, 2, 3);
        assert_eq!(first.add(&second).unwrap(), interval(-1, 6, 13, 6));
        assert_eq!(first.subtract(&second).unwrap(), interval(-7, 6, 7, 6));
        assert!(first.contains(&rational(0, 1)).unwrap());
        assert!(!first.contains(&rational(2, 1)).unwrap());
        assert!(first.contains_interval(&interval(0, 1, 1, 1)));
        assert!(!first.contains_interval(&interval(0, 1, 2, 1)));
    }

    #[test]
    fn multiplication_handles_every_sign_configuration() {
        assert_eq!(
            interval(-2, 1, 3, 1).multiply(&interval(-5, 1, -1, 1)),
            Ok(interval(-15, 1, 10, 1))
        );
        assert_eq!(
            interval(2, 1, 4, 1).multiply(&interval(3, 1, 5, 1)),
            Ok(interval(6, 1, 20, 1))
        );
        assert_eq!(
            interval(-4, 1, -2, 1).multiply(&interval(-3, 1, 6, 1)),
            Ok(interval(-24, 1, 12, 1))
        );
    }

    #[test]
    fn reciprocal_and_division_are_exact_and_fail_closed_at_zero() {
        assert_eq!(
            interval(2, 1, 4, 1).reciprocal().unwrap(),
            interval(1, 4, 1, 2)
        );
        assert_eq!(
            interval(-4, 1, -2, 1).reciprocal().unwrap(),
            interval(-1, 2, -1, 4)
        );
        assert_eq!(
            interval(2, 1, 4, 1)
                .divide(&interval(-2, 1, -1, 1))
                .unwrap(),
            interval(-4, 1, -1, 1)
        );
        for divisor in [
            interval(-1, 1, 1, 1),
            interval(0, 1, 2, 1),
            RationalInterval::try_point(BigRational::zero()).unwrap(),
        ] {
            assert_eq!(
                interval(1, 1, 2, 1).divide(&divisor),
                Err(NumericError::DivisionByZeroInterval)
            );
        }
    }

    #[test]
    fn scaling_reorders_negative_products() {
        let value = interval(1, 2, 3, 2);
        assert_eq!(value.scale(&rational(4, 1)), Ok(interval(2, 1, 6, 1)));
        assert_eq!(value.scale(&rational(-2, 1)), Ok(interval(-3, 1, -1, 1)));
        assert_eq!(
            value.scale(&BigRational::zero()),
            Ok(RationalInterval::try_point(BigRational::zero()).unwrap())
        );
    }

    #[test]
    fn ascending_horner_matches_hand_derived_monotone_example() {
        // p(x) = 1 + 2x + 3x^2 on [1,2]. Horner gives [6,17], which is
        // also the exact range because p is increasing there.
        let coefficients = [rational(1, 1), rational(2, 1), rational(3, 1)];
        assert_eq!(
            RationalInterval::horner_ascending(&coefficients, &interval(1, 1, 2, 1)),
            Ok(interval(6, 1, 17, 1))
        );
        assert_eq!(
            RationalInterval::horner_ascending(&[], &interval(-10, 1, 10, 1)),
            Ok(RationalInterval::try_point(BigRational::zero()).unwrap())
        );
    }

    #[test]
    fn malformed_public_rationals_fail_without_panicking() {
        let one = rational(1, 1);
        let interval = RationalInterval::try_point(one.clone()).unwrap();
        let zero_denominator = BigRational::new_raw(BigInt::from(1), BigInt::from(0));
        let negative_denominator = BigRational::new_raw(BigInt::from(1), BigInt::from(-2));
        let unreduced = BigRational::new_raw(BigInt::from(2), BigInt::from(4));

        let construction = std::panic::catch_unwind(|| {
            RationalInterval::new(zero_denominator.clone(), one.clone())
        });
        assert_eq!(
            construction.unwrap(),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            RationalInterval::try_point(negative_denominator),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            interval.contains(&unreduced),
            Err(NumericError::NonCanonicalRational)
        );
        assert_eq!(
            interval.scale(&zero_denominator),
            Err(NumericError::InvalidRationalDenominator)
        );
        assert_eq!(
            RationalInterval::horner_ascending(&[unreduced], &interval),
            Err(NumericError::NonCanonicalRational)
        );
    }

    #[test]
    fn interval_operations_cannot_store_components_above_the_hard_cap() {
        let large = BigRational::from_integer(
            BigInt::from(1_u8) << (crate::HARD_MAX_RATIONAL_COMPONENT_BITS as usize - 1),
        );
        let point = RationalInterval::try_point(large).unwrap();
        assert!(matches!(
            point.multiply(&point),
            Err(NumericError::RationalComponentBitLimitExceeded { .. })
        ));
    }

    #[test]
    fn horner_coefficient_count_is_hard_capped_before_scanning_values() {
        let coefficients = vec![BigRational::zero(); HARD_MAX_INTERVAL_HORNER_COEFFICIENTS + 1];
        let argument = interval(0, 1, 1, 1);
        assert_eq!(
            RationalInterval::horner_ascending(&coefficients, &argument),
            Err(NumericError::PolynomialCoefficientCountLimitExceeded {
                coefficient_count: HARD_MAX_INTERVAL_HORNER_COEFFICIENTS + 1,
                limit: HARD_MAX_INTERVAL_HORNER_COEFFICIENTS,
            })
        );
    }
}
