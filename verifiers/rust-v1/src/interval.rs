use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::NumericError;

/// A closed interval with normalized arbitrary-size rational endpoints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalInterval {
    lower: BigRational,
    upper: BigRational,
}

impl RationalInterval {
    pub fn new(lower: BigRational, upper: BigRational) -> Result<Self, NumericError> {
        if lower > upper {
            return Err(NumericError::InvalidIntervalBounds);
        }
        Ok(Self { lower, upper })
    }

    pub fn point(value: BigRational) -> Self {
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

    pub fn contains(&self, value: &BigRational) -> bool {
        self.lower <= *value && *value <= self.upper
    }

    pub fn contains_interval(&self, other: &Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    pub fn width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            lower: &self.lower + &other.lower,
            upper: &self.upper + &other.upper,
        }
    }

    pub fn subtract(&self, other: &Self) -> Self {
        Self {
            lower: &self.lower - &other.upper,
            upper: &self.upper - &other.lower,
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let products = [
            &self.lower * &other.lower,
            &self.lower * &other.upper,
            &self.upper * &other.lower,
            &self.upper * &other.upper,
        ];
        let lower = products.iter().min().expect("four products").clone();
        let upper = products.iter().max().expect("four products").clone();
        Self { lower, upper }
    }

    pub fn reciprocal(&self) -> Result<Self, NumericError> {
        if self.contains(&BigRational::zero()) {
            return Err(NumericError::DivisionByZeroInterval);
        }
        let one = BigRational::one();
        Self::new(&one / &self.upper, &one / &self.lower)
    }

    pub fn divide(&self, other: &Self) -> Result<Self, NumericError> {
        Ok(self.multiply(&other.reciprocal()?))
    }

    pub fn scale(&self, scalar: &BigRational) -> Self {
        if scalar >= &BigRational::zero() {
            Self {
                lower: &self.lower * scalar,
                upper: &self.upper * scalar,
            }
        } else {
            Self {
                lower: &self.upper * scalar,
                upper: &self.lower * scalar,
            }
        }
    }

    /// Evaluate `c[0] + c[1] x + ... + c[n] x^n` by interval Horner.
    ///
    /// The empty coefficient slice denotes the zero polynomial.
    pub fn horner_ascending(coefficients: &[BigRational], argument: &Self) -> Self {
        let Some(highest) = coefficients.last() else {
            return Self::point(BigRational::zero());
        };
        let mut accumulator = Self::point(highest.clone());
        for coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
            accumulator = accumulator
                .multiply(argument)
                .add(&Self::point(coefficient.clone()));
        }
        accumulator
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
    }

    #[test]
    fn addition_subtraction_and_containment_use_complete_endpoints() {
        let first = interval(-1, 2, 3, 2);
        let second = interval(1, 3, 2, 3);
        assert_eq!(first.add(&second), interval(-1, 6, 13, 6));
        assert_eq!(first.subtract(&second), interval(-7, 6, 7, 6));
        assert!(first.contains(&rational(0, 1)));
        assert!(!first.contains(&rational(2, 1)));
        assert!(first.contains_interval(&interval(0, 1, 1, 1)));
        assert!(!first.contains_interval(&interval(0, 1, 2, 1)));
    }

    #[test]
    fn multiplication_handles_every_sign_configuration() {
        assert_eq!(
            interval(-2, 1, 3, 1).multiply(&interval(-5, 1, -1, 1)),
            interval(-15, 1, 10, 1)
        );
        assert_eq!(
            interval(2, 1, 4, 1).multiply(&interval(3, 1, 5, 1)),
            interval(6, 1, 20, 1)
        );
        assert_eq!(
            interval(-4, 1, -2, 1).multiply(&interval(-3, 1, 6, 1)),
            interval(-24, 1, 12, 1)
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
            RationalInterval::point(BigRational::zero()),
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
        assert_eq!(value.scale(&rational(4, 1)), interval(2, 1, 6, 1));
        assert_eq!(value.scale(&rational(-2, 1)), interval(-3, 1, -1, 1));
        assert_eq!(
            value.scale(&BigRational::zero()),
            RationalInterval::point(BigRational::zero())
        );
    }

    #[test]
    fn ascending_horner_matches_hand_derived_monotone_example() {
        // p(x) = 1 + 2x + 3x^2 on [1,2]. Horner gives [6,17], which is
        // also the exact range because p is increasing there.
        let coefficients = [rational(1, 1), rational(2, 1), rational(3, 1)];
        assert_eq!(
            RationalInterval::horner_ascending(&coefficients, &interval(1, 1, 2, 1)),
            interval(6, 1, 17, 1)
        );
        assert_eq!(
            RationalInterval::horner_ascending(&[], &interval(-10, 1, 10, 1)),
            RationalInterval::point(BigRational::zero())
        );
    }
}
