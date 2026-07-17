//! Resource-bounded first-order interval dual arithmetic.
//!
//! This is a small arithmetic kernel: a value interval and a fixed-length
//! vector of first-derivative intervals.  It makes no vector-field or
//! certificate-level claim.  Every constructor and operation is fallible,
//! dimensions are capped before allocation, and scalar/interval resource
//! checks are delegated to [`RationalInterval`].

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{sqrt_enclosure_dyadic, NumericError, RationalInterval};

/// Non-configurable maximum number of first-derivative coordinates.
pub const HARD_MAX_DUAL_DIMENSION: usize = 64;

/// A first-order dual number whose value and derivatives are rational
/// intervals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntervalDual {
    value: RationalInterval,
    derivatives: Vec<RationalInterval>,
}

impl IntervalDual {
    /// Construct a constant in a validated derivative dimension.
    pub fn constant(value: RationalInterval, dimension: usize) -> Result<Self, NumericError> {
        validate_dimension(dimension)?;
        let zero = rational_point(BigRational::zero())?;
        Self::from_parts(value, vec![zero; dimension])
    }

    /// Construct an independent variable with one unit basis derivative.
    pub fn variable(
        value: RationalInterval,
        dimension: usize,
        index: usize,
    ) -> Result<Self, NumericError> {
        validate_dimension(dimension)?;
        if index >= dimension {
            return Err(NumericError::DualIndexOutOfBounds { index, dimension });
        }

        let zero = rational_point(BigRational::zero())?;
        let mut derivatives = vec![zero; dimension];
        let basis = derivatives
            .get_mut(index)
            .ok_or(NumericError::DualIndexOutOfBounds { index, dimension })?;
        *basis = rational_point(BigRational::one())?;
        Self::from_parts(value, derivatives)
    }

    pub fn value(&self) -> &RationalInterval {
        &self.value
    }

    pub fn derivatives(&self) -> &[RationalInterval] {
        &self.derivatives
    }

    pub fn derivative(&self, index: usize) -> Result<&RationalInterval, NumericError> {
        self.derivatives
            .get(index)
            .ok_or(NumericError::DualIndexOutOfBounds {
                index,
                dimension: self.dimension(),
            })
    }

    pub fn dimension(&self) -> usize {
        self.derivatives.len()
    }

    pub fn add(&self, other: &Self) -> Result<Self, NumericError> {
        self.ensure_same_dimension(other)?;
        let value = self.value.add(&other.value)?;
        let mut derivatives = Vec::with_capacity(self.dimension());
        for (left, right) in self.derivatives.iter().zip(&other.derivatives) {
            derivatives.push(left.add(right)?);
        }
        Self::from_parts(value, derivatives)
    }

    pub fn subtract(&self, other: &Self) -> Result<Self, NumericError> {
        self.ensure_same_dimension(other)?;
        let value = self.value.subtract(&other.value)?;
        let mut derivatives = Vec::with_capacity(self.dimension());
        for (left, right) in self.derivatives.iter().zip(&other.derivatives) {
            derivatives.push(left.subtract(right)?);
        }
        Self::from_parts(value, derivatives)
    }

    pub fn multiply(&self, other: &Self) -> Result<Self, NumericError> {
        self.ensure_same_dimension(other)?;
        let value = self.value.multiply(&other.value)?;
        let mut derivatives = Vec::with_capacity(self.dimension());
        for (left_derivative, right_derivative) in self.derivatives.iter().zip(&other.derivatives) {
            let left_term = left_derivative.multiply(&other.value)?;
            let right_term = self.value.multiply(right_derivative)?;
            derivatives.push(left_term.add(&right_term)?);
        }
        Self::from_parts(value, derivatives)
    }

    pub fn scale(&self, scalar: &BigRational) -> Result<Self, NumericError> {
        self.validate_internal_dimension()?;
        let value = self.value.scale(scalar)?;
        let mut derivatives = Vec::with_capacity(self.dimension());
        for derivative in &self.derivatives {
            derivatives.push(derivative.scale(scalar)?);
        }
        Self::from_parts(value, derivatives)
    }

    /// Compute `1/self`, rejecting a value interval that contains zero.
    pub fn reciprocal(&self) -> Result<Self, NumericError> {
        self.validate_internal_dimension()?;
        let value = self.value.reciprocal()?;
        let inverse_squared = value.multiply(&value)?;
        let negative_one = BigRational::from_integer(BigInt::from(-1));
        let mut derivatives = Vec::with_capacity(self.dimension());
        for derivative in &self.derivatives {
            derivatives.push(
                derivative
                    .multiply(&inverse_squared)?
                    .scale(&negative_one)?,
            );
        }
        Self::from_parts(value, derivatives)
    }

    pub fn divide(&self, other: &Self) -> Result<Self, NumericError> {
        // Establish this before reciprocal work or any derivative zip.
        self.ensure_same_dimension(other)?;
        self.multiply(&other.reciprocal()?)
    }

    /// Compute the dual square using `(x^2)' = 2*x*x'`.
    pub fn square(&self) -> Result<Self, NumericError> {
        self.validate_internal_dimension()?;
        // Preserve the dependency between the two factors.  Generic interval
        // multiplication is sound but maps [-1,1]^2 to [-1,1], which can make
        // sums of squared distances spuriously cross zero.
        let lower_square = self.value.lower() * self.value.lower();
        let upper_square = self.value.upper() * self.value.upper();
        let maximum_square = if lower_square >= upper_square {
            lower_square.clone()
        } else {
            upper_square.clone()
        };
        let zero = BigRational::zero();
        let value = if self.value.contains(&zero)? {
            RationalInterval::new(zero, maximum_square)?
        } else {
            let minimum_square = if lower_square <= upper_square {
                lower_square
            } else {
                upper_square
            };
            RationalInterval::new(minimum_square, maximum_square)?
        };
        let two = BigRational::from_integer(BigInt::from(2));
        let mut derivatives = Vec::with_capacity(self.dimension());
        for derivative in &self.derivatives {
            derivatives.push(self.value.multiply(derivative)?.scale(&two)?);
        }
        Self::from_parts(value, derivatives)
    }

    /// Monotonically enclose `sqrt(self)` at the caller's dyadic precision.
    ///
    /// The input value interval must be strictly positive because this method
    /// also computes the derivative.  The result value is assembled as
    /// `[sqrt(lower).lower, sqrt(upper).upper]`.  The derivative coefficient
    /// is formed explicitly as `(1 / sqrt_interval) / 2` before examining the
    /// derivative vector.  Thus a coarse root enclosure containing zero fails
    /// closed even in dimension zero.
    pub fn sqrt_dyadic(&self, precision_bits: usize) -> Result<Self, NumericError> {
        self.validate_internal_dimension()?;
        if self.value.lower().numer().sign() != Sign::Plus {
            return Err(NumericError::DualSquareRootRequiresStrictlyPositiveValue);
        }

        let lower_root = sqrt_enclosure_dyadic(self.value.lower(), precision_bits)?;
        let upper_root = sqrt_enclosure_dyadic(self.value.upper(), precision_bits)?;
        let value = RationalInterval::new(
            lower_root.interval().lower().clone(),
            upper_root.interval().upper().clone(),
        )?;

        // Reciprocal is intentionally evaluated before the derivative loop.
        // A positive input may still have a zero lower root at coarse dyadic
        // precision, in which case no finite derivative enclosure is proved.
        let half = BigRational::new(BigInt::one(), BigInt::from(2));
        let derivative_coefficient = value.reciprocal()?.scale(&half)?;
        let mut derivatives = Vec::with_capacity(self.dimension());
        for derivative in &self.derivatives {
            derivatives.push(derivative.multiply(&derivative_coefficient)?);
        }
        Self::from_parts(value, derivatives)
    }

    fn from_parts(
        value: RationalInterval,
        derivatives: Vec<RationalInterval>,
    ) -> Result<Self, NumericError> {
        validate_dimension(derivatives.len())?;
        Ok(Self { value, derivatives })
    }

    fn validate_internal_dimension(&self) -> Result<(), NumericError> {
        validate_dimension(self.dimension())
    }

    fn ensure_same_dimension(&self, other: &Self) -> Result<(), NumericError> {
        self.validate_internal_dimension()?;
        other.validate_internal_dimension()?;
        if self.dimension() != other.dimension() {
            return Err(NumericError::DualDimensionMismatch {
                left_dimension: self.dimension(),
                right_dimension: other.dimension(),
            });
        }
        Ok(())
    }
}

fn validate_dimension(dimension: usize) -> Result<(), NumericError> {
    if dimension > HARD_MAX_DUAL_DIMENSION {
        return Err(NumericError::DualDimensionLimitExceeded {
            dimension,
            limit: HARD_MAX_DUAL_DIMENSION,
        });
    }
    Ok(())
}

fn rational_point(value: BigRational) -> Result<RationalInterval, NumericError> {
    RationalInterval::try_point(value)
}

#[cfg(test)]
mod tests {
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

    fn point(numerator: i64, denominator: i64) -> RationalInterval {
        interval(numerator, denominator, numerator, denominator)
    }

    #[test]
    fn constructors_and_accessors_build_valid_basis_vectors() {
        let constant = IntervalDual::constant(interval(1, 2, 3, 2), 2).unwrap();
        assert_eq!(constant.value(), &interval(1, 2, 3, 2));
        assert_eq!(constant.dimension(), 2);
        assert_eq!(constant.derivatives(), &[point(0, 1), point(0, 1)]);

        let variable = IntervalDual::variable(point(3, 1), 3, 1).unwrap();
        assert_eq!(variable.derivative(0).unwrap(), &point(0, 1));
        assert_eq!(variable.derivative(1).unwrap(), &point(1, 1));
        assert_eq!(variable.derivative(2).unwrap(), &point(0, 1));
        assert_eq!(
            variable.derivative(3),
            Err(NumericError::DualIndexOutOfBounds {
                index: 3,
                dimension: 3,
            })
        );
    }

    #[test]
    fn addition_subtraction_and_scaling_are_componentwise() {
        let first = IntervalDual::variable(point(2, 1), 2, 0).unwrap();
        let second = IntervalDual::variable(point(3, 1), 2, 1).unwrap();
        let sum = first.add(&second).unwrap();
        assert_eq!(sum.value(), &point(5, 1));
        assert_eq!(sum.derivatives(), &[point(1, 1), point(1, 1)]);

        let difference = first.subtract(&second).unwrap();
        assert_eq!(difference.value(), &point(-1, 1));
        assert_eq!(difference.derivatives(), &[point(1, 1), point(-1, 1)]);

        let scaled = difference.scale(&rational(-2, 1)).unwrap();
        assert_eq!(scaled.value(), &point(2, 1));
        assert_eq!(scaled.derivatives(), &[point(-2, 1), point(2, 1)]);
    }

    #[test]
    fn product_rule_encloses_value_and_each_basis_derivative() {
        let first = IntervalDual::variable(interval(2, 1, 3, 1), 2, 0).unwrap();
        let second = IntervalDual::variable(interval(4, 1, 5, 1), 2, 1).unwrap();
        let product = first.multiply(&second).unwrap();
        assert_eq!(product.value(), &interval(8, 1, 15, 1));
        assert_eq!(product.derivative(0).unwrap(), &interval(4, 1, 5, 1));
        assert_eq!(product.derivative(1).unwrap(), &interval(2, 1, 3, 1));
    }

    #[test]
    fn reciprocal_quotient_and_square_match_hand_derived_point_rules() {
        let numerator = IntervalDual::variable(point(6, 1), 2, 0).unwrap();
        let denominator = IntervalDual::variable(point(2, 1), 2, 1).unwrap();

        let reciprocal = denominator.reciprocal().unwrap();
        assert_eq!(reciprocal.value(), &point(1, 2));
        assert_eq!(reciprocal.derivative(0).unwrap(), &point(0, 1));
        assert_eq!(reciprocal.derivative(1).unwrap(), &point(-1, 4));

        let quotient = numerator.divide(&denominator).unwrap();
        assert_eq!(quotient.value(), &point(3, 1));
        assert_eq!(quotient.derivative(0).unwrap(), &point(1, 2));
        assert_eq!(quotient.derivative(1).unwrap(), &point(-3, 2));

        let ranged = IntervalDual::variable(interval(2, 1, 3, 1), 1, 0).unwrap();
        let square = ranged.square().unwrap();
        assert_eq!(square.value(), &interval(4, 1, 9, 1));
        assert_eq!(square.derivative(0).unwrap(), &interval(4, 1, 6, 1));

        let zero_crossing = IntervalDual::variable(interval(-2, 1, 3, 1), 1, 0).unwrap();
        let zero_crossing_square = zero_crossing.square().unwrap();
        assert_eq!(zero_crossing_square.value(), &interval(0, 1, 9, 1));
        assert_eq!(
            zero_crossing_square.derivative(0).unwrap(),
            &interval(-4, 1, 6, 1)
        );

        let negative = IntervalDual::variable(interval(-3, 1, -2, 1), 1, 0).unwrap();
        let negative_square = negative.square().unwrap();
        assert_eq!(negative_square.value(), &interval(4, 1, 9, 1));
        assert_eq!(
            negative_square.derivative(0).unwrap(),
            &interval(-6, 1, -4, 1)
        );
    }

    #[test]
    fn monotone_sqrt_uses_endpoint_enclosures_and_chain_rule() {
        let point_dual = IntervalDual::variable(point(4, 1), 1, 0).unwrap();
        let point_root = point_dual.sqrt_dyadic(8).unwrap();
        assert_eq!(point_root.value(), &point(2, 1));
        assert_eq!(point_root.derivative(0).unwrap(), &point(1, 4));

        let ranged = IntervalDual::variable(interval(4, 1, 9, 1), 1, 0).unwrap();
        let root = ranged.sqrt_dyadic(8).unwrap();
        assert_eq!(root.value(), &interval(2, 1, 3, 1));
        assert_eq!(root.derivative(0).unwrap(), &interval(1, 6, 1, 4));
    }

    #[test]
    fn dimensions_are_bounded_and_must_match_before_arithmetic() {
        let value = point(1, 1);
        assert_eq!(
            IntervalDual::constant(value.clone(), HARD_MAX_DUAL_DIMENSION + 1),
            Err(NumericError::DualDimensionLimitExceeded {
                dimension: HARD_MAX_DUAL_DIMENSION + 1,
                limit: HARD_MAX_DUAL_DIMENSION,
            })
        );
        assert!(IntervalDual::constant(value.clone(), HARD_MAX_DUAL_DIMENSION).is_ok());
        assert_eq!(
            IntervalDual::variable(value.clone(), 2, 2),
            Err(NumericError::DualIndexOutOfBounds {
                index: 2,
                dimension: 2,
            })
        );

        let one_dimensional = IntervalDual::constant(value.clone(), 1).unwrap();
        let two_dimensional = IntervalDual::constant(value, 2).unwrap();
        let mismatch = Err(NumericError::DualDimensionMismatch {
            left_dimension: 1,
            right_dimension: 2,
        });
        assert_eq!(one_dimensional.add(&two_dimensional), mismatch.clone());
        assert_eq!(one_dimensional.subtract(&two_dimensional), mismatch.clone());
        assert_eq!(one_dimensional.multiply(&two_dimensional), mismatch.clone());
        assert_eq!(one_dimensional.divide(&two_dimensional), mismatch);
    }

    #[test]
    fn reciprocal_and_sqrt_domains_fail_closed() {
        let zero_crossing = IntervalDual::variable(interval(-1, 1, 1, 1), 1, 0).unwrap();
        assert_eq!(
            zero_crossing.reciprocal(),
            Err(NumericError::DivisionByZeroInterval)
        );

        for value in [interval(-1, 1, 4, 1), interval(0, 1, 4, 1)] {
            let dual = IntervalDual::constant(value, 0).unwrap();
            assert_eq!(
                dual.sqrt_dyadic(8),
                Err(NumericError::DualSquareRootRequiresStrictlyPositiveValue)
            );
        }

        // Strict positivity alone is insufficient at a coarse dyadic grid:
        // the assembled root enclosure contains zero, so its reciprocal must
        // fail even though this constant has no derivative coordinates.
        let coarse = IntervalDual::constant(interval(1, 4, 1, 2), 0).unwrap();
        assert_eq!(
            coarse.sqrt_dyadic(0),
            Err(NumericError::DivisionByZeroInterval)
        );
    }

    #[test]
    fn interval_derivative_contains_a_centered_finite_difference() {
        let lower = rational(199, 100);
        let upper = rational(201, 100);
        let input = IntervalDual::variable(
            RationalInterval::new(lower.clone(), upper.clone()).unwrap(),
            1,
            0,
        )
        .unwrap();
        let square = input.square().unwrap();
        let finite_difference = ((&upper * &upper) - (&lower * &lower)) / (&upper - &lower);
        assert!(square
            .derivative(0)
            .unwrap()
            .contains(&finite_difference)
            .unwrap());
        assert_eq!(finite_difference, rational(4, 1));
    }

    #[test]
    fn malformed_scalars_are_rejected_without_panicking() {
        let dual = IntervalDual::variable(point(2, 1), 1, 0).unwrap();
        let zero_denominator = BigRational::new_raw(BigInt::one(), BigInt::zero());
        let unreduced = BigRational::new_raw(BigInt::from(2), BigInt::from(4));

        let zero_result = std::panic::catch_unwind(|| dual.scale(&zero_denominator));
        assert_eq!(
            zero_result.unwrap(),
            Err(NumericError::InvalidRationalDenominator)
        );
        let unreduced_result = std::panic::catch_unwind(|| dual.scale(&unreduced));
        assert_eq!(
            unreduced_result.unwrap(),
            Err(NumericError::NonCanonicalRational)
        );
    }
}
