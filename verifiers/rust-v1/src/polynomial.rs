//! Bounded exact-rational vector polynomials with interval Horner evaluation.
//!
//! Coefficients are stored in the raw-v1 order: the outer index is ascending
//! monomial degree and each row is one fixed-dimensional vector coefficient.
//! This module supplies arithmetic only; chart-specific minimum degrees and
//! record-shape obligations remain semantic replay concerns.

use core::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    rational_input::validate_public_rational, NumericError, RationalInterval,
    HARD_MAX_INTERVAL_HORNER_COEFFICIENTS,
};

/// Maximum number of components in one vector-valued polynomial.
pub const HARD_MAX_POLYNOMIAL_DIMENSION: usize = 64;

/// Maximum outer monomial degree.  The corresponding coefficient-count cap
/// matches the scalar interval-Horner primitive.
pub const HARD_MAX_POLYNOMIAL_DEGREE: usize = HARD_MAX_INTERVAL_HORNER_COEFFICIENTS - 1;

/// Maximum conservative operation count admitted by one public polynomial
/// construction or evaluation.
pub const HARD_MAX_POLYNOMIAL_WORK_UNITS: usize = 65_536;

// Exact derivative evaluation can require coefficient scaling, point
// admission, interval multiplication, and interval addition.  Admission uses
// this worst-case factor so every stored polynomial is evaluable by every
// public method without relaxing the work ceiling later.
const MAX_WORK_UNITS_PER_COEFFICIENT: usize = 4;

/// Fail-closed errors specific to polynomial shape and bounded work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PolynomialError {
    Numeric(NumericError),
    EmptyCoefficientTable,
    ZeroDimension,
    DimensionLimitExceeded {
        dimension: usize,
        limit: usize,
    },
    DegreeLimitExceeded {
        degree: usize,
        limit: usize,
    },
    CoefficientDimensionMismatch {
        degree_index: usize,
        expected: usize,
        actual: usize,
    },
    WorkLimitExceeded {
        required: usize,
        limit: usize,
    },
}

impl fmt::Display for PolynomialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => write!(formatter, "polynomial numeric failure: {source}"),
            Self::EmptyCoefficientTable => {
                formatter.write_str("polynomial coefficient table is empty")
            }
            Self::ZeroDimension => formatter.write_str("polynomial vector dimension is zero"),
            Self::DimensionLimitExceeded { dimension, limit } => write!(
                formatter,
                "polynomial dimension {dimension} exceeds hard limit {limit}"
            ),
            Self::DegreeLimitExceeded { degree, limit } => write!(
                formatter,
                "polynomial degree {degree} exceeds hard limit {limit}"
            ),
            Self::CoefficientDimensionMismatch {
                degree_index,
                expected,
                actual,
            } => write!(
                formatter,
                "coefficient row {degree_index} has dimension {actual}, expected {expected}"
            ),
            Self::WorkLimitExceeded { required, limit } => write!(
                formatter,
                "polynomial operation requires {required} work units, hard limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for PolynomialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::EmptyCoefficientTable
            | Self::ZeroDimension
            | Self::DimensionLimitExceeded { .. }
            | Self::DegreeLimitExceeded { .. }
            | Self::CoefficientDimensionMismatch { .. }
            | Self::WorkLimitExceeded { .. } => None,
        }
    }
}

impl From<NumericError> for PolynomialError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// A fixed-dimensional exact-rational polynomial in ascending degree order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactRationalPolynomial {
    coefficients_by_degree: Vec<Vec<BigRational>>,
    dimension: usize,
}

impl ExactRationalPolynomial {
    /// Admit a nonempty rectangular degree-major coefficient table.
    pub fn from_degree_major_coefficients(
        coefficients_by_degree: Vec<Vec<BigRational>>,
    ) -> Result<Self, PolynomialError> {
        let coefficient_count = coefficients_by_degree.len();
        if coefficient_count == 0 {
            return Err(PolynomialError::EmptyCoefficientTable);
        }
        let degree = coefficient_count - 1;
        if degree > HARD_MAX_POLYNOMIAL_DEGREE {
            return Err(PolynomialError::DegreeLimitExceeded {
                degree,
                limit: HARD_MAX_POLYNOMIAL_DEGREE,
            });
        }

        let dimension = coefficients_by_degree[0].len();
        validate_dimension(dimension)?;
        enforce_work_limit(dimension, coefficient_count, MAX_WORK_UNITS_PER_COEFFICIENT)?;
        for (degree_index, row) in coefficients_by_degree.iter().enumerate() {
            if row.len() != dimension {
                return Err(PolynomialError::CoefficientDimensionMismatch {
                    degree_index,
                    expected: dimension,
                    actual: row.len(),
                });
            }
            for coefficient in row {
                validate_public_rational(coefficient)?;
            }
        }

        Ok(Self {
            coefficients_by_degree,
            dimension,
        })
    }

    pub fn coefficients_by_degree(&self) -> &[Vec<BigRational>] {
        &self.coefficients_by_degree
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn coefficient_count(&self) -> usize {
        self.coefficients_by_degree.len()
    }

    pub fn degree(&self) -> usize {
        self.coefficient_count() - 1
    }

    /// Evaluate every component over one interval argument by Horner's rule.
    pub fn evaluate(
        &self,
        argument: &RationalInterval,
    ) -> Result<Vec<RationalInterval>, PolynomialError> {
        enforce_work_limit(self.dimension, self.coefficient_count(), 2)?;
        let highest = &self.coefficients_by_degree[self.degree()];
        let mut values = Vec::with_capacity(self.dimension);
        for (component, highest_coefficient) in highest.iter().enumerate() {
            let mut accumulator = RationalInterval::try_point(highest_coefficient.clone())?;
            for row in self.coefficients_by_degree[..self.degree()].iter().rev() {
                accumulator = accumulator
                    .multiply(argument)?
                    .add(&RationalInterval::try_point(row[component].clone())?)?;
            }
            values.push(accumulator);
        }
        Ok(values)
    }

    /// Construct the exact formal derivative coefficient table.
    pub fn derivative(&self) -> Result<Self, PolynomialError> {
        if self.degree() == 0 {
            return Self::from_degree_major_coefficients(vec![vec![
                BigRational::zero();
                self.dimension
            ]]);
        }
        enforce_work_limit(self.dimension, self.degree(), 2)?;
        let mut derivative = Vec::with_capacity(self.degree());
        for degree in 1..self.coefficient_count() {
            let multiplier = BigRational::from_integer(BigInt::from(degree));
            let mut row = Vec::with_capacity(self.dimension);
            for coefficient in &self.coefficients_by_degree[degree] {
                row.push(coefficient * &multiplier);
            }
            derivative.push(row);
        }
        Self::from_degree_major_coefficients(derivative)
    }

    /// Evaluate the formal derivative directly, without changing coefficient
    /// order or introducing non-rational arithmetic.
    pub fn evaluate_derivative(
        &self,
        argument: &RationalInterval,
    ) -> Result<Vec<RationalInterval>, PolynomialError> {
        if self.degree() == 0 {
            return Ok(vec![
                RationalInterval::try_point(BigRational::zero())?;
                self.dimension
            ]);
        }
        enforce_work_limit(
            self.dimension,
            self.degree(),
            MAX_WORK_UNITS_PER_COEFFICIENT,
        )?;
        let highest_degree = self.degree();
        let highest_multiplier = BigRational::from_integer(BigInt::from(highest_degree));
        let mut values = Vec::with_capacity(self.dimension);
        for (component, coefficient) in self.coefficients_by_degree[highest_degree]
            .iter()
            .enumerate()
        {
            let highest = coefficient * &highest_multiplier;
            let mut accumulator = RationalInterval::try_point(highest)?;
            for degree in (1..highest_degree).rev() {
                let multiplier = BigRational::from_integer(BigInt::from(degree));
                let coefficient = &self.coefficients_by_degree[degree][component] * multiplier;
                accumulator = accumulator
                    .multiply(argument)?
                    .add(&RationalInterval::try_point(coefficient)?)?;
            }
            values.push(accumulator);
        }
        Ok(values)
    }
}

fn validate_dimension(dimension: usize) -> Result<(), PolynomialError> {
    if dimension == 0 {
        return Err(PolynomialError::ZeroDimension);
    }
    if dimension > HARD_MAX_POLYNOMIAL_DIMENSION {
        return Err(PolynomialError::DimensionLimitExceeded {
            dimension,
            limit: HARD_MAX_POLYNOMIAL_DIMENSION,
        });
    }
    Ok(())
}

fn enforce_work_limit(
    dimension: usize,
    coefficient_count: usize,
    factor: usize,
) -> Result<(), PolynomialError> {
    let required = dimension
        .checked_mul(coefficient_count)
        .and_then(|work| work.checked_mul(factor))
        .unwrap_or(usize::MAX);
    if required > HARD_MAX_POLYNOMIAL_WORK_UNITS {
        return Err(PolynomialError::WorkLimitExceeded {
            required,
            limit: HARD_MAX_POLYNOMIAL_WORK_UNITS,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::*;
    use crate::HARD_MAX_RATIONAL_COMPONENT_BITS;

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

    fn example_polynomial() -> ExactRationalPolynomial {
        // (1 + 2t + 3t^2, -1 + 4t^2)
        ExactRationalPolynomial::from_degree_major_coefficients(vec![
            vec![rational(1, 1), rational(-1, 1)],
            vec![rational(2, 1), rational(0, 1)],
            vec![rational(3, 1), rational(4, 1)],
        ])
        .unwrap()
    }

    fn exact_scalar_horner(coefficients: &[BigRational], argument: &BigRational) -> BigRational {
        let mut accumulator = coefficients
            .last()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        for coefficient in coefficients[..coefficients.len().saturating_sub(1)]
            .iter()
            .rev()
        {
            accumulator = accumulator * argument + coefficient;
        }
        accumulator
    }

    fn exact_scalar_derivative(
        coefficients: &[BigRational],
        argument: &BigRational,
    ) -> BigRational {
        if coefficients.len() <= 1 {
            return BigRational::zero();
        }
        let derivative: Vec<BigRational> = coefficients
            .iter()
            .enumerate()
            .skip(1)
            .map(|(degree, coefficient)| coefficient * BigInt::from(degree))
            .collect();
        exact_scalar_horner(&derivative, argument)
    }

    #[test]
    fn degree_major_shape_and_accessors_are_explicit() {
        let polynomial = example_polynomial();
        assert_eq!(polynomial.dimension(), 2);
        assert_eq!(polynomial.degree(), 2);
        assert_eq!(polynomial.coefficient_count(), 3);
        assert_eq!(
            polynomial.coefficients_by_degree()[1],
            vec![rational(2, 1), rational(0, 1)]
        );

        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(vec![]),
            Err(PolynomialError::EmptyCoefficientTable)
        );
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![]]),
            Err(PolynomialError::ZeroDimension)
        );
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(vec![
                vec![rational(1, 1), rational(2, 1)],
                vec![rational(3, 1)],
            ]),
            Err(PolynomialError::CoefficientDimensionMismatch {
                degree_index: 1,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn interval_horner_matches_hand_derived_vector_examples() {
        let polynomial = example_polynomial();
        assert_eq!(
            polynomial.evaluate(&interval(1, 1, 2, 1)).unwrap(),
            vec![interval(6, 1, 17, 1), interval(3, 1, 15, 1)]
        );
        assert_eq!(
            polynomial.evaluate(&point(2, 1)).unwrap(),
            vec![point(17, 1), point(15, 1)]
        );
    }

    #[test]
    fn exact_derivative_coefficients_and_direct_evaluation_agree() {
        let polynomial = example_polynomial();
        let derivative = polynomial.derivative().unwrap();
        assert_eq!(
            derivative.coefficients_by_degree(),
            &[
                vec![rational(2, 1), rational(0, 1)],
                vec![rational(6, 1), rational(8, 1)],
            ]
        );
        let argument = interval(1, 1, 2, 1);
        let expected = vec![interval(8, 1, 14, 1), interval(8, 1, 16, 1)];
        assert_eq!(derivative.evaluate(&argument).unwrap(), expected);
        assert_eq!(polynomial.evaluate_derivative(&argument).unwrap(), expected);

        let constant = ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![
            rational(7, 3),
            rational(-2, 5),
        ]])
        .unwrap();
        assert_eq!(
            constant.derivative().unwrap().coefficients_by_degree(),
            &[vec![BigRational::zero(), BigRational::zero()]]
        );
        assert_eq!(
            constant.evaluate_derivative(&argument).unwrap(),
            vec![point(0, 1), point(0, 1)]
        );
    }

    #[test]
    fn every_dimension_degree_and_work_cap_is_fail_closed() {
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![
                BigRational::zero();
                HARD_MAX_POLYNOMIAL_DIMENSION
                    + 1
            ]]),
            Err(PolynomialError::DimensionLimitExceeded {
                dimension: HARD_MAX_POLYNOMIAL_DIMENSION + 1,
                limit: HARD_MAX_POLYNOMIAL_DIMENSION,
            })
        );

        let excessive_degree = vec![vec![BigRational::zero()]; HARD_MAX_POLYNOMIAL_DEGREE + 2];
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(excessive_degree),
            Err(PolynomialError::DegreeLimitExceeded {
                degree: HARD_MAX_POLYNOMIAL_DEGREE + 1,
                limit: HARD_MAX_POLYNOMIAL_DEGREE,
            })
        );

        let excessive_work = vec![
            vec![BigRational::zero(); HARD_MAX_POLYNOMIAL_DIMENSION];
            HARD_MAX_POLYNOMIAL_WORK_UNITS
                / (HARD_MAX_POLYNOMIAL_DIMENSION
                    * MAX_WORK_UNITS_PER_COEFFICIENT)
                + 1
        ];
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(excessive_work),
            Err(PolynomialError::WorkLimitExceeded {
                required: HARD_MAX_POLYNOMIAL_WORK_UNITS
                    + HARD_MAX_POLYNOMIAL_DIMENSION * MAX_WORK_UNITS_PER_COEFFICIENT,
                limit: HARD_MAX_POLYNOMIAL_WORK_UNITS,
            })
        );

        let boundary = ExactRationalPolynomial::from_degree_major_coefficients(vec![
            vec![BigRational::zero(); HARD_MAX_POLYNOMIAL_DIMENSION];
            HARD_MAX_POLYNOMIAL_WORK_UNITS
                / (HARD_MAX_POLYNOMIAL_DIMENSION * MAX_WORK_UNITS_PER_COEFFICIENT)
        ])
        .unwrap();
        assert!(boundary.evaluate(&point(0, 1)).is_ok());
        assert!(boundary.derivative().is_ok());
        assert!(boundary.evaluate_derivative(&point(0, 1)).is_ok());
    }

    #[test]
    fn malformed_and_oversized_coefficients_never_reach_rational_arithmetic() {
        let malformed_cases = [
            (
                BigRational::new_raw(BigInt::one(), BigInt::zero()),
                NumericError::InvalidRationalDenominator,
            ),
            (
                BigRational::new_raw(BigInt::one(), BigInt::from(-2)),
                NumericError::InvalidRationalDenominator,
            ),
            (
                BigRational::new_raw(BigInt::from(2), BigInt::from(4)),
                NumericError::NonCanonicalRational,
            ),
        ];
        for (coefficient, expected) in malformed_cases {
            let result = std::panic::catch_unwind(|| {
                ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![coefficient]])
            });
            assert_eq!(result.unwrap(), Err(PolynomialError::Numeric(expected)));
        }

        let oversized =
            BigRational::from_integer(BigInt::one() << HARD_MAX_RATIONAL_COMPONENT_BITS as usize);
        assert_eq!(
            ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![oversized]]),
            Err(PolynomialError::Numeric(
                NumericError::RationalComponentBitLimitExceeded {
                    numerator_bits: HARD_MAX_RATIONAL_COMPONENT_BITS + 1,
                    denominator_bits: 1,
                    limit: HARD_MAX_RATIONAL_COMPONENT_BITS,
                }
            ))
        );
    }

    #[test]
    fn deterministic_property_stress_contains_exact_values_and_derivatives() {
        let outer_argument = interval(-3, 2, 5, 4);
        let inner_argument = interval(-1, 1, 1, 1);
        let sample_points = [
            rational(-3, 2),
            rational(-1, 1),
            rational(-1, 8),
            rational(0, 1),
            rational(1, 2),
            rational(5, 4),
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;

        for case_index in 0..128 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let coefficient_count = case_index % 6 + 1;
            let mut scalar_coefficients = Vec::with_capacity(coefficient_count);
            for _ in 0..coefficient_count {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let numerator = (state % 17) as i64 - 8;
                let denominator = (state % 5 + 1) as i64;
                scalar_coefficients.push(rational(numerator, denominator));
            }
            let table: Vec<Vec<BigRational>> = scalar_coefficients
                .iter()
                .cloned()
                .map(|coefficient| vec![coefficient])
                .collect();
            let polynomial =
                ExactRationalPolynomial::from_degree_major_coefficients(table).unwrap();
            let outer_value = polynomial.evaluate(&outer_argument).unwrap();
            let outer_derivative = polynomial.evaluate_derivative(&outer_argument).unwrap();
            let derivative_polynomial = polynomial.derivative().unwrap();
            assert_eq!(
                derivative_polynomial.evaluate(&outer_argument).unwrap(),
                outer_derivative
            );

            let inner_value = polynomial.evaluate(&inner_argument).unwrap();
            let inner_derivative = polynomial.evaluate_derivative(&inner_argument).unwrap();
            assert!(outer_value[0].contains_interval(&inner_value[0]));
            assert!(outer_derivative[0].contains_interval(&inner_derivative[0]));

            for point_value in &sample_points {
                let exact_value = exact_scalar_horner(&scalar_coefficients, point_value);
                let exact_derivative = exact_scalar_derivative(&scalar_coefficients, point_value);
                assert!(outer_value[0].contains(&exact_value).unwrap());
                assert!(outer_derivative[0].contains(&exact_derivative).unwrap());

                let point_interval = RationalInterval::try_point(point_value.clone()).unwrap();
                assert_eq!(
                    polynomial.evaluate(&point_interval).unwrap()[0],
                    RationalInterval::try_point(exact_value).unwrap()
                );
                assert_eq!(
                    polynomial.evaluate_derivative(&point_interval).unwrap()[0],
                    RationalInterval::try_point(exact_derivative).unwrap()
                );
            }
        }
    }
}
