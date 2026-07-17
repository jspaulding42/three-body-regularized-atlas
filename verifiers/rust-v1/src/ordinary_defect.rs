//! Exact-rational interval defects for ordinary planar polynomial arcs.
//!
//! Position and velocity polynomials use ascending degree-major coefficient
//! tables and each have exactly six components.  This module composes the
//! bounded polynomial and ordinary Newton-field kernels; it does not decode or
//! replay any certificate schema.

use core::fmt;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    evaluate_planar_three_body_ordinary_field,
    ordinary_field::preflight_planar_three_body_ordinary_field, ExactRationalPolynomial,
    NumericError, OrdinaryFieldError, PolynomialError, RationalInterval,
};

const CONFIGURATION_DIMENSION: usize = 6;
const STATE_DIMENSION: usize = 2 * CONFIGURATION_DIMENSION;
const BODY_COUNT: usize = 3;

/// Fail-closed errors from ordinary polynomial-defect evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryPolynomialDefectError {
    PositionDimensionMismatch { expected: usize, actual: usize },
    VelocityDimensionMismatch { expected: usize, actual: usize },
    PositionPolynomial(PolynomialError),
    VelocityPolynomial(PolynomialError),
    OrdinaryField(OrdinaryFieldError),
    Numeric(NumericError),
}

impl fmt::Display for OrdinaryPolynomialDefectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionDimensionMismatch { expected, actual } => write!(
                formatter,
                "ordinary position polynomial dimension is {actual}, expected {expected}"
            ),
            Self::VelocityDimensionMismatch { expected, actual } => write!(
                formatter,
                "ordinary velocity polynomial dimension is {actual}, expected {expected}"
            ),
            Self::PositionPolynomial(source) => {
                write!(formatter, "ordinary position polynomial failure: {source}")
            }
            Self::VelocityPolynomial(source) => {
                write!(formatter, "ordinary velocity polynomial failure: {source}")
            }
            Self::OrdinaryField(source) => {
                write!(formatter, "ordinary defect field failure: {source}")
            }
            Self::Numeric(source) => write!(formatter, "ordinary defect numeric failure: {source}"),
        }
    }
}

impl std::error::Error for OrdinaryPolynomialDefectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PositionPolynomial(source) => Some(source),
            Self::VelocityPolynomial(source) => Some(source),
            Self::OrdinaryField(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::PositionDimensionMismatch { .. } | Self::VelocityDimensionMismatch { .. } => None,
        }
    }
}

impl From<OrdinaryFieldError> for OrdinaryPolynomialDefectError {
    fn from(source: OrdinaryFieldError) -> Self {
        Self::OrdinaryField(source)
    }
}

impl From<NumericError> for OrdinaryPolynomialDefectError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Enclosure of the twelve ordinary ODE residuals and their exact endpoint
/// infinity bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryPolynomialDefectEnclosure {
    residuals: [RationalInterval; STATE_DIMENSION],
    maximum_absolute_endpoint_bound: BigRational,
}

impl OrdinaryPolynomialDefectEnclosure {
    /// Residual order is `q' - v` followed by `v' - a(q)`.
    pub fn residuals(&self) -> &[RationalInterval; STATE_DIMENSION] {
        &self.residuals
    }

    /// Exact maximum of the absolute values of all residual endpoints.
    pub fn maximum_absolute_endpoint_bound(&self) -> &BigRational {
        &self.maximum_absolute_endpoint_bound
    }
}

/// Evaluate the ordinary planar three-body defect of one polynomial arc.
///
/// Both polynomials must have the raw ordinary configuration dimension six.
/// The state passed to the Newton kernel is exactly `q[0..6], v[0..6]`.
pub fn evaluate_planar_three_body_ordinary_polynomial_defect(
    position: &ExactRationalPolynomial,
    velocity: &ExactRationalPolynomial,
    parameter: &RationalInterval,
    masses: &[BigRational; BODY_COUNT],
    sqrt_precision_bits: usize,
) -> Result<OrdinaryPolynomialDefectEnclosure, OrdinaryPolynomialDefectError> {
    validate_polynomial_dimensions(position, velocity)?;
    preflight_planar_three_body_ordinary_field(masses, sqrt_precision_bits)?;

    let position_values = position
        .evaluate(parameter)
        .map_err(OrdinaryPolynomialDefectError::PositionPolynomial)?;
    let velocity_values = velocity
        .evaluate(parameter)
        .map_err(OrdinaryPolynomialDefectError::VelocityPolynomial)?;
    let position_derivatives = position
        .evaluate_derivative(parameter)
        .map_err(OrdinaryPolynomialDefectError::PositionPolynomial)?;
    let velocity_derivatives = velocity
        .evaluate_derivative(parameter)
        .map_err(OrdinaryPolynomialDefectError::VelocityPolynomial)?;

    let state: [RationalInterval; STATE_DIMENSION] = std::array::from_fn(|index| {
        if index < CONFIGURATION_DIMENSION {
            position_values[index].clone()
        } else {
            velocity_values[index - CONFIGURATION_DIMENSION].clone()
        }
    });
    let field = evaluate_planar_three_body_ordinary_field(&state, masses, sqrt_precision_bits)?;

    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut residuals: [RationalInterval; STATE_DIMENSION] = std::array::from_fn(|_| zero.clone());
    for component in 0..CONFIGURATION_DIMENSION {
        residuals[component] =
            position_derivatives[component].subtract(&velocity_values[component])?;
        residuals[CONFIGURATION_DIMENSION + component] = velocity_derivatives[component]
            .subtract(&field.rhs()[CONFIGURATION_DIMENSION + component])?;
    }
    let maximum_absolute_endpoint_bound = maximum_absolute_endpoint(&residuals);

    Ok(OrdinaryPolynomialDefectEnclosure {
        residuals,
        maximum_absolute_endpoint_bound,
    })
}

fn validate_polynomial_dimensions(
    position: &ExactRationalPolynomial,
    velocity: &ExactRationalPolynomial,
) -> Result<(), OrdinaryPolynomialDefectError> {
    if position.dimension() != CONFIGURATION_DIMENSION {
        return Err(OrdinaryPolynomialDefectError::PositionDimensionMismatch {
            expected: CONFIGURATION_DIMENSION,
            actual: position.dimension(),
        });
    }
    if velocity.dimension() != CONFIGURATION_DIMENSION {
        return Err(OrdinaryPolynomialDefectError::VelocityDimensionMismatch {
            expected: CONFIGURATION_DIMENSION,
            actual: velocity.dimension(),
        });
    }
    Ok(())
}

fn maximum_absolute_endpoint(residuals: &[RationalInterval; STATE_DIMENSION]) -> BigRational {
    let mut maximum = BigRational::zero();
    for residual in residuals {
        for endpoint in [residual.lower(), residual.upper()] {
            let magnitude = if endpoint.numer().sign() == Sign::Minus {
                -endpoint.clone()
            } else {
                endpoint.clone()
            };
            if magnitude > maximum {
                maximum = magnitude;
            }
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::One;

    use super::*;
    use crate::{HARD_MAX_RATIONAL_COMPONENT_BITS, HARD_MAX_SQRT_PRECISION_BITS};

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn point(value: BigRational) -> RationalInterval {
        RationalInterval::try_point(value).unwrap()
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

    fn polynomial(rows: &[[(i64, i64); CONFIGURATION_DIMENSION]]) -> ExactRationalPolynomial {
        ExactRationalPolynomial::from_degree_major_coefficients(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|(numerator, denominator)| rational(*numerator, *denominator))
                        .collect()
                })
                .collect(),
        )
        .unwrap()
    }

    fn unit_masses() -> [BigRational; BODY_COUNT] {
        [rational(1, 1), rational(1, 1), rational(1, 1)]
    }

    fn collinear_positions() -> ExactRationalPolynomial {
        polynomial(&[[(-1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1)]])
    }

    fn constant_velocity() -> ExactRationalPolynomial {
        polynomial(&[[(1, 1), (2, 1), (-1, 1), (3, 1), (4, 1), (-2, 1)]])
    }

    fn zero_velocity() -> ExactRationalPolynomial {
        polynomial(&[[(0, 1); CONFIGURATION_DIMENSION]])
    }

    fn evaluate(
        position: &ExactRationalPolynomial,
        velocity: &ExactRationalPolynomial,
        parameter: &RationalInterval,
    ) -> Result<OrdinaryPolynomialDefectEnclosure, OrdinaryPolynomialDefectError> {
        evaluate_planar_three_body_ordinary_polynomial_defect(
            position,
            velocity,
            parameter,
            &unit_masses(),
            32,
        )
    }

    fn exact_component_value(
        polynomial: &ExactRationalPolynomial,
        component: usize,
        parameter: &BigRational,
    ) -> BigRational {
        let rows = polynomial.coefficients_by_degree();
        let mut accumulator = rows.last().unwrap()[component].clone();
        for row in rows[..rows.len() - 1].iter().rev() {
            accumulator = accumulator * parameter + &row[component];
        }
        accumulator
    }

    fn exact_component_derivative(
        polynomial: &ExactRationalPolynomial,
        component: usize,
        parameter: &BigRational,
    ) -> BigRational {
        if polynomial.degree() == 0 {
            return BigRational::zero();
        }
        let rows = polynomial.coefficients_by_degree();
        let mut accumulator =
            &rows[polynomial.degree()][component] * BigInt::from(polynomial.degree());
        for degree in (1..polynomial.degree()).rev() {
            accumulator = accumulator * parameter + &rows[degree][component] * BigInt::from(degree);
        }
        accumulator
    }

    #[test]
    fn collinear_point_defect_matches_hand_derived_values_and_bound() {
        let result = evaluate(
            &collinear_positions(),
            &constant_velocity(),
            &point(rational(0, 1)),
        )
        .unwrap();
        let expected = [
            rational(-1, 1),
            rational(-2, 1),
            rational(1, 1),
            rational(-3, 1),
            rational(-4, 1),
            rational(2, 1),
            rational(-5, 4),
            rational(0, 1),
            rational(0, 1),
            rational(0, 1),
            rational(5, 4),
            rational(0, 1),
        ];
        for (residual, expected_value) in result.residuals().iter().zip(expected) {
            assert_eq!(residual, &point(expected_value));
        }
        assert_eq!(result.maximum_absolute_endpoint_bound(), &rational(4, 1));
    }

    #[test]
    fn position_derivative_minus_velocity_block_can_vanish_exactly() {
        let position = polynomial(&[
            [(-1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1)],
            [(1, 1), (2, 1), (-1, 1), (3, 1), (4, 1), (-2, 1)],
        ]);
        let result = evaluate(&position, &constant_velocity(), &point(rational(0, 1))).unwrap();
        for residual in &result.residuals()[..CONFIGURATION_DIMENSION] {
            assert_eq!(residual, &point(rational(0, 1)));
        }
    }

    #[test]
    fn acceleration_defect_can_uniquely_determine_the_endpoint_bound() {
        let result = evaluate(
            &collinear_positions(),
            &zero_velocity(),
            &point(rational(0, 1)),
        )
        .unwrap();
        let expected_acceleration_defect = [
            rational(-5, 4),
            rational(0, 1),
            rational(0, 1),
            rational(0, 1),
            rational(5, 4),
            rational(0, 1),
        ];
        for residual in &result.residuals()[..CONFIGURATION_DIMENSION] {
            assert_eq!(residual, &point(rational(0, 1)));
        }
        for (residual, expected) in result.residuals()[CONFIGURATION_DIMENSION..]
            .iter()
            .zip(expected_acceleration_defect)
        {
            assert_eq!(residual, &point(expected));
        }
        assert_eq!(result.maximum_absolute_endpoint_bound(), &rational(5, 4));
    }

    #[test]
    fn unequal_masses_produce_asymmetric_acceleration_defect_and_bound() {
        let result = evaluate_planar_three_body_ordinary_polynomial_defect(
            &collinear_positions(),
            &zero_velocity(),
            &point(rational(0, 1)),
            &[rational(1, 1), rational(2, 1), rational(3, 1)],
            32,
        )
        .unwrap();
        let expected_acceleration_defect = [
            rational(-11, 4),
            rational(0, 1),
            rational(-2, 1),
            rational(0, 1),
            rational(9, 4),
            rational(0, 1),
        ];
        for residual in &result.residuals()[..CONFIGURATION_DIMENSION] {
            assert_eq!(residual, &point(rational(0, 1)));
        }
        for (residual, expected) in result.residuals()[CONFIGURATION_DIMENSION..]
            .iter()
            .zip(expected_acceleration_defect)
        {
            assert_eq!(residual, &point(expected));
        }
        assert_eq!(result.maximum_absolute_endpoint_bound(), &rational(11, 4));
    }

    #[test]
    fn interval_defect_contains_independently_evaluated_exact_point_defects() {
        let position = polynomial(&[
            [(-2, 1), (0, 1), (0, 1), (0, 1), (2, 1), (0, 1)],
            [(1, 1), (0, 1), (0, 1), (0, 1), (-1, 1), (0, 1)],
        ]);
        let velocity = polynomial(&[
            [(1, 2), (1, 3), (-1, 2), (-1, 3), (1, 4), (-1, 4)],
            [(1, 3), (0, 1), (-1, 4), (0, 1), (1, 5), (0, 1)],
            [(1, 7), (0, 1), (1, 9), (0, 1), (-1, 11), (0, 1)],
        ]);
        let enclosure = evaluate(&position, &velocity, &interval(-1, 4, 1, 4)).unwrap();

        for sample in [rational(-1, 4), rational(0, 1), rational(1, 4)] {
            let q: [BigRational; CONFIGURATION_DIMENSION] = std::array::from_fn(|component| {
                exact_component_value(&position, component, &sample)
            });
            let v: [BigRational; CONFIGURATION_DIMENSION] = std::array::from_fn(|component| {
                exact_component_value(&velocity, component, &sample)
            });
            let q_prime: [BigRational; CONFIGURATION_DIMENSION] =
                std::array::from_fn(|component| {
                    exact_component_derivative(&position, component, &sample)
                });
            let v_prime: [BigRational; CONFIGURATION_DIMENSION] =
                std::array::from_fn(|component| {
                    exact_component_derivative(&velocity, component, &sample)
                });
            let state: [RationalInterval; STATE_DIMENSION] = std::array::from_fn(|index| {
                if index < CONFIGURATION_DIMENSION {
                    point(q[index].clone())
                } else {
                    point(v[index - CONFIGURATION_DIMENSION].clone())
                }
            });
            let field =
                evaluate_planar_three_body_ordinary_field(&state, &unit_masses(), 32).unwrap();

            for component in 0..CONFIGURATION_DIMENSION {
                let q_defect = &q_prime[component] - &v[component];
                assert!(enclosure.residuals()[component]
                    .contains(&q_defect)
                    .unwrap());
                let acceleration = &field.rhs()[CONFIGURATION_DIMENSION + component];
                assert!(acceleration.is_point());
                let v_defect = &v_prime[component] - acceleration.lower();
                assert!(enclosure.residuals()[CONFIGURATION_DIMENSION + component]
                    .contains(&v_defect)
                    .unwrap());
            }
        }

        let independently_recomputed_bound = enclosure
            .residuals()
            .iter()
            .flat_map(|residual| [residual.lower(), residual.upper()])
            .map(|endpoint| {
                if endpoint.numer().sign() == Sign::Minus {
                    -endpoint.clone()
                } else {
                    endpoint.clone()
                }
            })
            .max()
            .unwrap();
        assert_eq!(
            enclosure.maximum_absolute_endpoint_bound(),
            &independently_recomputed_bound
        );
    }

    #[test]
    fn dimensions_masses_collisions_and_precision_fail_closed_without_panicking() {
        let five_dimensional = ExactRationalPolynomial::from_degree_major_coefficients(vec![vec![
            rational(0, 1);
            CONFIGURATION_DIMENSION - 1
        ]])
        .unwrap();
        let position = collinear_positions();
        let velocity = constant_velocity();
        let parameter = point(rational(0, 1));
        let malformed_masses = [
            BigRational::new_raw(BigInt::one(), BigInt::zero()),
            rational(1, 1),
            rational(1, 1),
        ];

        assert_eq!(
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &five_dimensional,
                &velocity,
                &parameter,
                &malformed_masses,
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(OrdinaryPolynomialDefectError::PositionDimensionMismatch {
                expected: CONFIGURATION_DIMENSION,
                actual: CONFIGURATION_DIMENSION - 1,
            })
        );
        assert_eq!(
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &position,
                &five_dimensional,
                &parameter,
                &unit_masses(),
                32,
            ),
            Err(OrdinaryPolynomialDefectError::VelocityDimensionMismatch {
                expected: CONFIGURATION_DIMENSION,
                actual: CONFIGURATION_DIMENSION - 1,
            })
        );

        // A quadratic at this admitted point would exceed the exact-rational
        // component cap during Horner evaluation.  Field preflight must reject
        // masses first, then precision when masses are valid, without touching
        // that polynomial arithmetic.
        let explosive_position = polynomial(&[
            [(-1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1)],
            [(0, 1); CONFIGURATION_DIMENSION],
            [(1, 1); CONFIGURATION_DIMENSION],
        ]);
        let huge_parameter = point(BigRational::from_integer(
            BigInt::one() << (HARD_MAX_RATIONAL_COMPONENT_BITS as usize - 1),
        ));
        assert_eq!(
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &explosive_position,
                &velocity,
                &huge_parameter,
                &malformed_masses,
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::Numeric(NumericError::InvalidRationalDenominator)
            ))
        );
        assert_eq!(
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &explosive_position,
                &velocity,
                &huge_parameter,
                &unit_masses(),
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::Numeric(NumericError::SquareRootPrecisionBitLimitExceeded {
                    precision_bits: HARD_MAX_SQRT_PRECISION_BITS + 1,
                    limit: HARD_MAX_SQRT_PRECISION_BITS,
                })
            ))
        );

        let invalid_mass_result = std::panic::catch_unwind(|| {
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &position,
                &velocity,
                &parameter,
                &malformed_masses,
                32,
            )
        });
        assert_eq!(
            invalid_mass_result.unwrap(),
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::Numeric(NumericError::InvalidRationalDenominator)
            ))
        );

        let zero_mass_result = evaluate_planar_three_body_ordinary_polynomial_defect(
            &position,
            &velocity,
            &parameter,
            &[rational(1, 1), rational(0, 1), rational(1, 1)],
            32,
        );
        assert_eq!(
            zero_mass_result,
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::NonPositiveMass { index: 1 }
            ))
        );

        let collision = polynomial(&[[(0, 1); CONFIGURATION_DIMENSION]]);
        assert_eq!(
            evaluate(&collision, &velocity, &parameter),
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::PairDistanceNotSeparated { pair: [0, 1] }
            ))
        );

        assert_eq!(
            evaluate_planar_three_body_ordinary_polynomial_defect(
                &position,
                &velocity,
                &parameter,
                &unit_masses(),
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(OrdinaryPolynomialDefectError::OrdinaryField(
                OrdinaryFieldError::Numeric(NumericError::SquareRootPrecisionBitLimitExceeded {
                    precision_bits: HARD_MAX_SQRT_PRECISION_BITS + 1,
                    limit: HARD_MAX_SQRT_PRECISION_BITS,
                })
            ))
        );
    }
}
