//! Bounded interval evaluation of the planar three-body Newton field.
//!
//! This module evaluates one ordinary-coordinate vector field and its
//! first-order interval Jacobian.  It does not replay a chart, tube, or
//! continuation certificate.

use core::fmt;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    rational_input::validate_public_rational, IntervalDual, NumericError, RationalInterval,
    HARD_MAX_SQRT_PRECISION_BITS,
};

const STATE_DIMENSION: usize = 12;
const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const CANONICAL_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Fail-closed errors from ordinary Newton-field evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryFieldError {
    Numeric(NumericError),
    NonPositiveMass { index: usize },
    PairDistanceNotSeparated { pair: [usize; 2] },
}

impl fmt::Display for OrdinaryFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => write!(formatter, "ordinary-field numeric failure: {source}"),
            Self::NonPositiveMass { index } => {
                write!(formatter, "mass[{index}] is not strictly positive")
            }
            Self::PairDistanceNotSeparated { pair } => write!(
                formatter,
                "ordinary position boxes do not separate pair ({},{})",
                pair[0], pair[1]
            ),
        }
    }
}

impl std::error::Error for OrdinaryFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::NonPositiveMass { .. } | Self::PairDistanceNotSeparated { .. } => None,
        }
    }
}

impl From<NumericError> for OrdinaryFieldError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Interval enclosure of the 12-component RHS, its 12x12 Jacobian, and the
/// induced-infinity row-sum upper bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryFieldEnclosure {
    rhs: [RationalInterval; STATE_DIMENSION],
    jacobian: [[RationalInterval; STATE_DIMENSION]; STATE_DIMENSION],
    lipschitz_infinity_upper: BigRational,
}

impl OrdinaryFieldEnclosure {
    pub fn rhs(&self) -> &[RationalInterval; STATE_DIMENSION] {
        &self.rhs
    }

    pub fn jacobian(&self) -> &[[RationalInterval; STATE_DIMENSION]; STATE_DIMENSION] {
        &self.jacobian
    }

    pub fn lipschitz_infinity_upper(&self) -> &BigRational {
        &self.lipschitz_infinity_upper
    }
}

/// Evaluate the planar Newton field in raw ordinary state order:
/// `q(3x2), v(3x2)`.
///
/// The acceleration is
/// `a_i = sum_{j != i} m_j (q_j-q_i) / ||q_j-q_i||^3`.
pub fn evaluate_planar_three_body_ordinary_field(
    state: &[RationalInterval; STATE_DIMENSION],
    masses: &[BigRational; BODY_COUNT],
    sqrt_precision_bits: usize,
) -> Result<OrdinaryFieldEnclosure, OrdinaryFieldError> {
    validate_masses(masses)?;
    if sqrt_precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(NumericError::SquareRootPrecisionBitLimitExceeded {
            precision_bits: sqrt_precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        }
        .into());
    }

    let variables = seed_state_variables(state)?;
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let zero_dual = IntervalDual::constant(zero, STATE_DIMENSION)?;
    let mut accelerations = [
        zero_dual.clone(),
        zero_dual.clone(),
        zero_dual.clone(),
        zero_dual.clone(),
        zero_dual.clone(),
        zero_dual,
    ];

    for (first, second) in CANONICAL_PAIRS {
        let displacement = [
            variables[position_index(second, 0)].subtract(&variables[position_index(first, 0)])?,
            variables[position_index(second, 1)].subtract(&variables[position_index(first, 1)])?,
        ];
        let distance_squared = displacement[0].square()?.add(&displacement[1].square()?)?;
        if distance_squared.value().lower().numer().sign() != Sign::Plus {
            return Err(OrdinaryFieldError::PairDistanceNotSeparated {
                pair: [first, second],
            });
        }

        let distance = distance_squared.sqrt_dyadic(sqrt_precision_bits)?;
        let inverse_distance_cubed = distance_squared.multiply(&distance)?.reciprocal()?;

        for (axis, displacement_component) in displacement.iter().enumerate() {
            let geometric_term = displacement_component.multiply(&inverse_distance_cubed)?;
            let first_term = geometric_term.scale(&masses[second])?;
            let second_term = geometric_term.scale(&masses[first])?;
            let first_index = position_index(first, axis);
            let second_index = position_index(second, axis);
            accelerations[first_index] = accelerations[first_index].add(&first_term)?;
            accelerations[second_index] = accelerations[second_index].subtract(&second_term)?;
        }
    }

    let dual_rhs = [
        variables[6].clone(),
        variables[7].clone(),
        variables[8].clone(),
        variables[9].clone(),
        variables[10].clone(),
        variables[11].clone(),
        accelerations[0].clone(),
        accelerations[1].clone(),
        accelerations[2].clone(),
        accelerations[3].clone(),
        accelerations[4].clone(),
        accelerations[5].clone(),
    ];
    let rhs = std::array::from_fn(|row| dual_rhs[row].value().clone());
    let jacobian = std::array::from_fn(|row| {
        std::array::from_fn(|column| dual_rhs[row].derivatives()[column].clone())
    });
    let lipschitz_infinity_upper = infinity_row_sum_upper(&jacobian)?;

    Ok(OrdinaryFieldEnclosure {
        rhs,
        jacobian,
        lipschitz_infinity_upper,
    })
}

fn validate_masses(masses: &[BigRational; BODY_COUNT]) -> Result<(), OrdinaryFieldError> {
    for (index, mass) in masses.iter().enumerate() {
        validate_public_rational(mass)?;
        if mass.numer().sign() != Sign::Plus {
            return Err(OrdinaryFieldError::NonPositiveMass { index });
        }
    }
    Ok(())
}

fn seed_state_variables(
    state: &[RationalInterval; STATE_DIMENSION],
) -> Result<[IntervalDual; STATE_DIMENSION], NumericError> {
    Ok([
        IntervalDual::variable(state[0].clone(), STATE_DIMENSION, 0)?,
        IntervalDual::variable(state[1].clone(), STATE_DIMENSION, 1)?,
        IntervalDual::variable(state[2].clone(), STATE_DIMENSION, 2)?,
        IntervalDual::variable(state[3].clone(), STATE_DIMENSION, 3)?,
        IntervalDual::variable(state[4].clone(), STATE_DIMENSION, 4)?,
        IntervalDual::variable(state[5].clone(), STATE_DIMENSION, 5)?,
        IntervalDual::variable(state[6].clone(), STATE_DIMENSION, 6)?,
        IntervalDual::variable(state[7].clone(), STATE_DIMENSION, 7)?,
        IntervalDual::variable(state[8].clone(), STATE_DIMENSION, 8)?,
        IntervalDual::variable(state[9].clone(), STATE_DIMENSION, 9)?,
        IntervalDual::variable(state[10].clone(), STATE_DIMENSION, 10)?,
        IntervalDual::variable(state[11].clone(), STATE_DIMENSION, 11)?,
    ])
}

const fn position_index(body: usize, axis: usize) -> usize {
    body * PLANE_DIMENSION + axis
}

fn infinity_row_sum_upper(
    jacobian: &[[RationalInterval; STATE_DIMENSION]; STATE_DIMENSION],
) -> Result<BigRational, NumericError> {
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut maximum = BigRational::zero();
    for row in jacobian {
        let mut row_sum = zero.clone();
        for entry in row {
            let magnitude = interval_absolute_upper(entry);
            row_sum = row_sum.add(&RationalInterval::try_point(magnitude)?)?;
        }
        if row_sum.upper() > &maximum {
            maximum = row_sum.upper().clone();
        }
    }
    Ok(maximum)
}

fn interval_absolute_upper(interval: &RationalInterval) -> BigRational {
    let lower_magnitude = if interval.lower().numer().sign() == Sign::Minus {
        -interval.lower().clone()
    } else {
        interval.lower().clone()
    };
    let upper_magnitude = if interval.upper().numer().sign() == Sign::Minus {
        -interval.upper().clone()
    } else {
        interval.upper().clone()
    };
    if lower_magnitude >= upper_magnitude {
        lower_magnitude
    } else {
        upper_magnitude
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
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

    fn unit_masses() -> [BigRational; BODY_COUNT] {
        [rational(1, 1), rational(1, 1), rational(1, 1)]
    }

    fn collinear_point_state() -> [RationalInterval; STATE_DIMENSION] {
        [
            point(-1, 1),
            point(0, 1),
            point(0, 1),
            point(0, 1),
            point(1, 1),
            point(0, 1),
            point(1, 1),
            point(2, 1),
            point(-1, 1),
            point(3, 1),
            point(4, 1),
            point(-2, 1),
        ]
    }

    fn evaluate(state: &[RationalInterval; STATE_DIMENSION]) -> OrdinaryFieldEnclosure {
        evaluate_planar_three_body_ordinary_field(state, &unit_masses(), 32).unwrap()
    }

    #[test]
    fn collinear_point_rhs_matches_hand_derived_newton_acceleration() {
        let result = evaluate(&collinear_point_state());
        let expected = [
            point(1, 1),
            point(2, 1),
            point(-1, 1),
            point(3, 1),
            point(4, 1),
            point(-2, 1),
            point(5, 4),
            point(0, 1),
            point(0, 1),
            point(0, 1),
            point(-5, 4),
            point(0, 1),
        ];
        assert_eq!(result.rhs(), &expected);
    }

    #[test]
    fn unequal_masses_use_source_indices_and_preserve_weighted_acceleration_balance() {
        let masses = [rational(1, 1), rational(2, 1), rational(3, 1)];
        let result =
            evaluate_planar_three_body_ordinary_field(&collinear_point_state(), &masses, 32)
                .unwrap();
        assert_eq!(result.rhs()[6], point(11, 4));
        assert_eq!(result.rhs()[8], point(2, 1));
        assert_eq!(result.rhs()[10], point(-9, 4));
        for y_component in [7, 9, 11] {
            assert_eq!(result.rhs()[y_component], point(0, 1));
        }

        let weighted_sum = &masses[0] * result.rhs()[6].lower()
            + &masses[1] * result.rhs()[8].lower()
            + &masses[2] * result.rhs()[10].lower();
        assert_eq!(weighted_sum, BigRational::zero());
    }

    #[test]
    fn position_derivative_rows_are_the_q_prime_equals_v_identity_block() {
        let result = evaluate(&collinear_point_state());
        for row in 0..6 {
            for column in 0..STATE_DIMENSION {
                let expected = if column == row + 6 {
                    point(1, 1)
                } else {
                    point(0, 1)
                };
                assert_eq!(result.jacobian()[row][column], expected);
            }
        }
    }

    #[test]
    fn point_jacobian_matches_analytic_collinear_entries() {
        let result = evaluate(&collinear_point_state());
        let jacobian = result.jacobian();

        assert_eq!(jacobian[6][0], point(9, 4));
        assert_eq!(jacobian[6][2], point(-2, 1));
        assert_eq!(jacobian[6][4], point(-1, 4));
        assert_eq!(jacobian[7][1], point(-9, 8));
        assert_eq!(jacobian[7][3], point(1, 1));
        assert_eq!(jacobian[7][5], point(1, 8));
        for column in 6..STATE_DIMENSION {
            assert_eq!(jacobian[6][column], point(0, 1));
            assert_eq!(jacobian[7][column], point(0, 1));
        }
    }

    #[test]
    fn interval_jacobian_contains_an_exact_centered_finite_difference() {
        let h = rational(1, 1024);
        let base = rational(-1, 1);
        let lower = &base - &h;
        let upper = &base + &h;

        let mut minus_state = collinear_point_state();
        minus_state[0] = RationalInterval::try_point(lower.clone()).unwrap();
        let mut plus_state = collinear_point_state();
        plus_state[0] = RationalInterval::try_point(upper.clone()).unwrap();
        let minus = evaluate(&minus_state);
        let plus = evaluate(&plus_state);
        assert!(minus.rhs()[6].is_point());
        assert!(plus.rhs()[6].is_point());
        let finite_difference =
            (plus.rhs()[6].lower() - minus.rhs()[6].lower()) / (&upper - &lower);

        let mut box_state = collinear_point_state();
        box_state[0] = RationalInterval::new(lower, upper).unwrap();
        let boxed = evaluate(&box_state);
        assert!(boxed.jacobian()[6][0].contains(&finite_difference).unwrap());
        assert!(finite_difference > rational(2, 1));
        assert!(finite_difference < rational(5, 2));
    }

    #[test]
    fn translating_every_position_leaves_field_and_jacobian_unchanged() {
        let state = collinear_point_state();
        let baseline = evaluate(&state);
        let shifts = [rational(7, 3), rational(-5, 4)];
        let mut translated = state.clone();
        for body in 0..BODY_COUNT {
            for (axis, shift) in shifts.iter().enumerate() {
                let index = position_index(body, axis);
                translated[index] = RationalInterval::new(
                    translated[index].lower() + shift,
                    translated[index].upper() + shift,
                )
                .unwrap();
            }
        }
        assert_eq!(evaluate(&translated), baseline);
    }

    #[test]
    fn interval_state_rhs_contains_the_center_point_rhs() {
        let state = [
            interval(-11, 10, -9, 10),
            interval(-1, 20, 1, 20),
            interval(-1, 10, 1, 10),
            interval(-1, 20, 1, 20),
            interval(9, 10, 11, 10),
            interval(-1, 20, 1, 20),
            interval(9, 10, 11, 10),
            interval(19, 10, 21, 10),
            interval(-11, 10, -9, 10),
            interval(29, 10, 31, 10),
            interval(39, 10, 41, 10),
            interval(-21, 10, -19, 10),
        ];
        let boxed = evaluate(&state);
        let center = evaluate(&collinear_point_state());
        for component in 0..STATE_DIMENSION {
            assert!(boxed.rhs()[component].contains_interval(&center.rhs()[component]));
        }
        assert_eq!(boxed.rhs()[0], state[6]);
        assert_eq!(boxed.rhs()[5], state[11]);
    }

    #[test]
    fn lipschitz_bound_equals_independent_exact_row_sum_recomputation() {
        let result = evaluate(&collinear_point_state());
        let mut recomputed = BigRational::zero();
        for row in result.jacobian() {
            let mut sum = BigRational::zero();
            for entry in row {
                sum += interval_absolute_upper(entry);
            }
            if sum > recomputed {
                recomputed = sum;
            }
        }
        assert_eq!(result.lipschitz_infinity_upper(), &recomputed);
        assert!(recomputed > BigRational::zero());
    }

    #[test]
    fn overlapping_position_boxes_are_rejected_with_the_first_pair() {
        let mut state = collinear_point_state();
        state[0] = interval(-1, 1, 0, 1);
        state[2] = interval(0, 1, 1, 1);
        assert_eq!(
            evaluate_planar_three_body_ordinary_field(&state, &unit_masses(), 32),
            Err(OrdinaryFieldError::PairDistanceNotSeparated { pair: [0, 1] })
        );
    }

    #[test]
    fn invalid_masses_fail_preflight_without_panicking() {
        let state = collinear_point_state();
        let invalid_cases = [
            (
                [rational(1, 1), rational(0, 1), rational(1, 1)],
                OrdinaryFieldError::NonPositiveMass { index: 1 },
            ),
            (
                [rational(1, 1), rational(1, 1), rational(-1, 1)],
                OrdinaryFieldError::NonPositiveMass { index: 2 },
            ),
            (
                [
                    BigRational::new_raw(BigInt::one(), BigInt::zero()),
                    rational(1, 1),
                    rational(1, 1),
                ],
                OrdinaryFieldError::Numeric(NumericError::InvalidRationalDenominator),
            ),
            (
                [
                    BigRational::new_raw(BigInt::one(), BigInt::from(-2)),
                    rational(1, 1),
                    rational(1, 1),
                ],
                OrdinaryFieldError::Numeric(NumericError::InvalidRationalDenominator),
            ),
            (
                [
                    BigRational::new_raw(BigInt::from(2), BigInt::from(4)),
                    rational(1, 1),
                    rational(1, 1),
                ],
                OrdinaryFieldError::Numeric(NumericError::NonCanonicalRational),
            ),
        ];

        for (masses, expected) in invalid_cases {
            let result = std::panic::catch_unwind(|| {
                evaluate_planar_three_body_ordinary_field(&state, &masses, 32)
            });
            assert_eq!(result.unwrap(), Err(expected));
        }

        let oversized =
            BigRational::from_integer(BigInt::one() << HARD_MAX_RATIONAL_COMPONENT_BITS as usize);
        let masses = [oversized, rational(1, 1), rational(1, 1)];
        let result = std::panic::catch_unwind(|| {
            evaluate_planar_three_body_ordinary_field(&state, &masses, 32)
        });
        assert_eq!(
            result.unwrap(),
            Err(OrdinaryFieldError::Numeric(
                NumericError::RationalComponentBitLimitExceeded {
                    numerator_bits: HARD_MAX_RATIONAL_COMPONENT_BITS + 1,
                    denominator_bits: 1,
                    limit: HARD_MAX_RATIONAL_COMPONENT_BITS,
                }
            ))
        );
    }

    #[test]
    fn oversized_sqrt_precision_is_propagated_before_field_arithmetic() {
        assert_eq!(
            evaluate_planar_three_body_ordinary_field(
                &collinear_point_state(),
                &unit_masses(),
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(OrdinaryFieldError::Numeric(
                NumericError::SquareRootPrecisionBitLimitExceeded {
                    precision_bits: HARD_MAX_SQRT_PRECISION_BITS + 1,
                    limit: HARD_MAX_SQRT_PRECISION_BITS,
                }
            ))
        );
    }
}
