//! Independent exact-rational interval-dual planar Levi-Civita field.
//!
//! This is an arithmetic kernel only. Evaluating it does not certify an LC
//! chart, tube, entry, constraint, gauge, projection, exit, or chain step.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError, PlanarLcMassProfile},
    ExactBinary64, IntervalDual, NumericError, PlanarLcChartInput, PlanarLcIntervalState,
    RationalInterval, HARD_MAX_SQRT_PRECISION_BITS, LC_H, LC_UX, LC_UY, LC_VX, LC_VY, LC_WX, LC_WY,
    LC_YX, LC_YY, LC_ZX, LC_ZY, PLANAR_LC_LIFTED_STATE_DIMENSION,
};

pub const PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcThirdDenominator {
    First,
    Second,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcFieldError {
    MassDecode {
        index: usize,
        source: NumericError,
    },
    MassProfile(MassProfileError),
    ThirdBodyCollision {
        denominator: PlanarLcThirdDenominator,
    },
    SqrtPrecisionExceeded {
        requested: usize,
        limit: usize,
    },
    InternalDimension {
        expected: usize,
        actual: usize,
    },
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC field arithmetic failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MassDecode { source, .. } => Some(source),
            Self::MassProfile(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::ThirdBodyCollision { .. }
            | Self::SqrtPrecisionExceeded { .. }
            | Self::InternalDimension { .. } => None,
        }
    }
}

impl From<NumericError> for PlanarLcFieldError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

impl From<MassProfileError> for PlanarLcFieldError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcFieldEnclosure {
    rhs: [RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
    jacobian:
        [[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]; PLANAR_LC_LIFTED_STATE_DIMENSION],
    lipschitz_infinity_row_sum_upper: BigRational,
    denominator_squares: [RationalInterval; 2],
    mass_profile: PlanarLcMassProfile,
}

impl PlanarLcFieldEnclosure {
    pub fn rhs(&self) -> &[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION] {
        &self.rhs
    }

    pub fn jacobian(
        &self,
    ) -> &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]; PLANAR_LC_LIFTED_STATE_DIMENSION]
    {
        &self.jacobian
    }

    pub fn lipschitz_infinity_row_sum_upper(&self) -> &BigRational {
        &self.lipschitz_infinity_row_sum_upper
    }

    pub fn denominator_squares(&self) -> &[RationalInterval; 2] {
        &self.denominator_squares
    }

    pub fn mass_profile(&self) -> &PlanarLcMassProfile {
        &self.mass_profile
    }
}

pub fn evaluate_planar_lc_field(
    chart: &PlanarLcChartInput,
    state: &PlanarLcIntervalState,
    sqrt_precision_bits: usize,
) -> Result<PlanarLcFieldEnclosure, PlanarLcFieldError> {
    if sqrt_precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(PlanarLcFieldError::SqrtPrecisionExceeded {
            requested: sqrt_precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        });
    }
    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;
    let dimension = PLANAR_LC_LIFTED_STATE_DIMENSION;
    let mut duals = Vec::with_capacity(dimension);
    for (index, component) in state.components().iter().enumerate() {
        duals.push(IntervalDual::variable(component.clone(), dimension, index)?);
    }
    let state_dual: [IntervalDual; PLANAR_LC_LIFTED_STATE_DIMENSION] =
        duals.try_into().map_err(|values: Vec<IntervalDual>| {
            PlanarLcFieldError::InternalDimension {
                expected: dimension,
                actual: values.len(),
            }
        })?;
    let x = &state_dual[LC_ZX];
    let y = &state_dual[LC_ZY];
    let wx = &state_dual[LC_WX];
    let wy = &state_dual[LC_WY];
    let h = &state_dual[LC_H];
    let ux = &state_dual[LC_UX];
    let uy = &state_dual[LC_UY];
    let yx = &state_dual[LC_YX];
    let yy = &state_dual[LC_YY];
    let vx = &state_dual[LC_VX];
    let vy = &state_dual[LC_VY];

    let rho = x.square()?.add(&y.square()?)?;
    let two = BigRational::from_integer(BigInt::from(2));
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    let quarter = BigRational::new(BigInt::from(1), BigInt::from(4));
    let minus_one = BigRational::from_integer(BigInt::from(-1));
    let qx = x.square()?.subtract(&y.square()?)?;
    let qy = x.multiply(y)?.scale(&two)?;
    let d1x = yx.add(&qx.scale(mass_profile.alpha().exact())?)?;
    let d1y = yy.add(&qy.scale(mass_profile.alpha().exact())?)?;
    let d2x = yx.subtract(&qx.scale(mass_profile.beta().exact())?)?;
    let d2y = yy.subtract(&qy.scale(mass_profile.beta().exact())?)?;
    let (f1, norm1) = inverse_cube_vector(
        [&d1x, &d1y],
        sqrt_precision_bits,
        PlanarLcThirdDenominator::First,
    )?;
    let (f2, norm2) = inverse_cube_vector(
        [&d2x, &d2y],
        sqrt_precision_bits,
        PlanarLcThirdDenominator::Second,
    )?;
    let third_mass = masses[mass_profile.third_index()].rational();
    let px = f2[0].subtract(&f1[0])?.scale(third_mass)?;
    let py = f2[1].subtract(&f1[1])?.scale(third_mass)?;
    let ltp_x = x.multiply(&px)?.add(&y.multiply(&py)?)?.scale(&two)?;
    let ltp_y = y
        .multiply(&px)?
        .scale(&minus_one)?
        .add(&x.multiply(&py)?)?
        .scale(&two)?;
    let wdot_x = h
        .multiply(x)?
        .scale(&half)?
        .add(&rho.multiply(&ltp_x)?.scale(&quarter)?)?;
    let wdot_y = h
        .multiply(y)?
        .scale(&half)?
        .add(&rho.multiply(&ltp_y)?.scale(&quarter)?)?;
    let qprime_x = x.multiply(wx)?.subtract(&y.multiply(wy)?)?.scale(&two)?;
    let qprime_y = y.multiply(wx)?.add(&x.multiply(wy)?)?.scale(&two)?;
    let hdot = qprime_x.multiply(&px)?.add(&qprime_y.multiply(&py)?)?;
    let rdd_x = f1[0]
        .scale(mass_profile.center_first().exact())?
        .add(&f2[0].scale(mass_profile.center_second().exact())?)?;
    let rdd_y = f1[1]
        .scale(mass_profile.center_first().exact())?
        .add(&f2[1].scale(mass_profile.center_second().exact())?)?;
    let ydd_x = f1[0]
        .scale(mass_profile.offset_first().exact())?
        .scale(&minus_one)?
        .subtract(&f2[0].scale(mass_profile.offset_second().exact())?)?;
    let ydd_y = f1[1]
        .scale(mass_profile.offset_first().exact())?
        .scale(&minus_one)?
        .subtract(&f2[1].scale(mass_profile.offset_second().exact())?)?;
    let rhs_dual = vec![
        wx.clone(),
        wy.clone(),
        wdot_x,
        wdot_y,
        hdot,
        rho.multiply(ux)?,
        rho.multiply(uy)?,
        rho.multiply(&rdd_x)?,
        rho.multiply(&rdd_y)?,
        rho.multiply(vx)?,
        rho.multiply(vy)?,
        rho.multiply(&ydd_x)?,
        rho.multiply(&ydd_y)?,
        rho,
    ];
    if rhs_dual.len() != dimension {
        return Err(PlanarLcFieldError::InternalDimension {
            expected: dimension,
            actual: rhs_dual.len(),
        });
    }
    let rhs_vec: Vec<RationalInterval> = rhs_dual.iter().map(|item| item.value().clone()).collect();
    let rhs = rhs_vec
        .try_into()
        .map_err(
            |values: Vec<RationalInterval>| PlanarLcFieldError::InternalDimension {
                expected: dimension,
                actual: values.len(),
            },
        )?;
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut jacobian = std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    for (row, item) in rhs_dual.iter().enumerate() {
        for column in 0..dimension {
            jacobian[row][column] = item.derivative(column)?.clone();
        }
    }
    let lipschitz_infinity_row_sum_upper = maximum_row_sum(&jacobian);
    Ok(PlanarLcFieldEnclosure {
        rhs,
        jacobian,
        lipschitz_infinity_row_sum_upper,
        denominator_squares: [norm1, norm2],
        mass_profile,
    })
}

pub fn evaluate_planar_lc_field_default(
    chart: &PlanarLcChartInput,
    state: &PlanarLcIntervalState,
) -> Result<PlanarLcFieldEnclosure, PlanarLcFieldError> {
    evaluate_planar_lc_field(chart, state, PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS)
}

fn reconstruct_masses(
    chart: &PlanarLcChartInput,
) -> Result<[ExactBinary64; 3], PlanarLcFieldError> {
    let mut values = Vec::with_capacity(3);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        values.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| PlanarLcFieldError::MassDecode { index, source })?,
        );
    }
    values.try_into().map_err(
        |values: Vec<ExactBinary64>| PlanarLcFieldError::InternalDimension {
            expected: 3,
            actual: values.len(),
        },
    )
}

fn inverse_cube_vector(
    vector: [&IntervalDual; 2],
    precision: usize,
    denominator: PlanarLcThirdDenominator,
) -> Result<([IntervalDual; 2], RationalInterval), PlanarLcFieldError> {
    let norm_square = vector[0].square()?.add(&vector[1].square()?)?;
    if norm_square.value().lower().numer().sign() != Sign::Plus {
        return Err(PlanarLcFieldError::ThirdBodyCollision { denominator });
    }
    let root = norm_square.sqrt_dyadic(precision)?;
    let cube = norm_square.multiply(&root)?;
    let inverse_cube = cube.reciprocal()?;
    Ok((
        [
            vector[0].multiply(&inverse_cube)?,
            vector[1].multiply(&inverse_cube)?,
        ],
        norm_square.value().clone(),
    ))
}

fn maximum_row_sum(
    jacobian: &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION];
         PLANAR_LC_LIFTED_STATE_DIMENSION],
) -> BigRational {
    let mut maximum = BigRational::zero();
    for row in jacobian {
        let mut sum = BigRational::zero();
        for entry in row {
            let lower = entry.lower().abs();
            let upper = entry.upper().abs();
            sum += if lower >= upper { lower } else { upper };
        }
        if sum > maximum {
            maximum = sum;
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checked_real_binary64_from_json, parse_wire_json, planar_lc_chart_input_from_wire,
        raw_schema::{decode_raw_chain, SchemaProfile, SegmentWire},
        PlanarLcStatePolynomial, DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS, LC_RX,
        LC_RY, LC_T,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn r(value: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn chain() -> crate::raw_schema::RawPlanarChainWire {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
    }

    fn unit_chart() -> PlanarLcChartInput {
        let chain = chain();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        let mut wire = segment.lc_chart.clone();
        let one = checked_real_binary64_from_json("1.0", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        wire.masses = vec![one.clone(), one.clone(), one];
        planar_lc_chart_input_from_wire(&wire).unwrap()
    }

    fn separated_state() -> PlanarLcIntervalState {
        let mut values = std::array::from_fn(|_| BigRational::zero());
        values[LC_YX] = r(1);
        values[LC_WX] = r(2);
        values[LC_WY] = r(3);
        PlanarLcIntervalState::from_exact_components(values).unwrap()
    }

    fn nontrivial_hand_state() -> PlanarLcIntervalState {
        let mut values = std::array::from_fn(|_| BigRational::zero());
        values[LC_ZX] = r(1);
        values[LC_WX] = r(1);
        values[LC_H] = r(2);
        values[LC_UX] = r(2);
        values[LC_UY] = r(3);
        values[LC_YX] = r(3);
        values[LC_VX] = r(4);
        values[LC_VY] = r(5);
        PlanarLcIntervalState::from_exact_components(values).unwrap()
    }

    #[test]
    fn separated_perfect_square_point_has_hand_derived_ordered_rhs() {
        let chart = unit_chart();
        let field = evaluate_planar_lc_field_default(&chart, &separated_state()).unwrap();
        assert_eq!(
            field.denominator_squares()[0],
            RationalInterval::try_point(r(1)).unwrap()
        );
        assert_eq!(
            field.denominator_squares()[1],
            RationalInterval::try_point(r(1)).unwrap()
        );
        assert_eq!(
            field.rhs()[LC_ZX],
            RationalInterval::try_point(r(2)).unwrap()
        );
        assert_eq!(
            field.rhs()[LC_ZY],
            RationalInterval::try_point(r(3)).unwrap()
        );
        for index in LC_WX..PLANAR_LC_LIFTED_STATE_DIMENSION {
            assert_eq!(
                field.rhs()[index],
                RationalInterval::try_point(r(0)).unwrap()
            );
        }
    }

    #[test]
    fn nontrivial_rational_point_matches_every_ordered_field_block() {
        let chart = unit_chart();
        let field = evaluate_planar_lc_field_default(&chart, &nontrivial_hand_state()).unwrap();
        let expected = [
            r(1),
            r(0),
            BigRational::new(BigInt::from(1273), BigInt::from(1225)),
            r(0),
            BigRational::new(BigInt::from(192), BigInt::from(1225)),
            r(2),
            r(3),
            BigRational::new(BigInt::from(148), BigInt::from(1225)),
            r(0),
            r(4),
            r(5),
            BigRational::new(BigInt::from(-444), BigInt::from(1225)),
            r(0),
            r(1),
        ];
        for (actual, expected) in field.rhs().iter().zip(expected) {
            assert_eq!(actual, &RationalInterval::try_point(expected).unwrap());
        }
    }

    #[test]
    fn all_canonical_pairs_reconstruct_and_revalidate_mass_profiles() {
        let chain = chain();
        let mut pairs = Vec::new();
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap();
            let polynomial = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
            let state = polynomial
                .evaluate(
                    &RationalInterval::try_point(chart.parameter_interval().lower().clone())
                        .unwrap(),
                )
                .unwrap();
            let field = evaluate_planar_lc_field_default(&chart, &state).unwrap();
            let masses = std::array::from_fn(|index| {
                ExactBinary64::from_bits(chart.mass_binary64_bits()[index]).unwrap()
            });
            field
                .mass_profile()
                .validate_against(&masses, chart.pair())
                .unwrap();
            pairs.push(chart.pair());
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs, vec![[0, 1], [0, 2], [1, 2]]);
    }

    #[test]
    fn third_collision_and_precision_exhaustion_fail_closed() {
        let chart = unit_chart();
        let zero = PlanarLcIntervalState::from_exact_components(std::array::from_fn(|_| {
            BigRational::zero()
        }))
        .unwrap();
        assert!(matches!(
            evaluate_planar_lc_field_default(&chart, &zero),
            Err(PlanarLcFieldError::ThirdBodyCollision { .. })
        ));
        assert_eq!(
            evaluate_planar_lc_field(&chart, &separated_state(), HARD_MAX_SQRT_PRECISION_BITS + 1),
            Err(PlanarLcFieldError::SqrtPrecisionExceeded {
                requested: HARD_MAX_SQRT_PRECISION_BITS + 1,
                limit: HARD_MAX_SQRT_PRECISION_BITS
            })
        );
    }

    #[test]
    fn deck_equivariance_and_unused_center_time_columns_are_exact() {
        let chart = unit_chart();
        let state = nontrivial_hand_state();
        let original = evaluate_planar_lc_field_default(&chart, &state).unwrap();
        let deck_state = state.deck_transform().unwrap();
        let deck_field = evaluate_planar_lc_field_default(&chart, &deck_state).unwrap();
        let rhs_as_state =
            PlanarLcIntervalState::from_interval_components(original.rhs().clone()).unwrap();
        assert_eq!(
            deck_field.rhs(),
            rhs_as_state.deck_transform().unwrap().components()
        );
        let zero = RationalInterval::try_point(BigRational::zero()).unwrap();
        for row in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
            for column in [LC_RX, LC_RY, LC_T] {
                assert_eq!(original.jacobian()[row][column], zero);
            }
        }
    }

    #[test]
    fn centered_difference_is_contained_by_interval_jacobian() {
        let chart = unit_chart();
        let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1024));
        let mut box_components = nontrivial_hand_state().components().clone();
        box_components[LC_YX] = RationalInterval::new(&r(3) - &epsilon, &r(3) + &epsilon).unwrap();
        let box_state = PlanarLcIntervalState::from_interval_components(box_components).unwrap();
        let box_field = evaluate_planar_lc_field_default(&chart, &box_state).unwrap();
        let mut plus = nontrivial_hand_state().components().clone();
        plus[LC_YX] = RationalInterval::try_point(&r(3) + &epsilon).unwrap();
        let mut minus = nontrivial_hand_state().components().clone();
        minus[LC_YX] = RationalInterval::try_point(&r(3) - &epsilon).unwrap();
        let plus_field = evaluate_planar_lc_field_default(
            &chart,
            &PlanarLcIntervalState::from_interval_components(plus).unwrap(),
        )
        .unwrap();
        let minus_field = evaluate_planar_lc_field_default(
            &chart,
            &PlanarLcIntervalState::from_interval_components(minus).unwrap(),
        )
        .unwrap();
        let two_epsilon = &epsilon * BigInt::from(2);
        let secant = plus_field.rhs()[LC_UX]
            .subtract(&minus_field.rhs()[LC_UX])
            .unwrap()
            .scale(&two_epsilon.recip())
            .unwrap();
        assert!(!secant.contains(&BigRational::zero()).unwrap());
        assert!(box_field.jacobian()[LC_UX][LC_YX].contains_interval(&secant));
    }

    #[test]
    fn reported_lipschitz_bound_is_exact_maximum_row_sum() {
        let field =
            evaluate_planar_lc_field_default(&unit_chart(), &nontrivial_hand_state()).unwrap();
        let mut independently_computed_maximum = BigRational::zero();
        for row in field.jacobian() {
            let row_sum = row.iter().fold(BigRational::zero(), |sum, entry| {
                let lower = entry.lower().abs();
                let upper = entry.upper().abs();
                sum + if lower >= upper { lower } else { upper }
            });
            if row_sum > independently_computed_maximum {
                independently_computed_maximum = row_sum;
            }
        }
        assert!(independently_computed_maximum > BigRational::zero());
        assert_eq!(
            field.lipschitz_infinity_row_sum_upper(),
            &independently_computed_maximum
        );
    }
}
