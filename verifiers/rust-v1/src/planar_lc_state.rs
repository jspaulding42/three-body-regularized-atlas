//! Exact arithmetic for the admitted planar Levi-Civita lifted state.
//!
//! This module only assembles and evaluates bounded rational polynomials and
//! derived algebraic expressions. It certifies no chart, tube, entry,
//! constraint, field equation, gauge, lift, projection, exit, or chain step.

use core::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::{
    ExactRationalPolynomial, NumericError, PlanarLcChartInput, PolynomialError, RationalInterval,
};

pub const PLANAR_LC_LIFTED_STATE_DIMENSION: usize = 14;
pub const PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION: usize = 13;
pub const PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES: [&str; PLANAR_LC_LIFTED_STATE_DIMENSION] = [
    "zx", "zy", "wx", "wy", "h", "Rx", "Ry", "Ux", "Uy", "yx", "yy", "Vx", "Vy", "t",
];
pub const PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_COMPONENT_NAMES: [&str;
    PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION] = [
    "zx", "zy", "wx", "wy", "h", "Rx", "Ry", "Ux", "Uy", "yx", "yy", "Vx", "Vy",
];

pub const LC_ZX: usize = 0;
pub const LC_ZY: usize = 1;
pub const LC_WX: usize = 2;
pub const LC_WY: usize = 3;
pub const LC_H: usize = 4;
pub const LC_RX: usize = 5;
pub const LC_RY: usize = 6;
pub const LC_UX: usize = 7;
pub const LC_UY: usize = 8;
pub const LC_YX: usize = 9;
pub const LC_YY: usize = 10;
pub const LC_VX: usize = 11;
pub const LC_VY: usize = 12;
pub const LC_T: usize = 13;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcStateError {
    CoefficientCountMismatch,
    StateDimensionMismatch { expected: usize, actual: usize },
    NegativeInflation,
    NonPointAnchorComponent { index: usize },
    Polynomial(PolynomialError),
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC state arithmetic failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial(source) => Some(source),
            Self::Numeric(source) => Some(source),
            _ => None,
        }
    }
}

impl From<PolynomialError> for PlanarLcStateError {
    fn from(source: PolynomialError) -> Self {
        Self::Polynomial(source)
    }
}

impl From<NumericError> for PlanarLcStateError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcStatePolynomial {
    polynomial: ExactRationalPolynomial,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcIntervalState {
    components: [RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcAnchorPoint {
    lifted_without_time: [BigRational; PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION],
    physical_time: BigRational,
}

impl PlanarLcAnchorPoint {
    pub fn lifted_without_time(
        &self,
    ) -> &[BigRational; PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION] {
        &self.lifted_without_time
    }

    pub fn physical_time(&self) -> &BigRational {
        &self.physical_time
    }
}

impl PlanarLcStatePolynomial {
    pub fn from_chart(chart: &PlanarLcChartInput) -> Result<Self, PlanarLcStateError> {
        let families = [
            chart.z_polynomial(),
            chart.z_velocity_polynomial(),
            chart.pair_energy_polynomial(),
            chart.binary_center_polynomial(),
            chart.binary_center_velocity_polynomial(),
            chart.third_offset_polynomial(),
            chart.third_offset_velocity_polynomial(),
            chart.physical_time_polynomial(),
        ];
        let count = families[0].coefficient_count();
        if families
            .iter()
            .any(|family| family.coefficient_count() != count)
        {
            return Err(PlanarLcStateError::CoefficientCountMismatch);
        }
        let mut coefficients = Vec::with_capacity(count);
        for degree in 0..count {
            let mut row = Vec::with_capacity(PLANAR_LC_LIFTED_STATE_DIMENSION);
            for family in families {
                let family_row = family
                    .coefficients_by_degree()
                    .get(degree)
                    .ok_or(PlanarLcStateError::CoefficientCountMismatch)?;
                row.extend(family_row.iter().cloned());
            }
            if row.len() != PLANAR_LC_LIFTED_STATE_DIMENSION {
                return Err(PlanarLcStateError::StateDimensionMismatch {
                    expected: PLANAR_LC_LIFTED_STATE_DIMENSION,
                    actual: row.len(),
                });
            }
            coefficients.push(row);
        }
        Ok(Self {
            polynomial: ExactRationalPolynomial::from_degree_major_coefficients(coefficients)?,
        })
    }

    pub fn polynomial(&self) -> &ExactRationalPolynomial {
        &self.polynomial
    }

    pub fn evaluate(
        &self,
        parameter: &RationalInterval,
    ) -> Result<PlanarLcIntervalState, PlanarLcStateError> {
        PlanarLcIntervalState::try_from_vector(self.polynomial.evaluate(parameter)?)
    }

    pub fn evaluate_derivative(
        &self,
        parameter: &RationalInterval,
    ) -> Result<PlanarLcIntervalState, PlanarLcStateError> {
        PlanarLcIntervalState::try_from_vector(self.polynomial.evaluate_derivative(parameter)?)
    }

    pub fn evaluate_anchor_point(
        &self,
        parameter: &BigRational,
    ) -> Result<PlanarLcAnchorPoint, PlanarLcStateError> {
        let state = self.evaluate(&RationalInterval::try_point(parameter.clone())?)?;
        for (index, component) in state.components.iter().enumerate() {
            if !component.is_point() {
                return Err(PlanarLcStateError::NonPointAnchorComponent { index });
            }
        }
        let lifted_without_time =
            std::array::from_fn(|index| state.components[index].lower().clone());
        Ok(PlanarLcAnchorPoint {
            lifted_without_time,
            physical_time: state.components[LC_T].lower().clone(),
        })
    }
}

impl PlanarLcIntervalState {
    fn try_from_vector(values: Vec<RationalInterval>) -> Result<Self, PlanarLcStateError> {
        let actual = values.len();
        let components =
            values
                .try_into()
                .map_err(|_| PlanarLcStateError::StateDimensionMismatch {
                    expected: PLANAR_LC_LIFTED_STATE_DIMENSION,
                    actual,
                })?;
        Ok(Self { components })
    }

    pub fn from_exact_components(
        values: [BigRational; PLANAR_LC_LIFTED_STATE_DIMENSION],
    ) -> Result<Self, PlanarLcStateError> {
        let mut components = Vec::with_capacity(PLANAR_LC_LIFTED_STATE_DIMENSION);
        for value in values {
            components.push(RationalInterval::try_point(value)?);
        }
        Self::try_from_vector(components)
    }

    pub fn from_interval_components(
        components: [RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
    ) -> Result<Self, PlanarLcStateError> {
        let mut admitted = Vec::with_capacity(PLANAR_LC_LIFTED_STATE_DIMENSION);
        for component in components {
            admitted.push(RationalInterval::new(
                component.lower().clone(),
                component.upper().clone(),
            )?);
        }
        Self::try_from_vector(admitted)
    }

    pub fn components(&self) -> &[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION] {
        &self.components
    }

    pub fn lifted_without_time(
        &self,
    ) -> [RationalInterval; PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION] {
        std::array::from_fn(|index| self.components[index].clone())
    }

    pub fn physical_time(&self) -> &RationalInterval {
        &self.components[LC_T]
    }

    pub fn inflate(&self, radius: &BigRational) -> Result<Self, PlanarLcStateError> {
        RationalInterval::try_point(radius.clone())?;
        if radius.is_negative() {
            return Err(PlanarLcStateError::NegativeInflation);
        }
        let inflation = RationalInterval::new(-radius.clone(), radius.clone())?;
        let mut values = Vec::with_capacity(PLANAR_LC_LIFTED_STATE_DIMENSION);
        for component in &self.components {
            values.push(component.add(&inflation)?);
        }
        Self::try_from_vector(values)
    }

    pub fn rho(&self) -> Result<RationalInterval, PlanarLcStateError> {
        Ok(interval_square(&self.components[LC_ZX])?
            .add(&interval_square(&self.components[LC_ZY])?)?)
    }

    pub fn lc_square(&self) -> Result<[RationalInterval; 2], PlanarLcStateError> {
        let zx2 = interval_square(&self.components[LC_ZX])?;
        let zy2 = interval_square(&self.components[LC_ZY])?;
        let two = BigRational::from_integer(BigInt::from(2));
        Ok([
            zx2.subtract(&zy2)?,
            self.components[LC_ZX]
                .multiply(&self.components[LC_ZY])?
                .scale(&two)?,
        ])
    }

    pub fn pair_energy_constraint(
        &self,
        pair_mass: &BigRational,
    ) -> Result<RationalInterval, PlanarLcStateError> {
        RationalInterval::try_point(pair_mass.clone())?;
        let two = BigRational::from_integer(BigInt::from(2));
        let w2 = interval_square(&self.components[LC_WX])?
            .add(&interval_square(&self.components[LC_WY])?)?;
        let mass = RationalInterval::try_point(pair_mass.clone())?;
        Ok(w2
            .scale(&two)?
            .subtract(&mass)?
            .subtract(&self.rho()?.multiply(&self.components[LC_H])?)?)
    }

    /// The LC deck involution `(z,w)->(-z,-w)`; every other component is
    /// retained exactly.
    pub fn deck_transform(&self) -> Result<Self, PlanarLcStateError> {
        let minus_one = -BigRational::one();
        let mut values = self.components.clone();
        for component in &mut values[..=LC_WY] {
            *component = component.scale(&minus_one)?;
        }
        Ok(Self { components: values })
    }
}

fn interval_square(value: &RationalInterval) -> Result<RationalInterval, PlanarLcStateError> {
    let lower_square = value.lower() * value.lower();
    let upper_square = value.upper() * value.upper();
    let upper = if lower_square >= upper_square {
        lower_square.clone()
    } else {
        upper_square.clone()
    };
    let lower = if value.contains(&BigRational::zero())? {
        BigRational::zero()
    } else if lower_square <= upper_square {
        lower_square
    } else {
        upper_square
    };
    Ok(RationalInterval::new(lower, upper)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parse_wire_json, planar_lc_chart_input_from_wire,
        raw_schema::{decode_raw_chain, SchemaProfile, SegmentWire},
        DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn rational(value: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn first_chart() -> (crate::raw_schema::PlanarLcChartWire, PlanarLcChartInput) {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        let wire = segment.lc_chart.clone();
        let chart = planar_lc_chart_input_from_wire(&wire).unwrap();
        (wire, chart)
    }

    #[test]
    fn lifted_order_and_mass_bits_are_exactly_source_bound_for_all_pairs() {
        assert_eq!(PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES[LC_T], "t");
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let mut pairs = Vec::new();
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap();
            assert_eq!(
                chart.mass_binary64_bits(),
                &std::array::from_fn(|index| segment.lc_chart.masses[index].bits())
            );
            let state = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
            let row = &state.polynomial().coefficients_by_degree()[0];
            let expected = [
                segment.lc_chart.z_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.z_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.z_velocity_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.z_velocity_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.pair_energy_coefficients[0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.binary_center_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.binary_center_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.binary_center_velocity_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.binary_center_velocity_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.third_offset_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.third_offset_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.third_offset_velocity_coefficients[0][0]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.third_offset_velocity_coefficients[0][1]
                    .binary64_rational()
                    .clone(),
                segment.lc_chart.physical_time_coefficients[0]
                    .binary64_rational()
                    .clone(),
            ];
            assert_eq!(row.as_slice(), expected.as_slice());
            pairs.push(chart.pair());
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs, vec![[0, 1], [0, 2], [1, 2]]);
    }

    #[test]
    fn endpoint_points_are_contained_and_anchor_splits_thirteen_plus_time() {
        let (_, chart) = first_chart();
        let state = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
        let enclosure = state.evaluate(chart.parameter_interval()).unwrap();
        for endpoint in [
            chart.parameter_interval().lower(),
            chart.parameter_interval().upper(),
        ] {
            let point = state
                .evaluate(&RationalInterval::try_point(endpoint.clone()).unwrap())
                .unwrap();
            for index in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
                assert!(enclosure.components()[index].contains_interval(&point.components()[index]));
            }
        }
        let anchor = state
            .evaluate_anchor_point(chart.parameter_interval().lower())
            .unwrap();
        assert_eq!(anchor.lifted_without_time().len(), 13);
        assert_eq!(
            anchor.physical_time(),
            &chart.physical_time_polynomial().coefficients_by_degree()[0][0]
        );
    }

    #[test]
    fn derivative_preserves_the_declared_component_order() {
        let (wire, chart) = first_chart();
        let state = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
        let zero = RationalInterval::try_point(BigRational::zero()).unwrap();
        let derivative = state.evaluate_derivative(&zero).unwrap();
        assert_eq!(
            derivative.components()[LC_ZX].lower(),
            wire.z_coefficients[1][0].binary64_rational()
        );
        assert_eq!(
            derivative.components()[LC_H].lower(),
            wire.pair_energy_coefficients[1].binary64_rational()
        );
        assert_eq!(
            derivative.components()[LC_T].lower(),
            wire.physical_time_coefficients[1].binary64_rational()
        );
    }

    #[test]
    fn hand_derived_rho_square_constraint_and_deck_involution_hold() {
        let mut values = std::array::from_fn(|_| BigRational::zero());
        values[LC_ZX] = rational(3);
        values[LC_ZY] = rational(4);
        values[LC_WX] = rational(5);
        values[LC_WY] = rational(12);
        values[LC_H] = rational(2);
        values[LC_T] = rational(7);
        let state = PlanarLcIntervalState::from_exact_components(values).unwrap();
        assert_eq!(state.rho().unwrap().lower(), &rational(25));
        let square = state.lc_square().unwrap();
        assert_eq!(square[0].lower(), &rational(-7));
        assert_eq!(square[1].lower(), &rational(24));
        assert_eq!(
            state.pair_energy_constraint(&rational(3)).unwrap().lower(),
            &rational(285)
        );
        let deck = state.deck_transform().unwrap();
        assert_eq!(deck.components()[LC_ZX].lower(), &rational(-3));
        assert_eq!(deck.components()[LC_ZY].lower(), &rational(-4));
        assert_eq!(deck.components()[LC_WX].lower(), &rational(-5));
        assert_eq!(deck.components()[LC_WY].lower(), &rational(-12));
        assert_eq!(deck.deck_transform().unwrap(), state);
        assert_eq!(deck.rho().unwrap(), state.rho().unwrap());
        assert_eq!(deck.lc_square().unwrap(), state.lc_square().unwrap());
        assert_eq!(
            deck.pair_energy_constraint(&rational(3)).unwrap(),
            state.pair_energy_constraint(&rational(3)).unwrap()
        );
        for index in LC_H..PLANAR_LC_LIFTED_STATE_DIMENSION {
            assert_eq!(deck.components()[index], state.components()[index]);
        }
    }

    #[test]
    fn inflation_is_symmetric_inclusive_and_negative_radius_fails() {
        let state = PlanarLcIntervalState::from_exact_components(std::array::from_fn(|index| {
            rational(index as i64)
        }))
        .unwrap();
        assert_eq!(state.inflate(&BigRational::zero()).unwrap(), state);
        let inflated = state.inflate(&rational(2)).unwrap();
        for index in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
            assert_eq!(
                inflated.components()[index].lower(),
                &rational(index as i64 - 2)
            );
            assert_eq!(
                inflated.components()[index].upper(),
                &rational(index as i64 + 2)
            );
        }
        assert_eq!(
            state.inflate(&rational(-1)),
            Err(PlanarLcStateError::NegativeInflation)
        );
    }

    #[test]
    fn crossing_zero_squares_keep_nonnegative_rho_and_sound_constraint() {
        let zero = RationalInterval::try_point(BigRational::zero()).unwrap();
        let mut components = std::array::from_fn(|_| zero.clone());
        components[LC_ZX] = RationalInterval::new(rational(-1), rational(1)).unwrap();
        components[LC_ZY] = RationalInterval::new(rational(-2), rational(3)).unwrap();
        components[LC_WX] = RationalInterval::new(rational(-1), rational(1)).unwrap();
        components[LC_H] = RationalInterval::try_point(rational(2)).unwrap();
        let state = PlanarLcIntervalState::from_interval_components(components).unwrap();
        assert_eq!(
            state.rho().unwrap(),
            RationalInterval::new(rational(0), rational(10)).unwrap()
        );
        assert_eq!(
            state.lc_square().unwrap()[0],
            RationalInterval::new(rational(-9), rational(1)).unwrap()
        );
        assert_eq!(
            state.pair_energy_constraint(&rational(3)).unwrap(),
            RationalInterval::new(rational(-23), rational(-1)).unwrap()
        );
    }
}
