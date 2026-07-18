//! Bounded exact-rational interval formal-series replay for planar LC charts.
//!
//! This module consumes only admitted chart coefficients and mass encodings.
//! It neither consumes claimed tolerances/tails/samples nor accepts a chart.
//! Its interval recurrences are not frozen binary64 status parity.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError, PlanarLcMassProfile},
    sqrt_enclosure_dyadic, ExactBinary64, NumericError, PlanarLcChartInput, PlanarLcStateError,
    PlanarLcStatePolynomial, RationalInterval, HARD_MAX_SQRT_PRECISION_BITS, LC_H, LC_UX, LC_UY,
    LC_VX, LC_VY, LC_WX, LC_WY, LC_YX, LC_YY, LC_ZX, LC_ZY, PLANAR_LC_LIFTED_STATE_DIMENSION,
};

const SERIES_WORK_MULTIPLIER: usize = 512;
pub const HARD_MAX_PLANAR_LC_SERIES_WORK_UNITS: usize = 4_000_000;
/// Fixed practical default; callers may request a higher bounded precision.
pub const PLANAR_LC_SERIES_DEFAULT_SQRT_PRECISION_BITS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcSeriesDenominator {
    First,
    Second,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcSeriesError {
    State(PlanarLcStateError),
    MassDecode {
        index: usize,
        source: NumericError,
    },
    MassProfile(MassProfileError),
    DegreeTooSmall {
        degree: usize,
        minimum: usize,
    },
    WorkLimitExceeded {
        required: usize,
        limit: usize,
    },
    SqrtPrecisionExceeded {
        requested: usize,
        limit: usize,
    },
    ThirdBodyCollision {
        denominator: PlanarLcSeriesDenominator,
    },
    PositiveSquareRootUnresolved {
        denominator: PlanarLcSeriesDenominator,
    },
    InternalShapeInvariant,
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcSeriesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "planar LC interval-series replay failed: {self:?}"
        )
    }
}

impl std::error::Error for PlanarLcSeriesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::MassDecode { source, .. } => Some(source),
            Self::MassProfile(source) => Some(source),
            Self::Numeric(source) => Some(source),
            _ => None,
        }
    }
}

impl From<PlanarLcStateError> for PlanarLcSeriesError {
    fn from(source: PlanarLcStateError) -> Self {
        Self::State(source)
    }
}

impl From<MassProfileError> for PlanarLcSeriesError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

impl From<NumericError> for PlanarLcSeriesError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcSeriesReplay {
    rhs_coefficients: Vec<[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]>,
    residual_coefficients: Vec<[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]>,
    maximum_absolute_residual_endpoint: BigRational,
    pair_energy_constraint_coefficients: Vec<RationalInterval>,
    maximum_absolute_constraint_endpoint: BigRational,
    denominator_square_constants: [BigRational; 2],
    mass_profile: PlanarLcMassProfile,
    accounted_work_units: usize,
}

impl PlanarLcSeriesReplay {
    pub fn rhs_coefficients(&self) -> &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]] {
        &self.rhs_coefficients
    }

    pub fn residual_coefficients(&self) -> &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]] {
        &self.residual_coefficients
    }

    pub fn maximum_absolute_residual_endpoint(&self) -> &BigRational {
        &self.maximum_absolute_residual_endpoint
    }

    pub fn pair_energy_constraint_coefficients(&self) -> &[RationalInterval] {
        &self.pair_energy_constraint_coefficients
    }

    pub fn maximum_absolute_constraint_endpoint(&self) -> &BigRational {
        &self.maximum_absolute_constraint_endpoint
    }

    pub fn denominator_square_constants(&self) -> &[BigRational; 2] {
        &self.denominator_square_constants
    }

    pub fn mass_profile(&self) -> &PlanarLcMassProfile {
        &self.mass_profile
    }

    pub const fn accounted_work_units(&self) -> usize {
        self.accounted_work_units
    }
}

pub fn replay_planar_lc_interval_series(
    chart: &PlanarLcChartInput,
    sqrt_precision_bits: usize,
) -> Result<PlanarLcSeriesReplay, PlanarLcSeriesError> {
    if sqrt_precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(PlanarLcSeriesError::SqrtPrecisionExceeded {
            requested: sqrt_precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        });
    }
    let state = PlanarLcStatePolynomial::from_chart(chart)?;
    let coefficient_count = state.polynomial().coefficient_count();
    let degree = state.polynomial().degree();
    if coefficient_count < 2 {
        return Err(PlanarLcSeriesError::DegreeTooSmall { degree, minimum: 1 });
    }
    let accounted_work_units = coefficient_count
        .checked_mul(coefficient_count)
        .and_then(|value| value.checked_mul(SERIES_WORK_MULTIPLIER))
        .unwrap_or(usize::MAX);
    if accounted_work_units > HARD_MAX_PLANAR_LC_SERIES_WORK_UNITS {
        return Err(PlanarLcSeriesError::WorkLimitExceeded {
            required: accounted_work_units,
            limit: HARD_MAX_PLANAR_LC_SERIES_WORK_UNITS,
        });
    }
    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;
    let coefficients = state.polynomial().coefficients_by_degree();
    let rhs_length = degree;
    let (rhs_coefficients, denominator_square_constants) = field_series(
        coefficients,
        rhs_length,
        &masses,
        &mass_profile,
        sqrt_precision_bits,
    )?;
    let mut residual_coefficients = Vec::with_capacity(rhs_length);
    for (series_degree, rhs) in rhs_coefficients.iter().enumerate() {
        let multiplier = BigRational::from_integer(BigInt::from(series_degree + 1));
        let mut residual = Vec::with_capacity(PLANAR_LC_LIFTED_STATE_DIMENSION);
        for (component, rhs_component) in rhs.iter().enumerate() {
            let derivative = RationalInterval::try_point(
                &coefficients[series_degree + 1][component] * &multiplier,
            )?;
            residual.push(derivative.subtract(rhs_component)?);
        }
        residual_coefficients.push(to_state_row(residual)?);
    }
    let maximum_absolute_residual_endpoint = maximum_absolute_rows(&residual_coefficients);
    let pair_energy_constraint_coefficients = constraint_series(
        coefficients,
        coefficient_count,
        mass_profile.pair_mass().exact(),
    )?;
    let maximum_absolute_constraint_endpoint =
        maximum_absolute_intervals(&pair_energy_constraint_coefficients);
    Ok(PlanarLcSeriesReplay {
        rhs_coefficients,
        residual_coefficients,
        maximum_absolute_residual_endpoint,
        pair_energy_constraint_coefficients,
        maximum_absolute_constraint_endpoint,
        denominator_square_constants,
        mass_profile,
        accounted_work_units,
    })
}

pub fn replay_planar_lc_interval_series_default(
    chart: &PlanarLcChartInput,
) -> Result<PlanarLcSeriesReplay, PlanarLcSeriesError> {
    replay_planar_lc_interval_series(chart, PLANAR_LC_SERIES_DEFAULT_SQRT_PRECISION_BITS)
}

type Series = Vec<RationalInterval>;

fn field_series(
    coefficients: &[Vec<BigRational>],
    length: usize,
    masses: &[ExactBinary64; 3],
    profile: &PlanarLcMassProfile,
    precision: usize,
) -> Result<
    (
        Vec<[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]>,
        [BigRational; 2],
    ),
    PlanarLcSeriesError,
> {
    let z0 = component_series(coefficients, LC_ZX, length)?;
    let z1 = component_series(coefficients, LC_ZY, length)?;
    let w0 = component_series(coefficients, LC_WX, length)?;
    let w1 = component_series(coefficients, LC_WY, length)?;
    let h = component_series(coefficients, LC_H, length)?;
    let u0 = component_series(coefficients, LC_UX, length)?;
    let u1 = component_series(coefficients, LC_UY, length)?;
    let y0 = component_series(coefficients, LC_YX, length)?;
    let y1 = component_series(coefficients, LC_YY, length)?;
    let v0 = component_series(coefficients, LC_VX, length)?;
    let v1 = component_series(coefficients, LC_VY, length)?;
    let two = BigRational::from_integer(BigInt::from(2));
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    let quarter = BigRational::new(BigInt::from(1), BigInt::from(4));
    let rho = add_series(&multiply_series(&z0, &z0)?, &multiply_series(&z1, &z1)?)?;
    let q0 = subtract_series(&multiply_series(&z0, &z0)?, &multiply_series(&z1, &z1)?)?;
    let q1 = scale_series(&multiply_series(&z0, &z1)?, &two)?;
    let d10 = add_series(&y0, &scale_series(&q0, profile.alpha().exact())?)?;
    let d11 = add_series(&y1, &scale_series(&q1, profile.alpha().exact())?)?;
    let d20 = subtract_series(&y0, &scale_series(&q0, profile.beta().exact())?)?;
    let d21 = subtract_series(&y1, &scale_series(&q1, profile.beta().exact())?)?;
    let (f10, f11, norm1) =
        inverse_cube_vector_series(&d10, &d11, precision, PlanarLcSeriesDenominator::First)?;
    let (f20, f21, norm2) =
        inverse_cube_vector_series(&d20, &d21, precision, PlanarLcSeriesDenominator::Second)?;
    let third_mass = masses[profile.third_index()].rational();
    let p0 = scale_series(&subtract_series(&f20, &f10)?, third_mass)?;
    let p1 = scale_series(&subtract_series(&f21, &f11)?, third_mass)?;
    let ltp0 = scale_series(
        &add_series(&multiply_series(&z0, &p0)?, &multiply_series(&z1, &p1)?)?,
        &two,
    )?;
    let ltp1 = scale_series(
        &subtract_series(&multiply_series(&z0, &p1)?, &multiply_series(&z1, &p0)?)?,
        &two,
    )?;
    let wdot0 = add_series(
        &scale_series(&multiply_series(&h, &z0)?, &half)?,
        &scale_series(&multiply_series(&rho, &ltp0)?, &quarter)?,
    )?;
    let wdot1 = add_series(
        &scale_series(&multiply_series(&h, &z1)?, &half)?,
        &scale_series(&multiply_series(&rho, &ltp1)?, &quarter)?,
    )?;
    let qprime0 = scale_series(
        &subtract_series(&multiply_series(&z0, &w0)?, &multiply_series(&z1, &w1)?)?,
        &two,
    )?;
    let qprime1 = scale_series(
        &add_series(&multiply_series(&z1, &w0)?, &multiply_series(&z0, &w1)?)?,
        &two,
    )?;
    let hdot = add_series(
        &multiply_series(&qprime0, &p0)?,
        &multiply_series(&qprime1, &p1)?,
    )?;
    let rdd0 = add_series(
        &scale_series(&f10, profile.center_first().exact())?,
        &scale_series(&f20, profile.center_second().exact())?,
    )?;
    let rdd1 = add_series(
        &scale_series(&f11, profile.center_first().exact())?,
        &scale_series(&f21, profile.center_second().exact())?,
    )?;
    let ydd0 = scale_series(
        &add_series(
            &scale_series(&f10, profile.offset_first().exact())?,
            &scale_series(&f20, profile.offset_second().exact())?,
        )?,
        &BigRational::from_integer(BigInt::from(-1)),
    )?;
    let ydd1 = scale_series(
        &add_series(
            &scale_series(&f11, profile.offset_first().exact())?,
            &scale_series(&f21, profile.offset_second().exact())?,
        )?,
        &BigRational::from_integer(BigInt::from(-1)),
    )?;
    let ordered = [
        w0,
        w1,
        wdot0,
        wdot1,
        hdot,
        multiply_series(&rho, &u0)?,
        multiply_series(&rho, &u1)?,
        multiply_series(&rho, &rdd0)?,
        multiply_series(&rho, &rdd1)?,
        multiply_series(&rho, &v0)?,
        multiply_series(&rho, &v1)?,
        multiply_series(&rho, &ydd0)?,
        multiply_series(&rho, &ydd1)?,
        rho,
    ];
    let mut rows = Vec::with_capacity(length);
    for degree in 0..length {
        rows.push(std::array::from_fn(|component| {
            ordered[component][degree].clone()
        }));
    }
    Ok((rows, [norm1, norm2]))
}

fn inverse_cube_vector_series(
    x: &Series,
    y: &Series,
    precision: usize,
    denominator: PlanarLcSeriesDenominator,
) -> Result<(Series, Series, BigRational), PlanarLcSeriesError> {
    let norm = add_series(&multiply_series(x, x)?, &multiply_series(y, y)?)?;
    if !norm[0].is_point() || norm[0].lower().numer().sign() != Sign::Plus {
        return Err(PlanarLcSeriesError::ThirdBodyCollision { denominator });
    }
    let constant = norm[0].lower().clone();
    let root = sqrt_series(&norm, precision, denominator)?;
    let cube = multiply_series(&norm, &root)?;
    let inverse = reciprocal_series(&cube, precision)?;
    Ok((
        multiply_series(x, &inverse)?,
        multiply_series(y, &inverse)?,
        constant,
    ))
}

fn sqrt_series(
    input: &Series,
    precision: usize,
    denominator: PlanarLcSeriesDenominator,
) -> Result<Series, PlanarLcSeriesError> {
    let enclosure = sqrt_enclosure_dyadic(input[0].lower(), precision)?;
    if enclosure.interval().lower().numer().sign() != Sign::Plus {
        return Err(PlanarLcSeriesError::PositiveSquareRootUnresolved { denominator });
    }
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut output = vec![zero.clone(); input.len()];
    output[0] = enclosure.interval().clone();
    for degree in 1..input.len() {
        let mut known = zero.clone();
        for left in 1..degree {
            known = known.add(&output[left].multiply(&output[degree - left])?)?;
        }
        let denominator_interval = output[0].scale(&BigRational::from_integer(BigInt::from(2)))?;
        output[degree] = outward_dyadic(
            &input[degree]
                .subtract(&known)?
                .divide(&denominator_interval)?,
            precision,
        )?;
    }
    Ok(output)
}

fn reciprocal_series(input: &Series, precision: usize) -> Result<Series, PlanarLcSeriesError> {
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut output = vec![zero.clone(); input.len()];
    output[0] = outward_dyadic(&input[0].reciprocal()?, precision)?;
    for degree in 1..input.len() {
        let mut known = zero.clone();
        for left in 1..=degree {
            known = known.add(&input[left].multiply(&output[degree - left])?)?;
        }
        output[degree] = outward_dyadic(
            &known
                .scale(&BigRational::from_integer(BigInt::from(-1)))?
                .divide(&input[0])?,
            precision,
        )?;
    }
    Ok(output)
}

fn constraint_series(
    coefficients: &[Vec<BigRational>],
    length: usize,
    pair_mass: &BigRational,
) -> Result<Series, PlanarLcSeriesError> {
    let wx = component_series(coefficients, LC_WX, length)?;
    let wy = component_series(coefficients, LC_WY, length)?;
    let zx = component_series(coefficients, LC_ZX, length)?;
    let zy = component_series(coefficients, LC_ZY, length)?;
    let h = component_series(coefficients, LC_H, length)?;
    let rho = add_series(&multiply_series(&zx, &zx)?, &multiply_series(&zy, &zy)?)?;
    let mut constraint = subtract_series(
        &scale_series(
            &add_series(&multiply_series(&wx, &wx)?, &multiply_series(&wy, &wy)?)?,
            &BigRational::from_integer(BigInt::from(2)),
        )?,
        &multiply_series(&rho, &h)?,
    )?;
    constraint[0] = constraint[0].subtract(&RationalInterval::try_point(pair_mass.clone())?)?;
    Ok(constraint)
}

fn component_series(
    coefficients: &[Vec<BigRational>],
    component: usize,
    length: usize,
) -> Result<Series, PlanarLcSeriesError> {
    let mut output = Vec::with_capacity(length);
    for row in coefficients.iter().take(length) {
        output.push(RationalInterval::try_point(
            row.get(component)
                .ok_or(PlanarLcSeriesError::InternalShapeInvariant)?
                .clone(),
        )?);
    }
    if output.len() != length {
        return Err(PlanarLcSeriesError::InternalShapeInvariant);
    }
    Ok(output)
}

fn add_series(left: &Series, right: &Series) -> Result<Series, PlanarLcSeriesError> {
    ensure_same_length(left, right)?;
    left.iter().zip(right).map(|(a, b)| Ok(a.add(b)?)).collect()
}

fn subtract_series(left: &Series, right: &Series) -> Result<Series, PlanarLcSeriesError> {
    ensure_same_length(left, right)?;
    left.iter()
        .zip(right)
        .map(|(a, b)| Ok(a.subtract(b)?))
        .collect()
}

fn scale_series(input: &Series, scalar: &BigRational) -> Result<Series, PlanarLcSeriesError> {
    input.iter().map(|value| Ok(value.scale(scalar)?)).collect()
}

fn multiply_series(left: &Series, right: &Series) -> Result<Series, PlanarLcSeriesError> {
    ensure_same_length(left, right)?;
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut output = vec![zero.clone(); left.len()];
    for degree in 0..left.len() {
        for left_degree in 0..=degree {
            output[degree] =
                output[degree].add(&left[left_degree].multiply(&right[degree - left_degree])?)?;
        }
    }
    Ok(output)
}

fn ensure_same_length(left: &Series, right: &Series) -> Result<(), PlanarLcSeriesError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(PlanarLcSeriesError::InternalShapeInvariant);
    }
    Ok(())
}

fn reconstruct_masses(
    chart: &PlanarLcChartInput,
) -> Result<[ExactBinary64; 3], PlanarLcSeriesError> {
    let mut masses = Vec::with_capacity(3);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        masses.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| PlanarLcSeriesError::MassDecode { index, source })?,
        );
    }
    masses
        .try_into()
        .map_err(|_| PlanarLcSeriesError::InternalShapeInvariant)
}

fn to_state_row(
    values: Vec<RationalInterval>,
) -> Result<[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION], PlanarLcSeriesError> {
    values
        .try_into()
        .map_err(|_| PlanarLcSeriesError::InternalShapeInvariant)
}

fn maximum_absolute_rows(
    rows: &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]],
) -> BigRational {
    rows.iter()
        .flat_map(|row| row.iter())
        .fold(BigRational::zero(), maximum_with_interval)
}

fn maximum_absolute_intervals(intervals: &[RationalInterval]) -> BigRational {
    intervals
        .iter()
        .fold(BigRational::zero(), maximum_with_interval)
}

fn maximum_with_interval(maximum: BigRational, interval: &RationalInterval) -> BigRational {
    let endpoint = interval.lower().abs().max(interval.upper().abs());
    maximum.max(endpoint)
}

fn outward_dyadic(
    interval: &RationalInterval,
    precision: usize,
) -> Result<RationalInterval, PlanarLcSeriesError> {
    if interval.is_point() {
        return Ok(interval.clone());
    }
    let denominator = BigInt::from(1) << precision;
    let lower_scaled = interval.lower().numer() << precision;
    let upper_scaled = interval.upper().numer() << precision;
    let mut lower_integer = &lower_scaled / interval.lower().denom();
    if (&lower_scaled % interval.lower().denom()).sign() == Sign::Minus {
        lower_integer -= 1;
    }
    let mut upper_integer = &upper_scaled / interval.upper().denom();
    if (&upper_scaled % interval.upper().denom()).sign() == Sign::Plus {
        upper_integer += 1;
    }
    Ok(RationalInterval::new(
        BigRational::new(lower_integer, denominator.clone()),
        BigRational::new(upper_integer, denominator),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checked_real_binary64_from_json, parse_json_number_lexeme, parse_wire_json,
        planar_lc_chart_input_from_wire,
        raw_schema::{decode_raw_chain, PlanarLcChartWire, SchemaProfile, SegmentWire},
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn real(value: &str) -> crate::raw_schema::Real {
        checked_real_binary64_from_json(value, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn chain() -> crate::raw_schema::RawPlanarChainWire {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
    }

    fn first_wire() -> PlanarLcChartWire {
        let chain = chain();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        segment.lc_chart.clone()
    }

    fn set_vector_series_zero(series: &mut Vec<Vec<crate::raw_schema::Real>>, count: usize) {
        *series = vec![vec![real("0.0"), real("0.0")]; count];
    }

    fn set_scalar_series_zero(series: &mut Vec<crate::raw_schema::Real>, count: usize) {
        *series = vec![real("0.0"); count];
    }

    fn hand_wire() -> PlanarLcChartWire {
        let mut wire = first_wire();
        wire.masses = vec![real("1.0"), real("1.0"), real("1.0")];
        let count = 3;
        set_vector_series_zero(&mut wire.z_coefficients, count);
        set_vector_series_zero(&mut wire.z_velocity_coefficients, count);
        set_scalar_series_zero(&mut wire.pair_energy_coefficients, count);
        set_vector_series_zero(&mut wire.binary_center_coefficients, count);
        set_vector_series_zero(&mut wire.binary_center_velocity_coefficients, count);
        set_vector_series_zero(&mut wire.third_offset_coefficients, count);
        set_vector_series_zero(&mut wire.third_offset_velocity_coefficients, count);
        set_scalar_series_zero(&mut wire.physical_time_coefficients, count);
        wire.z_coefficients[0][0] = real("1.0");
        wire.z_velocity_coefficients[0][0] = real("1.0");
        wire.pair_energy_coefficients[0] = real("2.0");
        wire.binary_center_velocity_coefficients[0] = vec![real("2.0"), real("3.0")];
        wire.third_offset_coefficients[0][0] = real("3.0");
        wire.third_offset_velocity_coefficients[0] = vec![real("4.0"), real("5.0")];
        wire
    }

    fn hand_chart() -> PlanarLcChartInput {
        planar_lc_chart_input_from_wire(&hand_wire()).unwrap()
    }

    fn truncate_wire(wire: &mut PlanarLcChartWire, count: usize) {
        wire.z_coefficients.truncate(count);
        wire.z_velocity_coefficients.truncate(count);
        wire.pair_energy_coefficients.truncate(count);
        wire.binary_center_coefficients.truncate(count);
        wire.binary_center_velocity_coefficients.truncate(count);
        wire.third_offset_coefficients.truncate(count);
        wire.third_offset_velocity_coefficients.truncate(count);
        wire.physical_time_coefficients.truncate(count);
    }

    fn point(numerator: i64, denominator: i64) -> RationalInterval {
        RationalInterval::try_point(BigRational::new(
            BigInt::from(numerator),
            BigInt::from(denominator),
        ))
        .unwrap()
    }

    #[test]
    fn low_degree_unit_mass_chart_matches_every_constant_rhs_block() {
        let replay = replay_planar_lc_interval_series_default(&hand_chart()).unwrap();
        assert_eq!(replay.rhs_coefficients().len(), 2);
        let expected = [
            point(1, 1),
            point(0, 1),
            point(1273, 1225),
            point(0, 1),
            point(192, 1225),
            point(2, 1),
            point(3, 1),
            point(148, 1225),
            point(0, 1),
            point(4, 1),
            point(5, 1),
            point(-444, 1225),
            point(0, 1),
            point(1, 1),
        ];
        for (actual, expected) in replay.rhs_coefficients()[0].iter().zip(expected) {
            assert!(actual.contains_interval(&expected));
        }
        assert_eq!(
            replay.rhs_coefficients()[1],
            std::array::from_fn(|_| point(0, 1))
        );
        assert_eq!(
            replay.denominator_square_constants(),
            &[
                BigRational::new(BigInt::from(49), BigInt::from(4)),
                BigRational::new(BigInt::from(25), BigInt::from(4)),
            ]
        );
        assert_eq!(
            replay.pair_energy_constraint_coefficients(),
            &[point(-2, 1), point(0, 1), point(0, 1)]
        );
    }

    #[test]
    fn minimum_two_coefficient_chart_has_one_recurrence_and_two_constraint_rows() {
        let mut wire = hand_wire();
        truncate_wire(&mut wire, 2);
        let replay = replay_planar_lc_interval_series_default(
            &planar_lc_chart_input_from_wire(&wire).unwrap(),
        )
        .unwrap();
        assert_eq!(replay.rhs_coefficients().len(), 1);
        assert_eq!(replay.residual_coefficients().len(), 1);
        assert_eq!(replay.pair_energy_constraint_coefficients().len(), 2);
    }

    #[test]
    fn outward_dyadic_pins_floor_ceil_containment_and_exact_points() {
        let positive = RationalInterval::new(
            BigRational::new(BigInt::from(1), BigInt::from(3)),
            BigRational::new(BigInt::from(2), BigInt::from(3)),
        )
        .unwrap();
        let positive_rounded = outward_dyadic(&positive, 2).unwrap();
        assert_eq!(
            positive_rounded,
            RationalInterval::new(
                BigRational::new(BigInt::from(1), BigInt::from(4)),
                BigRational::new(BigInt::from(3), BigInt::from(4)),
            )
            .unwrap()
        );
        assert!(positive_rounded.contains_interval(&positive));
        let signed = RationalInterval::new(
            BigRational::new(BigInt::from(-1), BigInt::from(3)),
            BigRational::new(BigInt::from(1), BigInt::from(3)),
        )
        .unwrap();
        let signed_rounded = outward_dyadic(&signed, 2).unwrap();
        assert_eq!(
            signed_rounded,
            RationalInterval::new(
                BigRational::new(BigInt::from(-2), BigInt::from(4)),
                BigRational::new(BigInt::from(2), BigInt::from(4)),
            )
            .unwrap()
        );
        assert!(signed_rounded.contains_interval(&signed));
        let exact = point(1, 3);
        assert_eq!(outward_dyadic(&exact, 2).unwrap(), exact);
    }

    #[test]
    fn outward_dyadic_rounds_a_negative_upper_endpoint_away_from_zero() {
        let negative = RationalInterval::new(
            BigRational::new(BigInt::from(-2), BigInt::from(3)),
            BigRational::new(BigInt::from(-1), BigInt::from(3)),
        )
        .unwrap();
        let negative_rounded = outward_dyadic(&negative, 2).unwrap();
        assert_eq!(
            negative_rounded,
            RationalInterval::new(
                BigRational::new(BigInt::from(-3), BigInt::from(4)),
                BigRational::new(BigInt::from(-1), BigInt::from(4)),
            )
            .unwrap()
        );
        assert!(negative_rounded.contains_interval(&negative));
    }

    #[test]
    fn all_three_canonical_pairs_replay_and_revalidate_mass_profiles() {
        let chain = chain();
        let mut pairs = Vec::new();
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap();
            let replay = replay_planar_lc_interval_series_default(&chart).unwrap();
            assert_eq!(
                replay.rhs_coefficients().len(),
                chart.z_polynomial().coefficient_count() - 1
            );
            assert_eq!(
                replay.residual_coefficients().len(),
                chart.z_polynomial().coefficient_count() - 1
            );
            assert_eq!(
                replay.pair_energy_constraint_coefficients().len(),
                chart.z_polynomial().coefficient_count()
            );
            assert!(replay.maximum_absolute_residual_endpoint() <= chart.coefficient_tolerance());
            assert!(replay.maximum_absolute_constraint_endpoint() <= chart.coefficient_tolerance());
            let masses = std::array::from_fn(|index| {
                ExactBinary64::from_bits(chart.mass_binary64_bits()[index]).unwrap()
            });
            replay
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
    fn both_reported_maxima_equal_independent_endpoint_recomputations() {
        let replay = replay_planar_lc_interval_series_default(&hand_chart()).unwrap();
        let residual_maximum = replay
            .residual_coefficients()
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|interval| [interval.lower().abs(), interval.upper().abs()])
            .max()
            .unwrap();
        let constraint_maximum = replay
            .pair_energy_constraint_coefficients()
            .iter()
            .flat_map(|interval| [interval.lower().abs(), interval.upper().abs()])
            .max()
            .unwrap();
        assert_eq!(
            replay.maximum_absolute_residual_endpoint(),
            &residual_maximum
        );
        assert_eq!(
            replay.maximum_absolute_constraint_endpoint(),
            &constraint_maximum
        );
    }

    #[test]
    fn denominator_collision_precision_and_work_exhaustion_fail_closed() {
        let mut collision = hand_wire();
        collision.z_coefficients[0][0] = real("0.0");
        collision.third_offset_coefficients[0][0] = real("0.0");
        let collision = planar_lc_chart_input_from_wire(&collision).unwrap();
        assert!(matches!(
            replay_planar_lc_interval_series_default(&collision),
            Err(PlanarLcSeriesError::ThirdBodyCollision { .. })
        ));
        assert_eq!(
            replay_planar_lc_interval_series(&hand_chart(), HARD_MAX_SQRT_PRECISION_BITS + 1),
            Err(PlanarLcSeriesError::SqrtPrecisionExceeded {
                requested: HARD_MAX_SQRT_PRECISION_BITS + 1,
                limit: HARD_MAX_SQRT_PRECISION_BITS,
            })
        );
        let mut oversized = hand_wire();
        let count = 91;
        set_vector_series_zero(&mut oversized.z_coefficients, count);
        set_vector_series_zero(&mut oversized.z_velocity_coefficients, count);
        set_scalar_series_zero(&mut oversized.pair_energy_coefficients, count);
        set_vector_series_zero(&mut oversized.binary_center_coefficients, count);
        set_vector_series_zero(&mut oversized.binary_center_velocity_coefficients, count);
        set_vector_series_zero(&mut oversized.third_offset_coefficients, count);
        set_vector_series_zero(&mut oversized.third_offset_velocity_coefficients, count);
        set_scalar_series_zero(&mut oversized.physical_time_coefficients, count);
        let oversized = planar_lc_chart_input_from_wire(&oversized).unwrap();
        assert!(matches!(
            replay_planar_lc_interval_series_default(&oversized),
            Err(PlanarLcSeriesError::WorkLimitExceeded { .. })
        ));
    }

    #[test]
    fn deck_transform_flips_exactly_the_first_four_rhs_and_residual_components() {
        let original = replay_planar_lc_interval_series_default(&hand_chart()).unwrap();
        let mut deck_wire = hand_wire();
        deck_wire.z_coefficients[0][0] = real("-1.0");
        deck_wire.z_velocity_coefficients[0][0] = real("-1.0");
        let deck = replay_planar_lc_interval_series_default(
            &planar_lc_chart_input_from_wire(&deck_wire).unwrap(),
        )
        .unwrap();
        let minus_one = BigRational::from_integer(BigInt::from(-1));
        for degree in 0..original.rhs_coefficients().len() {
            for component in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
                let expected_rhs = if component <= LC_WY {
                    original.rhs_coefficients()[degree][component]
                        .scale(&minus_one)
                        .unwrap()
                } else {
                    original.rhs_coefficients()[degree][component].clone()
                };
                let expected_residual = if component <= LC_WY {
                    original.residual_coefficients()[degree][component]
                        .scale(&minus_one)
                        .unwrap()
                } else {
                    original.residual_coefficients()[degree][component].clone()
                };
                assert_eq!(deck.rhs_coefficients()[degree][component], expected_rhs);
                assert_eq!(
                    deck.residual_coefficients()[degree][component],
                    expected_residual
                );
            }
        }
        assert_eq!(
            deck.pair_energy_constraint_coefficients(),
            original.pair_energy_constraint_coefficients()
        );
    }

    #[test]
    fn claims_are_ignored_but_a_consumed_coefficient_changes_replay() {
        let wire = hand_wire();
        let baseline = replay_planar_lc_interval_series_default(
            &planar_lc_chart_input_from_wire(&wire).unwrap(),
        )
        .unwrap();
        let mut claims = wire.clone();
        claims.tail_bound = real("7.0");
        claims.coefficient_tolerance = real("8.0");
        claims.regularized_residual_tolerance = real("9.0");
        claims.projected_residual_tolerance = real("10.0");
        claims.projection_rho_lower_bound = real("11.0");
        claims.parameter_interval = [real("2.0"), real("3.0")];
        claims.physical_time_interval = [real("4.0"), real("5.0")];
        claims.sample_count = crate::raw_schema::RawInteger::from_parsed_for_test(
            parse_json_number_lexeme("6", DEFAULT_JSON_NUMBER_LIMITS).unwrap(),
        );
        assert_ne!(claims.sample_count.value(), wire.sample_count.value());
        let claims = replay_planar_lc_interval_series_default(
            &planar_lc_chart_input_from_wire(&claims).unwrap(),
        )
        .unwrap();
        assert_eq!(claims, baseline);
        let mut coefficient = wire;
        coefficient.z_velocity_coefficients[1][0] = real("1.0");
        let coefficient = replay_planar_lc_interval_series_default(
            &planar_lc_chart_input_from_wire(&coefficient).unwrap(),
        )
        .unwrap();
        assert_ne!(coefficient, baseline);
    }
}
