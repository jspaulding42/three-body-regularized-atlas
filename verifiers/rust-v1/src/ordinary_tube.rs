//! Conditional ordinary-tube replay under a named exact-rational profile.
//!
//! This module proves only the six ordered ordinary-tube obligations.  It
//! makes no claim that a particular initial-value problem lies in the tube.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{
    evaluate_planar_three_body_ordinary_polynomial_defect, exp_enclosure_rational,
    ordinary_field::OrdinaryFieldError,
    ordinary_semantic::{ordinary_tube_binding_status, OrdinaryChartInput, OrdinaryTubeInput},
    rational_input::validate_public_rational,
    sqrt_enclosure_dyadic, ExpEnclosureError, NumericError, OrdinaryPolynomialDefectError,
    PolynomialError, RationalInterval, DEFAULT_EXP_ENCLOSURE_LIMITS,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const CONFIGURATION_DIMENSION: usize = BODY_COUNT * PLANE_DIMENSION;
const CANONICAL_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

pub const EXACT_RATIONAL_ORDINARY_TUBE_V04_PROFILE_ID: &str = "exact_rational_ordinary_tube_v04";

pub const ORDINARY_TUBE_OBLIGATION_IDS: [&str; 6] = [
    "ordinary_tube_identity_matches_chart",
    "ordinary_tube_inputs_finite",
    "ordinary_tube_polynomial_defect_within_cap",
    "ordinary_tube_collision_free",
    "ordinary_tube_lipschitz_within_cap",
    "ordinary_tube_gronwall_self_consistent",
];

/// Deterministic sound arithmetic profile for the exact-rational v0.4
/// verifier.  Its status parity with the historical binary64 implementation
/// is intentionally not claimed frozen; that remains OPEN-V1-08.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRationalOrdinaryTubeV04;

impl ExactRationalOrdinaryTubeV04 {
    pub const SQRT_PRECISION_BITS: usize = 256;
    pub const EXPONENT_UPPER_PRECISION_BITS: usize = 32;
    /// Degree 32 is sufficient because the exponential kernel range-reduces
    /// to an argument at most 1/2; its geometric Taylor-tail majorant is then
    /// below the declared 2^-128 profile ceiling.
    pub const EXP_TAYLOR_CUTOFF: usize = 32;
    pub const MAXIMUM_REDUCED_EXP_TAIL_BITS: usize = 128;

    pub fn maximum_reduced_exp_tail() -> BigRational {
        BigRational::new(
            BigInt::one(),
            BigInt::one() << Self::MAXIMUM_REDUCED_EXP_TAIL_BITS,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryTubeObligation {
    id: &'static str,
    satisfied: bool,
}

impl OrdinaryTubeObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Exact quantities and ordered Boolean ledger from one conditional replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryTubeReplay {
    obligations: [OrdinaryTubeObligation; 6],
    defect_upper: Option<BigRational>,
    nominal_pair_distance_lower: Option<BigRational>,
    tube_pair_distance_lower: Option<BigRational>,
    sqrt_two_upper: Option<BigRational>,
    body_lipschitz_upper: Option<[BigRational; BODY_COUNT]>,
    lipschitz_upper: Option<BigRational>,
    horizon: Option<BigRational>,
    exponential_argument_upper: Option<BigRational>,
    exponential_upper: Option<BigRational>,
    gronwall_upper: Option<BigRational>,
}

impl OrdinaryTubeReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_ORDINARY_TUBE_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[OrdinaryTubeObligation; 6] {
        &self.obligations
    }

    /// Conditional tube certification only; no IVP containment is asserted.
    pub fn certified(&self) -> bool {
        self.obligations
            .iter()
            .all(|obligation| obligation.satisfied)
    }

    pub fn defect_upper(&self) -> Option<&BigRational> {
        self.defect_upper.as_ref()
    }

    pub fn nominal_pair_distance_lower(&self) -> Option<&BigRational> {
        self.nominal_pair_distance_lower.as_ref()
    }

    pub fn tube_pair_distance_lower(&self) -> Option<&BigRational> {
        self.tube_pair_distance_lower.as_ref()
    }

    pub fn sqrt_two_upper(&self) -> Option<&BigRational> {
        self.sqrt_two_upper.as_ref()
    }

    pub fn body_lipschitz_upper(&self) -> Option<&[BigRational; BODY_COUNT]> {
        self.body_lipschitz_upper.as_ref()
    }

    pub fn lipschitz_upper(&self) -> Option<&BigRational> {
        self.lipschitz_upper.as_ref()
    }

    pub fn horizon(&self) -> Option<&BigRational> {
        self.horizon.as_ref()
    }

    pub fn exponential_argument_upper(&self) -> Option<&BigRational> {
        self.exponential_argument_upper.as_ref()
    }

    pub fn exponential_upper(&self) -> Option<&BigRational> {
        self.exponential_upper.as_ref()
    }

    pub fn gronwall_upper(&self) -> Option<&BigRational> {
        self.gronwall_upper.as_ref()
    }
}

/// Kernel/resource failures are separate from a sound false obligation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryTubeReplayError {
    Numeric(NumericError),
    Polynomial(PolynomialError),
    Defect(OrdinaryPolynomialDefectError),
    Exponential(ExpEnclosureError),
}

impl fmt::Display for OrdinaryTubeReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => write!(formatter, "ordinary tube numeric failure: {source}"),
            Self::Polynomial(source) => {
                write!(formatter, "ordinary tube polynomial failure: {source}")
            }
            Self::Defect(source) => write!(formatter, "ordinary tube defect failure: {source}"),
            Self::Exponential(source) => {
                write!(formatter, "ordinary tube exponential failure: {source}")
            }
        }
    }
}

impl std::error::Error for OrdinaryTubeReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::Polynomial(source) => Some(source),
            Self::Defect(source) => Some(source),
            Self::Exponential(source) => Some(source),
        }
    }
}

impl From<NumericError> for OrdinaryTubeReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

impl From<PolynomialError> for OrdinaryTubeReplayError {
    fn from(source: PolynomialError) -> Self {
        Self::Polynomial(source)
    }
}

impl From<ExpEnclosureError> for OrdinaryTubeReplayError {
    fn from(source: ExpEnclosureError) -> Self {
        Self::Exponential(source)
    }
}

/// Replay the six conditional ordinary-tube obligations under the named v0.4
/// exact-rational arithmetic profile.
pub fn replay_ordinary_tube_exact_rational_v04(
    chart: &OrdinaryChartInput,
    tube: &OrdinaryTubeInput,
) -> Result<OrdinaryTubeReplay, OrdinaryTubeReplayError> {
    let binding = ordinary_tube_binding_status(tube, chart);
    if !binding.bound() {
        return Ok(build_replay(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [
                binding.identity_matches(),
                binding.finite_inputs(),
                false,
                false,
                false,
                false,
            ],
        ));
    }

    let position_box = chart
        .position_polynomial()
        .evaluate(chart.parameter_interval())?;
    let nominal_pair_distance_lower = nominal_pair_distance_lower(&position_box)?;
    let sqrt_two_upper = sqrt_enclosure_dyadic(
        &BigRational::from_integer(BigInt::from(2_u8)),
        ExactRationalOrdinaryTubeV04::SQRT_PRECISION_BITS,
    )?
    .interval()
    .upper()
    .clone();
    let tube_pair_distance_lower = tube_pair_distance_lower(
        &nominal_pair_distance_lower,
        &sqrt_two_upper,
        tube.tube_radius(),
    )?;
    let horizon = exact_horizon(chart.parameter_interval(), tube.anchor_parameter())?;

    if nominal_pair_distance_lower.numer().sign() != Sign::Plus {
        return Ok(build_replay(
            None,
            Some(nominal_pair_distance_lower),
            Some(tube_pair_distance_lower),
            Some(sqrt_two_upper),
            None,
            None,
            Some(horizon),
            None,
            None,
            None,
            [true, true, false, false, false, false],
        ));
    }

    let defect = match evaluate_planar_three_body_ordinary_polynomial_defect(
        chart.position_polynomial(),
        chart.velocity_polynomial(),
        chart.parameter_interval(),
        chart.masses(),
        ExactRationalOrdinaryTubeV04::SQRT_PRECISION_BITS,
    ) {
        Ok(defect) => defect,
        Err(OrdinaryPolynomialDefectError::OrdinaryField(
            OrdinaryFieldError::PairDistanceNotSeparated { .. },
        )) => {
            return Ok(build_replay(
                None,
                Some(nominal_pair_distance_lower),
                Some(tube_pair_distance_lower),
                Some(sqrt_two_upper),
                None,
                None,
                Some(horizon),
                None,
                None,
                None,
                [true, true, false, false, false, false],
            ));
        }
        Err(source) => return Err(map_defect_error(source)),
    };
    let delta = defect.maximum_absolute_endpoint_bound().clone();
    let defect_within_cap = delta <= *tube.maximum_defect_bound();

    if tube_pair_distance_lower.numer().sign() != Sign::Plus {
        return Ok(build_replay(
            Some(delta),
            Some(nominal_pair_distance_lower),
            Some(tube_pair_distance_lower),
            Some(sqrt_two_upper),
            None,
            None,
            Some(horizon),
            None,
            None,
            None,
            [true, true, defect_within_cap, false, false, false],
        ));
    }

    let (body_lipschitz_upper, lipschitz_upper) =
        analytic_lipschitz_upper(chart.masses(), &sqrt_two_upper, &tube_pair_distance_lower)?;
    let lipschitz_within_cap = lipschitz_upper <= *tube.maximum_lipschitz_bound();
    let exact_exponent = checked_multiply(&lipschitz_upper, &horizon)?;
    let exponential_argument_upper = dyadic_upper(
        &exact_exponent,
        ExactRationalOrdinaryTubeV04::EXPONENT_UPPER_PRECISION_BITS,
    )?;
    let exponential = exp_enclosure_rational(
        &exponential_argument_upper,
        ExactRationalOrdinaryTubeV04::EXP_TAYLOR_CUTOFF,
        &ExactRationalOrdinaryTubeV04::maximum_reduced_exp_tail(),
        DEFAULT_EXP_ENCLOSURE_LIMITS,
    )?;
    let exponential_upper = exponential.upper_bound().clone();
    let gronwall_upper = gronwall_upper(
        &exponential_upper,
        tube.initial_error_bound(),
        &delta,
        &lipschitz_upper,
    )?;
    let gronwall_self_consistent = strictly_inside_radius(&gronwall_upper, tube.tube_radius());

    Ok(build_replay(
        Some(delta),
        Some(nominal_pair_distance_lower),
        Some(tube_pair_distance_lower),
        Some(sqrt_two_upper),
        Some(body_lipschitz_upper),
        Some(lipschitz_upper),
        Some(horizon),
        Some(exponential_argument_upper),
        Some(exponential_upper),
        Some(gronwall_upper),
        [
            true,
            true,
            defect_within_cap,
            true,
            lipschitz_within_cap,
            gronwall_self_consistent,
        ],
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_replay(
    defect_upper: Option<BigRational>,
    nominal_pair_distance_lower: Option<BigRational>,
    tube_pair_distance_lower: Option<BigRational>,
    sqrt_two_upper: Option<BigRational>,
    body_lipschitz_upper: Option<[BigRational; BODY_COUNT]>,
    lipschitz_upper: Option<BigRational>,
    horizon: Option<BigRational>,
    exponential_argument_upper: Option<BigRational>,
    exponential_upper: Option<BigRational>,
    gronwall_upper: Option<BigRational>,
    satisfied: [bool; 6],
) -> OrdinaryTubeReplay {
    let obligations = std::array::from_fn(|index| OrdinaryTubeObligation {
        id: ORDINARY_TUBE_OBLIGATION_IDS[index],
        satisfied: satisfied[index],
    });
    OrdinaryTubeReplay {
        obligations,
        defect_upper,
        nominal_pair_distance_lower,
        tube_pair_distance_lower,
        sqrt_two_upper,
        body_lipschitz_upper,
        lipschitz_upper,
        horizon,
        exponential_argument_upper,
        exponential_upper,
        gronwall_upper,
    }
}

fn map_defect_error(source: OrdinaryPolynomialDefectError) -> OrdinaryTubeReplayError {
    match source {
        OrdinaryPolynomialDefectError::PositionPolynomial(source)
        | OrdinaryPolynomialDefectError::VelocityPolynomial(source) => {
            OrdinaryTubeReplayError::Polynomial(source)
        }
        OrdinaryPolynomialDefectError::Numeric(source)
        | OrdinaryPolynomialDefectError::OrdinaryField(OrdinaryFieldError::Numeric(source)) => {
            OrdinaryTubeReplayError::Numeric(source)
        }
        source => OrdinaryTubeReplayError::Defect(source),
    }
}

fn nominal_pair_distance_lower(
    position_box: &[RationalInterval],
) -> Result<BigRational, OrdinaryTubeReplayError> {
    if position_box.len() != CONFIGURATION_DIMENSION {
        return Err(OrdinaryTubeReplayError::Defect(
            OrdinaryPolynomialDefectError::PositionDimensionMismatch {
                expected: CONFIGURATION_DIMENSION,
                actual: position_box.len(),
            },
        ));
    }
    let mut nominal: Option<BigRational> = None;
    for (first, second) in CANONICAL_PAIRS {
        let mut squared_floor = BigRational::zero();
        for axis in 0..PLANE_DIMENSION {
            let displacement = position_box[second * PLANE_DIMENSION + axis]
                .subtract(&position_box[first * PLANE_DIMENSION + axis])?;
            let absolute_minimum = interval_absolute_minimum(&displacement);
            let square = checked_multiply(&absolute_minimum, &absolute_minimum)?;
            squared_floor = checked_add(&squared_floor, &square)?;
        }
        let pair_floor = sqrt_enclosure_dyadic(
            &squared_floor,
            ExactRationalOrdinaryTubeV04::SQRT_PRECISION_BITS,
        )?
        .interval()
        .lower()
        .clone();
        nominal = Some(match nominal {
            Some(current) if current <= pair_floor => current,
            _ => pair_floor,
        });
    }
    Ok(nominal.unwrap_or_else(BigRational::zero))
}

fn interval_absolute_minimum(interval: &RationalInterval) -> BigRational {
    if interval.lower().numer().sign() == Sign::Plus {
        interval.lower().clone()
    } else if interval.upper().numer().sign() == Sign::Minus {
        -interval.upper().clone()
    } else {
        BigRational::zero()
    }
}

fn tube_pair_distance_lower(
    nominal: &BigRational,
    sqrt_two_upper: &BigRational,
    radius: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    let twice_sqrt_two = checked_multiply(
        &BigRational::from_integer(BigInt::from(2_u8)),
        sqrt_two_upper,
    )?;
    let reduction = checked_multiply(&twice_sqrt_two, radius)?;
    checked_subtract(nominal, &reduction)
}

fn analytic_lipschitz_upper(
    masses: &[BigRational; BODY_COUNT],
    sqrt_two_upper: &BigRational,
    distance_lower: &BigRational,
) -> Result<([BigRational; BODY_COUNT], BigRational), OrdinaryTubeReplayError> {
    let three_sqrt_two = checked_multiply(
        &BigRational::from_integer(BigInt::from(3_u8)),
        sqrt_two_upper,
    )?;
    let geometry = checked_add(&BigRational::one(), &three_sqrt_two)?;
    let twice_geometry =
        checked_multiply(&BigRational::from_integer(BigInt::from(2_u8)), &geometry)?;
    let distance_square = checked_multiply(distance_lower, distance_lower)?;
    let distance_cube = checked_multiply(&distance_square, distance_lower)?;

    let mut bounds: [BigRational; BODY_COUNT] = std::array::from_fn(|_| BigRational::zero());
    for (body, bound) in bounds.iter_mut().enumerate() {
        let mut other_mass_sum = BigRational::zero();
        for (index, mass) in masses.iter().enumerate() {
            if index != body {
                other_mass_sum = checked_add(&other_mass_sum, mass)?;
            }
        }
        let numerator = checked_multiply(&twice_geometry, &other_mass_sum)?;
        *bound = checked_divide(&numerator, &distance_cube)?;
    }
    let mut maximum = BigRational::one();
    for bound in &bounds {
        if bound > &maximum {
            maximum = bound.clone();
        }
    }
    Ok((bounds, maximum))
}

fn exact_horizon(
    interval: &RationalInterval,
    anchor: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    let left = checked_subtract(interval.lower(), anchor)?;
    let right = checked_subtract(interval.upper(), anchor)?;
    let left_magnitude = exact_absolute(&left);
    let right_magnitude = exact_absolute(&right);
    Ok(if left_magnitude >= right_magnitude {
        left_magnitude
    } else {
        right_magnitude
    })
}

/// Monotone exact ceiling onto a fixed dyadic grid.  The coarser denominator
/// keeps the fixed exponential proof inside its declared default resources.
fn dyadic_upper(
    value: &BigRational,
    precision_bits: usize,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    if value.numer().sign() == Sign::Minus {
        return Err(ExpEnclosureError::NegativeExponent.into());
    }
    let scale = BigInt::one() << precision_bits;
    let scaled_numerator = value.numer() * &scale;
    let quotient = &scaled_numerator / value.denom();
    let remainder = &scaled_numerator % value.denom();
    let upper_numerator = if remainder.is_zero() {
        quotient
    } else {
        quotient + BigInt::one()
    };
    admit_rational(BigRational::new(upper_numerator, scale))
}

fn gronwall_upper(
    exponential_upper: &BigRational,
    initial_error: &BigRational,
    defect_upper: &BigRational,
    lipschitz_upper: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    let propagated_initial = checked_multiply(exponential_upper, initial_error)?;
    let exponential_increment = checked_subtract(exponential_upper, &BigRational::one())?;
    let defect_numerator = checked_multiply(defect_upper, &exponential_increment)?;
    let propagated_defect = checked_divide(&defect_numerator, lipschitz_upper)?;
    checked_add(&propagated_initial, &propagated_defect)
}

fn strictly_inside_radius(error_upper: &BigRational, radius: &BigRational) -> bool {
    error_upper < radius
}

fn exact_absolute(value: &BigRational) -> BigRational {
    if value.numer().sign() == Sign::Minus {
        -value.clone()
    } else {
        value.clone()
    }
}

fn checked_add(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    admit_rational(left + right)
}

fn checked_subtract(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    admit_rational(left - right)
}

fn checked_multiply(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    admit_rational(left * right)
}

fn checked_divide(
    numerator: &BigRational,
    denominator: &BigRational,
) -> Result<BigRational, OrdinaryTubeReplayError> {
    if denominator.is_zero() {
        return Err(NumericError::DivisionByZeroInterval.into());
    }
    admit_rational(numerator / denominator)
}

fn admit_rational(value: BigRational) -> Result<BigRational, OrdinaryTubeReplayError> {
    validate_public_rational(&value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, parse_wire_json,
        raw_schema::{
            decode_raw_chain, OrdinaryChartWire, OrdinaryTubeWire, RawPlanarChainWire, Real,
            SchemaProfile, SegmentWire,
        },
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
    ));
    const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/failed-revisit.raw.json"
    ));

    fn parse_chain(bytes: &[u8]) -> RawPlanarChainWire {
        let ast = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(ast, SchemaProfile::V03Compatible).unwrap()
    }

    fn base_chain() -> &'static RawPlanarChainWire {
        static CHAIN: OnceLock<RawPlanarChainWire> = OnceLock::new();
        CHAIN.get_or_init(|| parse_chain(SUCCESS_RAW))
    }

    fn real(lexeme: &str) -> Real {
        checked_real_binary64_from_json(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn row(values: [[&str; PLANE_DIMENSION]; BODY_COUNT]) -> Vec<Vec<Real>> {
        values
            .into_iter()
            .map(|body| body.into_iter().map(real).collect())
            .collect()
    }

    fn simple_wires(masses: [&str; BODY_COUNT]) -> (OrdinaryChartWire, OrdinaryTubeWire) {
        let mut chart = base_chain().initial_chart.clone();
        chart.certificate_id = "simple-certificate".into();
        chart.chart_id = "simple-chart".into();
        chart.chart_type = "ordinary_taylor".into();
        chart.masses = masses.into_iter().map(real).collect();
        chart.position_coefficients = vec![
            row([["-1.0", "0.0"], ["0.0", "0.0"], ["1.0", "0.0"]]),
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
        ];
        chart.velocity_coefficients = vec![
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
        ];
        chart.parameter_interval = [real("0.0"), real("0.000001")];
        chart.physical_time_interval = [real("0.0"), real("0.000001")];
        chart.coefficient_tolerance = real("0.0");
        chart.residual_tolerance = real("0.0");
        chart.tail_bound = real("0.0");
        chart.source = "simple-source".into();

        let mut tube = base_chain().initial_tube.clone();
        tube.tube_id = "simple-tube".into();
        tube.chart_id = chart.chart_id.clone();
        tube.anchor_parameter = real("0.0");
        tube.initial_error_bound = real("0.0");
        tube.tube_radius = real("0.01");
        tube.max_defect_bound = real("1e100");
        tube.max_lipschitz_bound = real("1e100");
        tube.source = "simple-tube-source".into();
        (chart, tube)
    }

    fn replay_wires(
        chart_wire: &OrdinaryChartWire,
        tube_wire: &OrdinaryTubeWire,
    ) -> OrdinaryTubeReplay {
        let chart = ordinary_chart_input_from_wire(chart_wire).unwrap();
        let tube = ordinary_tube_input_from_wire(tube_wire);
        replay_ordinary_tube_exact_rational_v04(&chart, &tube).unwrap()
    }

    fn satisfaction(replay: &OrdinaryTubeReplay) -> [bool; 6] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    #[test]
    fn simple_and_unequal_masses_match_hand_derived_defect_and_lipschitz_formulas() {
        for (masses, expected_delta, other_mass_sums) in [
            (
                ["1.0", "1.0", "1.0"],
                BigRational::new(BigInt::from(5), BigInt::from(4)),
                [2_i64, 2, 2],
            ),
            (
                ["1.0", "2.0", "3.0"],
                BigRational::new(BigInt::from(11), BigInt::from(4)),
                [5_i64, 4, 3],
            ),
        ] {
            let (chart_wire, tube_wire) = simple_wires(masses);
            let replay = replay_wires(&chart_wire, &tube_wire);
            assert!(replay.certified());
            assert_eq!(
                replay.nominal_pair_distance_lower(),
                Some(&BigRational::one())
            );
            assert_eq!(replay.defect_upper(), Some(&expected_delta));

            let sqrt_two = replay.sqrt_two_upper().unwrap();
            let radius = tube_wire.tube_radius.binary64_rational();
            let expected_tube_floor =
                BigRational::one() - BigRational::from_integer(BigInt::from(2)) * sqrt_two * radius;
            assert_eq!(
                replay.tube_pair_distance_lower(),
                Some(&expected_tube_floor)
            );
            let geometry =
                BigRational::one() + BigRational::from_integer(BigInt::from(3)) * sqrt_two;
            let denominator = &expected_tube_floor * &expected_tube_floor * &expected_tube_floor;
            let expected_bounds: [BigRational; BODY_COUNT] = std::array::from_fn(|body| {
                BigRational::from_integer(BigInt::from(2 * other_mass_sums[body])) * &geometry
                    / &denominator
            });
            assert_eq!(replay.body_lipschitz_upper(), Some(&expected_bounds));
            let mut expected_maximum = expected_bounds.iter().max().unwrap().clone();
            if expected_maximum < BigRational::one() {
                expected_maximum = BigRational::one();
            }
            assert_eq!(replay.lipschitz_upper(), Some(&expected_maximum));
        }
    }

    #[test]
    fn exact_cap_equality_passes_and_the_next_lower_binary64_fails() {
        let (chart_wire, mut tube_wire) = simple_wires(["1.0", "1.0", "1.0"]);
        tube_wire.max_defect_bound = real("1.25");
        let equal_defect = replay_wires(&chart_wire, &tube_wire);
        assert!(equal_defect.obligations()[2].satisfied());
        tube_wire.max_defect_bound = real("1.2499999999999998");
        let below_defect = replay_wires(&chart_wire, &tube_wire);
        assert!(!below_defect.obligations()[2].satisfied());

        let (mut low_chart, mut low_tube) = simple_wires(["1e-100", "1e-100", "1e-100"]);
        low_chart.position_coefficients[0] =
            row([["-1e100", "0.0"], ["0.0", "0.0"], ["1e100", "0.0"]]);
        low_tube.max_lipschitz_bound = real("1.0");
        let equal_lipschitz = replay_wires(&low_chart, &low_tube);
        assert_eq!(equal_lipschitz.lipschitz_upper(), Some(&BigRational::one()));
        assert!(equal_lipschitz.obligations()[4].satisfied());
        low_tube.max_lipschitz_bound = real("0.9999999999999999");
        let below_lipschitz = replay_wires(&low_chart, &low_tube);
        assert!(!below_lipschitz.obligations()[4].satisfied());
    }

    #[test]
    fn zero_tube_distance_and_equal_gronwall_radius_are_strict_failures() {
        assert_eq!(ExactRationalOrdinaryTubeV04::SQRT_PRECISION_BITS, 256);
        assert_eq!(
            ExactRationalOrdinaryTubeV04::EXPONENT_UPPER_PRECISION_BITS,
            32
        );
        assert_eq!(ExactRationalOrdinaryTubeV04::EXP_TAYLOR_CUTOFF, 32);
        assert_eq!(
            ExactRationalOrdinaryTubeV04::MAXIMUM_REDUCED_EXP_TAIL_BITS,
            128
        );
        assert_eq!(
            ExactRationalOrdinaryTubeV04::maximum_reduced_exp_tail(),
            BigRational::new(BigInt::one(), BigInt::one() << 128_usize)
        );
        let sqrt_two = sqrt_enclosure_dyadic(
            &BigRational::from_integer(BigInt::from(2)),
            ExactRationalOrdinaryTubeV04::SQRT_PRECISION_BITS,
        )
        .unwrap()
        .interval()
        .upper()
        .clone();
        let radius = BigRational::one() / (BigRational::from_integer(BigInt::from(2)) * &sqrt_two);
        assert_eq!(
            tube_pair_distance_lower(&BigRational::one(), &sqrt_two, &radius).unwrap(),
            BigRational::zero()
        );
        assert!(!strictly_inside_radius(&radius, &radius));
        assert!(strictly_inside_radius(
            &(&radius - BigRational::new(BigInt::one(), BigInt::one() << 1024_usize)),
            &radius
        ));
    }

    #[test]
    fn primitive_tolerance_and_tail_mutations_do_not_enter_tube_math() {
        let (chart_wire, tube_wire) = simple_wires(["1.0", "1.0", "1.0"]);
        let baseline = replay_wires(&chart_wire, &tube_wire);
        let mut mutated = chart_wire;
        mutated.coefficient_tolerance = real("-1e200");
        mutated.residual_tolerance = real("1e200");
        mutated.tail_bound = real("-3.0");
        assert_eq!(replay_wires(&mutated, &tube_wire), baseline);
    }

    #[test]
    fn both_canonical_chains_have_six_certified_ordinary_tubes() {
        for (name, bytes) in [
            ("success", SUCCESS_RAW),
            ("failed-revisit", FAILED_REVISIT_RAW),
        ] {
            let chain = parse_chain(bytes);
            let mut certified_count = 0_usize;
            let mut replay_pair = |chart_wire: &OrdinaryChartWire, tube_wire: &OrdinaryTubeWire| {
                let replay = replay_wires(chart_wire, tube_wire);
                assert!(replay.certified(), "{name}: {replay:?}");
                assert_eq!(
                    replay
                        .obligations()
                        .iter()
                        .map(OrdinaryTubeObligation::id)
                        .collect::<Vec<_>>(),
                    ORDINARY_TUBE_OBLIGATION_IDS
                );
                certified_count += 1;
            };
            replay_pair(&chain.initial_chart, &chain.initial_tube);
            for segment in &chain.segments {
                match segment {
                    SegmentWire::OrdinaryBridge(segment) => {
                        replay_pair(&segment.target_chart, &segment.target_tube)
                    }
                    SegmentWire::PlanarLcPassage(segment) => {
                        replay_pair(&segment.target_chart, &segment.target_tube)
                    }
                }
            }
            assert_eq!(certified_count, 6, "{name}");
        }
    }

    #[test]
    fn separation_and_multiple_binding_failures_return_ordered_false_ledgers() {
        let (mut collision_chart, tube_wire) = simple_wires(["1.0", "1.0", "1.0"]);
        collision_chart.position_coefficients[0] =
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]);
        let collision = replay_wires(&collision_chart, &tube_wire);
        assert_eq!(
            satisfaction(&collision),
            [true, true, false, false, false, false]
        );
        assert!(collision.defect_upper().is_none());
        assert!(!collision.certified());

        let chart = ordinary_chart_input_from_wire(&collision_chart).unwrap();
        let mut multiple = tube_wire;
        multiple.chart_id = "wrong-chart".into();
        multiple.source.clear();
        multiple.anchor_parameter = real("2.0");
        multiple.tube_radius = real("-1.0");
        let tube = ordinary_tube_input_from_wire(&multiple);
        let replay = replay_ordinary_tube_exact_rational_v04(&chart, &tube).unwrap();
        assert_eq!(
            satisfaction(&replay),
            [false, false, false, false, false, false]
        );
        assert_eq!(
            replay
                .obligations()
                .iter()
                .map(OrdinaryTubeObligation::id)
                .collect::<Vec<_>>(),
            ORDINARY_TUBE_OBLIGATION_IDS
        );
        assert!(replay.nominal_pair_distance_lower().is_none());
    }

    #[test]
    fn replay_resource_exhaustion_is_typed_and_never_panics() {
        let (mut chart_wire, mut tube_wire) = simple_wires(["1.0", "1.0", "1.0"]);
        let one_row = row([["1.0", "1.0"], ["1.0", "1.0"], ["1.0", "1.0"]]);
        let zero_row = row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]);
        chart_wire.position_coefficients = vec![one_row; 100];
        chart_wire.velocity_coefficients = vec![zero_row; 100];
        chart_wire.parameter_interval = [real("1e308"), real("1.1e308")];
        tube_wire.anchor_parameter = real("1e308");
        let chart = ordinary_chart_input_from_wire(&chart_wire).unwrap();
        let tube = ordinary_tube_input_from_wire(&tube_wire);
        let result =
            std::panic::catch_unwind(|| replay_ordinary_tube_exact_rational_v04(&chart, &tube));
        assert!(matches!(
            result.unwrap(),
            Err(OrdinaryTubeReplayError::Polynomial(
                PolynomialError::Numeric(NumericError::RationalComponentBitLimitExceeded { .. })
            ))
        ));
    }
}
