//! Conditional primitive ordinary-chart replay under an exact interval-series
//! profile.
//!
//! The serialized `tail_bound` is only a claimed allowance.  This module does
//! not derive a Taylor remainder and does not claim frozen v0.3 binary64
//! status parity.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    ordinary_semantic::OrdinaryChartInput, rational_from_f64_bits, sqrt_enclosure_dyadic,
    NumericError, RationalInterval,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const CONFIGURATION_DIMENSION: usize = BODY_COUNT * PLANE_DIMENSION;
const STATE_DIMENSION: usize = 2 * CONFIGURATION_DIMENSION;
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
const UNIT_SPEED_FLOOR_BITS: u64 = 0x3d06_849b_86a1_2b9b; // binary64 1e-14

pub const EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID: &str =
    "exact_rational_ordinary_chart_claimed_tail_v04";
pub const HARD_MAX_ORDINARY_CHART_SERIES_WORK_UNITS: usize = 2_000_000;

pub const ORDINARY_CHART_OBLIGATION_IDS: [&str; 13] = [
    "ordinary_chart_type",
    "certificate_identity_present",
    "coefficient_array_shape",
    "finite_coefficients",
    "positive_masses",
    "finite_nonempty_time_intervals",
    "ordinary_physical_parameter_unit_speed",
    "finite_checker_tolerances",
    "initial_noncollision",
    "ordinary_taylor_coefficient_recurrence",
    "ordinary_taylor_exact_rational_residual_polynomials",
    "interval_taylor_model_newton_residual",
    "tail_bound_admissible",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRationalOrdinaryChartClaimedTailV04;

impl ExactRationalOrdinaryChartClaimedTailV04 {
    pub const INITIAL_SQRT_PRECISION_BITS: usize = 256;
    pub const MAX_SQRT_PRECISION_BITS: usize = 2_048;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChartObligation {
    id: &'static str,
    satisfied: bool,
}

impl OrdinaryChartObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChartReplay {
    obligations: [OrdinaryChartObligation; 13],
    unit_speed_width_difference: BigRational,
    unit_speed_tolerance: BigRational,
    maximum_coefficient_residual_upper: Option<BigRational>,
    maximum_polynomial_residual_upper: Option<BigRational>,
    residual_with_claimed_tail_upper: Option<BigRational>,
}

impl OrdinaryChartReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[OrdinaryChartObligation; 13] {
        &self.obligations
    }

    /// Whether every obligation in this conditional claimed-tail profile holds.
    ///
    /// This is not theorem-facing chart certification: the profile does not
    /// prove that the serialized tail allowance bounds the omitted remainder.
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(OrdinaryChartObligation::satisfied)
    }

    pub fn unit_speed_width_difference(&self) -> &BigRational {
        &self.unit_speed_width_difference
    }

    pub fn unit_speed_tolerance(&self) -> &BigRational {
        &self.unit_speed_tolerance
    }

    pub fn maximum_coefficient_residual_upper(&self) -> Option<&BigRational> {
        self.maximum_coefficient_residual_upper.as_ref()
    }

    pub fn maximum_polynomial_residual_upper(&self) -> Option<&BigRational> {
        self.maximum_polynomial_residual_upper.as_ref()
    }

    pub fn residual_with_claimed_tail_upper(&self) -> Option<&BigRational> {
        self.residual_with_claimed_tail_upper.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryChartReplayError {
    Numeric(NumericError),
    WorkLimitExceeded { required: usize, limit: usize },
    PositiveSquareRootResolutionLimitExceeded { precision_bits: usize },
    InternalShapeInvariant,
}

impl fmt::Display for OrdinaryChartReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => write!(formatter, "ordinary chart numeric failure: {source}"),
            Self::WorkLimitExceeded { required, limit } => write!(
                formatter,
                "ordinary chart series requires {required} work units, limit is {limit}"
            ),
            Self::PositiveSquareRootResolutionLimitExceeded { precision_bits } => write!(
                formatter,
                "positive ordinary-chart separation remained unresolved at {precision_bits} square-root precision bits"
            ),
            Self::InternalShapeInvariant => {
                formatter.write_str("validated ordinary chart violated its shape invariant")
            }
        }
    }
}

impl std::error::Error for OrdinaryChartReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::WorkLimitExceeded { .. }
            | Self::PositiveSquareRootResolutionLimitExceeded { .. }
            | Self::InternalShapeInvariant => None,
        }
    }
}

impl From<NumericError> for OrdinaryChartReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

pub fn replay_ordinary_chart_exact_rational_claimed_tail_v04(
    chart: &OrdinaryChartInput,
) -> Result<OrdinaryChartReplay, OrdinaryChartReplayError> {
    let parameter_width = chart.parameter_interval().width();
    let physical_width = chart.physical_time_interval().width();
    let unit_speed_width_difference = absolute(&(parameter_width - physical_width));
    let floor = rational_from_f64_bits(UNIT_SPEED_FLOOR_BITS)?;
    let unit_speed_tolerance = if chart.coefficient_tolerance().numer().sign() == Sign::Minus {
        floor
    } else if chart.coefficient_tolerance() > &floor {
        chart.coefficient_tolerance().clone()
    } else {
        floor
    };
    let unit_speed = unit_speed_width_difference <= unit_speed_tolerance;
    let finite_tolerances = chart.coefficient_tolerance().numer().sign() != Sign::Minus
        && chart.residual_tolerance().numer().sign() != Sign::Minus;
    let tail_admissible = chart.tail_bound().numer().sign() != Sign::Minus;
    let initial_noncollision = initial_noncollision(chart)?;

    let mut maximum_coefficient_residual_upper = None;
    let mut maximum_polynomial_residual_upper = None;
    let mut residual_with_claimed_tail_upper = None;
    let mut recurrence_certified = false;
    let mut residual_formed = false;
    let mut residual_certified = false;

    if finite_tolerances && initial_noncollision {
        let residuals = residual_coefficient_intervals(chart)?;
        let coefficient_upper = maximum_absolute_endpoint_rows(&residuals);
        recurrence_certified = coefficient_upper <= *chart.coefficient_tolerance();
        maximum_coefficient_residual_upper = Some(coefficient_upper);

        if recurrence_certified && unit_speed {
            let polynomial_upper = maximum_residual_horner(&residuals, chart.parameter_interval())?;
            let claimed_tail = if chart.tail_bound().numer().sign() == Sign::Minus {
                BigRational::zero()
            } else {
                chart.tail_bound().clone()
            };
            let with_tail = RationalInterval::try_point(polynomial_upper.clone())?
                .add(&RationalInterval::try_point(claimed_tail)?)?
                .upper()
                .clone();
            residual_formed = true;
            residual_certified = with_tail <= *chart.residual_tolerance();
            maximum_polynomial_residual_upper = Some(polynomial_upper);
            residual_with_claimed_tail_upper = Some(with_tail);
        }
    }

    let satisfied = [
        true,
        true,
        true,
        true,
        true,
        true,
        unit_speed,
        finite_tolerances,
        initial_noncollision,
        recurrence_certified,
        residual_formed,
        residual_certified,
        tail_admissible,
    ];
    Ok(OrdinaryChartReplay {
        obligations: std::array::from_fn(|index| OrdinaryChartObligation {
            id: ORDINARY_CHART_OBLIGATION_IDS[index],
            satisfied: satisfied[index],
        }),
        unit_speed_width_difference,
        unit_speed_tolerance,
        maximum_coefficient_residual_upper,
        maximum_polynomial_residual_upper,
        residual_with_claimed_tail_upper,
    })
}

fn initial_noncollision(chart: &OrdinaryChartInput) -> Result<bool, OrdinaryChartReplayError> {
    let rows = chart.position_polynomial().coefficients_by_degree();
    let Some(initial) = rows.first() else {
        return Err(OrdinaryChartReplayError::InternalShapeInvariant);
    };
    if initial.len() != CONFIGURATION_DIMENSION {
        return Err(OrdinaryChartReplayError::InternalShapeInvariant);
    }
    for (first, second) in PAIRS {
        let mut squared = BigRational::zero();
        for axis in 0..PLANE_DIMENSION {
            let difference = &initial[second * PLANE_DIMENSION + axis]
                - &initial[first * PLANE_DIMENSION + axis];
            squared += &difference * &difference;
        }
        if squared.is_zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn residual_coefficient_intervals(
    chart: &OrdinaryChartInput,
) -> Result<Vec<Vec<RationalInterval>>, OrdinaryChartReplayError> {
    let q = chart.position_polynomial().coefficients_by_degree();
    let v = chart.velocity_polynomial().coefficients_by_degree();
    if q.len() != v.len() || q.len() < 2 {
        return Err(OrdinaryChartReplayError::InternalShapeInvariant);
    }
    let order = q.len() - 1;
    enforce_work_limit(order)?;
    let acceleration = acceleration_series(q, chart.masses(), order)?;
    let mut residuals = Vec::with_capacity(order);
    for degree in 0..order {
        let multiplier = BigRational::from_integer(BigInt::from(degree + 1));
        let mut row = Vec::with_capacity(STATE_DIMENSION);
        for component in 0..CONFIGURATION_DIMENSION {
            row.push(RationalInterval::try_point(
                &q[degree + 1][component] * &multiplier - &v[degree][component],
            )?);
        }
        for component in 0..CONFIGURATION_DIMENSION {
            let derivative = RationalInterval::try_point(&v[degree + 1][component] * &multiplier)?;
            row.push(derivative.subtract(&acceleration[degree][component])?);
        }
        residuals.push(row);
    }
    Ok(residuals)
}

fn acceleration_series(
    q: &[Vec<BigRational>],
    masses: &[BigRational; BODY_COUNT],
    order: usize,
) -> Result<Vec<Vec<RationalInterval>>, OrdinaryChartReplayError> {
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut acceleration = vec![vec![zero.clone(); CONFIGURATION_DIMENSION]; order];
    for (first, second) in PAIRS {
        let mut delta = vec![vec![BigRational::zero(); PLANE_DIMENSION]; order];
        for degree in 0..order {
            for axis in 0..PLANE_DIMENSION {
                delta[degree][axis] = &q[degree][second * PLANE_DIMENSION + axis]
                    - &q[degree][first * PLANE_DIMENSION + axis];
            }
        }
        let mut squared_distance = vec![BigRational::zero(); order];
        for degree in 0..order {
            for axis in 0..PLANE_DIMENSION {
                for left_degree in 0..=degree {
                    squared_distance[degree] +=
                        &delta[left_degree][axis] * &delta[degree - left_degree][axis];
                }
            }
        }
        if squared_distance[0].numer().sign() != Sign::Plus {
            return Err(OrdinaryChartReplayError::InternalShapeInvariant);
        }
        let inverse_cube = inverse_three_halves_series(&squared_distance)?;
        for degree in 0..order {
            for axis in 0..PLANE_DIMENSION {
                let mut pair = zero.clone();
                for left_degree in 0..=degree {
                    pair = pair.add(
                        &inverse_cube[degree - left_degree].scale(&delta[left_degree][axis])?,
                    )?;
                }
                let first_component = first * PLANE_DIMENSION + axis;
                let second_component = second * PLANE_DIMENSION + axis;
                acceleration[degree][first_component] =
                    acceleration[degree][first_component].add(&pair.scale(&masses[second])?)?;
                acceleration[degree][second_component] = acceleration[degree][second_component]
                    .subtract(&pair.scale(&masses[first])?)?;
            }
        }
    }
    Ok(acceleration)
}

fn inverse_three_halves_series(
    base: &[BigRational],
) -> Result<Vec<RationalInterval>, OrdinaryChartReplayError> {
    let sqrt = positive_sqrt_enclosure(&base[0])?;
    let base_zero = RationalInterval::try_point(base[0].clone())?;
    let initial = base_zero.multiply(sqrt.interval())?.reciprocal()?;
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut out = vec![zero.clone(); base.len()];
    out[0] = initial;
    let exponent = BigRational::new(BigInt::from(-3), BigInt::from(2));
    for degree in 1..base.len() {
        let mut right_sum = zero.clone();
        for index in 1..=degree {
            let scalar = &base[index] * BigRational::from_integer(BigInt::from(index));
            right_sum = right_sum.add(&out[degree - index].scale(&scalar)?)?;
        }
        let right = right_sum.scale(&exponent)?;
        let mut left_known = zero.clone();
        for index in 1..degree {
            let scalar = &base[index] * BigRational::from_integer(BigInt::from(degree - index));
            left_known = left_known.add(&out[degree - index].scale(&scalar)?)?;
        }
        let denominator = RationalInterval::try_point(
            &base[0] * BigRational::from_integer(BigInt::from(degree)),
        )?;
        out[degree] = right.subtract(&left_known)?.divide(&denominator)?;
    }
    Ok(out)
}

fn positive_sqrt_enclosure(
    radicand: &BigRational,
) -> Result<crate::DyadicSqrtEnclosure, OrdinaryChartReplayError> {
    let mut precision = ExactRationalOrdinaryChartClaimedTailV04::INITIAL_SQRT_PRECISION_BITS;
    loop {
        let enclosure = sqrt_enclosure_dyadic(radicand, precision)?;
        if enclosure.interval().lower().numer().sign() == Sign::Plus {
            return Ok(enclosure);
        }
        if precision == ExactRationalOrdinaryChartClaimedTailV04::MAX_SQRT_PRECISION_BITS {
            return Err(
                OrdinaryChartReplayError::PositiveSquareRootResolutionLimitExceeded {
                    precision_bits: precision,
                },
            );
        }
        precision = precision
            .checked_mul(2)
            .unwrap_or(ExactRationalOrdinaryChartClaimedTailV04::MAX_SQRT_PRECISION_BITS)
            .min(ExactRationalOrdinaryChartClaimedTailV04::MAX_SQRT_PRECISION_BITS);
    }
}

fn maximum_residual_horner(
    residuals: &[Vec<RationalInterval>],
    argument: &RationalInterval,
) -> Result<BigRational, OrdinaryChartReplayError> {
    let mut maximum = BigRational::zero();
    for component in 0..STATE_DIMENSION {
        let mut accumulator = residuals
            .last()
            .ok_or(OrdinaryChartReplayError::InternalShapeInvariant)?[component]
            .clone();
        for row in residuals[..residuals.len() - 1].iter().rev() {
            accumulator = accumulator.multiply(argument)?.add(&row[component])?;
        }
        update_maximum(&mut maximum, &accumulator);
    }
    Ok(maximum)
}

fn maximum_absolute_endpoint_rows(rows: &[Vec<RationalInterval>]) -> BigRational {
    let mut maximum = BigRational::zero();
    for row in rows {
        for interval in row {
            update_maximum(&mut maximum, interval);
        }
    }
    maximum
}

fn update_maximum(maximum: &mut BigRational, interval: &RationalInterval) {
    for endpoint in [interval.lower(), interval.upper()] {
        let magnitude = absolute(endpoint);
        if magnitude > *maximum {
            *maximum = magnitude;
        }
    }
}

fn absolute(value: &BigRational) -> BigRational {
    if value.numer().sign() == Sign::Minus {
        -value.clone()
    } else {
        value.clone()
    }
}

fn enforce_work_limit(order: usize) -> Result<(), OrdinaryChartReplayError> {
    let required = order
        .checked_mul(order)
        .and_then(|value| value.checked_mul(96))
        .unwrap_or(usize::MAX);
    if required > HARD_MAX_ORDINARY_CHART_SERIES_WORK_UNITS {
        return Err(OrdinaryChartReplayError::WorkLimitExceeded {
            required,
            limit: HARD_MAX_ORDINARY_CHART_SERIES_WORK_UNITS,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire, parse_wire_json,
        raw_schema::{
            decode_raw_chain, OrdinaryChartWire, RawPlanarChainWire, Real, SchemaProfile,
        },
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
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

    fn simple_chart_wire(masses: [&str; BODY_COUNT]) -> OrdinaryChartWire {
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
        chart.coefficient_tolerance = real("1e100");
        chart.residual_tolerance = real("1e100");
        chart.tail_bound = real("0.0");
        chart.source = "simple-source".into();
        chart
    }

    fn replay_wire(wire: &OrdinaryChartWire) -> OrdinaryChartReplay {
        let chart = ordinary_chart_input_from_wire(wire).unwrap();
        replay_ordinary_chart_exact_rational_claimed_tail_v04(&chart).unwrap()
    }

    fn satisfaction(replay: &OrdinaryChartReplay) -> [bool; 13] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    #[test]
    fn static_collinear_equal_and_unequal_masses_have_hand_derived_bounds() {
        for (masses, expected) in [
            (
                ["1.0", "1.0", "1.0"],
                BigRational::new(BigInt::from(5), BigInt::from(4)),
            ),
            (
                ["1.0", "2.0", "3.0"],
                BigRational::new(BigInt::from(11), BigInt::from(4)),
            ),
        ] {
            let replay = replay_wire(&simple_chart_wire(masses));
            assert!(replay.conditional_profile_satisfied());
            assert_eq!(replay.maximum_coefficient_residual_upper(), Some(&expected));
            assert_eq!(replay.maximum_polynomial_residual_upper(), Some(&expected));
            assert_eq!(replay.residual_with_claimed_tail_upper(), Some(&expected));
        }
    }

    #[test]
    fn coefficient_and_residual_cap_equality_pass_and_next_lower_binary64_fails() {
        let mut coefficient_wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        coefficient_wire.coefficient_tolerance = real("1.25");
        let equal_coefficient = replay_wire(&coefficient_wire);
        assert!(equal_coefficient.obligations()[9].satisfied());
        coefficient_wire.coefficient_tolerance = real("1.2499999999999998");
        let lower_coefficient = replay_wire(&coefficient_wire);
        assert!(!lower_coefficient.obligations()[9].satisfied());
        assert!(!lower_coefficient.obligations()[10].satisfied());
        assert!(!lower_coefficient.obligations()[11].satisfied());

        let mut residual_wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        residual_wire.residual_tolerance = real("1.25");
        let equal_residual = replay_wire(&residual_wire);
        assert!(equal_residual.obligations()[11].satisfied());
        residual_wire.residual_tolerance = real("1.2499999999999998");
        let lower_residual = replay_wire(&residual_wire);
        assert!(lower_residual.obligations()[10].satisfied());
        assert!(!lower_residual.obligations()[11].satisfied());
    }

    #[test]
    fn negative_claimed_tail_is_independent_and_clamped_only_in_residual_bound() {
        let mut wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        wire.coefficient_tolerance = real("1.25");
        wire.residual_tolerance = real("1.25");
        wire.tail_bound = real("-3.0");
        let replay = replay_wire(&wire);
        assert_eq!(
            satisfaction(&replay),
            [true, true, true, true, true, true, true, true, true, true, true, true, false]
        );
        assert_eq!(
            replay.residual_with_claimed_tail_upper(),
            Some(&BigRational::new(BigInt::from(5), BigInt::from(4)))
        );
        assert!(!replay.conditional_profile_satisfied());
    }

    #[test]
    fn initial_collision_makes_exactly_obligations_nine_through_twelve_false() {
        let mut wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        wire.position_coefficients[0] = row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]);
        let replay = replay_wire(&wire);
        assert_eq!(
            satisfaction(&replay),
            [true, true, true, true, true, true, true, true, false, false, false, false, true]
        );
        assert!(replay.maximum_coefficient_residual_upper().is_none());
        assert!(replay.maximum_polynomial_residual_upper().is_none());
        assert!(replay.residual_with_claimed_tail_upper().is_none());
    }

    #[test]
    fn minimum_subnormal_initial_separation_replays_without_zero_sqrt_lower_bound() {
        let mut wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        wire.position_coefficients[0] = row([["0.0", "0.0"], ["5e-324", "0.0"], ["1.0", "0.0"]]);
        assert_eq!(wire.position_coefficients[0][1][0].bits(), 1);
        let chart = ordinary_chart_input_from_wire(&wire).unwrap();

        let replay = replay_ordinary_chart_exact_rational_claimed_tail_v04(&chart)
            .expect("adaptive precision must resolve a positive subnormal separation");

        assert!(replay.obligations()[8].satisfied());
        assert!(!replay.obligations()[9].satisfied());
        assert!(!replay.obligations()[10].satisfied());
        assert!(!replay.obligations()[11].satisfied());
        assert!(replay.maximum_coefficient_residual_upper().is_some());
    }

    #[test]
    fn series_work_exhaustion_is_typed_and_does_not_panic() {
        let mut wire = simple_chart_wire(["1.0", "1.0", "1.0"]);
        let zero = row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]);
        wire.position_coefficients.resize(146, zero.clone());
        wire.velocity_coefficients.resize(146, zero);
        let chart = ordinary_chart_input_from_wire(&wire).unwrap();
        let result = std::panic::catch_unwind(|| {
            replay_ordinary_chart_exact_rational_claimed_tail_v04(&chart)
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Err(OrdinaryChartReplayError::WorkLimitExceeded {
                required: 145 * 145 * 96,
                limit: HARD_MAX_ORDINARY_CHART_SERIES_WORK_UNITS,
            })
        );
    }
}
