//! Conditional planar LC chart ledger under an exact-rational claimed-tail
//! profile.
//!
//! The serialized tail is a claim, not a derived remainder. An all-true
//! ledger here is not frozen-v0.3 parity, convergence, chart acceptance, or a
//! continuation theorem.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    replay_planar_lc_interval_series_default, replay_planar_lc_projection_default, NumericError,
    PlanarLcChartInput, PlanarLcProjectionError, PlanarLcSeriesError, PlanarLcStateError,
    PlanarLcStatePolynomial, RationalInterval, PLANAR_LC_LIFTED_STATE_DIMENSION,
};

pub const EXACT_RATIONAL_PLANAR_LC_CHART_CLAIMED_TAIL_V04_PROFILE_ID: &str =
    "exact_rational_planar_lc_chart_claimed_tail_v04";
pub const PLANAR_LC_CHART_PHYSICAL_TIME_SUBDIVISIONS: usize = 64;

pub const PLANAR_LC_CHART_OBLIGATION_IDS: [&str; 15] = [
    "planar_levi_civita_binary_chart_type",
    "certificate_identity_present",
    "planar_lc_coefficient_array_shape",
    "finite_coefficients",
    "positive_masses",
    "binary_pair_valid",
    "finite_nonempty_time_intervals",
    "finite_checker_tolerances",
    "interval_physical_time_containment",
    "planar_lc_regularized_coefficient_recurrence",
    "planar_lc_pair_energy_constraint",
    "planar_lc_exact_rational_regularized_residual_polynomials",
    "interval_planar_lc_regularized_rhs_residual",
    "interval_projected_newton_residual_away_from_binary_collision",
    "tail_bound_admissible",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRationalPlanarLcChartClaimedTailV04;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcChartObligation {
    id: &'static str,
    satisfied: bool,
}

impl PlanarLcChartObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcChartReplayError {
    State(PlanarLcStateError),
    Series(PlanarLcSeriesError),
    Projection(PlanarLcProjectionError),
    InternalShapeInvariant,
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcChartReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC chart replay failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcChartReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::Series(source) => Some(source),
            Self::Projection(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::InternalShapeInvariant => None,
        }
    }
}

impl From<PlanarLcStateError> for PlanarLcChartReplayError {
    fn from(source: PlanarLcStateError) -> Self {
        Self::State(source)
    }
}

impl From<NumericError> for PlanarLcChartReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcChartReplay {
    obligations: [PlanarLcChartObligation; 15],
    physical_time_enclosure: RationalInterval,
    maximum_coefficient_residual: Option<BigRational>,
    maximum_pair_energy_constraint: Option<BigRational>,
    maximum_polynomial_residual: Option<BigRational>,
    polynomial_residual_with_claimed_tail: Option<BigRational>,
    maximum_projection_residual: Option<BigRational>,
    projection_residual_with_claimed_tail: Option<BigRational>,
}

impl PlanarLcChartReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PLANAR_LC_CHART_CLAIMED_TAIL_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[PlanarLcChartObligation; 15] {
        &self.obligations
    }

    /// Conditional compatibility only. This does not prove that the claimed
    /// tail bounds an omitted remainder and is not chart certification.
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(PlanarLcChartObligation::satisfied)
    }

    pub fn physical_time_enclosure(&self) -> &RationalInterval {
        &self.physical_time_enclosure
    }

    pub fn maximum_coefficient_residual(&self) -> Option<&BigRational> {
        self.maximum_coefficient_residual.as_ref()
    }

    pub fn maximum_pair_energy_constraint(&self) -> Option<&BigRational> {
        self.maximum_pair_energy_constraint.as_ref()
    }

    pub fn maximum_polynomial_residual(&self) -> Option<&BigRational> {
        self.maximum_polynomial_residual.as_ref()
    }

    pub fn polynomial_residual_with_claimed_tail(&self) -> Option<&BigRational> {
        self.polynomial_residual_with_claimed_tail.as_ref()
    }

    pub fn maximum_projection_residual(&self) -> Option<&BigRational> {
        self.maximum_projection_residual.as_ref()
    }

    pub fn projection_residual_with_claimed_tail(&self) -> Option<&BigRational> {
        self.projection_residual_with_claimed_tail.as_ref()
    }
}

pub fn replay_planar_lc_chart_exact_rational_claimed_tail_v04(
    chart: &PlanarLcChartInput,
) -> Result<PlanarLcChartReplay, PlanarLcChartReplayError> {
    let tail_admissible = chart.tail_bound().numer().sign() != Sign::Minus;
    let finite_tolerances = [
        chart.coefficient_tolerance(),
        chart.regularized_residual_tolerance(),
        chart.projected_residual_tolerance(),
        chart.projection_rho_lower_bound(),
    ]
    .iter()
    .all(|value| value.numer().sign() != Sign::Minus);
    let (physical_time_enclosure, physical_time_contained) =
        physical_time_enclosure(chart, tail_admissible)?;

    let series = if finite_tolerances {
        match replay_planar_lc_interval_series_default(chart) {
            Ok(replay) => Some(replay),
            Err(error) if series_domain_failure(&error) => None,
            Err(error) => return Err(PlanarLcChartReplayError::Series(error)),
        }
    } else {
        None
    };
    let maximum_coefficient_residual = series
        .as_ref()
        .map(|replay| replay.maximum_absolute_residual_endpoint().clone());
    let maximum_pair_energy_constraint = series
        .as_ref()
        .map(|replay| replay.maximum_absolute_constraint_endpoint().clone());
    let recurrence = maximum_coefficient_residual
        .as_ref()
        .is_some_and(|maximum| maximum <= chart.coefficient_tolerance());
    let constraint = recurrence
        && maximum_pair_energy_constraint
            .as_ref()
            .is_some_and(|maximum| maximum <= chart.coefficient_tolerance());
    let residual_polynomials_formed = recurrence && finite_tolerances && series.is_some();
    let mut maximum_polynomial_residual = None;
    let mut polynomial_residual_with_claimed_tail = None;
    let mut interval_regularized_residual = false;
    let nonnegative_tail = if tail_admissible {
        chart.tail_bound().clone()
    } else {
        BigRational::zero()
    };
    if residual_polynomials_formed {
        let residuals = series
            .as_ref()
            .ok_or(PlanarLcChartReplayError::InternalShapeInvariant)?
            .residual_coefficients();
        let maximum = maximum_interval_polynomial_rows(residuals, chart.parameter_interval())?;
        let with_tail = &maximum + &nonnegative_tail;
        interval_regularized_residual = with_tail <= *chart.regularized_residual_tolerance();
        maximum_polynomial_residual = Some(maximum);
        polynomial_residual_with_claimed_tail = Some(with_tail);
    }

    let mut maximum_projection_residual = None;
    let mut projection_residual_with_claimed_tail = None;
    let mut interval_projection_residual = false;
    if finite_tolerances && recurrence {
        let state =
            PlanarLcStatePolynomial::from_chart(chart)?.evaluate(chart.parameter_interval())?;
        match replay_planar_lc_projection_default(chart, &state) {
            Ok(replay) => {
                let maximum = replay.maximum_absolute_residual_endpoint().clone();
                let with_tail = &maximum + &nonnegative_tail;
                interval_projection_residual = with_tail <= *chart.projected_residual_tolerance();
                maximum_projection_residual = Some(maximum);
                projection_residual_with_claimed_tail = Some(with_tail);
            }
            Err(error) if projection_domain_failure(&error) => {}
            Err(error) => return Err(PlanarLcChartReplayError::Projection(error)),
        }
    }

    let satisfied = [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        finite_tolerances,
        physical_time_contained,
        recurrence,
        constraint,
        residual_polynomials_formed,
        interval_regularized_residual,
        interval_projection_residual,
        tail_admissible,
    ];
    Ok(PlanarLcChartReplay {
        obligations: std::array::from_fn(|index| PlanarLcChartObligation {
            id: PLANAR_LC_CHART_OBLIGATION_IDS[index],
            satisfied: satisfied[index],
        }),
        physical_time_enclosure,
        maximum_coefficient_residual,
        maximum_pair_energy_constraint,
        maximum_polynomial_residual,
        polynomial_residual_with_claimed_tail,
        maximum_projection_residual,
        projection_residual_with_claimed_tail,
    })
}

fn physical_time_enclosure(
    chart: &PlanarLcChartInput,
    tail_admissible: bool,
) -> Result<(RationalInterval, bool), PlanarLcChartReplayError> {
    let parameter = chart.parameter_interval();
    let width = parameter.width();
    let subdivisions =
        BigRational::from_integer(BigInt::from(PLANAR_LC_CHART_PHYSICAL_TIME_SUBDIVISIONS));
    let step = width / subdivisions;
    let mut hull: Option<RationalInterval> = None;
    for index in 0..PLANAR_LC_CHART_PHYSICAL_TIME_SUBDIVISIONS {
        let left = parameter.lower() + &step * BigRational::from_integer(BigInt::from(index));
        let right = parameter.lower() + &step * BigRational::from_integer(BigInt::from(index + 1));
        let piece = RationalInterval::new(left, right)?;
        let value = scalar_polynomial_horner(
            chart.physical_time_polynomial().coefficients_by_degree(),
            &piece,
        )?;
        hull = Some(match hull {
            Some(current) => RationalInterval::new(
                current.lower().min(value.lower()).clone(),
                current.upper().max(value.upper()).clone(),
            )?,
            None => value,
        });
    }
    let hull = hull.ok_or(PlanarLcChartReplayError::InternalShapeInvariant)?;
    let declared = if tail_admissible {
        RationalInterval::new(
            chart.physical_time_interval().lower() - chart.tail_bound(),
            chart.physical_time_interval().upper() + chart.tail_bound(),
        )?
    } else {
        chart.physical_time_interval().clone()
    };
    let contained = tail_admissible && declared.contains_interval(&hull);
    Ok((hull, contained))
}

fn scalar_polynomial_horner(
    rows: &[Vec<BigRational>],
    argument: &RationalInterval,
) -> Result<RationalInterval, PlanarLcChartReplayError> {
    let highest = rows
        .last()
        .and_then(|row| row.first())
        .ok_or(PlanarLcChartReplayError::InternalShapeInvariant)?;
    if rows.iter().any(|row| row.len() != 1) {
        return Err(PlanarLcChartReplayError::InternalShapeInvariant);
    }
    let mut accumulator = RationalInterval::try_point(highest.clone())?;
    for row in rows[..rows.len() - 1].iter().rev() {
        accumulator = accumulator
            .multiply(argument)?
            .add(&RationalInterval::try_point(row[0].clone())?)?;
    }
    Ok(accumulator)
}

fn maximum_interval_polynomial_rows(
    rows: &[[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION]],
    argument: &RationalInterval,
) -> Result<BigRational, PlanarLcChartReplayError> {
    let highest = rows
        .last()
        .ok_or(PlanarLcChartReplayError::InternalShapeInvariant)?;
    let mut maximum = BigRational::zero();
    for component in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
        let mut accumulator = highest[component].clone();
        for row in rows[..rows.len() - 1].iter().rev() {
            accumulator = accumulator.multiply(argument)?.add(&row[component])?;
        }
        maximum = maximum.max(accumulator.lower().abs().max(accumulator.upper().abs()));
    }
    Ok(maximum)
}

fn series_domain_failure(error: &PlanarLcSeriesError) -> bool {
    matches!(
        error,
        PlanarLcSeriesError::ThirdBodyCollision { .. }
            | PlanarLcSeriesError::PositiveSquareRootUnresolved { .. }
    )
}

fn projection_domain_failure(error: &PlanarLcProjectionError) -> bool {
    matches!(
        error,
        PlanarLcProjectionError::NegativeProjectionFloor
            | PlanarLcProjectionError::RhoFloorNotStrict
            | PlanarLcProjectionError::NewtonCollision { .. }
            | PlanarLcProjectionError::PositiveSquareRootUnresolved { .. }
            | PlanarLcProjectionError::Field(crate::PlanarLcFieldError::ThirdBodyCollision { .. })
    )
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

    fn chart(wire: &PlanarLcChartWire) -> PlanarLcChartInput {
        planar_lc_chart_input_from_wire(wire).unwrap()
    }

    fn truth(replay: &PlanarLcChartReplay) -> Vec<bool> {
        replay
            .obligations()
            .iter()
            .map(PlanarLcChartObligation::satisfied)
            .collect()
    }

    #[test]
    fn all_canonical_charts_have_the_normative_ordered_all_true_ledger() {
        let chain = chain();
        let mut pairs = Vec::new();
        let mut count = 0;
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = chart(&segment.lc_chart);
            let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart).unwrap();
            assert_eq!(
                replay
                    .obligations()
                    .iter()
                    .map(PlanarLcChartObligation::id)
                    .collect::<Vec<_>>(),
                PLANAR_LC_CHART_OBLIGATION_IDS
            );
            assert_eq!(truth(&replay), vec![true; 15]);
            assert!(replay.conditional_profile_satisfied());
            pairs.push(chart.pair());
            count += 1;
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs, vec![[0, 1], [0, 2], [1, 2]]);
        assert_eq!(count, 4);
    }

    #[test]
    fn tightened_physical_time_interval_fails_only_containment() {
        let mut wire = first_wire();
        wire.physical_time_interval[1] = real("0.000001");
        let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&wire)).unwrap();
        let mut expected = vec![true; 15];
        expected[8] = false;
        assert_eq!(truth(&replay), expected);
    }

    #[test]
    fn coefficient_mutation_fails_recurrence_and_its_historical_dependents() {
        let mut wire = first_wire();
        wire.z_velocity_coefficients[1][0] = real("1.0");
        let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&wire)).unwrap();
        assert!(!replay.obligations()[9].satisfied());
        assert!(!replay.obligations()[10].satisfied());
        assert!(!replay.obligations()[11].satisfied());
        assert!(!replay.obligations()[12].satisfied());
        assert!(!replay.obligations()[13].satisfied());
    }

    #[test]
    fn negative_checker_claims_have_the_exact_dependency_patterns() {
        for field in 0..4 {
            let mut wire = first_wire();
            match field {
                0 => wire.coefficient_tolerance = real("-1.0"),
                1 => wire.regularized_residual_tolerance = real("-1.0"),
                2 => wire.projected_residual_tolerance = real("-1.0"),
                3 => wire.projection_rho_lower_bound = real("-1.0"),
                _ => unreachable!(),
            }
            let replay =
                replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&wire)).unwrap();
            assert!(!replay.obligations()[7].satisfied());
            for index in 9..=13 {
                assert!(!replay.obligations()[index].satisfied());
            }
            assert!(replay.maximum_coefficient_residual().is_none());
            assert!(replay.maximum_pair_energy_constraint().is_none());
            assert!(replay.obligations()[14].satisfied());
        }
    }

    #[test]
    fn negative_tail_only_fails_time_containment_and_tail_while_residuals_clamp_zero() {
        let mut wire = first_wire();
        wire.tail_bound = real("-1.0");
        let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&wire)).unwrap();
        let mut expected = vec![true; 15];
        expected[8] = false;
        expected[14] = false;
        assert_eq!(truth(&replay), expected);
        assert_eq!(
            replay.polynomial_residual_with_claimed_tail(),
            replay.maximum_polynomial_residual()
        );
        assert_eq!(
            replay.projection_residual_with_claimed_tail(),
            replay.maximum_projection_residual()
        );
    }

    #[test]
    fn rho_floor_domain_failure_only_falsifies_projection_obligation() {
        let mut wire = first_wire();
        wire.projection_rho_lower_bound = real("2.0");
        let admitted = chart(&wire);
        let state = PlanarLcStatePolynomial::from_chart(&admitted)
            .unwrap()
            .evaluate(admitted.parameter_interval())
            .unwrap();
        assert!(state.rho().unwrap().lower() <= &BigRational::from_integer(BigInt::from(2)));
        let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&admitted).unwrap();
        let mut expected = vec![true; 15];
        expected[13] = false;
        assert_eq!(truth(&replay), expected);
    }

    #[test]
    fn subdivision_tightens_time_and_diagnostics_recompute_independently() {
        let mut time_wire = first_wire();
        time_wire.parameter_interval = [real("0.0"), real("1.0")];
        time_wire.physical_time_interval = [real("0.0"), real("0.3")];
        time_wire.tail_bound = real("0.0");
        for coefficient in &mut time_wire.physical_time_coefficients {
            *coefficient = real("0.0");
        }
        time_wire.physical_time_coefficients[1] = real("1.0");
        time_wire.physical_time_coefficients[2] = real("-1.0");
        let time_chart = chart(&time_wire);
        let subdivided = physical_time_enclosure(&time_chart, true).unwrap().0;
        let unsplit = scalar_polynomial_horner(
            time_chart
                .physical_time_polynomial()
                .coefficients_by_degree(),
            time_chart.parameter_interval(),
        )
        .unwrap();
        assert!(subdivided.width() < unsplit.width());
        assert!(!time_chart
            .physical_time_interval()
            .contains_interval(&unsplit));
        let time_replay =
            replay_planar_lc_chart_exact_rational_claimed_tail_v04(&time_chart).unwrap();
        assert!(time_replay.obligations()[8].satisfied());
        let chart = chart(&first_wire());
        let replay = replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart).unwrap();
        let series = replay_planar_lc_interval_series_default(&chart).unwrap();
        assert_eq!(
            replay.maximum_coefficient_residual(),
            Some(series.maximum_absolute_residual_endpoint())
        );
        assert_eq!(
            replay.maximum_pair_energy_constraint(),
            Some(series.maximum_absolute_constraint_endpoint())
        );
        let polynomial_maximum = maximum_interval_polynomial_rows(
            series.residual_coefficients(),
            chart.parameter_interval(),
        )
        .unwrap();
        let polynomial_with_tail = &polynomial_maximum + chart.tail_bound();
        assert_eq!(
            replay.maximum_polynomial_residual(),
            Some(&polynomial_maximum)
        );
        assert_eq!(
            replay.polynomial_residual_with_claimed_tail(),
            Some(&polynomial_with_tail)
        );
        let state = PlanarLcStatePolynomial::from_chart(&chart)
            .unwrap()
            .evaluate(chart.parameter_interval())
            .unwrap();
        let projection = replay_planar_lc_projection_default(&chart, &state).unwrap();
        assert_eq!(
            replay.maximum_projection_residual(),
            Some(projection.maximum_absolute_residual_endpoint())
        );
    }

    #[test]
    fn sample_and_source_are_diagnostics_only_but_series_work_failure_is_typed() {
        let wire = first_wire();
        let baseline =
            replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&wire)).unwrap();
        let mut metadata = wire.clone();
        metadata.source = "changed-but-nonempty".to_string();
        metadata.sample_count = crate::raw_schema::RawInteger::from_parsed_for_test(
            parse_json_number_lexeme("6", DEFAULT_JSON_NUMBER_LIMITS).unwrap(),
        );
        assert_eq!(
            replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&metadata)).unwrap(),
            baseline
        );
        let mut oversized = wire;
        let count = 91;
        let zero = real("0.0");
        oversized.z_coefficients = vec![vec![zero.clone(), zero.clone()]; count];
        oversized.z_velocity_coefficients = vec![vec![zero.clone(), zero.clone()]; count];
        oversized.pair_energy_coefficients = vec![zero.clone(); count];
        oversized.binary_center_coefficients = vec![vec![zero.clone(), zero.clone()]; count];
        oversized.binary_center_velocity_coefficients =
            vec![vec![zero.clone(), zero.clone()]; count];
        oversized.third_offset_coefficients = vec![vec![zero.clone(), zero.clone()]; count];
        oversized.third_offset_velocity_coefficients =
            vec![vec![zero.clone(), zero.clone()]; count];
        oversized.physical_time_coefficients = vec![zero; count];
        assert!(matches!(
            replay_planar_lc_chart_exact_rational_claimed_tail_v04(&chart(&oversized)),
            Err(PlanarLcChartReplayError::Series(
                PlanarLcSeriesError::WorkLimitExceeded { .. }
            ))
        ));
    }
}
