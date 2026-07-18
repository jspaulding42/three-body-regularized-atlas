//! Conditional exact-rational planar LC a-posteriori tube replay.
//!
//! This profile proves only its nine ordered tube obligations. It does not
//! establish IVP containment, lift/entry validity, frozen binary64 parity, or
//! a continuation-chain step.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::{
    evaluate_planar_lc_field, evaluate_planar_lc_polynomial_defect_default, exp_enclosure_rational,
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError},
    rational_input::validate_public_rational,
    ExactBinary64, ExpEnclosureError, NumericError, PlanarLcChartInput, PlanarLcFieldError,
    PlanarLcPolynomialDefectError, PlanarLcStateError, PlanarLcStatePolynomial, PlanarLcTubeInput,
    RationalInterval, DEFAULT_EXP_ENCLOSURE_LIMITS,
};

pub const EXACT_RATIONAL_PLANAR_LC_TUBE_V04_PROFILE_ID: &str = "exact_rational_planar_lc_tube_v04";

pub const PLANAR_LC_TUBE_OBLIGATION_IDS: [&str; 9] = [
    "planar_lc_tube_identity_matches_chart",
    "planar_lc_tube_lifted_serialization_admissible",
    "planar_lc_tube_outward_mass_arithmetic_certified",
    "planar_lc_tube_inputs_finite",
    "planar_lc_tube_direct_lifted_defect_within_cap",
    "planar_lc_tube_separated_third_body",
    "planar_lc_tube_interval_jacobian_within_cap",
    "planar_lc_tube_pair_energy_constraint_when_required",
    "planar_lc_tube_gronwall_self_consistent",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRationalPlanarLcTubeV04;

impl ExactRationalPlanarLcTubeV04 {
    pub const EXPONENT_UPPER_PRECISION_BITS: usize = 32;
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
pub struct PlanarLcTubeObligation {
    id: &'static str,
    satisfied: bool,
}

impl PlanarLcTubeObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcTubeReplayError {
    State(PlanarLcStateError),
    Defect(PlanarLcPolynomialDefectError),
    Field(PlanarLcFieldError),
    MassDecode { index: usize, source: NumericError },
    MassProfile(MassProfileError),
    Exponential(ExpEnclosureError),
    InternalShapeInvariant,
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcTubeReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC tube replay failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcTubeReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::Defect(source) => Some(source),
            Self::Field(source) => Some(source),
            Self::MassDecode { source, .. } => Some(source),
            Self::MassProfile(source) => Some(source),
            Self::Exponential(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::InternalShapeInvariant => None,
        }
    }
}

impl From<PlanarLcStateError> for PlanarLcTubeReplayError {
    fn from(source: PlanarLcStateError) -> Self {
        Self::State(source)
    }
}

impl From<NumericError> for PlanarLcTubeReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

impl From<MassProfileError> for PlanarLcTubeReplayError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

impl From<ExpEnclosureError> for PlanarLcTubeReplayError {
    fn from(source: ExpEnclosureError) -> Self {
        Self::Exponential(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcTubeReplay {
    obligations: [PlanarLcTubeObligation; 9],
    defect_upper: Option<BigRational>,
    lipschitz_upper: Option<BigRational>,
    third_body_squared_distance_floors: Option<[BigRational; 2]>,
    horizon: Option<BigRational>,
    exponential_argument_upper: Option<BigRational>,
    exponential_upper: Option<BigRational>,
    gronwall_upper: Option<BigRational>,
    pair_energy_constraint_residual: Option<BigRational>,
}

impl PlanarLcTubeReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PLANAR_LC_TUBE_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[PlanarLcTubeObligation; 9] {
        &self.obligations
    }

    /// Conditional tube certification only. No IVP containment, lift, entry,
    /// exit, or chain claim follows from this Boolean.
    pub fn certified(&self) -> bool {
        self.obligations
            .iter()
            .all(PlanarLcTubeObligation::satisfied)
    }

    pub fn defect_upper(&self) -> Option<&BigRational> {
        self.defect_upper.as_ref()
    }

    pub fn lipschitz_upper(&self) -> Option<&BigRational> {
        self.lipschitz_upper.as_ref()
    }

    pub fn third_body_squared_distance_floors(&self) -> Option<&[BigRational; 2]> {
        self.third_body_squared_distance_floors.as_ref()
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

    pub fn pair_energy_constraint_residual(&self) -> Option<&BigRational> {
        self.pair_energy_constraint_residual.as_ref()
    }
}

pub fn replay_planar_lc_tube_exact_rational_v04(
    chart: &PlanarLcChartInput,
    tube: &PlanarLcTubeInput,
) -> Result<PlanarLcTubeReplay, PlanarLcTubeReplayError> {
    let identity = !tube.tube_id().is_empty() && tube.chart_id() == chart.chart_id();
    let polynomial = PlanarLcStatePolynomial::from_chart(chart)?;
    let serialized = true;
    let finite_inputs = !tube.tube_id().is_empty()
        && !tube.chart_id().is_empty()
        && !tube.source().is_empty()
        && tube.anchor_parameter() >= chart.parameter_interval().lower()
        && tube.anchor_parameter() <= chart.parameter_interval().upper()
        && tube.initial_error_bound().numer().sign() != Sign::Minus
        && tube.tube_radius().numer().sign() == Sign::Plus
        && tube.maximum_defect_bound().numer().sign() != Sign::Minus
        && tube.maximum_lipschitz_bound().numer().sign() != Sign::Minus;

    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;
    let mass_certified = true;
    if !(identity && serialized && finite_inputs) {
        return Ok(build_replay(
            [
                identity,
                serialized,
                mass_certified,
                finite_inputs,
                false,
                false,
                false,
                false,
                false,
            ],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
    }

    let horizon = exact_horizon(chart.parameter_interval(), tube.anchor_parameter())?;
    let constraint_residual = if tube.require_pair_energy_constraint() {
        let anchor = RationalInterval::try_point(tube.anchor_parameter().clone())?;
        let anchor_state = polynomial.evaluate(&anchor)?;
        let constraint = anchor_state.pair_energy_constraint(mass_profile.pair_mass().exact())?;
        if !constraint.is_point() {
            return Err(PlanarLcTubeReplayError::InternalShapeInvariant);
        }
        Some(constraint.lower().abs())
    } else {
        None
    };
    let constraint_satisfied = !tube.require_pair_energy_constraint()
        || constraint_residual
            .as_ref()
            .is_some_and(BigRational::is_zero);
    let defect = match evaluate_planar_lc_polynomial_defect_default(chart) {
        Ok(defect) => defect,
        Err(error) if defect_domain_failure(&error) => {
            return Ok(build_replay(
                [
                    true,
                    true,
                    true,
                    true,
                    false,
                    false,
                    false,
                    constraint_satisfied,
                    false,
                ],
                None,
                None,
                None,
                Some(horizon),
                None,
                None,
                None,
                constraint_residual,
            ));
        }
        Err(error) => return Err(PlanarLcTubeReplayError::Defect(error)),
    };
    let delta = defect.maximum_absolute_endpoint_bound().clone();
    let defect_within_cap = within_cap(&delta, tube.maximum_defect_bound());
    let state = polynomial.evaluate(chart.parameter_interval())?;
    let inflated = state.inflate(tube.tube_radius())?;
    let field = match evaluate_planar_lc_field(
        chart,
        &inflated,
        crate::PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS,
    ) {
        Ok(field) => field,
        Err(PlanarLcFieldError::ThirdBodyCollision { .. }) => {
            return Ok(build_replay(
                [
                    true,
                    true,
                    true,
                    true,
                    defect_within_cap,
                    false,
                    false,
                    constraint_satisfied,
                    false,
                ],
                Some(delta),
                None,
                None,
                Some(horizon),
                None,
                None,
                None,
                constraint_residual,
            ));
        }
        Err(error) => return Err(PlanarLcTubeReplayError::Field(error)),
    };
    let squared_distance_floors = [
        field.denominator_squares()[0].lower().clone(),
        field.denominator_squares()[1].lower().clone(),
    ];
    let separated = squared_distance_floors
        .iter()
        .all(|floor| floor.numer().sign() == Sign::Plus);
    let lipschitz_upper = field.lipschitz_infinity_row_sum_upper().clone();
    let lipschitz_within_cap =
        separated && within_cap(&lipschitz_upper, tube.maximum_lipschitz_bound());

    let mut exponential_argument_upper = None;
    let mut exponential_upper = None;
    let mut gronwall = None;
    let mut gronwall_satisfied = false;
    if separated {
        let exact_exponent = checked_multiply(&lipschitz_upper, &horizon)?;
        let exponent_upper = dyadic_upper(
            &exact_exponent,
            ExactRationalPlanarLcTubeV04::EXPONENT_UPPER_PRECISION_BITS,
        )?;
        let exponential = exp_enclosure_rational(
            &exponent_upper,
            ExactRationalPlanarLcTubeV04::EXP_TAYLOR_CUTOFF,
            &ExactRationalPlanarLcTubeV04::maximum_reduced_exp_tail(),
            DEFAULT_EXP_ENCLOSURE_LIMITS,
        )?;
        let exp_upper = exponential.upper_bound().clone();
        let gronwall_upper = gronwall_upper(
            &exp_upper,
            tube.initial_error_bound(),
            &delta,
            &lipschitz_upper,
            &horizon,
        )?;
        gronwall_satisfied = strictly_inside_radius(&gronwall_upper, tube.tube_radius());
        exponential_argument_upper = Some(exponent_upper);
        exponential_upper = Some(exp_upper);
        gronwall = Some(gronwall_upper);
    }
    Ok(build_replay(
        [
            true,
            true,
            true,
            true,
            defect_within_cap,
            separated,
            lipschitz_within_cap,
            constraint_satisfied,
            gronwall_satisfied,
        ],
        Some(delta),
        Some(lipschitz_upper),
        Some(squared_distance_floors),
        Some(horizon),
        exponential_argument_upper,
        exponential_upper,
        gronwall,
        constraint_residual,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_replay(
    satisfied: [bool; 9],
    defect_upper: Option<BigRational>,
    lipschitz_upper: Option<BigRational>,
    third_body_squared_distance_floors: Option<[BigRational; 2]>,
    horizon: Option<BigRational>,
    exponential_argument_upper: Option<BigRational>,
    exponential_upper: Option<BigRational>,
    gronwall_upper: Option<BigRational>,
    pair_energy_constraint_residual: Option<BigRational>,
) -> PlanarLcTubeReplay {
    PlanarLcTubeReplay {
        obligations: std::array::from_fn(|index| PlanarLcTubeObligation {
            id: PLANAR_LC_TUBE_OBLIGATION_IDS[index],
            satisfied: satisfied[index],
        }),
        defect_upper,
        lipschitz_upper,
        third_body_squared_distance_floors,
        horizon,
        exponential_argument_upper,
        exponential_upper,
        gronwall_upper,
        pair_energy_constraint_residual,
    }
}

fn reconstruct_masses(
    chart: &PlanarLcChartInput,
) -> Result<[ExactBinary64; 3], PlanarLcTubeReplayError> {
    let mut masses = Vec::with_capacity(3);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        masses.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| PlanarLcTubeReplayError::MassDecode { index, source })?,
        );
    }
    masses
        .try_into()
        .map_err(|_| PlanarLcTubeReplayError::InternalShapeInvariant)
}

fn defect_domain_failure(error: &PlanarLcPolynomialDefectError) -> bool {
    matches!(
        error,
        PlanarLcPolynomialDefectError::Field(PlanarLcFieldError::ThirdBodyCollision { .. })
    )
}

fn strictly_inside_radius(bound: &BigRational, radius: &BigRational) -> bool {
    bound < radius
}

fn within_cap(value: &BigRational, cap: &BigRational) -> bool {
    value <= cap
}

fn exact_horizon(
    interval: &RationalInterval,
    anchor: &BigRational,
) -> Result<BigRational, PlanarLcTubeReplayError> {
    let left = (interval.lower() - anchor).abs();
    let right = (interval.upper() - anchor).abs();
    admit_rational(left.max(right))
}

fn dyadic_upper(
    value: &BigRational,
    precision_bits: usize,
) -> Result<BigRational, PlanarLcTubeReplayError> {
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
    horizon: &BigRational,
) -> Result<BigRational, PlanarLcTubeReplayError> {
    if lipschitz_upper.is_zero() {
        return checked_add(initial_error, &checked_multiply(defect_upper, horizon)?);
    }
    let propagated_initial = checked_multiply(exponential_upper, initial_error)?;
    let increment = exponential_upper - BigRational::one();
    let propagated_defect = checked_divide(
        &checked_multiply(defect_upper, &increment)?,
        lipschitz_upper,
    )?;
    checked_add(&propagated_initial, &propagated_defect)
}

fn checked_add(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, PlanarLcTubeReplayError> {
    admit_rational(left + right)
}

fn checked_multiply(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, PlanarLcTubeReplayError> {
    admit_rational(left * right)
}

fn checked_divide(
    numerator: &BigRational,
    denominator: &BigRational,
) -> Result<BigRational, PlanarLcTubeReplayError> {
    if denominator.is_zero() {
        return Err(NumericError::DivisionByZeroInterval.into());
    }
    admit_rational(numerator / denominator)
}

fn admit_rational(value: BigRational) -> Result<BigRational, PlanarLcTubeReplayError> {
    validate_public_rational(&value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checked_real_binary64_from_json, parse_json_number_lexeme, parse_wire_json,
        planar_lc_chart_input_from_wire, planar_lc_tube_input_from_wire,
        raw_schema::{
            decode_raw_chain, PlanarLcChartWire, PlanarLcPassageSegmentWire, PlanarLcTubeWire,
            RawPlanarChainWire, SchemaProfile, SegmentWire,
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

    fn real(value: &str) -> crate::raw_schema::Real {
        checked_real_binary64_from_json(value, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn parse_chain(bytes: &[u8]) -> RawPlanarChainWire {
        let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
    }

    fn first_passage() -> PlanarLcPassageSegmentWire {
        parse_chain(SUCCESS_RAW)
            .segments
            .into_iter()
            .find_map(|segment| match segment {
                SegmentWire::PlanarLcPassage(passage) => Some(*passage),
                SegmentWire::OrdinaryBridge(_) => None,
            })
            .unwrap()
    }

    fn replay_wires(
        chart_wire: &PlanarLcChartWire,
        tube_wire: &PlanarLcTubeWire,
    ) -> PlanarLcTubeReplay {
        let chart = planar_lc_chart_input_from_wire(chart_wire).unwrap();
        let tube = planar_lc_tube_input_from_wire(tube_wire, &chart).unwrap();
        replay_planar_lc_tube_exact_rational_v04(&chart, &tube).unwrap()
    }

    fn truth(replay: &PlanarLcTubeReplay) -> [bool; 9] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    fn set_vector_series_zero(series: &mut Vec<Vec<crate::raw_schema::Real>>) {
        let count = series.len();
        *series = vec![vec![real("0.0"), real("0.0")]; count];
    }

    fn set_scalar_series_zero(series: &mut Vec<crate::raw_schema::Real>) {
        let count = series.len();
        *series = vec![real("0.0"); count];
    }

    fn simple_wires() -> (PlanarLcChartWire, PlanarLcTubeWire) {
        let passage = first_passage();
        let mut chart = passage.lc_chart;
        chart.masses = vec![real("1.0"), real("1.0"), real("1.0")];
        set_vector_series_zero(&mut chart.z_coefficients);
        set_vector_series_zero(&mut chart.z_velocity_coefficients);
        set_scalar_series_zero(&mut chart.pair_energy_coefficients);
        set_vector_series_zero(&mut chart.binary_center_coefficients);
        set_vector_series_zero(&mut chart.binary_center_velocity_coefficients);
        set_vector_series_zero(&mut chart.third_offset_coefficients);
        set_vector_series_zero(&mut chart.third_offset_velocity_coefficients);
        set_scalar_series_zero(&mut chart.physical_time_coefficients);
        chart.z_coefficients[0][0] = real("1.0");
        chart.pair_energy_coefficients[0] = real("-2.0");
        chart.third_offset_coefficients[0][0] = real("3.0");
        let mut tube = passage.lc_tube;
        tube.chart_id = chart.chart_id.clone();
        tube.initial_error_bound = real("0.0");
        tube.tube_radius = real("0.000001");
        tube.max_defect_bound = real("1e100");
        tube.max_lipschitz_bound = real("1e100");
        (chart, tube)
    }

    #[test]
    fn canonical_tubes_from_both_reference_chains_have_the_normative_ledger() {
        let mut count = 0;
        for bytes in [SUCCESS_RAW, FAILED_REVISIT_RAW] {
            for segment in parse_chain(bytes).segments {
                let SegmentWire::PlanarLcPassage(passage) = segment else {
                    continue;
                };
                let replay = replay_wires(&passage.lc_chart, &passage.lc_tube);
                assert_eq!(
                    replay
                        .obligations()
                        .iter()
                        .map(PlanarLcTubeObligation::id)
                        .collect::<Vec<_>>(),
                    PLANAR_LC_TUBE_OBLIGATION_IDS
                );
                assert_eq!(truth(&replay), [true; 9]);
                assert!(replay.certified());
                count += 1;
            }
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn pair_energy_constraint_is_checked_exactly_and_independently() {
        let (chart, mut tube) = simple_wires();
        tube.require_pair_energy_constraint = true;
        let passing = replay_wires(&chart, &tube);
        assert!(passing.obligations()[7].satisfied());
        assert_eq!(
            passing.pair_energy_constraint_residual(),
            Some(&BigRational::zero())
        );

        let mut failing_chart = chart;
        failing_chart.pair_energy_coefficients[0] = real("-1.0");
        let failing = replay_wires(&failing_chart, &tube);
        assert!(!failing.obligations()[7].satisfied());
        assert_eq!(
            failing.pair_energy_constraint_residual(),
            Some(&BigRational::one())
        );
    }

    #[test]
    fn third_body_collision_keeps_disabled_constraint_true_only_in_its_row() {
        let (mut chart, mut tube) = simple_wires();
        chart.third_offset_coefficients[0][0] = real("-0.5");
        tube.require_pair_energy_constraint = false;
        let replay = replay_wires(&chart, &tube);
        assert_eq!(
            truth(&replay),
            [true, true, true, true, false, false, false, true, false]
        );
        assert_eq!(replay.pair_energy_constraint_residual(), None);
    }

    #[test]
    fn unconsumed_chart_claims_do_not_change_tube_replay() {
        let (chart, tube) = simple_wires();
        let baseline = replay_wires(&chart, &tube);
        let mut claims = chart;
        claims.coefficient_tolerance = real("7.0");
        claims.regularized_residual_tolerance = real("8.0");
        claims.projected_residual_tolerance = real("9.0");
        claims.tail_bound = real("10.0");
        claims.projection_rho_lower_bound = real("11.0");
        claims.sample_count = crate::raw_schema::RawInteger::from_parsed_for_test(
            parse_json_number_lexeme("6", DEFAULT_JSON_NUMBER_LIMITS).unwrap(),
        );
        claims.source = "ignored-source-claim".into();
        assert_eq!(replay_wires(&claims, &tube), baseline);
        let mut tube_claim = tube;
        tube_claim.source = "also-ignored-tube-source".into();
        assert_eq!(replay_wires(&claims, &tube_claim), baseline);
    }

    #[test]
    fn diagnostics_equal_independent_direct_recomputations() {
        let (chart_wire, tube_wire) = simple_wires();
        let chart = planar_lc_chart_input_from_wire(&chart_wire).unwrap();
        let tube = planar_lc_tube_input_from_wire(&tube_wire, &chart).unwrap();
        let replay = replay_planar_lc_tube_exact_rational_v04(&chart, &tube).unwrap();

        let defect = evaluate_planar_lc_polynomial_defect_default(&chart).unwrap();
        assert_eq!(
            replay.defect_upper(),
            Some(defect.maximum_absolute_endpoint_bound())
        );
        let polynomial = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
        let state = polynomial.evaluate(chart.parameter_interval()).unwrap();
        let inflated = state.inflate(tube.tube_radius()).unwrap();
        let field = evaluate_planar_lc_field(
            &chart,
            &inflated,
            crate::PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS,
        )
        .unwrap();
        assert_eq!(
            replay.lipschitz_upper(),
            Some(field.lipschitz_infinity_row_sum_upper())
        );
        assert_eq!(
            replay.third_body_squared_distance_floors(),
            Some(&[
                field.denominator_squares()[0].lower().clone(),
                field.denominator_squares()[1].lower().clone(),
            ])
        );
        assert_eq!(
            replay.horizon(),
            Some(&exact_horizon(chart.parameter_interval(), tube.anchor_parameter()).unwrap())
        );
    }

    #[test]
    fn defect_and_lipschitz_caps_are_inclusive_and_independent() {
        let (chart, tube) = simple_wires();
        let baseline = replay_wires(&chart, &tube);
        assert!(baseline.obligations()[4].satisfied());
        assert!(baseline.obligations()[6].satisfied());

        let mut zero_defect = tube.clone();
        zero_defect.max_defect_bound = real("0.0");
        let defect_replay = replay_wires(&chart, &zero_defect);
        assert!(!defect_replay.obligations()[4].satisfied());
        assert!(defect_replay.obligations()[6].satisfied());

        let mut zero_lipschitz = tube;
        zero_lipschitz.max_lipschitz_bound = real("0.0");
        let lipschitz_replay = replay_wires(&chart, &zero_lipschitz);
        assert!(lipschitz_replay.obligations()[4].satisfied());
        assert!(!lipschitz_replay.obligations()[6].satisfied());

        assert!(within_cap(
            baseline.defect_upper().unwrap(),
            baseline.defect_upper().unwrap()
        ));
        assert!(within_cap(
            baseline.lipschitz_upper().unwrap(),
            baseline.lipschitz_upper().unwrap()
        ));
    }

    #[test]
    fn gronwall_zero_lipschitz_limit_and_strict_radius_boundary_are_pinned() {
        let e0 = BigRational::new(BigInt::from(1), BigInt::from(10));
        let delta = BigRational::new(BigInt::from(1), BigInt::from(5));
        let horizon = BigRational::new(BigInt::from(3), BigInt::from(2));
        let bound = gronwall_upper(
            &BigRational::one(),
            &e0,
            &delta,
            &BigRational::zero(),
            &horizon,
        )
        .unwrap();
        assert_eq!(bound, &e0 + &delta * &horizon);
        assert!(!strictly_inside_radius(&bound, &bound));
        assert!(strictly_inside_radius(
            &bound,
            &(&bound + BigRational::new(BigInt::one(), BigInt::from(1_000_000)))
        ));
    }
}
