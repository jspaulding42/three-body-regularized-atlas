//! Exact initial-value binding and proof-oriented ordinary-root replay.
//!
//! The binding profile evaluates the admitted ordinary chart at the raw root
//! parameter with exact rational arithmetic.  The validated-root profile then
//! composes that replay with the a-posteriori ordinary-tube replay.  It does
//! not consume the chart's claimed Taylor tail: the tube defect, separation,
//! Lipschitz, and Gronwall obligations are the proof layer used here.

use core::fmt;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    ordinary_semantic::{OrdinaryChartInput, OrdinaryTubeInput},
    ordinary_tube::{
        replay_ordinary_tube_exact_rational_v04, OrdinaryTubeReplay, OrdinaryTubeReplayError,
    },
    rational_input::validate_public_rational,
    raw_schema::RootBindingWire,
    NumericError, PolynomialError, RationalInterval,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const CONFIGURATION_DIMENSION: usize = BODY_COUNT * PLANE_DIMENSION;

pub const EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID: &str =
    "exact_rational_initial_value_binding_v04";

pub const INITIAL_VALUE_BINDING_OBLIGATION_IDS: [&str; 8] = [
    "initial_value_binding_identity_present",
    "initial_value_binding_chart_present",
    "initial_value_problem_finite_positive_mass_state",
    "initial_value_binding_tolerances_finite",
    "initial_value_binding_chart_masses_match",
    "initial_value_binding_parameter_inside_chart",
    "initial_value_binding_state_shape_matches",
    "initial_value_binding_polynomial_state_matches",
];

pub const EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID: &str =
    "exact_rational_validated_ordinary_root_v04";

pub const VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS: [&str; 8] = [
    "validated_ordinary_chart_serialization_admissible",
    "validated_ordinary_chart_exact_unit_speed",
    "validated_ordinary_root_exact_time_anchor",
    "validated_ordinary_ivp_binding_checked",
    "validated_ordinary_tube_checked",
    "validated_ordinary_component_chart_ids_match",
    "validated_ordinary_anchor_parameter_matches_binding",
    "validated_ordinary_actual_initial_error_covered",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitialValueBindingObligation {
    id: &'static str,
    satisfied: bool,
}

impl InitialValueBindingObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Ordered binding ledger and exact discrepancies at the requested root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitialValueBindingReplay {
    obligations: [InitialValueBindingObligation; 8],
    time_gap: Option<BigRational>,
    maximum_position_gap: Option<BigRational>,
    maximum_velocity_gap: Option<BigRational>,
}

impl InitialValueBindingReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[InitialValueBindingObligation; 8] {
        &self.obligations
    }

    pub fn profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(|obligation| obligation.satisfied)
    }

    pub fn time_gap(&self) -> Option<&BigRational> {
        self.time_gap.as_ref()
    }

    pub fn maximum_position_gap(&self) -> Option<&BigRational> {
        self.maximum_position_gap.as_ref()
    }

    pub fn maximum_velocity_gap(&self) -> Option<&BigRational> {
        self.maximum_velocity_gap.as_ref()
    }
}

/// Kernel/resource failures are distinct from sound false obligations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InitialValueBindingReplayError {
    Numeric(NumericError),
    PositionPolynomial(PolynomialError),
    VelocityPolynomial(PolynomialError),
    InternalInvariant { detail: &'static str },
}

impl fmt::Display for InitialValueBindingReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => {
                write!(formatter, "initial-value binding numeric failure: {source}")
            }
            Self::PositionPolynomial(source) => {
                write!(
                    formatter,
                    "initial-value position polynomial failure: {source}"
                )
            }
            Self::VelocityPolynomial(source) => {
                write!(
                    formatter,
                    "initial-value velocity polynomial failure: {source}"
                )
            }
            Self::InternalInvariant { detail } => {
                write!(
                    formatter,
                    "initial-value binding internal invariant failed: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for InitialValueBindingReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::PositionPolynomial(source) | Self::VelocityPolynomial(source) => Some(source),
            Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for InitialValueBindingReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedOrdinaryRootObligation {
    id: &'static str,
    satisfied: bool,
}

impl ValidatedOrdinaryRootObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Ordered root ledger and the exact component replays it composes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedOrdinaryRootReplay {
    obligations: [ValidatedOrdinaryRootObligation; 8],
    binding_replay: InitialValueBindingReplay,
    tube_replay: OrdinaryTubeReplay,
    actual_initial_error: Option<BigRational>,
    root_clock_origin: Option<BigRational>,
}

impl ValidatedOrdinaryRootReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[ValidatedOrdinaryRootObligation; 8] {
        &self.obligations
    }

    pub fn validated_root_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(|obligation| obligation.satisfied)
    }

    pub fn binding_replay(&self) -> &InitialValueBindingReplay {
        &self.binding_replay
    }

    pub fn tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.tube_replay
    }

    pub fn actual_initial_error(&self) -> Option<&BigRational> {
        self.actual_initial_error.as_ref()
    }

    /// Exact additive clock origin `initial_time - chart_parameter`, retained
    /// only after the chart's affine physical-time map matches the root time.
    pub fn root_clock_origin(&self) -> Option<&BigRational> {
        self.root_clock_origin.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidatedOrdinaryRootReplayError {
    Binding(InitialValueBindingReplayError),
    Tube(OrdinaryTubeReplayError),
}

impl fmt::Display for ValidatedOrdinaryRootReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(source) => write!(formatter, "validated-root binding failure: {source}"),
            Self::Tube(source) => write!(formatter, "validated-root tube failure: {source}"),
        }
    }
}

impl std::error::Error for ValidatedOrdinaryRootReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Binding(source) => Some(source),
            Self::Tube(source) => Some(source),
        }
    }
}

impl From<InitialValueBindingReplayError> for ValidatedOrdinaryRootReplayError {
    fn from(source: InitialValueBindingReplayError) -> Self {
        Self::Binding(source)
    }
}

impl From<OrdinaryTubeReplayError> for ValidatedOrdinaryRootReplayError {
    fn from(source: OrdinaryTubeReplayError) -> Self {
        Self::Tube(source)
    }
}

/// Replay the eight initial-value binding obligations with exact dyadic input
/// values and exact rational polynomial evaluation.
pub fn replay_initial_value_binding_exact_rational_v04(
    binding: &RootBindingWire,
    chart: &OrdinaryChartInput,
) -> Result<InitialValueBindingReplay, InitialValueBindingReplayError> {
    let identity_present = !binding.binding_id.is_empty() && !binding.chart_id.is_empty();
    let chart_present = binding.chart_id == chart.chart_id();
    let planar_shape = planar_state_shape_matches(binding);
    let positive_mass_state = planar_shape
        && binding.masses.len() == BODY_COUNT
        && binding
            .masses
            .iter()
            .all(|mass| mass.binary64_rational().numer().sign() == Sign::Plus);
    let tolerances_finite = [
        &binding.time_tolerance,
        &binding.position_tolerance,
        &binding.velocity_tolerance,
    ]
    .into_iter()
    .all(|tolerance| tolerance.binary64_rational().numer().sign() != Sign::Minus);
    let chart_masses_match = chart_present
        && binding.masses.len() == BODY_COUNT
        && binding
            .masses
            .iter()
            .zip(chart.masses())
            .all(|(binding_mass, chart_mass)| binding_mass.binary64_rational() == chart_mass);
    let parameter = binding.chart_parameter.binary64_rational();
    let parameter_inside_chart = chart_present
        && chart.parameter_interval().lower() <= parameter
        && parameter <= chart.parameter_interval().upper();
    let state_shape_matches = chart_present
        && positive_mass_state
        && tolerances_finite
        && chart_masses_match
        && parameter_inside_chart;

    if !state_shape_matches {
        return Ok(build_binding_replay(
            [
                identity_present,
                chart_present,
                positive_mass_state,
                tolerances_finite,
                chart_masses_match,
                parameter_inside_chart,
                state_shape_matches,
                false,
            ],
            None,
            None,
            None,
        ));
    }

    let chart_time = exact_affine_physical_time(chart, parameter)?;
    let time_gap =
        checked_absolute_difference(binding.initial_time.binary64_rational(), &chart_time)?;
    let argument = RationalInterval::try_point(parameter.clone())?;
    let positions = chart
        .position_polynomial()
        .evaluate(&argument)
        .map_err(InitialValueBindingReplayError::PositionPolynomial)?;
    let velocities = chart
        .velocity_polynomial()
        .evaluate(&argument)
        .map_err(InitialValueBindingReplayError::VelocityPolynomial)?;
    let maximum_position_gap = maximum_state_gap(&binding.positions, &positions)?;
    let maximum_velocity_gap = maximum_state_gap(&binding.velocities, &velocities)?;
    let polynomial_state_matches = time_gap <= *binding.time_tolerance.binary64_rational()
        && maximum_position_gap <= *binding.position_tolerance.binary64_rational()
        && maximum_velocity_gap <= *binding.velocity_tolerance.binary64_rational();

    Ok(build_binding_replay(
        [
            identity_present,
            chart_present,
            positive_mass_state,
            tolerances_finite,
            chart_masses_match,
            parameter_inside_chart,
            state_shape_matches,
            polynomial_state_matches,
        ],
        Some(time_gap),
        Some(maximum_position_gap),
        Some(maximum_velocity_gap),
    ))
}

/// Compose an admitted chart, exact IVP binding, and exact ordinary-tube
/// replay into the eight proof-oriented validated-root obligations.
pub fn replay_validated_ordinary_root_exact_rational_v04(
    binding: &RootBindingWire,
    chart: &OrdinaryChartInput,
    tube: &OrdinaryTubeInput,
) -> Result<ValidatedOrdinaryRootReplay, ValidatedOrdinaryRootReplayError> {
    let binding_replay = replay_initial_value_binding_exact_rational_v04(binding, chart)?;
    let tube_replay = replay_ordinary_tube_exact_rational_v04(chart, tube)?;

    // Construction of `OrdinaryChartInput` is the serialization-admission
    // witness.  This profile intentionally does not replay the claimed tail.
    let serialization_admissible = true;
    let exact_unit_speed =
        chart.parameter_interval().width() == chart.physical_time_interval().width();
    let exact_time_anchor = binding_replay.time_gap().is_some_and(BigRational::is_zero);
    let root_clock_origin = if exact_time_anchor && exact_unit_speed {
        Some(
            checked_subtract(
                binding.initial_time.binary64_rational(),
                binding.chart_parameter.binary64_rational(),
            )
            .map_err(ValidatedOrdinaryRootReplayError::Binding)?,
        )
    } else {
        None
    };
    let binding_checked = binding_replay.profile_satisfied();
    let tube_checked = tube_replay.certified();
    let component_chart_ids_match =
        binding.chart_id == chart.chart_id() && tube.chart_id() == chart.chart_id();
    let anchor_parameter_matches_binding =
        tube.anchor_parameter() == binding.chart_parameter.binary64_rational();
    let actual_initial_error = match (
        binding_replay.maximum_position_gap(),
        binding_replay.maximum_velocity_gap(),
    ) {
        (Some(position), Some(velocity)) => Some(std::cmp::max(position, velocity).clone()),
        _ => None,
    };
    let actual_initial_error_covered = actual_initial_error
        .as_ref()
        .is_some_and(|error| error <= tube.initial_error_bound());

    Ok(ValidatedOrdinaryRootReplay {
        obligations: std::array::from_fn(|index| ValidatedOrdinaryRootObligation {
            id: VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS[index],
            satisfied: [
                serialization_admissible,
                exact_unit_speed,
                exact_time_anchor,
                binding_checked,
                tube_checked,
                component_chart_ids_match,
                anchor_parameter_matches_binding,
                actual_initial_error_covered,
            ][index],
        }),
        binding_replay,
        tube_replay,
        actual_initial_error,
        root_clock_origin,
    })
}

fn planar_state_shape_matches(binding: &RootBindingWire) -> bool {
    binding.positions.len() == BODY_COUNT
        && binding.velocities.len() == BODY_COUNT
        && binding
            .positions
            .iter()
            .all(|body| body.len() == PLANE_DIMENSION)
        && binding
            .velocities
            .iter()
            .all(|body| body.len() == PLANE_DIMENSION)
}

fn exact_affine_physical_time(
    chart: &OrdinaryChartInput,
    parameter: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    let parameter_width = chart.parameter_interval().width();
    let physical_width = chart.physical_time_interval().width();
    let offset = checked_subtract(parameter, chart.parameter_interval().lower())?;
    let scaled = checked_multiply(&offset, &physical_width)?;
    let normalized = checked_divide(&scaled, &parameter_width)?;
    checked_add(chart.physical_time_interval().lower(), &normalized)
}

fn maximum_state_gap(
    state: &[Vec<crate::raw_schema::Real>],
    evaluated: &[RationalInterval],
) -> Result<BigRational, InitialValueBindingReplayError> {
    if evaluated.len() != CONFIGURATION_DIMENSION {
        return Err(InitialValueBindingReplayError::InternalInvariant {
            detail: "ordinary polynomial dimension is not planar three-body",
        });
    }
    if evaluated.iter().any(|value| !value.is_point()) {
        return Err(InitialValueBindingReplayError::InternalInvariant {
            detail: "point polynomial evaluation produced a non-point interval",
        });
    }
    let mut gap = BigRational::zero();
    for (body_index, body) in state.iter().enumerate() {
        for (axis_index, component) in body.iter().enumerate() {
            let polynomial = evaluated[body_index * PLANE_DIMENSION + axis_index].lower();
            let component_gap =
                checked_absolute_difference(component.binary64_rational(), polynomial)?;
            if component_gap > gap {
                gap = component_gap;
            }
        }
    }
    Ok(gap)
}

fn build_binding_replay(
    satisfaction: [bool; 8],
    time_gap: Option<BigRational>,
    maximum_position_gap: Option<BigRational>,
    maximum_velocity_gap: Option<BigRational>,
) -> InitialValueBindingReplay {
    InitialValueBindingReplay {
        obligations: std::array::from_fn(|index| InitialValueBindingObligation {
            id: INITIAL_VALUE_BINDING_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        time_gap,
        maximum_position_gap,
        maximum_velocity_gap,
    }
}

fn checked_add(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    admit_rational(left + right)
}

fn checked_subtract(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    admit_rational(left - right)
}

fn checked_multiply(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    admit_rational(left * right)
}

fn checked_divide(
    numerator: &BigRational,
    denominator: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    if denominator.is_zero() {
        return Err(NumericError::DivisionByZeroInterval.into());
    }
    admit_rational(numerator / denominator)
}

fn checked_absolute_difference(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, InitialValueBindingReplayError> {
    let difference = checked_subtract(left, right)?;
    if difference.numer().sign() == Sign::Minus {
        admit_rational(-difference)
    } else {
        Ok(difference)
    }
}

fn admit_rational(value: BigRational) -> Result<BigRational, InitialValueBindingReplayError> {
    validate_public_rational(&value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use num_bigint::BigInt;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, parse_wire_json,
        raw_schema::{decode_raw_chain, RawPlanarChainWire, Real, SchemaProfile},
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

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn row(values: [[&str; PLANE_DIMENSION]; BODY_COUNT]) -> Vec<Vec<Real>> {
        values
            .into_iter()
            .map(|body| body.into_iter().map(real).collect())
            .collect()
    }

    fn binding_satisfaction(replay: &InitialValueBindingReplay) -> [bool; 8] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    fn validated_satisfaction(replay: &ValidatedOrdinaryRootReplay) -> [bool; 8] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    #[test]
    fn both_canonical_roots_satisfy_binding_and_validated_root_profiles() {
        for (name, bytes) in [
            ("success", SUCCESS_RAW),
            ("failed-revisit", FAILED_REVISIT_RAW),
        ] {
            let chain = parse_chain(bytes);
            let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
            let tube = ordinary_tube_input_from_wire(&chain.initial_tube);
            let binding =
                replay_initial_value_binding_exact_rational_v04(&chain.root_binding, &chart)
                    .unwrap();
            assert_eq!(
                binding.profile_id(),
                EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID
            );
            assert_eq!(binding_satisfaction(&binding), [true; 8], "{name}");
            assert_eq!(
                binding
                    .obligations()
                    .iter()
                    .map(InitialValueBindingObligation::id)
                    .collect::<Vec<_>>(),
                INITIAL_VALUE_BINDING_OBLIGATION_IDS
            );
            assert_eq!(binding.time_gap(), Some(&BigRational::zero()));
            assert_eq!(binding.maximum_position_gap(), Some(&BigRational::zero()));
            assert_eq!(binding.maximum_velocity_gap(), Some(&BigRational::zero()));

            let validated = replay_validated_ordinary_root_exact_rational_v04(
                &chain.root_binding,
                &chart,
                &tube,
            )
            .unwrap();
            assert_eq!(
                validated.profile_id(),
                EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID
            );
            assert_eq!(validated_satisfaction(&validated), [true; 8], "{name}");
            assert_eq!(
                validated
                    .obligations()
                    .iter()
                    .map(ValidatedOrdinaryRootObligation::id)
                    .collect::<Vec<_>>(),
                VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS
            );
            assert_eq!(validated.actual_initial_error(), Some(&BigRational::zero()));
            assert_eq!(validated.root_clock_origin(), Some(&BigRational::zero()));
            assert!(validated.validated_root_satisfied());
        }
    }

    #[test]
    fn semantic_mutations_are_false_obligations_and_ragged_state_never_panics() {
        let chain = base_chain();
        let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();

        let mut identity = chain.root_binding.clone();
        identity.binding_id.clear();
        let replay = replay_initial_value_binding_exact_rational_v04(&identity, &chart).unwrap();
        assert!(!replay.obligations()[0].satisfied());
        assert!(replay.obligations()[7].satisfied());
        assert!(!replay.profile_satisfied());

        let mut empty_source = chain.root_binding.clone();
        empty_source.source.clear();
        let replay =
            replay_initial_value_binding_exact_rational_v04(&empty_source, &chart).unwrap();
        assert!(replay.profile_satisfied());

        let mut wrong_chart = chain.root_binding.clone();
        wrong_chart.chart_id = "absent-chart".into();
        let replay = replay_initial_value_binding_exact_rational_v04(&wrong_chart, &chart).unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, false, true, true, false, false, false, false]
        );
        assert!(replay.time_gap().is_none());

        let mut wrong_masses = chain.root_binding.clone();
        wrong_masses.masses.pop();
        let replay =
            replay_initial_value_binding_exact_rational_v04(&wrong_masses, &chart).unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, true, false, true, false, true, false, false]
        );

        let mut unequal_masses = chain.root_binding.clone();
        unequal_masses.masses[0] = real("0.2");
        let replay =
            replay_initial_value_binding_exact_rational_v04(&unequal_masses, &chart).unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, true, true, true, false, true, false, false]
        );

        let mut outside = chain.root_binding.clone();
        outside.chart_parameter = real("-1.0");
        let replay = replay_initial_value_binding_exact_rational_v04(&outside, &chart).unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, true, true, true, true, false, false, false]
        );

        let mut negative_tolerance = chain.root_binding.clone();
        negative_tolerance.position_tolerance = real("-1.0");
        let replay =
            replay_initial_value_binding_exact_rational_v04(&negative_tolerance, &chart).unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, true, true, false, true, true, false, false]
        );

        let mut mismatch = chain.root_binding.clone();
        mismatch.positions[0][0] = real("0.25");
        let replay = replay_initial_value_binding_exact_rational_v04(&mismatch, &chart).unwrap();
        assert_eq!(replay.maximum_position_gap(), Some(&rational(1, 4)));
        assert!(!replay.obligations()[7].satisfied());

        let mut velocity_dominant = chain.root_binding.clone();
        velocity_dominant.velocities[0][1] = real("0.25");
        velocity_dominant.velocity_tolerance = real("0.25");
        let replay =
            replay_initial_value_binding_exact_rational_v04(&velocity_dominant, &chart).unwrap();
        assert!(replay.profile_satisfied());
        assert_eq!(replay.maximum_position_gap(), Some(&BigRational::zero()));
        assert_eq!(replay.maximum_velocity_gap(), Some(&rational(1, 4)));

        let mut ragged = chain.root_binding.clone();
        ragged.positions[0].pop();
        let result = std::panic::catch_unwind(|| {
            replay_initial_value_binding_exact_rational_v04(&ragged, &chart)
        });
        let replay = result.expect("ragged wire must not panic").unwrap();
        assert_eq!(
            binding_satisfaction(&replay),
            [true, true, false, true, true, true, false, false]
        );
    }

    #[test]
    fn exact_state_cap_is_inclusive_and_next_lower_binary64_fails() {
        let chain = base_chain();
        let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
        let mut binding = chain.root_binding.clone();
        binding.positions[0][0] = real("0.25");
        binding.position_tolerance = real("0.25");
        let equality = replay_initial_value_binding_exact_rational_v04(&binding, &chart).unwrap();
        assert_eq!(equality.maximum_position_gap(), Some(&rational(1, 4)));
        assert!(equality.profile_satisfied());

        binding.position_tolerance = real("0.24999999999999997");
        let below = replay_initial_value_binding_exact_rational_v04(&binding, &chart).unwrap();
        assert!(!below.obligations()[7].satisfied());
        assert!(!below.profile_satisfied());
    }

    #[test]
    fn planar_body_axis_flattening_matches_patterned_exact_values() {
        let chain = base_chain();
        let mut chart_wire = chain.initial_chart.clone();
        let position_pattern = [["1.0", "2.0"], ["3.0", "4.0"], ["5.0", "6.0"]];
        let velocity_pattern = [["7.0", "8.0"], ["9.0", "10.0"], ["11.0", "12.0"]];
        chart_wire.position_coefficients[0] = position_pattern
            .iter()
            .map(|body| body.iter().map(|value| real(value)).collect())
            .collect();
        chart_wire.velocity_coefficients[0] = velocity_pattern
            .iter()
            .map(|body| body.iter().map(|value| real(value)).collect())
            .collect();
        let chart = ordinary_chart_input_from_wire(&chart_wire).unwrap();
        let mut binding = chain.root_binding.clone();
        binding.positions = position_pattern
            .iter()
            .map(|body| body.iter().map(|value| real(value)).collect())
            .collect();
        binding.velocities = velocity_pattern
            .iter()
            .map(|body| body.iter().map(|value| real(value)).collect())
            .collect();
        let replay = replay_initial_value_binding_exact_rational_v04(&binding, &chart).unwrap();
        assert!(replay.profile_satisfied());
        assert_eq!(replay.maximum_position_gap(), Some(&BigRational::zero()));
        assert_eq!(replay.maximum_velocity_gap(), Some(&BigRational::zero()));
    }

    #[test]
    fn validated_root_rejects_uncovered_error_unequal_width_and_inexact_time_anchor() {
        let chain = base_chain();
        let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
        let tube = ordinary_tube_input_from_wire(&chain.initial_tube);

        let mut uncovered_binding = chain.root_binding.clone();
        uncovered_binding.positions[0][0] = real("0.25");
        uncovered_binding.position_tolerance = real("0.25");
        let uncovered =
            replay_validated_ordinary_root_exact_rational_v04(&uncovered_binding, &chart, &tube)
                .unwrap();
        assert!(uncovered.binding_replay().profile_satisfied());
        assert_eq!(uncovered.actual_initial_error(), Some(&rational(1, 4)));
        assert!(!uncovered.obligations()[7].satisfied());
        assert!(!uncovered.validated_root_satisfied());

        let mut unequal_wire = chain.initial_chart.clone();
        unequal_wire.physical_time_interval[1] = real("0.000000000001");
        let unequal_chart = ordinary_chart_input_from_wire(&unequal_wire).unwrap();
        let unequal = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &unequal_chart,
            &tube,
        )
        .unwrap();
        assert!(!unequal.obligations()[1].satisfied());
        assert!(unequal.root_clock_origin().is_none());
        assert!(!unequal.validated_root_satisfied());

        let mut shifted_time = chain.root_binding.clone();
        shifted_time.initial_time = real("0.25");
        shifted_time.time_tolerance = real("0.25");
        let shifted_binding =
            replay_initial_value_binding_exact_rational_v04(&shifted_time, &chart).unwrap();
        assert!(shifted_binding.profile_satisfied());
        assert_eq!(shifted_binding.time_gap(), Some(&rational(1, 4)));
        let shifted =
            replay_validated_ordinary_root_exact_rational_v04(&shifted_time, &chart, &tube)
                .unwrap();
        assert!(!shifted.obligations()[2].satisfied());
        assert!(shifted.obligations()[3].satisfied());
        assert!(shifted.root_clock_origin().is_none());
        assert!(!shifted.validated_root_satisfied());
    }

    #[test]
    fn validated_root_exact_boundaries_ids_and_claimed_tail_noninterference_are_pinned() {
        let chain = base_chain();
        let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();

        let exact_error = real("8.881784197001252e-16");
        let next_lower = real("8.881784197001251e-16");
        assert_eq!(next_lower.bits() + 1, exact_error.bits());
        let mut binding = chain.root_binding.clone();
        binding.positions[0][0] = exact_error.clone();
        binding.position_tolerance = exact_error.clone();
        let mut tube_wire = chain.initial_tube.clone();
        tube_wire.initial_error_bound = exact_error;
        let tube = ordinary_tube_input_from_wire(&tube_wire);
        let equality =
            replay_validated_ordinary_root_exact_rational_v04(&binding, &chart, &tube).unwrap();
        assert!(equality.validated_root_satisfied());
        assert!(equality.obligations()[7].satisfied());

        tube_wire.initial_error_bound = next_lower;
        let tube = ordinary_tube_input_from_wire(&tube_wire);
        let below =
            replay_validated_ordinary_root_exact_rational_v04(&binding, &chart, &tube).unwrap();
        assert!(below.tube_replay().certified());
        assert!(!below.obligations()[7].satisfied());

        let right = chain.initial_chart.parameter_interval[1].clone();
        let mut at_right = chain.root_binding.clone();
        at_right.chart_parameter = right.clone();
        let replay = replay_initial_value_binding_exact_rational_v04(&at_right, &chart).unwrap();
        assert!(replay.obligations()[5].satisfied());
        let adjacent_outside = real("9.094947017729284e-13");
        assert_eq!(right.bits() + 1, adjacent_outside.bits());
        at_right.chart_parameter = adjacent_outside;
        let replay = replay_initial_value_binding_exact_rational_v04(&at_right, &chart).unwrap();
        assert!(!replay.obligations()[5].satisfied());

        let mut adjacent_anchor_wire = chain.initial_tube.clone();
        adjacent_anchor_wire.anchor_parameter = real("5e-324");
        assert_eq!(adjacent_anchor_wire.anchor_parameter.bits(), 1);
        let adjacent_anchor = ordinary_tube_input_from_wire(&adjacent_anchor_wire);
        let replay = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &chart,
            &adjacent_anchor,
        )
        .unwrap();
        assert!(replay.tube_replay().certified());
        assert!(!replay.obligations()[6].satisfied());

        let mut wrong_id_wire = chain.initial_tube.clone();
        wrong_id_wire.chart_id = "different-chart".into();
        let wrong_id_tube = ordinary_tube_input_from_wire(&wrong_id_wire);
        let replay = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &chart,
            &wrong_id_tube,
        )
        .unwrap();
        assert!(!replay.obligations()[4].satisfied());
        assert!(!replay.obligations()[5].satisfied());

        let mut negative_claims_wire = chain.initial_chart.clone();
        negative_claims_wire.coefficient_tolerance = real("-1.0");
        negative_claims_wire.residual_tolerance = real("-1.0");
        negative_claims_wire.tail_bound = real("-1.0");
        let negative_claims = ordinary_chart_input_from_wire(&negative_claims_wire).unwrap();
        let original_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
        let replay = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &negative_claims,
            &original_tube,
        )
        .unwrap();
        assert!(replay.validated_root_satisfied());
    }

    #[test]
    fn exact_affine_time_preserves_a_nonzero_clock_origin_away_from_the_left_endpoint() {
        let chain = base_chain();
        let mut chart_wire = chain.initial_chart.clone();
        chart_wire.certificate_id = "shifted-clock-certificate".into();
        chart_wire.chart_id = "shifted-clock-chart".into();
        chart_wire.masses = ["1.0", "1.0", "1.0"].into_iter().map(real).collect();
        chart_wire.position_coefficients = vec![
            row([["-1.0", "0.0"], ["0.0", "0.0"], ["1.0", "0.0"]]),
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
        ];
        chart_wire.velocity_coefficients = vec![
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
        ];
        chart_wire.parameter_interval = [real("1.0"), real("1.00000095367431640625")];
        chart_wire.physical_time_interval = [real("3.0"), real("3.00000095367431640625")];
        chart_wire.coefficient_tolerance = real("-1.0");
        chart_wire.residual_tolerance = real("-1.0");
        chart_wire.tail_bound = real("-1.0");
        let chart = ordinary_chart_input_from_wire(&chart_wire).unwrap();

        let mut binding = chain.root_binding.clone();
        binding.binding_id = "shifted-clock-binding".into();
        binding.chart_id = chart_wire.chart_id.clone();
        binding.masses = chart_wire.masses.clone();
        binding.chart_parameter = real("1.000000476837158203125");
        binding.initial_time = real("3.000000476837158203125");
        binding.positions = chart_wire.position_coefficients[0].clone();
        binding.velocities = chart_wire.velocity_coefficients[0].clone();
        binding.time_tolerance = real("0.0");
        binding.position_tolerance = real("0.0");
        binding.velocity_tolerance = real("0.0");

        let binding_replay =
            replay_initial_value_binding_exact_rational_v04(&binding, &chart).unwrap();
        assert!(binding_replay.profile_satisfied());
        assert_eq!(binding_replay.time_gap(), Some(&BigRational::zero()));
        assert_eq!(
            binding_replay.maximum_position_gap(),
            Some(&BigRational::zero())
        );
        assert_eq!(
            binding_replay.maximum_velocity_gap(),
            Some(&BigRational::zero())
        );

        let mut tube_wire = chain.initial_tube.clone();
        tube_wire.tube_id = "shifted-clock-tube".into();
        tube_wire.chart_id = chart_wire.chart_id;
        tube_wire.anchor_parameter = binding.chart_parameter.clone();
        tube_wire.initial_error_bound = real("0.0");
        tube_wire.tube_radius = real("0.01");
        tube_wire.max_defect_bound = real("1e100");
        tube_wire.max_lipschitz_bound = real("1e100");
        let tube = ordinary_tube_input_from_wire(&tube_wire);
        let root =
            replay_validated_ordinary_root_exact_rational_v04(&binding, &chart, &tube).unwrap();
        assert_eq!(validated_satisfaction(&root), [true; 8]);
        assert!(root.validated_root_satisfied());
        assert_eq!(
            root.root_clock_origin(),
            Some(&BigRational::from_integer(BigInt::from(2)))
        );
    }
}
