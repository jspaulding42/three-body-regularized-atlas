//! Exact local composition for one parent-carried ordinary-to-ordinary bridge.
//!
//! This separately named profile replays both conditional tube proofs, then
//! checks the exact endpoint enclosure and additive clock cocycle.  It is a
//! local component: it neither admits a raw-chain root nor folds a chain, and
//! it does not consume either chart's claimed Taylor-tail replay.

use core::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{
    ordinary_semantic::{OrdinaryChartInput, OrdinaryTubeInput},
    ordinary_tube::{
        replay_ordinary_tube_exact_rational_v04, OrdinaryTubeReplay, OrdinaryTubeReplayError,
    },
    rational_input::validate_public_rational,
    raw_schema::OrdinaryBridgeTransitionWire,
    NumericError, PolynomialError, RationalInterval,
};

const STATE_DIMENSION: usize = 12;
const TRANSITION_RECORD_TYPE: &str = "ordinary_bridge_transition";
const TRANSITION_SOURCE: &str = "private_carried_ordinary_bridge_v1";

pub const EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID: &str =
    "exact_rational_carried_ordinary_bridge_v04";
pub const ORDINARY_AUTONOMOUS_UNIQUENESS_BRIDGE_KERNEL_V1_ID: &str =
    "ordinary_autonomous_uniqueness_bridge_kernel_v1";

pub const ORDINARY_BRIDGE_OBLIGATION_IDS: [&str; 9] = [
    "ordinary_bridge_exact_raw_schemas",
    "ordinary_bridge_identifiers_match_and_are_unique",
    "ordinary_bridge_parent_clock_origin_is_exact_interval",
    "ordinary_bridge_common_planar_mass_problem",
    "ordinary_bridge_source_and_target_tubes_freshly_certified",
    "ordinary_bridge_exact_right_to_left_endpoint_handoff",
    "ordinary_bridge_complete_source_endpoint_enclosure_reconstructed",
    "ordinary_bridge_target_initial_ball_contains_complete_source_endpoint",
    "ordinary_bridge_target_clock_origin_exactly_derived",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryBridgeObligation {
    id: &'static str,
    satisfied: bool,
}

impl OrdinaryBridgeObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Ordered local bridge ledger, component tube replays, and exact derived
/// endpoint/clock quantities.  Derived quantities remain absent until all
/// first six obligations have passed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryBridgeReplay {
    obligations: [OrdinaryBridgeObligation; 9],
    source_tube_replay: Option<OrdinaryTubeReplay>,
    target_tube_replay: Option<OrdinaryTubeReplay>,
    source_endpoint_state_box: Option<Vec<RationalInterval>>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_target_anchor_gap: Option<BigRational>,
    target_clock_origin: Option<RationalInterval>,
}

impl OrdinaryBridgeReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID
    }

    /// Identifier of the trusted autonomy/local-uniqueness lemma whose
    /// hypotheses this finite replay checks.  Replay does not re-prove it.
    pub const fn analytic_kernel_id(&self) -> &'static str {
        ORDINARY_AUTONOMOUS_UNIQUENESS_BRIDGE_KERNEL_V1_ID
    }

    pub fn obligations(&self) -> &[OrdinaryBridgeObligation; 9] {
        &self.obligations
    }

    /// Whether this local component profile is satisfied.  This is not a
    /// complete raw-chain replay claim.
    pub fn local_bridge_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(OrdinaryBridgeObligation::satisfied)
    }

    pub fn source_tube_replay(&self) -> Option<&OrdinaryTubeReplay> {
        self.source_tube_replay.as_ref()
    }

    pub fn target_tube_replay(&self) -> Option<&OrdinaryTubeReplay> {
        self.target_tube_replay.as_ref()
    }

    pub fn source_endpoint_state_box(&self) -> Option<&[RationalInterval]> {
        self.source_endpoint_state_box.as_deref()
    }

    /// Target polynomial center at the exact target anchor in `q` body-major
    /// order followed by `v` body-major order.
    pub fn target_anchor_centers(&self) -> Option<&[BigRational]> {
        self.target_anchor_centers.as_deref()
    }

    pub fn maximum_target_anchor_gap(&self) -> Option<&BigRational> {
        self.maximum_target_anchor_gap.as_ref()
    }

    pub fn target_clock_origin(&self) -> Option<&RationalInterval> {
        self.target_clock_origin.as_ref()
    }
}

/// Arithmetic/resource failures are distinct from sound false obligations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryBridgeReplayError {
    Numeric(NumericError),
    SourcePositionPolynomial(PolynomialError),
    SourceVelocityPolynomial(PolynomialError),
    TargetPositionPolynomial(PolynomialError),
    TargetVelocityPolynomial(PolynomialError),
    SourceTube(OrdinaryTubeReplayError),
    TargetTube(OrdinaryTubeReplayError),
    InternalInvariant { detail: &'static str },
}

impl fmt::Display for OrdinaryBridgeReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(source) => write!(formatter, "ordinary bridge numeric failure: {source}"),
            Self::SourcePositionPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary bridge source-position polynomial failure: {source}"
                )
            }
            Self::SourceVelocityPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary bridge source-velocity polynomial failure: {source}"
                )
            }
            Self::TargetPositionPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary bridge target-position polynomial failure: {source}"
                )
            }
            Self::TargetVelocityPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary bridge target-velocity polynomial failure: {source}"
                )
            }
            Self::SourceTube(source) => {
                write!(formatter, "ordinary bridge source-tube failure: {source}")
            }
            Self::TargetTube(source) => {
                write!(formatter, "ordinary bridge target-tube failure: {source}")
            }
            Self::InternalInvariant { detail } => {
                write!(
                    formatter,
                    "ordinary bridge internal invariant failed: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for OrdinaryBridgeReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numeric(source) => Some(source),
            Self::SourcePositionPolynomial(source)
            | Self::SourceVelocityPolynomial(source)
            | Self::TargetPositionPolynomial(source)
            | Self::TargetVelocityPolynomial(source) => Some(source),
            Self::SourceTube(source) | Self::TargetTube(source) => Some(source),
            Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for OrdinaryBridgeReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Replay one exact-rational local ordinary bridge from already admitted
/// planar semantic inputs and a parent-derived clock interval.
pub fn replay_carried_ordinary_bridge_exact_rational_v04(
    transition: &OrdinaryBridgeTransitionWire,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
    target_chart: &OrdinaryChartInput,
    target_tube: &OrdinaryTubeInput,
    parent_source_clock_origin: &RationalInterval,
) -> Result<OrdinaryBridgeReplay, OrdinaryBridgeReplayError> {
    // Construction of both `OrdinaryChartInput` values is the chart-schema
    // witness.  Tube/transition schemas remain ordered Boolean obligations.
    let raw_schemas =
        transition_schema(transition) && tube_schema(source_tube) && tube_schema(target_tube);
    let identifiers = identifiers_match_and_are_unique(
        transition,
        source_chart,
        source_tube,
        target_chart,
        target_tube,
    );
    // `RationalInterval` construction is the exact finite ordered-interval
    // witness; no independent clock evidence is read from the transition.
    let parent_clock_valid = true;
    let common_problem = source_chart.masses() == target_chart.masses();
    let endpoint_handoff = transition.source_parameter.binary64_rational()
        == source_chart.parameter_interval().upper()
        && transition.target_parameter.binary64_rational()
            == target_chart.parameter_interval().lower()
        && source_tube.anchor_parameter() == source_chart.parameter_interval().lower()
        && target_tube.anchor_parameter() == target_chart.parameter_interval().lower();
    if !raw_schemas {
        return Ok(build_replay(
            None,
            None,
            [
                false,
                identifiers,
                parent_clock_valid,
                common_problem,
                false,
                endpoint_handoff,
                false,
                false,
                false,
            ],
            None,
            None,
            None,
            None,
        ));
    }

    // Both nested tube profiles are freshly replayed after schema admission.
    // False nested obligations stay Boolean; resource failures remain typed.
    let source_tube_replay = replay_ordinary_tube_exact_rational_v04(source_chart, source_tube)
        .map_err(OrdinaryBridgeReplayError::SourceTube)?;
    let target_tube_replay = replay_ordinary_tube_exact_rational_v04(target_chart, target_tube)
        .map_err(OrdinaryBridgeReplayError::TargetTube)?;
    let tubes_certified = source_tube_replay.certified() && target_tube_replay.certified();

    let first_six = [
        raw_schemas,
        identifiers,
        parent_clock_valid,
        common_problem,
        tubes_certified,
        endpoint_handoff,
    ];
    if !first_six.iter().all(|value| *value) {
        return Ok(build_replay(
            Some(source_tube_replay),
            Some(target_tube_replay),
            [
                raw_schemas,
                identifiers,
                parent_clock_valid,
                common_problem,
                tubes_certified,
                endpoint_handoff,
                false,
                false,
                false,
            ],
            None,
            None,
            None,
            None,
        ));
    }

    let source_parameter = transition.source_parameter.binary64_rational();
    let target_parameter = transition.target_parameter.binary64_rational();
    let source_centers = evaluate_state(source_chart, source_parameter, true)?;
    let target_centers = evaluate_state(target_chart, target_parameter, false)?;
    let source_radius = source_tube_replay.gronwall_upper().ok_or(
        OrdinaryBridgeReplayError::InternalInvariant {
            detail: "certified source tube has no Gronwall upper bound",
        },
    )?;
    let source_box = source_centers
        .iter()
        .map(|center| {
            RationalInterval::new(
                checked_subtract(center, source_radius)?,
                checked_add(center, source_radius)?,
            )
            .map_err(OrdinaryBridgeReplayError::Numeric)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let endpoint_reconstructed = source_box.len() == STATE_DIMENSION;
    let maximum_gap = maximum_box_center_gap(&source_box, &target_centers)?;
    let target_contains =
        target_initial_ball_contains(&maximum_gap, target_tube.initial_error_bound())?;
    let delta = checked_subtract(source_parameter, target_parameter)?;
    let target_clock = translate_clock(parent_source_clock_origin, &delta)?;

    Ok(build_replay(
        Some(source_tube_replay),
        Some(target_tube_replay),
        [
            raw_schemas,
            identifiers,
            parent_clock_valid,
            common_problem,
            tubes_certified,
            endpoint_handoff,
            endpoint_reconstructed,
            target_contains,
            true,
        ],
        Some(source_box),
        Some(target_centers),
        Some(maximum_gap),
        Some(target_clock),
    ))
}

fn transition_schema(transition: &OrdinaryBridgeTransitionWire) -> bool {
    transition.schema_version.value() == &BigInt::one()
        && transition.record_type == TRANSITION_RECORD_TYPE
        && transition.source == TRANSITION_SOURCE
        && [
            &transition.transition_id,
            &transition.source_chart_id,
            &transition.source_tube_id,
            &transition.target_chart_id,
            &transition.target_tube_id,
        ]
        .iter()
        .all(|value| !value.is_empty())
}

fn tube_schema(tube: &OrdinaryTubeInput) -> bool {
    !tube.tube_id().is_empty()
        && !tube.chart_id().is_empty()
        && !tube.source().is_empty()
        && tube.initial_error_bound() >= &BigRational::zero()
        && tube.tube_radius() > &BigRational::zero()
        && tube.maximum_defect_bound() >= &BigRational::zero()
        && tube.maximum_lipschitz_bound() >= &BigRational::zero()
}

fn identifiers_match_and_are_unique(
    transition: &OrdinaryBridgeTransitionWire,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
    target_chart: &OrdinaryChartInput,
    target_tube: &OrdinaryTubeInput,
) -> bool {
    let identifiers = [
        transition.transition_id.as_str(),
        source_chart.certificate_id(),
        source_chart.chart_id(),
        source_tube.tube_id(),
        target_chart.certificate_id(),
        target_chart.chart_id(),
        target_tube.tube_id(),
    ];
    let unique = identifiers
        .iter()
        .enumerate()
        .all(|(index, value)| !value.is_empty() && !identifiers[..index].contains(value));
    unique
        && transition.source_chart_id == source_chart.chart_id()
        && transition.source_tube_id == source_tube.tube_id()
        && transition.target_chart_id == target_chart.chart_id()
        && transition.target_tube_id == target_tube.tube_id()
        && source_tube.chart_id() == source_chart.chart_id()
        && target_tube.chart_id() == target_chart.chart_id()
}

fn evaluate_state(
    chart: &OrdinaryChartInput,
    parameter: &BigRational,
    source: bool,
) -> Result<Vec<BigRational>, OrdinaryBridgeReplayError> {
    let argument = RationalInterval::try_point(parameter.clone())?;
    let positions = chart
        .position_polynomial()
        .evaluate(&argument)
        .map_err(|error| {
            if source {
                OrdinaryBridgeReplayError::SourcePositionPolynomial(error)
            } else {
                OrdinaryBridgeReplayError::TargetPositionPolynomial(error)
            }
        })?;
    let velocities = chart
        .velocity_polynomial()
        .evaluate(&argument)
        .map_err(|error| {
            if source {
                OrdinaryBridgeReplayError::SourceVelocityPolynomial(error)
            } else {
                OrdinaryBridgeReplayError::TargetVelocityPolynomial(error)
            }
        })?;
    let mut state = Vec::with_capacity(positions.len() + velocities.len());
    for value in positions.into_iter().chain(velocities) {
        if !value.is_point() {
            return Err(OrdinaryBridgeReplayError::InternalInvariant {
                detail: "point polynomial evaluation produced a non-point interval",
            });
        }
        state.push(value.lower().clone());
    }
    if state.len() != STATE_DIMENSION {
        return Err(OrdinaryBridgeReplayError::InternalInvariant {
            detail: "ordinary bridge state is not planar three-body",
        });
    }
    Ok(state)
}

fn maximum_box_center_gap(
    source_box: &[RationalInterval],
    target_centers: &[BigRational],
) -> Result<BigRational, OrdinaryBridgeReplayError> {
    if source_box.len() != STATE_DIMENSION || target_centers.len() != STATE_DIMENSION {
        return Err(OrdinaryBridgeReplayError::InternalInvariant {
            detail: "ordinary bridge endpoint state dimension mismatch",
        });
    }
    let mut maximum = BigRational::zero();
    for (interval, center) in source_box.iter().zip(target_centers) {
        let lower_gap = checked_absolute_difference(interval.lower(), center)?;
        let upper_gap = checked_absolute_difference(interval.upper(), center)?;
        maximum = maximum.max(lower_gap).max(upper_gap);
    }
    Ok(maximum)
}

fn target_initial_ball_contains(
    maximum_gap: &BigRational,
    target_initial_error: &BigRational,
) -> Result<bool, OrdinaryBridgeReplayError> {
    validate_public_rational(maximum_gap)?;
    validate_public_rational(target_initial_error)?;
    Ok(maximum_gap <= target_initial_error)
}

fn translate_clock(
    parent: &RationalInterval,
    delta: &BigRational,
) -> Result<RationalInterval, OrdinaryBridgeReplayError> {
    validate_public_rational(delta)?;
    parent
        .add(&RationalInterval::try_point(delta.clone())?)
        .map_err(OrdinaryBridgeReplayError::Numeric)
}

fn checked_add(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryBridgeReplayError> {
    admit_rational(left + right)
}

fn checked_subtract(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryBridgeReplayError> {
    admit_rational(left - right)
}

fn checked_absolute_difference(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, OrdinaryBridgeReplayError> {
    let difference = checked_subtract(left, right)?;
    if difference < BigRational::zero() {
        admit_rational(-difference)
    } else {
        Ok(difference)
    }
}

fn admit_rational(value: BigRational) -> Result<BigRational, OrdinaryBridgeReplayError> {
    validate_public_rational(&value)?;
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn build_replay(
    source_tube_replay: Option<OrdinaryTubeReplay>,
    target_tube_replay: Option<OrdinaryTubeReplay>,
    satisfaction: [bool; 9],
    source_endpoint_state_box: Option<Vec<RationalInterval>>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_target_anchor_gap: Option<BigRational>,
    target_clock_origin: Option<RationalInterval>,
) -> OrdinaryBridgeReplay {
    OrdinaryBridgeReplay {
        obligations: std::array::from_fn(|index| OrdinaryBridgeObligation {
            id: ORDINARY_BRIDGE_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        source_tube_replay,
        target_tube_replay,
        source_endpoint_state_box,
        target_anchor_centers,
        maximum_target_anchor_gap,
        target_clock_origin,
    }
}

#[cfg(test)]
mod tests {
    use std::{panic::catch_unwind, sync::OnceLock};

    use num_bigint::BigInt;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, parse_wire_json, rational_from_f64,
        raw_schema::{
            decode_raw_chain, OrdinaryChartWire, OrdinaryTubeWire, RawPlanarChainWire,
            SchemaProfile, SegmentWire,
        },
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));
    const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
    ));

    #[derive(Clone)]
    struct Fixture {
        transition: OrdinaryBridgeTransitionWire,
        source_chart: OrdinaryChartWire,
        source_tube: OrdinaryTubeWire,
        target_chart: OrdinaryChartWire,
        target_tube: OrdinaryTubeWire,
        zero_integer: crate::raw_schema::RawInteger,
        one_integer: crate::raw_schema::RawInteger,
    }

    fn decode(bytes: &[u8]) -> RawPlanarChainWire {
        let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
    }

    fn fixture_from(bytes: &[u8]) -> Fixture {
        let chain = decode(bytes);
        let SegmentWire::OrdinaryBridge(bridge) = &chain.segments[0] else {
            panic!("first canonical segment must be an ordinary bridge");
        };
        let zero_integer = chain
            .segments
            .iter()
            .find_map(|segment| match segment {
                SegmentWire::PlanarLcPassage(segment) => segment
                    .lc_chart
                    .pair
                    .iter()
                    .find(|value| value.value().is_zero()),
                SegmentWire::OrdinaryBridge(_) => None,
            })
            .expect("canonical LC pair supplies an exact zero integer")
            .clone();
        let one_integer = chain.schema_version.clone();
        Fixture {
            transition: bridge.transition.clone(),
            source_chart: chain.initial_chart,
            source_tube: chain.initial_tube,
            target_chart: bridge.target_chart.clone(),
            target_tube: bridge.target_tube.clone(),
            zero_integer,
            one_integer,
        }
    }

    fn success_fixture() -> &'static Fixture {
        static FIXTURE: OnceLock<Fixture> = OnceLock::new();
        FIXTURE.get_or_init(|| fixture_from(SUCCESS_RAW))
    }

    fn real(lexeme: &str) -> crate::raw_schema::Real {
        checked_real_binary64_from_json(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn point(value: BigRational) -> RationalInterval {
        RationalInterval::try_point(value).unwrap()
    }

    fn replay(fixture: &Fixture, parent: &RationalInterval) -> OrdinaryBridgeReplay {
        let source_chart = ordinary_chart_input_from_wire(&fixture.source_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&fixture.source_tube);
        let target_chart = ordinary_chart_input_from_wire(&fixture.target_chart).unwrap();
        let target_tube = ordinary_tube_input_from_wire(&fixture.target_tube);
        replay_carried_ordinary_bridge_exact_rational_v04(
            &fixture.transition,
            &source_chart,
            &source_tube,
            &target_chart,
            &target_tube,
            parent,
        )
        .unwrap()
    }

    fn assert_only_gate_false(result: &OrdinaryBridgeReplay, gate: usize) {
        for (index, obligation) in result.obligations().iter().enumerate().take(6) {
            assert_eq!(
                obligation.satisfied(),
                index != gate,
                "unexpected preliminary gate at {index}: {}",
                obligation.id()
            );
        }
        assert!(result.source_endpoint_state_box().is_none());
        assert!(result.target_anchor_centers().is_none());
        assert!(result.maximum_target_anchor_gap().is_none());
        assert!(result.target_clock_origin().is_none());
    }

    #[test]
    fn both_canonical_first_bridges_pass_with_frozen_ids_kernel_and_clock() {
        let zero = point(BigRational::zero());
        for bytes in [SUCCESS_RAW, FAILED_REVISIT_RAW] {
            let result = replay(&fixture_from(bytes), &zero);
            assert!(result.local_bridge_satisfied());
            assert_eq!(
                result.profile_id(),
                "exact_rational_carried_ordinary_bridge_v04"
            );
            assert_eq!(
                result.analytic_kernel_id(),
                "ordinary_autonomous_uniqueness_bridge_kernel_v1"
            );
            assert_eq!(
                result
                    .obligations()
                    .iter()
                    .map(OrdinaryBridgeObligation::id)
                    .collect::<Vec<_>>(),
                ORDINARY_BRIDGE_OBLIGATION_IDS
            );
            assert!(result.source_tube_replay().unwrap().certified());
            assert!(result.target_tube_replay().unwrap().certified());
            assert_eq!(
                result.source_endpoint_state_box().unwrap().len(),
                STATE_DIMENSION
            );
            assert_eq!(
                result.target_anchor_centers().unwrap().len(),
                STATE_DIMENSION
            );
            let expected = point(BigRational::new(BigInt::one(), BigInt::one() << 40));
            assert_eq!(result.target_clock_origin(), Some(&expected));
        }
    }

    #[test]
    fn inclusive_containment_accepts_equality_and_rejects_the_next_lower_binary64() {
        let gap = BigRational::one();
        let equal = BigRational::one();
        let next_lower = rational_from_f64(f64::from_bits(1.0_f64.to_bits() - 1)).unwrap();
        assert!(target_initial_ball_contains(&gap, &equal).unwrap());
        assert!(!target_initial_ball_contains(&gap, &next_lower).unwrap());
    }

    #[test]
    fn failed_containment_retains_the_locally_derived_clock_and_endpoint_evidence() {
        let mut fixture = success_fixture().clone();
        fixture.target_tube.initial_error_bound = real("0.0");
        let result = replay(&fixture, &point(BigRational::zero()));

        assert!(result.source_tube_replay().unwrap().certified());
        assert!(result.target_tube_replay().unwrap().certified());
        assert!(result.obligations()[..7]
            .iter()
            .all(OrdinaryBridgeObligation::satisfied));
        assert!(!result.obligations()[7].satisfied());
        assert!(result.obligations()[8].satisfied());
        assert!(!result.local_bridge_satisfied());
        assert_eq!(
            result.target_clock_origin(),
            Some(&point(
                BigRational::new(BigInt::one(), BigInt::one() << 40,)
            ))
        );
        assert_eq!(
            result.source_endpoint_state_box().unwrap().len(),
            STATE_DIMENSION
        );
        assert_eq!(
            result.target_anchor_centers().unwrap().len(),
            STATE_DIMENSION
        );
        assert!(result.maximum_target_anchor_gap().is_some());
    }

    #[test]
    fn schema_constants_and_nonempty_fields_fail_without_nested_replay() {
        let base = success_fixture();
        let zero = point(BigRational::zero());
        let mut cases = Vec::new();

        let mut wrong_version = base.clone();
        wrong_version.transition.schema_version = base.zero_integer.clone();
        cases.push(wrong_version);
        let mut wrong_type = base.clone();
        wrong_type.transition.record_type = "ordinary_bridge_transition_v2".to_owned();
        cases.push(wrong_type);
        let mut wrong_source = base.clone();
        wrong_source.transition.source = "unreviewed".to_owned();
        cases.push(wrong_source);
        let mut empty_id = base.clone();
        empty_id.transition.source_chart_id.clear();
        cases.push(empty_id);
        let mut empty_tube_source = base.clone();
        empty_tube_source.source_tube.source.clear();
        cases.push(empty_tube_source);

        for case in cases {
            let result = replay(&case, &zero);
            assert!(!result.obligations()[0].satisfied());
            assert!(result.source_tube_replay().is_none());
            assert!(result.target_tube_replay().is_none());
            assert!(result.source_endpoint_state_box().is_none());
            assert!(result.target_clock_origin().is_none());
        }
    }

    #[test]
    fn identity_mass_endpoint_and_tube_gates_fail_deterministically() {
        let base = success_fixture();
        let zero = point(BigRational::zero());

        let mut identity = base.clone();
        identity.transition.target_chart_id = "wrong-target-chart".to_owned();
        assert_only_gate_false(&replay(&identity, &zero), 1);

        let mut duplicate = base.clone();
        duplicate.transition.transition_id = duplicate.source_chart.chart_id.clone();
        assert_only_gate_false(&replay(&duplicate, &zero), 1);

        let mut mass = base.clone();
        mass.target_chart.masses[0] = real("0.10000000000000002");
        assert_only_gate_false(&replay(&mass, &zero), 3);

        let mut wrong_source_endpoint = base.clone();
        wrong_source_endpoint.transition.source_parameter = real("0.0");
        assert_only_gate_false(&replay(&wrong_source_endpoint, &zero), 5);

        let mut wrong_target_endpoint = base.clone();
        wrong_target_endpoint.transition.target_parameter = real("9.5367431640625e-7");
        assert_only_gate_false(&replay(&wrong_target_endpoint, &zero), 5);

        let mut wrong_source_anchor = base.clone();
        wrong_source_anchor.source_tube.anchor_parameter = real("9.094947017729282e-13");
        assert_only_gate_false(&replay(&wrong_source_anchor, &zero), 5);

        let mut wrong_target_anchor = base.clone();
        wrong_target_anchor.target_tube.anchor_parameter = real("9.5367431640625e-7");
        assert_only_gate_false(&replay(&wrong_target_anchor, &zero), 5);

        let mut tube = base.clone();
        tube.source_tube.max_defect_bound = real("0.0");
        assert_only_gate_false(&replay(&tube, &zero), 4);
    }

    #[test]
    fn parent_nonpoint_interval_is_translated_exactly() {
        let parent = RationalInterval::new(
            BigRational::new(BigInt::from(-1), BigInt::from(3)),
            BigRational::new(BigInt::from(2), BigInt::from(5)),
        )
        .unwrap();
        let delta = BigRational::new(BigInt::one(), BigInt::one() << 40);
        let expected = parent.add(&point(delta)).unwrap();
        let result = replay(success_fixture(), &parent);
        assert!(result.local_bridge_satisfied());
        assert_eq!(result.target_clock_origin(), Some(&expected));
    }

    #[test]
    fn unconsumed_chart_claims_time_metadata_and_positive_sample_count_are_irrelevant() {
        let base = success_fixture();
        let parent = point(BigRational::zero());
        let baseline = replay(base, &parent);
        let mut changed = base.clone();
        for chart in [&mut changed.source_chart, &mut changed.target_chart] {
            chart.coefficient_tolerance = real("-1.0");
            chart.residual_tolerance = real("-2.0");
            chart.tail_bound = real("-3.0");
            chart.sample_count = base.one_integer.clone();
        }
        changed.source_chart.physical_time_interval = [real("100.0"), real("101.0")];
        changed.target_chart.physical_time_interval = [real("-200.0"), real("-198.0")];
        assert_eq!(replay(&changed, &parent), baseline);
    }

    #[test]
    fn malformed_and_oversized_rationals_return_typed_errors_without_panicking() {
        let malformed = BigRational::new_raw(BigInt::one(), BigInt::zero());
        let malformed_result =
            catch_unwind(|| translate_clock(&point(BigRational::zero()), &malformed));
        assert!(malformed_result.is_ok());
        assert!(matches!(
            malformed_result.unwrap(),
            Err(OrdinaryBridgeReplayError::Numeric(
                NumericError::InvalidRationalDenominator
            ))
        ));

        let oversized = BigRational::from_integer(
            BigInt::one() << crate::HARD_MAX_RATIONAL_COMPONENT_BITS as usize,
        );
        let oversized_result =
            catch_unwind(|| target_initial_ball_contains(&oversized, &BigRational::zero()));
        assert!(oversized_result.is_ok());
        assert!(matches!(
            oversized_result.unwrap(),
            Err(OrdinaryBridgeReplayError::Numeric(
                NumericError::RationalComponentBitLimitExceeded { .. }
            ))
        ));

        // A valid semantic chart with a constant collision-free state but an
        // extreme horizon reaches the bounded exponential kernel.  The public
        // bridge entry point must preserve that nested resource failure.
        let mut resource = success_fixture().clone();
        let zero_real = real("0.0");
        for row in resource
            .source_chart
            .position_coefficients
            .iter_mut()
            .skip(1)
        {
            for body in row {
                for component in body {
                    *component = zero_real.clone();
                }
            }
        }
        for row in &mut resource.source_chart.velocity_coefficients {
            for body in row {
                for component in body {
                    *component = zero_real.clone();
                }
            }
        }
        let maximum = real("1.7976931348623157e308");
        resource.source_chart.parameter_interval[1] = maximum.clone();
        resource.transition.source_parameter = maximum;
        let source_chart = ordinary_chart_input_from_wire(&resource.source_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&resource.source_tube);
        let target_chart = ordinary_chart_input_from_wire(&resource.target_chart).unwrap();
        let target_tube = ordinary_tube_input_from_wire(&resource.target_tube);
        let replay_result = catch_unwind(|| {
            replay_carried_ordinary_bridge_exact_rational_v04(
                &resource.transition,
                &source_chart,
                &source_tube,
                &target_chart,
                &target_tube,
                &point(BigRational::zero()),
            )
        });
        assert!(replay_result.is_ok());
        assert!(matches!(
            replay_result.unwrap(),
            Err(OrdinaryBridgeReplayError::SourceTube(_))
        ));
    }
}
