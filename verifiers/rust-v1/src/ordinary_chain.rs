//! Bounded exact-rational replay of the ordinary-only prefix of one raw chain.
//!
//! This v0.4 profile begins at raw top-level obligation 6. It deliberately
//! does not reproduce outer raw-byte/schema/namespace admission, and it stops
//! at the first planar-LC segment. Root admission consumes the conditional
//! claimed-tail ordinary-chart profile as a fail-closed compatibility gate.
//! That gate retains an unproved allowance; local theorem support instead
//! comes from exact binding, direct-defect tubes, and the named analytic
//! kernel. OPEN-V1-06 therefore remains open, and this is not frozen-v0.3
//! arithmetic/status parity.

use core::fmt;

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    ordinary_chart_input_from_wire, ordinary_tube_binding_status, ordinary_tube_input_from_wire,
    rational_input::validate_public_rational,
    raw_schema::{RawPlanarChainWire, SegmentWire},
    replay_carried_ordinary_bridge_exact_rational_v04,
    replay_ordinary_chart_exact_rational_claimed_tail_v04, replay_ordinary_tube_exact_rational_v04,
    replay_validated_ordinary_root_exact_rational_v04, NumericError, OrdinaryBridgeReplay,
    OrdinaryBridgeReplayError, OrdinaryChartInput, OrdinaryChartReplay, OrdinaryChartReplayError,
    OrdinarySemanticError, OrdinaryTubeInput, OrdinaryTubeReplay, OrdinaryTubeReplayError,
    PolynomialError, RationalInterval, ValidatedOrdinaryRootReplay,
    ValidatedOrdinaryRootReplayError,
};

const POSITION_DIMENSION: usize = 6;

pub const EXACT_RATIONAL_RAW_ORDINARY_ONLY_CHAIN_V04_PROFILE_ID: &str =
    "exact_rational_raw_ordinary_only_chain_v04";

/// Hard orchestration bound checked before any segment replay.
pub const HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS: usize = 256;

pub const RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS: [&str; 8] = [
    "raw_planar_chain_root_exact_point_left_anchor",
    "raw_planar_chain_root_freshly_certified",
    "raw_planar_chain_all_segments_freshly_folded",
    "raw_planar_chain_target_not_before_current_left_clock",
    "raw_planar_chain_fixed_time_preimage_exactly_derived",
    "raw_planar_chain_fixed_time_preimage_inside_forward_current_domain",
    "raw_planar_chain_target_state_exactly_evaluated_and_inflated",
    "raw_planar_chain_final_component_width_within_requested_bound",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawOrdinaryOnlyChainObligation {
    id: &'static str,
    satisfied: bool,
}

impl RawOrdinaryOnlyChainObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChainClockLedgerEntry {
    vertex_index: usize,
    chart_id: String,
    clock_origin: RationalInterval,
}

impl OrdinaryChainClockLedgerEntry {
    pub const fn vertex_index(&self) -> usize {
        self.vertex_index
    }

    pub fn chart_id(&self) -> &str {
        &self.chart_id
    }

    pub fn clock_origin(&self) -> &RationalInterval {
        &self.clock_origin
    }
}

/// Why replay stopped after retaining the longest certified prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinaryChainReplayFailure {
    SegmentLimitExceeded {
        actual: usize,
        limit: usize,
    },
    TargetChartSemantic {
        segment_index: usize,
        source: OrdinarySemanticError,
    },
    Bridge {
        segment_index: usize,
        source: OrdinaryBridgeReplayError,
    },
    FinalTube {
        source: OrdinaryTubeReplayError,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChainFinalEnclosure {
    physical_time: BigRational,
    parameter_preimage: RationalInterval,
    position_intervals: Vec<RationalInterval>,
    velocity_intervals: Vec<RationalInterval>,
}

/// A certified enclosure at the right edge of the longest committed ordinary
/// prefix. This remains useful when the next segment is unsupported or false.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChainRightFrontier {
    committed_segment_count: usize,
    chart_id: String,
    right_parameter: BigRational,
    physical_time_interval: RationalInterval,
    position_intervals: Vec<RationalInterval>,
    velocity_intervals: Vec<RationalInterval>,
    tube_replay: OrdinaryTubeReplay,
}

impl OrdinaryChainRightFrontier {
    pub const fn committed_segment_count(&self) -> usize {
        self.committed_segment_count
    }

    pub fn chart_id(&self) -> &str {
        &self.chart_id
    }

    pub fn right_parameter(&self) -> &BigRational {
        &self.right_parameter
    }

    pub fn physical_time_interval(&self) -> &RationalInterval {
        &self.physical_time_interval
    }

    pub fn position_intervals(&self) -> &[RationalInterval] {
        &self.position_intervals
    }

    pub fn velocity_intervals(&self) -> &[RationalInterval] {
        &self.velocity_intervals
    }

    pub fn tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.tube_replay
    }
}

impl OrdinaryChainFinalEnclosure {
    pub(crate) fn new(
        physical_time: BigRational,
        parameter_preimage: RationalInterval,
        position_intervals: Vec<RationalInterval>,
        velocity_intervals: Vec<RationalInterval>,
    ) -> Self {
        Self {
            physical_time,
            parameter_preimage,
            position_intervals,
            velocity_intervals,
        }
    }

    pub fn physical_time(&self) -> &BigRational {
        &self.physical_time
    }

    pub fn parameter_preimage(&self) -> &RationalInterval {
        &self.parameter_preimage
    }

    pub fn position_intervals(&self) -> &[RationalInterval] {
        &self.position_intervals
    }

    pub fn velocity_intervals(&self) -> &[RationalInterval] {
        &self.velocity_intervals
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawOrdinaryOnlyChainReplay {
    obligations: [RawOrdinaryOnlyChainObligation; 8],
    initial_chart_replay: Option<OrdinaryChartReplay>,
    root_replay: Option<ValidatedOrdinaryRootReplay>,
    bridge_replays: Vec<OrdinaryBridgeReplay>,
    clock_ledger: Vec<OrdinaryChainClockLedgerEntry>,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    unsupported_segment_index: Option<usize>,
    replay_failure: Option<OrdinaryChainReplayFailure>,
    target_preimage: Option<RationalInterval>,
    final_tube_replay: Option<OrdinaryTubeReplay>,
    final_enclosure: Option<OrdinaryChainFinalEnclosure>,
    right_frontier: Option<OrdinaryChainRightFrontier>,
    maximum_component_width: Option<BigRational>,
    mathematical_to_target: bool,
}

impl RawOrdinaryOnlyChainReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_RAW_ORDINARY_ONLY_CHAIN_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[RawOrdinaryOnlyChainObligation; 8] {
        &self.obligations
    }

    pub fn initial_chart_replay(&self) -> Option<&OrdinaryChartReplay> {
        self.initial_chart_replay.as_ref()
    }

    pub fn root_replay(&self) -> Option<&ValidatedOrdinaryRootReplay> {
        self.root_replay.as_ref()
    }

    pub fn bridge_replays(&self) -> &[OrdinaryBridgeReplay] {
        &self.bridge_replays
    }

    pub fn clock_ledger(&self) -> &[OrdinaryChainClockLedgerEntry] {
        &self.clock_ledger
    }

    pub const fn certified_segment_count(&self) -> usize {
        self.certified_segment_count
    }

    pub const fn failed_segment_index(&self) -> Option<usize> {
        self.failed_segment_index
    }

    pub const fn unsupported_segment_index(&self) -> Option<usize> {
        self.unsupported_segment_index
    }

    pub fn replay_failure(&self) -> Option<&OrdinaryChainReplayFailure> {
        self.replay_failure.as_ref()
    }

    pub fn target_preimage(&self) -> Option<&RationalInterval> {
        self.target_preimage.as_ref()
    }

    pub fn final_tube_replay(&self) -> Option<&OrdinaryTubeReplay> {
        self.final_tube_replay.as_ref()
    }

    pub fn final_enclosure(&self) -> Option<&OrdinaryChainFinalEnclosure> {
        self.final_enclosure.as_ref()
    }

    pub fn right_frontier(&self) -> Option<&OrdinaryChainRightFrontier> {
        self.right_frontier.as_ref()
    }

    pub fn maximum_component_width(&self) -> Option<&BigRational> {
        self.maximum_component_width.as_ref()
    }

    /// Whether obligations 6 through 12 establish a state enclosure at `T`.
    /// The requested display-width gate (obligation 13) is intentionally not
    /// part of this mathematical predicate.
    pub const fn mathematical_to_target(&self) -> bool {
        self.mathematical_to_target
    }

    pub fn profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(|obligation| obligation.satisfied)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawOrdinaryOnlyChainReplayError {
    InitialChartSemantic(OrdinarySemanticError),
    InitialChartReplay(OrdinaryChartReplayError),
    Root(ValidatedOrdinaryRootReplayError),
    FrontierTube(OrdinaryTubeReplayError),
    PositionPolynomial(PolynomialError),
    VelocityPolynomial(PolynomialError),
    Numeric(NumericError),
    InternalInvariant { detail: &'static str },
}

impl fmt::Display for RawOrdinaryOnlyChainReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitialChartSemantic(source) => {
                write!(
                    formatter,
                    "ordinary-chain initial chart semantic failure: {source}"
                )
            }
            Self::InitialChartReplay(source) => {
                write!(
                    formatter,
                    "ordinary-chain initial chart replay failure: {source}"
                )
            }
            Self::Root(source) => write!(formatter, "ordinary-chain root replay failure: {source}"),
            Self::FrontierTube(source) => {
                write!(
                    formatter,
                    "ordinary-chain frontier tube replay failure: {source}"
                )
            }
            Self::PositionPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary-chain position evaluation failure: {source}"
                )
            }
            Self::VelocityPolynomial(source) => {
                write!(
                    formatter,
                    "ordinary-chain velocity evaluation failure: {source}"
                )
            }
            Self::Numeric(source) => write!(formatter, "ordinary-chain numeric failure: {source}"),
            Self::InternalInvariant { detail } => {
                write!(
                    formatter,
                    "ordinary-chain internal invariant failed: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for RawOrdinaryOnlyChainReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InitialChartSemantic(source) => Some(source),
            Self::InitialChartReplay(source) => Some(source),
            Self::Root(source) => Some(source),
            Self::FrontierTube(source) => Some(source),
            Self::PositionPolynomial(source) | Self::VelocityPolynomial(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for RawOrdinaryOnlyChainReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Replay the bounded v0.4 ordinary-only chain checkpoint.
///
/// This function assumes a caller has already obtained a typed
/// [`RawPlanarChainWire`]. It does not implement raw byte acceptance, global
/// identifier-namespace checking, LC replay, or a complete raw-chain fold.
pub fn replay_raw_ordinary_only_chain_exact_rational_v04(
    chain: &RawPlanarChainWire,
) -> Result<RawOrdinaryOnlyChainReplay, RawOrdinaryOnlyChainReplayError> {
    let initial_chart = ordinary_chart_input_from_wire(&chain.initial_chart)
        .map_err(RawOrdinaryOnlyChainReplayError::InitialChartSemantic)?;
    let initial_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
    let root_exact = strict_root_exact_gate(chain, &initial_chart, &initial_tube);

    let mut initial_chart_replay = None;
    let mut root_replay = None;
    let mut bridge_replays = Vec::new();
    let mut clock_ledger = Vec::new();
    let mut certified_segment_count = 0;
    let mut failed_segment_index = None;
    let mut unsupported_segment_index = None;
    let mut replay_failure = None;
    let mut target_preimage = None;
    let mut final_tube_replay = None;
    let mut final_enclosure = None;
    let mut right_frontier = None;
    let mut maximum_component_width = None;

    let mut root_certified = false;
    let mut current_chart = initial_chart;
    let mut current_tube = initial_tube;
    let mut current_clock = None;

    if root_exact {
        let chart_replay = replay_ordinary_chart_exact_rational_claimed_tail_v04(&current_chart)
            .map_err(RawOrdinaryOnlyChainReplayError::InitialChartReplay)?;
        let validated_root = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &current_chart,
            &current_tube,
        )
        .map_err(RawOrdinaryOnlyChainReplayError::Root)?;
        root_certified = chart_replay.conditional_profile_satisfied()
            && validated_root.validated_root_satisfied();

        if root_certified {
            let origin = validated_root.root_clock_origin().ok_or(
                RawOrdinaryOnlyChainReplayError::InternalInvariant {
                    detail: "satisfied validated root has no exact clock origin",
                },
            )?;
            let point = RationalInterval::try_point(origin.clone())?;
            clock_ledger.push(OrdinaryChainClockLedgerEntry {
                vertex_index: 0,
                chart_id: current_chart.chart_id().to_owned(),
                clock_origin: point.clone(),
            });
            current_clock = Some(point);
        }
        initial_chart_replay = Some(chart_replay);
        root_replay = Some(validated_root);
    }

    if root_certified && chain.segments.len() > HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS {
        replay_failure = Some(OrdinaryChainReplayFailure::SegmentLimitExceeded {
            actual: chain.segments.len(),
            limit: HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
        });
    } else if root_certified {
        for (segment_index, segment) in chain.segments.iter().enumerate() {
            match segment {
                SegmentWire::OrdinaryBridge(segment) => {
                    let target_chart = match ordinary_chart_input_from_wire(&segment.target_chart) {
                        Ok(chart) => chart,
                        Err(source) => {
                            failed_segment_index = Some(segment_index);
                            replay_failure =
                                Some(OrdinaryChainReplayFailure::TargetChartSemantic {
                                    segment_index,
                                    source,
                                });
                            break;
                        }
                    };
                    let target_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                    let parent_clock = current_clock.as_ref().ok_or(
                        RawOrdinaryOnlyChainReplayError::InternalInvariant {
                            detail: "admitted ordinary prefix has no current clock",
                        },
                    )?;
                    let bridge = match replay_carried_ordinary_bridge_exact_rational_v04(
                        &segment.transition,
                        &current_chart,
                        &current_tube,
                        &target_chart,
                        &target_tube,
                        parent_clock,
                    ) {
                        Ok(bridge) => bridge,
                        Err(source) => {
                            failed_segment_index = Some(segment_index);
                            replay_failure = Some(OrdinaryChainReplayFailure::Bridge {
                                segment_index,
                                source,
                            });
                            break;
                        }
                    };
                    let commits = bridge.local_bridge_satisfied();
                    let committed_clock = if commits {
                        Some(
                            bridge
                                .target_clock_origin()
                                .ok_or(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                                    detail: "satisfied bridge has no target clock",
                                })?
                                .clone(),
                        )
                    } else {
                        None
                    };
                    bridge_replays.push(bridge);
                    if !commits {
                        failed_segment_index = Some(segment_index);
                        break;
                    }

                    current_chart = target_chart;
                    current_tube = target_tube;
                    current_clock = committed_clock;
                    certified_segment_count += 1;
                    clock_ledger.push(OrdinaryChainClockLedgerEntry {
                        vertex_index: certified_segment_count,
                        chart_id: current_chart.chart_id().to_owned(),
                        clock_origin: current_clock
                            .as_ref()
                            .ok_or(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                                detail: "committed bridge lost its target clock",
                            })?
                            .clone(),
                    });
                }
                SegmentWire::PlanarLcPassage(_) => {
                    unsupported_segment_index = Some(segment_index);
                    break;
                }
            }
        }
    }

    let all_segments_folded = root_certified
        && failed_segment_index.is_none()
        && unsupported_segment_index.is_none()
        && replay_failure.is_none()
        && certified_segment_count == chain.segments.len();

    let mut target_not_before = false;
    let mut preimage_derived = false;
    let mut preimage_inside = false;
    let mut state_evaluated = false;
    let mut width_within = false;

    if all_segments_folded {
        let clock =
            current_clock
                .as_ref()
                .ok_or(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                    detail: "fully folded ordinary prefix has no current clock",
                })?;
        let target = chain.requested_target_time.binary64_rational();
        let domain = current_chart.parameter_interval();
        let current_left_physical = checked_add(domain.lower(), clock.upper())?;
        target_not_before = target >= &current_left_physical;

        if target_not_before {
            let candidate = RationalInterval::new(
                checked_subtract(target, clock.upper())?,
                checked_subtract(target, clock.lower())?,
            )?;
            preimage_derived = true;
            preimage_inside = domain.contains_interval(&candidate);
            target_preimage = Some(candidate.clone());

            if preimage_inside {
                match replay_ordinary_tube_exact_rational_v04(&current_chart, &current_tube) {
                    Ok(tube_replay) => {
                        if tube_replay.certified() {
                            let radius = tube_replay.gronwall_upper().ok_or(
                                RawOrdinaryOnlyChainReplayError::InternalInvariant {
                                    detail: "certified final tube has no Gronwall upper bound",
                                },
                            )?;
                            let position_intervals = evaluate_and_inflate(
                                current_chart.position_polynomial(),
                                &candidate,
                                radius,
                                true,
                            )?;
                            let velocity_intervals = evaluate_and_inflate(
                                current_chart.velocity_polynomial(),
                                &candidate,
                                radius,
                                false,
                            )?;
                            if position_intervals.len() != POSITION_DIMENSION
                                || velocity_intervals.len() != POSITION_DIMENSION
                            {
                                return Err(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                                    detail: "ordinary final enclosure is not planar three-body",
                                });
                            }
                            let maximum_width = position_intervals
                                .iter()
                                .chain(&velocity_intervals)
                                .map(RationalInterval::width)
                                .max()
                                .unwrap_or_else(BigRational::zero);
                            validate_public_rational(&maximum_width)?;
                            width_within = &maximum_width
                                <= chain.requested_maximum_component_width.binary64_rational();
                            final_enclosure = Some(OrdinaryChainFinalEnclosure::new(
                                target.clone(),
                                candidate,
                                position_intervals,
                                velocity_intervals,
                            ));
                            maximum_component_width = Some(maximum_width);
                            state_evaluated = true;
                        }
                        final_tube_replay = Some(tube_replay);
                    }
                    Err(source) => {
                        replay_failure = Some(OrdinaryChainReplayFailure::FinalTube { source });
                    }
                }
            }
        }
    }

    // A fixed-time enclosure is the stronger result, including when only its
    // requested-width presentation gate fails. Otherwise retain a certified
    // right-edge enclosure for the longest committed ordinary prefix.
    if root_certified
        && final_enclosure.is_none()
        && !matches!(
            replay_failure,
            Some(OrdinaryChainReplayFailure::FinalTube { .. })
        )
    {
        let clock =
            current_clock
                .as_ref()
                .ok_or(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                    detail: "certified ordinary prefix has no current clock",
                })?;
        right_frontier = reconstruct_ordinary_right_frontier_exact_rational_v04(
            &current_chart,
            &current_tube,
            clock,
            certified_segment_count,
        )?;
    }

    let satisfaction = [
        root_exact,
        root_certified,
        all_segments_folded,
        target_not_before,
        preimage_derived,
        preimage_inside,
        state_evaluated,
        width_within,
    ];
    let mathematical_to_target = satisfaction[..7].iter().all(|value| *value);

    Ok(RawOrdinaryOnlyChainReplay {
        obligations: std::array::from_fn(|index| RawOrdinaryOnlyChainObligation {
            id: RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        initial_chart_replay,
        root_replay,
        bridge_replays,
        clock_ledger,
        certified_segment_count,
        failed_segment_index,
        unsupported_segment_index,
        replay_failure,
        target_preimage,
        final_tube_replay,
        final_enclosure,
        right_frontier,
        maximum_component_width,
        mathematical_to_target,
    })
}

pub(crate) fn strict_root_exact_gate(
    chain: &RawPlanarChainWire,
    chart: &OrdinaryChartInput,
    tube: &OrdinaryTubeInput,
) -> bool {
    let binding = &chain.root_binding;
    let planar_state = binding.positions.len() == 3
        && binding.positions.iter().all(|row| row.len() == 2)
        && binding.velocities.len() == 3
        && binding.velocities.iter().all(|row| row.len() == 2);
    let masses_match = binding.masses.len() == 3
        && binding
            .masses
            .iter()
            .zip(chart.masses())
            .all(|(left, right)| left.binary64_rational() == right);
    !binding.binding_id.is_empty()
        && !binding.chart_id.is_empty()
        && !binding.source.is_empty()
        && binding.masses.len() == 3
        && binding
            .masses
            .iter()
            .all(|mass| mass.binary64_rational().is_positive())
        && planar_state
        && !chart.chart_id().is_empty()
        && !chart.source().is_empty()
        && !tube.tube_id().is_empty()
        && !tube.chart_id().is_empty()
        && !tube.source().is_empty()
        && ordinary_tube_binding_status(tube, chart).bound()
        && binding.chart_id == chart.chart_id()
        && chart.chart_id() == tube.chart_id()
        && masses_match
        && binding.time_tolerance.binary64_rational().is_zero()
        && binding.position_tolerance.binary64_rational().is_zero()
        && binding.velocity_tolerance.binary64_rational().is_zero()
        && binding.chart_parameter.binary64_rational() == chart.parameter_interval().lower()
        && binding.chart_parameter.binary64_rational() == tube.anchor_parameter()
}

pub(crate) fn evaluate_and_inflate(
    polynomial: &crate::ExactRationalPolynomial,
    argument: &RationalInterval,
    radius: &BigRational,
    position: bool,
) -> Result<Vec<RationalInterval>, RawOrdinaryOnlyChainReplayError> {
    validate_public_rational(radius)?;
    let values = polynomial.evaluate(argument).map_err(|source| {
        if position {
            RawOrdinaryOnlyChainReplayError::PositionPolynomial(source)
        } else {
            RawOrdinaryOnlyChainReplayError::VelocityPolynomial(source)
        }
    })?;
    let inflation = RationalInterval::new(-radius.clone(), radius.clone())?;
    values
        .iter()
        .map(|value| value.add(&inflation).map_err(Into::into))
        .collect()
}

pub(crate) fn checked_add(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, RawOrdinaryOnlyChainReplayError> {
    admit_rational(left + right)
}

pub(crate) fn checked_subtract(
    left: &BigRational,
    right: &BigRational,
) -> Result<BigRational, RawOrdinaryOnlyChainReplayError> {
    admit_rational(left - right)
}

fn admit_rational(value: BigRational) -> Result<BigRational, RawOrdinaryOnlyChainReplayError> {
    validate_public_rational(&value)?;
    Ok(value)
}

/// Freshly replay and reconstruct the right edge of one committed ordinary
/// chart.  Both chain orchestrators share this exact implementation so
/// retention semantics cannot drift.
pub(crate) fn reconstruct_ordinary_right_frontier_exact_rational_v04(
    chart: &OrdinaryChartInput,
    tube: &OrdinaryTubeInput,
    clock: &RationalInterval,
    committed_segment_count: usize,
) -> Result<Option<OrdinaryChainRightFrontier>, RawOrdinaryOnlyChainReplayError> {
    let tube_replay = replay_ordinary_tube_exact_rational_v04(chart, tube)
        .map_err(RawOrdinaryOnlyChainReplayError::FrontierTube)?;
    if !tube_replay.certified() {
        return Ok(None);
    }
    let radius =
        tube_replay
            .gronwall_upper()
            .ok_or(RawOrdinaryOnlyChainReplayError::InternalInvariant {
                detail: "certified frontier tube has no Gronwall upper bound",
            })?;
    let right_parameter = chart.parameter_interval().upper().clone();
    let argument = RationalInterval::try_point(right_parameter.clone())?;
    let position_intervals =
        evaluate_and_inflate(chart.position_polynomial(), &argument, radius, true)?;
    let velocity_intervals =
        evaluate_and_inflate(chart.velocity_polynomial(), &argument, radius, false)?;
    if position_intervals.len() != POSITION_DIMENSION
        || velocity_intervals.len() != POSITION_DIMENSION
    {
        return Err(RawOrdinaryOnlyChainReplayError::InternalInvariant {
            detail: "ordinary frontier enclosure is not planar three-body",
        });
    }
    let physical_time_interval = RationalInterval::new(
        checked_add(&right_parameter, clock.lower())?,
        checked_add(&right_parameter, clock.upper())?,
    )?;
    Ok(Some(OrdinaryChainRightFrontier {
        committed_segment_count,
        chart_id: chart.chart_id().to_owned(),
        right_parameter,
        physical_time_interval,
        position_intervals,
        velocity_intervals,
        tube_replay,
    }))
}
