//! Bounded exact-rational replay of one admitted mixed ordinary/planar-LC
//! chain after the outer raw-v1 admission boundary.
//!
//! This profile covers only the existing ordered top-level obligations 6--13.
//! It transactionally folds freshly replayed local bridge and LC-exit profiles,
//! derives every clock internally, and implements the frozen retained-frontier
//! priority.  It does not establish outer obligations 1--5, raw hashing,
//! serialization, frozen arithmetic parity, or verifier independence.

use core::fmt;

use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    ordinary_chain::{
        checked_add, checked_subtract, evaluate_and_inflate,
        reconstruct_ordinary_right_frontier_exact_rational_v04, strict_root_exact_gate,
    },
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire,
    planar_lc_exit::replay_admission_bound_planar_lc_right_exact_rational_v04,
    rational_input::validate_public_rational,
    raw_schema::SegmentWire,
    replay_carried_ordinary_bridge_exact_rational_v04,
    replay_carried_planar_lc_exit_exact_rational_v04,
    replay_ordinary_chart_exact_rational_claimed_tail_v04, replay_ordinary_tube_exact_rational_v04,
    replay_validated_ordinary_root_exact_rational_v04, CanonicalRawV1Admission,
    CarriedPlanarLcExitError, CarriedPlanarLcExitReplay, NumericError, OrdinaryBridgeReplay,
    OrdinaryBridgeReplayError, OrdinaryChainFinalEnclosure, OrdinaryChainRightFrontier,
    OrdinaryChartInput, OrdinaryChartReplay, OrdinaryChartReplayError, OrdinarySemanticError,
    OrdinaryTubeReplay, OrdinaryTubeReplayError, PolynomialError, RationalInterval,
    RawOrdinaryOnlyChainReplayError, ValidatedOrdinaryRootReplay, ValidatedOrdinaryRootReplayError,
    HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS, RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS,
};

const POSITION_DIMENSION: usize = 6;

pub const EXACT_RATIONAL_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID: &str =
    "exact_rational_raw_mixed_planar_chain_v04";
pub const RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS: [&str; 8] = RAW_ORDINARY_ONLY_CHAIN_OBLIGATION_IDS;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawMixedPlanarChainObligation {
    id: &'static str,
    satisfied: bool,
}

impl RawMixedPlanarChainObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedPlanarSegmentKind {
    OrdinaryBridge,
    PlanarLcPassage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedOrdinarySegmentReplay {
    segment_index: usize,
    replay: OrdinaryBridgeReplay,
}

impl MixedOrdinarySegmentReplay {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    pub fn replay(&self) -> &OrdinaryBridgeReplay {
        &self.replay
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedPlanarLcSegmentReplay {
    segment_index: usize,
    replay: CarriedPlanarLcExitReplay,
}

impl MixedPlanarLcSegmentReplay {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    pub fn replay(&self) -> &CarriedPlanarLcExitReplay {
        &self.replay
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedPlanarSegmentReplay {
    Ordinary(Box<MixedOrdinarySegmentReplay>),
    PlanarLc(Box<MixedPlanarLcSegmentReplay>),
}

impl MixedPlanarSegmentReplay {
    pub const fn segment_index(&self) -> usize {
        match self {
            Self::Ordinary(value) => value.segment_index,
            Self::PlanarLc(value) => value.segment_index,
        }
    }

    pub const fn kind(&self) -> MixedPlanarSegmentKind {
        match self {
            Self::Ordinary(_) => MixedPlanarSegmentKind::OrdinaryBridge,
            Self::PlanarLc(_) => MixedPlanarSegmentKind::PlanarLcPassage,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedChainClockLedgerEntry {
    vertex_index: usize,
    chart_id: String,
    clock_origin: RationalInterval,
}

impl MixedChainClockLedgerEntry {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedOrdinaryCocycle {
    segment_index: usize,
    transition_id: String,
    source_chart_id: String,
    target_chart_id: String,
    source_clock_origin: RationalInterval,
    parameter_translation: BigRational,
    target_clock_origin: RationalInterval,
}

impl MixedOrdinaryCocycle {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }
    pub fn transition_id(&self) -> &str {
        &self.transition_id
    }
    pub fn source_chart_id(&self) -> &str {
        &self.source_chart_id
    }
    pub fn target_chart_id(&self) -> &str {
        &self.target_chart_id
    }
    pub fn source_clock_origin(&self) -> &RationalInterval {
        &self.source_clock_origin
    }
    /// Exact `e-a` translation from source right to target left parameter.
    pub fn parameter_translation(&self) -> &BigRational {
        &self.parameter_translation
    }
    pub fn target_clock_origin(&self) -> &RationalInterval {
        &self.target_clock_origin
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedPlanarLcCocycle {
    segment_index: usize,
    entry_transition_id: String,
    exit_transition_id: String,
    source_chart_id: String,
    lc_chart_id: String,
    target_chart_id: String,
    source_clock_origin: RationalInterval,
    entry_time_interval: RationalInterval,
    exit_time_interval: RationalInterval,
    target_clock_origin: RationalInterval,
    pair: [usize; 2],
    selected_gauge_assignment: Vec<u8>,
}

impl MixedPlanarLcCocycle {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }
    pub fn entry_transition_id(&self) -> &str {
        &self.entry_transition_id
    }
    pub fn exit_transition_id(&self) -> &str {
        &self.exit_transition_id
    }
    pub fn source_chart_id(&self) -> &str {
        &self.source_chart_id
    }
    pub fn lc_chart_id(&self) -> &str {
        &self.lc_chart_id
    }
    pub fn target_chart_id(&self) -> &str {
        &self.target_chart_id
    }
    pub fn source_clock_origin(&self) -> &RationalInterval {
        &self.source_clock_origin
    }
    pub fn entry_time_interval(&self) -> &RationalInterval {
        &self.entry_time_interval
    }
    pub fn exit_time_interval(&self) -> &RationalInterval {
        &self.exit_time_interval
    }
    pub fn target_clock_origin(&self) -> &RationalInterval {
        &self.target_clock_origin
    }
    pub const fn pair(&self) -> [usize; 2] {
        self.pair
    }
    pub fn selected_gauge_assignment(&self) -> &[u8] {
        &self.selected_gauge_assignment
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedChainCocycleLedgerEntry {
    Ordinary(Box<MixedOrdinaryCocycle>),
    PlanarLc(Box<MixedPlanarLcCocycle>),
}

impl MixedChainCocycleLedgerEntry {
    pub const fn segment_index(&self) -> usize {
        match self {
            Self::Ordinary(value) => value.segment_index,
            Self::PlanarLc(value) => value.segment_index,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedChainLiftedLcRightFrontier {
    failed_segment_index: usize,
    lc_chart_id: String,
    pair: [usize; 2],
    right_parameter: BigRational,
    physical_time_interval: RationalInterval,
    lifted_state: crate::PlanarLcIntervalState,
}

impl MixedChainLiftedLcRightFrontier {
    pub const fn failed_segment_index(&self) -> usize {
        self.failed_segment_index
    }
    pub fn lc_chart_id(&self) -> &str {
        &self.lc_chart_id
    }
    pub const fn pair(&self) -> [usize; 2] {
        self.pair
    }
    pub fn right_parameter(&self) -> &BigRational {
        &self.right_parameter
    }
    pub fn physical_time_interval(&self) -> &RationalInterval {
        &self.physical_time_interval
    }
    pub fn lifted_state(&self) -> &crate::PlanarLcIntervalState {
        &self.lifted_state
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedChainRetainedRegion {
    OrdinaryFixedTime(Box<OrdinaryChainFinalEnclosure>),
    LiftedPlanarLcRight(Box<MixedChainLiftedLcRightFrontier>),
    CurrentOrdinaryRight(Box<OrdinaryChainRightFrontier>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedChainReplayFailure {
    InitialChartSemantic {
        source: OrdinarySemanticError,
    },
    SegmentLimitExceeded {
        actual: usize,
        limit: usize,
    },
    LocalObligations {
        segment_index: usize,
        kind: MixedPlanarSegmentKind,
    },
    OrdinaryTargetSemantic {
        segment_index: usize,
        source: OrdinarySemanticError,
    },
    OrdinaryBridgeKernel {
        segment_index: usize,
        source: OrdinaryBridgeReplayError,
    },
    PlanarLcExitKernel {
        segment_index: usize,
        source: CarriedPlanarLcExitError,
    },
    FinalTube {
        source: OrdinaryTubeReplayError,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawMixedPlanarChainReplay {
    obligations: [RawMixedPlanarChainObligation; 8],
    initial_chart_replay: Option<OrdinaryChartReplay>,
    root_replay: Option<ValidatedOrdinaryRootReplay>,
    segment_replays: Vec<MixedPlanarSegmentReplay>,
    clock_ledger: Vec<MixedChainClockLedgerEntry>,
    cocycle_ledger: Vec<MixedChainCocycleLedgerEntry>,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    failed_local_obligation_ids: Vec<&'static str>,
    replay_failure: Option<MixedChainReplayFailure>,
    current_chart: Option<OrdinaryChartInput>,
    current_clock_origin: Option<RationalInterval>,
    target_preimage: Option<RationalInterval>,
    final_tube_replay: Option<OrdinaryTubeReplay>,
    final_enclosure: Option<OrdinaryChainFinalEnclosure>,
    maximum_component_width: Option<BigRational>,
    covered_physical_interval: Option<RationalInterval>,
    retained_region: Option<MixedChainRetainedRegion>,
    mathematical_to_target: bool,
}

impl RawMixedPlanarChainReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID
    }
    pub fn obligations(&self) -> &[RawMixedPlanarChainObligation; 8] {
        &self.obligations
    }
    pub fn initial_chart_replay(&self) -> Option<&OrdinaryChartReplay> {
        self.initial_chart_replay.as_ref()
    }
    pub fn root_replay(&self) -> Option<&ValidatedOrdinaryRootReplay> {
        self.root_replay.as_ref()
    }
    pub fn segment_replays(&self) -> &[MixedPlanarSegmentReplay] {
        &self.segment_replays
    }
    pub fn clock_ledger(&self) -> &[MixedChainClockLedgerEntry] {
        &self.clock_ledger
    }
    pub fn cocycle_ledger(&self) -> &[MixedChainCocycleLedgerEntry] {
        &self.cocycle_ledger
    }
    pub const fn certified_segment_count(&self) -> usize {
        self.certified_segment_count
    }
    pub const fn failed_segment_index(&self) -> Option<usize> {
        self.failed_segment_index
    }
    pub fn failed_local_obligation_ids(&self) -> &[&'static str] {
        &self.failed_local_obligation_ids
    }
    pub fn replay_failure(&self) -> Option<&MixedChainReplayFailure> {
        self.replay_failure.as_ref()
    }
    pub fn current_chart(&self) -> Option<&OrdinaryChartInput> {
        self.current_chart.as_ref()
    }
    pub fn current_clock_origin(&self) -> Option<&RationalInterval> {
        self.current_clock_origin.as_ref()
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
    pub fn maximum_component_width(&self) -> Option<&BigRational> {
        self.maximum_component_width.as_ref()
    }
    pub fn covered_physical_interval(&self) -> Option<&RationalInterval> {
        self.covered_physical_interval.as_ref()
    }
    pub fn retained_region(&self) -> Option<&MixedChainRetainedRegion> {
        self.retained_region.as_ref()
    }
    pub const fn mathematical_to_target(&self) -> bool {
        self.mathematical_to_target
    }
    pub fn profile_satisfied(&self) -> bool {
        self.obligations.iter().all(|row| row.satisfied)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawMixedPlanarChainReplayError {
    InitialChartSemantic(OrdinarySemanticError),
    InitialChartReplay(OrdinaryChartReplayError),
    Root(ValidatedOrdinaryRootReplayError),
    OrdinaryShared(RawOrdinaryOnlyChainReplayError),
    PositionPolynomial(PolynomialError),
    VelocityPolynomial(PolynomialError),
    Numeric(NumericError),
    InternalInvariant { detail: &'static str },
}

impl fmt::Display for RawMixedPlanarChainReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "mixed planar chain replay failed: {self:?}")
    }
}

impl std::error::Error for RawMixedPlanarChainReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InitialChartSemantic(source) => Some(source),
            Self::InitialChartReplay(source) => Some(source),
            Self::Root(source) => Some(source),
            Self::OrdinaryShared(source) => Some(source),
            Self::PositionPolynomial(source) | Self::VelocityPolynomial(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for RawMixedPlanarChainReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Fold an admitted finite raw-v1 segment word under the bounded v0.4 mixed
/// profile.  A successful local replay is committed atomically; failed target
/// records and clocks remain diagnostic only.
pub fn replay_raw_mixed_planar_chain_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
) -> Result<RawMixedPlanarChainReplay, RawMixedPlanarChainReplayError> {
    let chain = admission.wire();
    let initial_chart = match ordinary_chart_input_from_wire(&chain.initial_chart) {
        Ok(value) => value,
        Err(source) if source.is_certificate_defect() => {
            return Ok(initial_chart_semantic_failure_replay(source));
        }
        Err(source) => {
            return Err(RawMixedPlanarChainReplayError::InitialChartSemantic(source));
        }
    };
    let initial_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
    let root_exact = strict_root_exact_gate(chain, &initial_chart, &initial_tube);

    let mut initial_chart_replay = None;
    let mut root_replay = None;
    let mut segment_replays = Vec::new();
    let mut clock_ledger = Vec::new();
    let mut cocycle_ledger = Vec::new();
    let mut certified_segment_count = 0;
    let mut failed_segment_index = None;
    let mut failed_local_obligation_ids = Vec::new();
    let mut replay_failure = None;
    let mut target_preimage = None;
    let mut final_tube_replay = None;
    let mut final_enclosure = None;
    let mut maximum_component_width = None;
    let mut failed_lc_frontier = None;

    let mut root_certified = false;
    let mut current_chart = initial_chart;
    let mut current_tube = initial_tube;
    let mut current_clock = None;

    if root_exact {
        let chart_replay = replay_ordinary_chart_exact_rational_claimed_tail_v04(&current_chart)
            .map_err(RawMixedPlanarChainReplayError::InitialChartReplay)?;
        let validated_root = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &current_chart,
            &current_tube,
        )
        .map_err(RawMixedPlanarChainReplayError::Root)?;
        root_certified = chart_replay.conditional_profile_satisfied()
            && validated_root.validated_root_satisfied();
        if root_certified {
            let origin = validated_root.root_clock_origin().ok_or(
                RawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "satisfied validated root has no exact clock origin",
                },
            )?;
            let point = RationalInterval::try_point(origin.clone())?;
            clock_ledger.push(MixedChainClockLedgerEntry {
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
        replay_failure = Some(MixedChainReplayFailure::SegmentLimitExceeded {
            actual: chain.segments.len(),
            limit: HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
        });
    } else if root_certified {
        for (segment_index, segment) in chain.segments.iter().enumerate() {
            let parent_clock = current_clock.as_ref().ok_or(
                RawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "certified mixed prefix has no current clock",
                },
            )?;
            match segment {
                SegmentWire::OrdinaryBridge(segment) => {
                    let target_chart = match ordinary_chart_input_from_wire(&segment.target_chart) {
                        Ok(value) => value,
                        Err(source) => {
                            failed_segment_index = Some(segment_index);
                            replay_failure =
                                Some(MixedChainReplayFailure::OrdinaryTargetSemantic {
                                    segment_index,
                                    source,
                                });
                            break;
                        }
                    };
                    let target_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                    let bridge = match replay_carried_ordinary_bridge_exact_rational_v04(
                        &segment.transition,
                        &current_chart,
                        &current_tube,
                        &target_chart,
                        &target_tube,
                        parent_clock,
                    ) {
                        Ok(value) => value,
                        Err(source) => {
                            failed_segment_index = Some(segment_index);
                            replay_failure = Some(MixedChainReplayFailure::OrdinaryBridgeKernel {
                                segment_index,
                                source,
                            });
                            break;
                        }
                    };
                    let commits = bridge.local_bridge_satisfied();
                    if !commits {
                        failed_local_obligation_ids = bridge
                            .obligations()
                            .iter()
                            .filter_map(|row| (!row.satisfied()).then_some(row.id()))
                            .collect();
                    }
                    let committed_clock = bridge.target_clock_origin().cloned();
                    segment_replays.push(MixedPlanarSegmentReplay::Ordinary(Box::new(
                        MixedOrdinarySegmentReplay {
                            segment_index,
                            replay: bridge,
                        },
                    )));
                    if !commits {
                        failed_segment_index = Some(segment_index);
                        replay_failure = Some(MixedChainReplayFailure::LocalObligations {
                            segment_index,
                            kind: MixedPlanarSegmentKind::OrdinaryBridge,
                        });
                        break;
                    }
                    let target_clock = committed_clock.ok_or(
                        RawMixedPlanarChainReplayError::InternalInvariant {
                            detail: "satisfied ordinary bridge has no target clock",
                        },
                    )?;
                    let translation = checked_subtract(
                        segment.transition.source_parameter.binary64_rational(),
                        segment.transition.target_parameter.binary64_rational(),
                    )
                    .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?;
                    cocycle_ledger.push(MixedChainCocycleLedgerEntry::Ordinary(Box::new(
                        MixedOrdinaryCocycle {
                            segment_index,
                            transition_id: segment.transition.transition_id.clone(),
                            source_chart_id: current_chart.chart_id().to_owned(),
                            target_chart_id: target_chart.chart_id().to_owned(),
                            source_clock_origin: parent_clock.clone(),
                            parameter_translation: translation,
                            target_clock_origin: target_clock.clone(),
                        },
                    )));
                    current_chart = target_chart;
                    current_tube = target_tube;
                    current_clock = Some(target_clock.clone());
                    certified_segment_count += 1;
                    clock_ledger.push(MixedChainClockLedgerEntry {
                        vertex_index: certified_segment_count,
                        chart_id: current_chart.chart_id().to_owned(),
                        clock_origin: target_clock,
                    });
                }
                SegmentWire::PlanarLcPassage(segment) => {
                    let exit = match replay_carried_planar_lc_exit_exact_rational_v04(
                        admission,
                        segment_index,
                        parent_clock,
                    ) {
                        Ok(value) => value,
                        Err(source) => {
                            failed_lc_frontier = reconstruct_failed_lc_frontier(
                                admission,
                                segment_index,
                                parent_clock,
                                None,
                            )?;
                            failed_segment_index = Some(segment_index);
                            replay_failure = Some(MixedChainReplayFailure::PlanarLcExitKernel {
                                segment_index,
                                source,
                            });
                            break;
                        }
                    };
                    let commits = exit.conditional_profile_satisfied();
                    if !commits {
                        failed_local_obligation_ids = exit
                            .obligations()
                            .iter()
                            .filter_map(|row| (!row.satisfied()).then_some(row.id()))
                            .collect();
                    }
                    let committed_clock = exit.target_clock_origin().cloned();
                    if !commits {
                        failed_lc_frontier = reconstruct_failed_lc_frontier(
                            admission,
                            segment_index,
                            parent_clock,
                            Some(&exit),
                        )?;
                    }
                    let entry_time = exit.entry_replay().entry_time_interval().cloned();
                    let exit_time = exit.exit_time_interval().cloned();
                    let selected_gauge = exit
                        .entry_replay()
                        .selected_assignment()
                        .map(<[u8]>::to_vec);
                    segment_replays.push(MixedPlanarSegmentReplay::PlanarLc(Box::new(
                        MixedPlanarLcSegmentReplay {
                            segment_index,
                            replay: exit,
                        },
                    )));
                    if !commits {
                        failed_segment_index = Some(segment_index);
                        replay_failure = Some(MixedChainReplayFailure::LocalObligations {
                            segment_index,
                            kind: MixedPlanarSegmentKind::PlanarLcPassage,
                        });
                        break;
                    }
                    let target_chart = ordinary_chart_input_from_wire(&segment.target_chart)
                        .map_err(RawMixedPlanarChainReplayError::InitialChartSemantic)?;
                    let target_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                    let target_clock = committed_clock.ok_or(
                        RawMixedPlanarChainReplayError::InternalInvariant {
                            detail: "satisfied planar LC exit has no target clock",
                        },
                    )?;
                    cocycle_ledger.push(MixedChainCocycleLedgerEntry::PlanarLc(Box::new(
                        MixedPlanarLcCocycle {
                            segment_index,
                            entry_transition_id: segment.entry_transition.transition_id.clone(),
                            exit_transition_id: segment.exit_transition.transition_id.clone(),
                            source_chart_id: current_chart.chart_id().to_owned(),
                            lc_chart_id: segment.lc_chart.chart_id.clone(),
                            target_chart_id: target_chart.chart_id().to_owned(),
                            source_clock_origin: parent_clock.clone(),
                            entry_time_interval: entry_time.ok_or(
                                RawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC entry has no D_in",
                                },
                            )?,
                            exit_time_interval: exit_time.ok_or(
                                RawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC exit has no D_out",
                                },
                            )?,
                            target_clock_origin: target_clock.clone(),
                            pair: canonical_pair(segment)?,
                            selected_gauge_assignment: selected_gauge.ok_or(
                                RawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC entry has no selected gauge",
                                },
                            )?,
                        },
                    )));
                    current_chart = target_chart;
                    current_tube = target_tube;
                    current_clock = Some(target_clock.clone());
                    certified_segment_count += 1;
                    clock_ledger.push(MixedChainClockLedgerEntry {
                        vertex_index: certified_segment_count,
                        chart_id: current_chart.chart_id().to_owned(),
                        clock_origin: target_clock,
                    });
                }
            }
        }
    }

    let all_segments_folded = root_certified
        && replay_failure.is_none()
        && failed_segment_index.is_none()
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
                .ok_or(RawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "fully folded mixed chain has no current clock",
                })?;
        let target = chain.requested_target_time.binary64_rational();
        let domain = current_chart.parameter_interval();
        let current_left_physical = checked_add(domain.lower(), clock.upper())
            .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?;
        target_not_before = target >= &current_left_physical;
        if target_not_before {
            let candidate = RationalInterval::new(
                checked_subtract(target, clock.upper())
                    .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?,
                checked_subtract(target, clock.lower())
                    .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?,
            )?;
            preimage_derived = true;
            preimage_inside = domain.contains_interval(&candidate);
            target_preimage = Some(candidate.clone());
            if preimage_inside {
                match replay_ordinary_tube_exact_rational_v04(&current_chart, &current_tube) {
                    Ok(tube_replay) => {
                        if tube_replay.certified() {
                            let radius = tube_replay.gronwall_upper().ok_or(
                                RawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "certified final tube has no Gronwall upper bound",
                                },
                            )?;
                            let position_intervals = evaluate_and_inflate(
                                current_chart.position_polynomial(),
                                &candidate,
                                radius,
                                true,
                            )
                            .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?;
                            let velocity_intervals = evaluate_and_inflate(
                                current_chart.velocity_polynomial(),
                                &candidate,
                                radius,
                                false,
                            )
                            .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?;
                            if position_intervals.len() != POSITION_DIMENSION
                                || velocity_intervals.len() != POSITION_DIMENSION
                            {
                                return Err(RawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "mixed final enclosure is not planar three-body",
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
                        replay_failure = Some(MixedChainReplayFailure::FinalTube { source });
                    }
                }
            }
        }
    }

    let ordinary_frontier = if root_certified
        && final_enclosure.is_none()
        && !matches!(
            replay_failure,
            Some(MixedChainReplayFailure::FinalTube { .. })
        ) {
        Some(
            reconstruct_ordinary_right_frontier_exact_rational_v04(
                &current_chart,
                &current_tube,
                current_clock.as_ref().ok_or(
                    RawMixedPlanarChainReplayError::InternalInvariant {
                        detail: "certified mixed prefix has no current clock for retention",
                    },
                )?,
                certified_segment_count,
            )
            .map_err(RawMixedPlanarChainReplayError::OrdinaryShared)?,
        )
        .flatten()
    } else {
        None
    };

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
    let initial_time = chain.root_binding.initial_time.binary64_rational();
    let covered_physical_interval = if mathematical_to_target {
        Some(RationalInterval::new(
            initial_time.clone(),
            chain.requested_target_time.binary64_rational().clone(),
        )?)
    } else if ordinary_frontier.is_some() || failed_lc_frontier.is_some() {
        let mut covered_upper = initial_time.clone();
        if let Some(frontier) = &ordinary_frontier {
            covered_upper = covered_upper.max(frontier.physical_time_interval().lower().clone());
        }
        if let Some(frontier) = &failed_lc_frontier {
            covered_upper = covered_upper.max(frontier.physical_time_interval().lower().clone());
        }
        Some(RationalInterval::new(initial_time.clone(), covered_upper)?)
    } else {
        None
    };
    let retained_region = if mathematical_to_target && !width_within {
        Some(MixedChainRetainedRegion::OrdinaryFixedTime(Box::new(
            final_enclosure
                .as_ref()
                .ok_or(RawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "width-only failure has no fixed-time enclosure",
                })?
                .clone(),
        )))
    } else if mathematical_to_target {
        None
    } else if let Some(frontier) = failed_lc_frontier {
        Some(MixedChainRetainedRegion::LiftedPlanarLcRight(Box::new(
            frontier,
        )))
    } else {
        ordinary_frontier
            .map(Box::new)
            .map(MixedChainRetainedRegion::CurrentOrdinaryRight)
    };

    // Populate the private typed foundation for every retained failure while
    // preserving the v0.4 replay object and all public outcomes unchanged.
    let _failure_disposition = replay_failure
        .as_ref()
        .map(MixedChainReplayFailure::disposition);

    Ok(RawMixedPlanarChainReplay {
        obligations: std::array::from_fn(|index| RawMixedPlanarChainObligation {
            id: RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        initial_chart_replay,
        root_replay,
        segment_replays,
        clock_ledger,
        cocycle_ledger,
        certified_segment_count,
        failed_segment_index,
        failed_local_obligation_ids,
        replay_failure,
        current_chart: root_certified.then_some(current_chart),
        current_clock_origin: current_clock,
        target_preimage,
        final_tube_replay,
        final_enclosure,
        maximum_component_width,
        covered_physical_interval,
        retained_region,
        mathematical_to_target,
    })
}

fn initial_chart_semantic_failure_replay(
    source: OrdinarySemanticError,
) -> RawMixedPlanarChainReplay {
    RawMixedPlanarChainReplay {
        obligations: std::array::from_fn(|index| RawMixedPlanarChainObligation {
            id: RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[index],
            satisfied: false,
        }),
        initial_chart_replay: None,
        root_replay: None,
        segment_replays: Vec::new(),
        clock_ledger: Vec::new(),
        cocycle_ledger: Vec::new(),
        certified_segment_count: 0,
        failed_segment_index: None,
        failed_local_obligation_ids: Vec::new(),
        replay_failure: Some(MixedChainReplayFailure::InitialChartSemantic { source }),
        current_chart: None,
        current_clock_origin: None,
        target_preimage: None,
        final_tube_replay: None,
        final_enclosure: None,
        maximum_component_width: None,
        covered_physical_interval: None,
        retained_region: None,
        mathematical_to_target: false,
    }
}

fn reconstruct_failed_lc_frontier(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_clock: &RationalInterval,
    exit_replay: Option<&CarriedPlanarLcExitReplay>,
) -> Result<Option<MixedChainLiftedLcRightFrontier>, RawMixedPlanarChainReplayError> {
    if let Some(replay) = exit_replay {
        if let Some(state) = replay.lifted_exit_slice() {
            return build_lc_frontier(admission, segment_index, state.clone()).map(Some);
        }
        if !replay.entry_replay().conditional_profile_satisfied()
            || !replay.lc_tube_replay().certified()
        {
            return Ok(None);
        }
    }
    match replay_admission_bound_planar_lc_right_exact_rational_v04(
        admission,
        segment_index,
        parent_clock,
    ) {
        Ok(replay) => replay
            .lifted_right_slice()
            .cloned()
            .map(|state| {
                let physical_time_interval = state.physical_time().clone();
                Ok(MixedChainLiftedLcRightFrontier {
                    failed_segment_index: segment_index,
                    lc_chart_id: replay.lc_chart_id().to_owned(),
                    pair: replay.pair(),
                    right_parameter: replay.right_parameter().clone(),
                    physical_time_interval,
                    lifted_state: state,
                })
            })
            .transpose(),
        // The primary local error remains the structured failure.  Failure to
        // independently justify a later frontier merely selects the ordinary
        // fallback; it cannot turn rejected evidence into a verifier error.
        Err(_) => Ok(None),
    }
}

fn build_lc_frontier(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    state: crate::PlanarLcIntervalState,
) -> Result<MixedChainLiftedLcRightFrontier, RawMixedPlanarChainReplayError> {
    let passage = admission.planar_lc_passage(segment_index).ok_or(
        RawMixedPlanarChainReplayError::InternalInvariant {
            detail: "LC replay has no admission-bound passage",
        },
    )?;
    Ok(MixedChainLiftedLcRightFrontier {
        failed_segment_index: segment_index,
        lc_chart_id: passage.lc_chart.chart_id.clone(),
        pair: canonical_pair(passage)?,
        right_parameter: passage.lc_chart.parameter_interval[1]
            .binary64_rational()
            .clone(),
        physical_time_interval: state.physical_time().clone(),
        lifted_state: state,
    })
}

fn canonical_pair(
    passage: &crate::raw_schema::PlanarLcPassageSegmentWire,
) -> Result<[usize; 2], RawMixedPlanarChainReplayError> {
    use num_traits::ToPrimitive;
    let left = passage.lc_chart.pair[0].value().to_usize();
    let right = passage.lc_chart.pair[1].value().to_usize();
    match (left, right) {
        (Some(left), Some(right)) if left < right && right < 3 => Ok([left, right]),
        _ => Err(RawMixedPlanarChainReplayError::InternalInvariant {
            detail: "certified LC replay lost its canonical pair",
        }),
    }
}
