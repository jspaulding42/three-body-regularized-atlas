//! Proof-grade transactional mixed ordinary/planar-LC chain replay.
//!
//! This profile retains compatibility claimed-tail chart replays as typed
//! diagnostics only.  Its eight decisive rows are derived from the exact root,
//! direct ordinary bridges, proof-grade LC exits, and terminal enclosure.

use core::fmt;

use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    mixed_chain::{
        MixedChainClockLedgerEntry, MixedChainCocycleLedgerEntry, MixedChainLiftedLcRightFrontier,
        MixedChainReplayFailure, MixedChainRetainedRegion, MixedOrdinaryCocycle,
        MixedOrdinarySegmentReplay, MixedPlanarLcCocycle, MixedPlanarSegmentKind,
    },
    ordinary_chain::{
        checked_add, checked_subtract, evaluate_and_inflate,
        reconstruct_ordinary_right_frontier_exact_rational_v04, strict_root_exact_gate,
    },
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire,
    planar_lc_exit::replay_proof_grade_admission_bound_planar_lc_right_exact_rational_v04,
    rational_input::validate_public_rational,
    raw_schema::SegmentWire,
    replay_carried_ordinary_bridge_exact_rational_v04,
    replay_ordinary_chart_exact_rational_claimed_tail_v04, replay_ordinary_tube_exact_rational_v04,
    replay_proof_grade_carried_planar_lc_exit_exact_rational_v04,
    replay_validated_ordinary_root_exact_rational_v04, CanonicalRawV1Admission, NumericError,
    OrdinaryChainFinalEnclosure, OrdinaryChartInput, OrdinaryChartReplay, OrdinaryChartReplayError,
    OrdinarySemanticError, OrdinaryTubeReplay, PolynomialError,
    ProofGradeCarriedPlanarLcExitReplay, RationalInterval, RawOrdinaryOnlyChainReplayError,
    ValidatedOrdinaryRootReplay, ValidatedOrdinaryRootReplayError,
    HARD_MAX_RAW_ORDINARY_CHAIN_SEGMENTS,
};

const POSITION_DIMENSION: usize = 6;

pub const EXACT_RATIONAL_PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID: &str =
    "exact_rational_proof_grade_raw_mixed_planar_chain_v04";

pub const PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS: [&str; 8] = [
    "proof_grade_raw_planar_chain_root_exact_point_left_anchor",
    "proof_grade_raw_planar_chain_root_direct_tube_and_binding_certified",
    "proof_grade_raw_planar_chain_all_segments_direct_evidence_folded",
    "proof_grade_raw_planar_chain_target_not_before_current_left_clock",
    "proof_grade_raw_planar_chain_fixed_time_preimage_exactly_derived",
    "proof_grade_raw_planar_chain_fixed_time_preimage_inside_forward_current_domain",
    "proof_grade_raw_planar_chain_target_state_directly_evaluated_and_inflated",
    "proof_grade_raw_planar_chain_final_component_width_within_requested_bound",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeRawMixedPlanarChainObligation {
    id: &'static str,
    satisfied: bool,
}

impl ProofGradeRawMixedPlanarChainObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeMixedPlanarLcSegmentReplay {
    segment_index: usize,
    replay: ProofGradeCarriedPlanarLcExitReplay,
}

impl ProofGradeMixedPlanarLcSegmentReplay {
    pub(crate) const fn new(
        segment_index: usize,
        replay: ProofGradeCarriedPlanarLcExitReplay,
    ) -> Self {
        Self {
            segment_index,
            replay,
        }
    }

    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    pub fn replay(&self) -> &ProofGradeCarriedPlanarLcExitReplay {
        &self.replay
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProofGradeMixedPlanarSegmentReplay {
    Ordinary(Box<MixedOrdinarySegmentReplay>),
    PlanarLc(Box<ProofGradeMixedPlanarLcSegmentReplay>),
}

impl ProofGradeMixedPlanarSegmentReplay {
    pub const fn segment_index(&self) -> usize {
        match self {
            Self::Ordinary(value) => value.segment_index(),
            Self::PlanarLc(value) => value.segment_index(),
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
pub struct ProofGradeRawMixedPlanarChainReplay {
    obligations: [ProofGradeRawMixedPlanarChainObligation; 8],
    initial_claimed_tail_chart_diagnostic:
        Option<Result<OrdinaryChartReplay, OrdinaryChartReplayError>>,
    root_replay: Option<ValidatedOrdinaryRootReplay>,
    segment_replays: Vec<ProofGradeMixedPlanarSegmentReplay>,
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

impl ProofGradeRawMixedPlanarChainReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID
    }
    pub fn obligations(&self) -> &[ProofGradeRawMixedPlanarChainObligation; 8] {
        &self.obligations
    }
    pub fn initial_claimed_tail_chart_diagnostic(
        &self,
    ) -> Option<&Result<OrdinaryChartReplay, OrdinaryChartReplayError>> {
        self.initial_claimed_tail_chart_diagnostic.as_ref()
    }
    pub fn root_replay(&self) -> Option<&ValidatedOrdinaryRootReplay> {
        self.root_replay.as_ref()
    }
    pub fn segment_replays(&self) -> &[ProofGradeMixedPlanarSegmentReplay] {
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
pub enum ProofGradeRawMixedPlanarChainReplayError {
    InitialChartSemantic(OrdinarySemanticError),
    Root(ValidatedOrdinaryRootReplayError),
    OrdinaryShared(RawOrdinaryOnlyChainReplayError),
    PositionPolynomial(PolynomialError),
    VelocityPolynomial(PolynomialError),
    Numeric(NumericError),
    InternalInvariant { detail: &'static str },
}

impl fmt::Display for ProofGradeRawMixedPlanarChainReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "proof-grade mixed planar chain replay failed: {self:?}"
        )
    }
}

impl std::error::Error for ProofGradeRawMixedPlanarChainReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InitialChartSemantic(source) => Some(source),
            Self::Root(source) => Some(source),
            Self::OrdinaryShared(source) => Some(source),
            Self::PositionPolynomial(source) | Self::VelocityPolynomial(source) => Some(source),
            Self::Numeric(source) => Some(source),
            Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for ProofGradeRawMixedPlanarChainReplayError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Fold an admitted segment word using only the direct root and local direct
/// evidence.  Claimed-tail replays are retained as non-decisive diagnostics.
pub fn replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
) -> Result<ProofGradeRawMixedPlanarChainReplay, ProofGradeRawMixedPlanarChainReplayError> {
    let chain = admission.wire();
    let initial_chart = match ordinary_chart_input_from_wire(&chain.initial_chart) {
        Ok(value) => value,
        Err(source) if source.is_certificate_defect() => {
            return Ok(initial_chart_semantic_failure_replay(source));
        }
        Err(source) => {
            return Err(ProofGradeRawMixedPlanarChainReplayError::InitialChartSemantic(source))
        }
    };
    let initial_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
    let root_exact = strict_root_exact_gate(chain, &initial_chart, &initial_tube);

    let mut initial_claimed_tail_chart_diagnostic = None;
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
        // Deliberately do not branch on, unwrap, or inspect this diagnostic.
        initial_claimed_tail_chart_diagnostic = Some(
            replay_ordinary_chart_exact_rational_claimed_tail_v04(&current_chart),
        );
        let validated_root = replay_validated_ordinary_root_exact_rational_v04(
            &chain.root_binding,
            &current_chart,
            &current_tube,
        )
        .map_err(ProofGradeRawMixedPlanarChainReplayError::Root)?;
        root_certified = validated_root.validated_root_satisfied();
        if root_certified {
            let origin = validated_root.root_clock_origin().ok_or(
                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "satisfied validated root has no exact clock origin",
                },
            )?;
            let point = RationalInterval::try_point(origin.clone())?;
            clock_ledger.push(MixedChainClockLedgerEntry::new(
                0,
                current_chart.chart_id().to_owned(),
                point.clone(),
            ));
            current_clock = Some(point);
        }
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
                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                    detail: "certified proof-grade mixed prefix has no current clock",
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
                    segment_replays.push(ProofGradeMixedPlanarSegmentReplay::Ordinary(Box::new(
                        MixedOrdinarySegmentReplay::new(segment_index, bridge),
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
                        ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                            detail: "satisfied ordinary bridge has no target clock",
                        },
                    )?;
                    let translation = checked_subtract(
                        segment.transition.source_parameter.binary64_rational(),
                        segment.transition.target_parameter.binary64_rational(),
                    )
                    .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?;
                    cocycle_ledger.push(MixedChainCocycleLedgerEntry::Ordinary(Box::new(
                        MixedOrdinaryCocycle::new(
                            segment_index,
                            segment.transition.transition_id.clone(),
                            current_chart.chart_id().to_owned(),
                            target_chart.chart_id().to_owned(),
                            parent_clock.clone(),
                            translation,
                            target_clock.clone(),
                        ),
                    )));
                    current_chart = target_chart;
                    current_tube = target_tube;
                    current_clock = Some(target_clock.clone());
                    certified_segment_count += 1;
                    clock_ledger.push(MixedChainClockLedgerEntry::new(
                        certified_segment_count,
                        current_chart.chart_id().to_owned(),
                        target_clock,
                    ));
                }
                SegmentWire::PlanarLcPassage(segment) => {
                    let exit = match replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
                        admission,
                        segment_index,
                        parent_clock,
                    ) {
                        Ok(value) => value,
                        Err(source) => {
                            failed_lc_frontier = reconstruct_failed_proof_grade_lc_frontier(
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
                        failed_lc_frontier = reconstruct_failed_proof_grade_lc_frontier(
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
                    segment_replays.push(ProofGradeMixedPlanarSegmentReplay::PlanarLc(Box::new(
                        ProofGradeMixedPlanarLcSegmentReplay::new(segment_index, exit),
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
                        .map_err(ProofGradeRawMixedPlanarChainReplayError::InitialChartSemantic)?;
                    let target_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                    let target_clock = committed_clock.ok_or(
                        ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                            detail: "satisfied planar LC exit has no target clock",
                        },
                    )?;
                    cocycle_ledger.push(MixedChainCocycleLedgerEntry::PlanarLc(Box::new(
                        MixedPlanarLcCocycle::new(
                            segment_index,
                            segment.entry_transition.transition_id.clone(),
                            segment.exit_transition.transition_id.clone(),
                            current_chart.chart_id().to_owned(),
                            segment.lc_chart.chart_id.clone(),
                            target_chart.chart_id().to_owned(),
                            parent_clock.clone(),
                            entry_time.ok_or(
                                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC entry has no D_in",
                                },
                            )?,
                            exit_time.ok_or(
                                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC exit has no D_out",
                                },
                            )?,
                            target_clock.clone(),
                            canonical_pair(segment)?,
                            selected_gauge.ok_or(
                                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "satisfied planar LC entry has no selected gauge",
                                },
                            )?,
                        ),
                    )));
                    current_chart = target_chart;
                    current_tube = target_tube;
                    current_clock = Some(target_clock.clone());
                    certified_segment_count += 1;
                    clock_ledger.push(MixedChainClockLedgerEntry::new(
                        certified_segment_count,
                        current_chart.chart_id().to_owned(),
                        target_clock,
                    ));
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
        let clock = current_clock.as_ref().ok_or(
            ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                detail: "fully folded proof-grade mixed chain has no current clock",
            },
        )?;
        let target = chain.requested_target_time.binary64_rational();
        let domain = current_chart.parameter_interval();
        let current_left_physical = checked_add(domain.lower(), clock.upper())
            .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?;
        target_not_before = target >= &current_left_physical;
        if target_not_before {
            let candidate = RationalInterval::new(
                checked_subtract(target, clock.upper())
                    .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?,
                checked_subtract(target, clock.lower())
                    .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?,
            )?;
            preimage_derived = true;
            preimage_inside = domain.contains_interval(&candidate);
            target_preimage = Some(candidate.clone());
            if preimage_inside {
                match replay_ordinary_tube_exact_rational_v04(&current_chart, &current_tube) {
                    Ok(tube_replay) => {
                        if tube_replay.certified() {
                            let radius = tube_replay.gronwall_upper().ok_or(
                                ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "certified final tube has no Gronwall upper bound",
                                },
                            )?;
                            let position_intervals = evaluate_and_inflate(
                                current_chart.position_polynomial(),
                                &candidate,
                                radius,
                                true,
                            )
                            .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?;
                            let velocity_intervals = evaluate_and_inflate(
                                current_chart.velocity_polynomial(),
                                &candidate,
                                radius,
                                false,
                            )
                            .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?;
                            if position_intervals.len() != POSITION_DIMENSION
                                || velocity_intervals.len() != POSITION_DIMENSION
                            {
                                return Err(ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                                    detail: "proof-grade mixed final enclosure is not planar three-body",
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
                    ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                        detail:
                            "certified proof-grade mixed prefix has no current clock for retention",
                    },
                )?,
                certified_segment_count,
            )
            .map_err(ProofGradeRawMixedPlanarChainReplayError::OrdinaryShared)?,
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
                .ok_or(
                    ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                        detail: "width-only failure has no fixed-time enclosure",
                    },
                )?
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

    let _failure_disposition = replay_failure
        .as_ref()
        .map(MixedChainReplayFailure::disposition);

    Ok(ProofGradeRawMixedPlanarChainReplay {
        obligations: std::array::from_fn(|index| ProofGradeRawMixedPlanarChainObligation {
            id: PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        initial_claimed_tail_chart_diagnostic,
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
) -> ProofGradeRawMixedPlanarChainReplay {
    ProofGradeRawMixedPlanarChainReplay {
        obligations: std::array::from_fn(|index| ProofGradeRawMixedPlanarChainObligation {
            id: PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[index],
            satisfied: false,
        }),
        initial_claimed_tail_chart_diagnostic: None,
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

fn reconstruct_failed_proof_grade_lc_frontier(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_clock: &RationalInterval,
    exit_replay: Option<&ProofGradeCarriedPlanarLcExitReplay>,
) -> Result<Option<MixedChainLiftedLcRightFrontier>, ProofGradeRawMixedPlanarChainReplayError> {
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
    match replay_proof_grade_admission_bound_planar_lc_right_exact_rational_v04(
        admission,
        segment_index,
        parent_clock,
    ) {
        Ok(replay) => replay
            .lifted_right_slice()
            .cloned()
            .map(|state| {
                let physical_time_interval = state.physical_time().clone();
                Ok(MixedChainLiftedLcRightFrontier::new(
                    segment_index,
                    replay.lc_chart_id().to_owned(),
                    replay.pair(),
                    replay.right_parameter().clone(),
                    physical_time_interval,
                    state,
                ))
            })
            .transpose(),
        // A missing proof right cannot turn locally rejected evidence into an
        // execution error; ordinary retention remains the fail-closed fallback.
        Err(_) => Ok(None),
    }
}

fn build_lc_frontier(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    state: crate::PlanarLcIntervalState,
) -> Result<MixedChainLiftedLcRightFrontier, ProofGradeRawMixedPlanarChainReplayError> {
    let passage = admission.planar_lc_passage(segment_index).ok_or(
        ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
            detail: "LC replay has no admission-bound passage",
        },
    )?;
    Ok(MixedChainLiftedLcRightFrontier::new(
        segment_index,
        passage.lc_chart.chart_id.clone(),
        canonical_pair(passage)?,
        passage.lc_chart.parameter_interval[1]
            .binary64_rational()
            .clone(),
        state.physical_time().clone(),
        state,
    ))
}

fn canonical_pair(
    passage: &crate::raw_schema::PlanarLcPassageSegmentWire,
) -> Result<[usize; 2], ProofGradeRawMixedPlanarChainReplayError> {
    use num_traits::ToPrimitive;
    let left = passage.lc_chart.pair[0].value().to_usize();
    let right = passage.lc_chart.pair[1].value().to_usize();
    match (left, right) {
        (Some(left), Some(right)) if left < right && right < 3 => Ok([left, right]),
        _ => Err(
            ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                detail: "certified LC replay lost its canonical pair",
            },
        ),
    }
}
