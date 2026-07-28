//! Conditional exact-rational carried planar-LC exit composer.
//!
//! This local profile consumes an opaque canonical raw-v1 admission and a
//! parent-derived source clock.  It freshly replays the carried entry and both
//! endpoint tubes, reconstructs the complete inflated LC right slice, projects
//! that slice to Cartesian coordinates, and derives the target clock.

use core::fmt;

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire,
    outward_mass::PLANAR_LC_MASS_KERNEL_ID, planar_lc_entry_input_from_admission,
    project_planar_lc_full_state_exact_rational, raw_schema::SegmentWire,
    raw_v1_defining_namespace_unique, replay_carried_planar_lc_entry_exact_rational_v04,
    replay_ordinary_tube_exact_rational_v04, replay_planar_lc_tube_exact_rational_v04,
    replay_proof_grade_carried_planar_lc_entry_exact_rational_v04, CanonicalRawV1Admission,
    CarriedPlanarLcEntryError, CarriedPlanarLcEntryReplay, NumericError, OrdinaryChartInput,
    OrdinarySemanticError, OrdinaryTubeInput, OrdinaryTubeReplay, OrdinaryTubeReplayError,
    PlanarLcCartesianStateProjection, PlanarLcEntryInput, PlanarLcProjectionError,
    PlanarLcSemanticError, PlanarLcStateError, PlanarLcStatePolynomial, PlanarLcTubeReplay,
    PlanarLcTubeReplayError, PolynomialError, ProofGradeCarriedPlanarLcEntryReplay,
    RationalInterval, PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID,
    PLANAR_LC_LIFTED_STATE_DIMENSION,
};

pub const EXACT_RATIONAL_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID: &str =
    "exact_rational_carried_planar_lc_exit_v04";
pub const EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID: &str =
    "exact_rational_proof_grade_carried_planar_lc_exit_v04";
pub const PLANAR_LC_ANALYTIC_KERNEL_V1_ID: &str = "planar_lc_analytic_kernel_v1";
pub const PARENT_CARRIED_ORDINARY_SOLUTION_INVARIANT_V1_ID: &str =
    "parent_carried_ordinary_solution_invariant_v1";

pub const CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS: [&str; 21] = [
    "carried_lc_exit_exact_raw_schemas",
    "carried_lc_exit_identifiers_match_and_are_unique",
    "carried_lc_exit_parent_source_invariant_is_explicit_condition",
    "carried_lc_exit_entry_freshly_replayed_and_certified",
    "carried_lc_exit_common_planar_mass_problem",
    "carried_lc_exit_pair_is_canonical_ascending",
    "carried_lc_exit_outward_mass_arithmetic_certified",
    "carried_lc_exit_exact_right_to_left_endpoint_handoff",
    "carried_lc_exit_constrained_entry_branch_carried",
    "carried_lc_exit_constraint_invariance_kernel",
    "carried_lc_exit_lc_tube_freshly_certified",
    "carried_lc_exit_third_body_separated",
    "carried_lc_exit_target_ordinary_tube_freshly_certified",
    "carried_lc_exit_strict_physical_clock_kernel",
    "carried_lc_exit_complete_inflated_slice_reconstructed",
    "carried_lc_exit_complete_slice_rho_positive",
    "carried_lc_exit_complete_cartesian_projection_reconstructed",
    "carried_lc_exit_deck_equivariant_newton_projection_kernel",
    "carried_lc_exit_target_initial_ball_contains_complete_projection",
    "carried_lc_exit_time_interval_derived_from_component_fourteen",
    "carried_lc_exit_target_clock_origin_exactly_derived",
];

/// The proof-grade exit uses the same direct exit obligations as the
/// compatibility profile.  Its nested entry instead retains the two
/// claimed-tail chart replays as non-decisive diagnostics.
pub const PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS: [&str; 21] = [
    "proof_grade_carried_lc_exit_exact_raw_schemas",
    "proof_grade_carried_lc_exit_identifiers_match_and_are_unique",
    "proof_grade_carried_lc_exit_parent_source_invariant_is_explicit_condition",
    "proof_grade_carried_lc_exit_entry_freshly_replayed_and_certified",
    "proof_grade_carried_lc_exit_common_planar_mass_problem",
    "proof_grade_carried_lc_exit_pair_is_canonical_ascending",
    "proof_grade_carried_lc_exit_outward_mass_arithmetic_certified",
    "proof_grade_carried_lc_exit_exact_right_to_left_endpoint_handoff",
    "proof_grade_carried_lc_exit_constrained_entry_branch_carried",
    "proof_grade_carried_lc_exit_constraint_invariance_kernel",
    "proof_grade_carried_lc_exit_lc_tube_freshly_certified",
    "proof_grade_carried_lc_exit_third_body_separated",
    "proof_grade_carried_lc_exit_target_ordinary_tube_freshly_certified",
    "proof_grade_carried_lc_exit_strict_physical_clock_kernel",
    "proof_grade_carried_lc_exit_complete_inflated_slice_reconstructed",
    "proof_grade_carried_lc_exit_complete_slice_rho_positive",
    "proof_grade_carried_lc_exit_complete_cartesian_projection_reconstructed",
    "proof_grade_carried_lc_exit_deck_equivariant_newton_projection_kernel",
    "proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection",
    "proof_grade_carried_lc_exit_time_interval_derived_from_component_fourteen",
    "proof_grade_carried_lc_exit_target_clock_origin_exactly_derived",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CarriedPlanarLcExitObligation {
    id: &'static str,
    satisfied: bool,
}

impl CarriedPlanarLcExitObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }
    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeCarriedPlanarLcExitObligation {
    id: &'static str,
    satisfied: bool,
}

impl ProofGradeCarriedPlanarLcExitObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }
    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CarriedPlanarLcExitReplay {
    obligations: [CarriedPlanarLcExitObligation; 21],
    entry_replay: CarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    target_tube_replay: OrdinaryTubeReplay,
    lifted_exit_slice: Option<crate::PlanarLcIntervalState>,
    exit_rho_interval: Option<RationalInterval>,
    cartesian_projection: Option<PlanarLcCartesianStateProjection>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_projected_anchor_gap: Option<BigRational>,
    exit_time_interval: Option<RationalInterval>,
    target_clock_origin: Option<RationalInterval>,
}

impl CarriedPlanarLcExitReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID
    }
    pub const fn analytic_kernel_id(&self) -> &'static str {
        PLANAR_LC_ANALYTIC_KERNEL_V1_ID
    }
    pub const fn mass_kernel_id(&self) -> &'static str {
        PLANAR_LC_MASS_KERNEL_ID
    }
    pub const fn parent_invariant_id(&self) -> &'static str {
        PARENT_CARRIED_ORDINARY_SOLUTION_INVARIANT_V1_ID
    }
    pub fn obligations(&self) -> &[CarriedPlanarLcExitObligation; 21] {
        &self.obligations
    }
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(CarriedPlanarLcExitObligation::satisfied)
    }
    pub fn entry_replay(&self) -> &CarriedPlanarLcEntryReplay {
        &self.entry_replay
    }
    pub fn lc_tube_replay(&self) -> &PlanarLcTubeReplay {
        &self.lc_tube_replay
    }
    pub fn target_tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.target_tube_replay
    }
    pub fn lifted_exit_slice(&self) -> Option<&crate::PlanarLcIntervalState> {
        self.lifted_exit_slice.as_ref()
    }
    pub fn exit_rho_interval(&self) -> Option<&RationalInterval> {
        self.exit_rho_interval.as_ref()
    }
    pub fn cartesian_projection(&self) -> Option<&PlanarLcCartesianStateProjection> {
        self.cartesian_projection.as_ref()
    }
    pub fn target_anchor_centers(&self) -> Option<&[BigRational]> {
        self.target_anchor_centers.as_deref()
    }
    pub fn maximum_projected_anchor_gap(&self) -> Option<&BigRational> {
        self.maximum_projected_anchor_gap.as_ref()
    }
    pub fn exit_time_interval(&self) -> Option<&RationalInterval> {
        self.exit_time_interval.as_ref()
    }
    pub fn target_clock_origin(&self) -> Option<&RationalInterval> {
        self.target_clock_origin.as_ref()
    }
}

/// Proof-grade `N -> LC -> N` evidence.  The nested carried entry retains
/// claimed-tail chart outcomes as diagnostics; those outcomes deliberately do
/// not gate any of the direct right-slice, projection, containment, or clock
/// evidence recorded here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeCarriedPlanarLcExitReplay {
    obligations: [ProofGradeCarriedPlanarLcExitObligation; 21],
    entry_replay: ProofGradeCarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    target_tube_replay: OrdinaryTubeReplay,
    lifted_exit_slice: Option<crate::PlanarLcIntervalState>,
    exit_rho_interval: Option<RationalInterval>,
    cartesian_projection: Option<PlanarLcCartesianStateProjection>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_projected_anchor_gap: Option<BigRational>,
    exit_time_interval: Option<RationalInterval>,
    target_clock_origin: Option<RationalInterval>,
}

impl ProofGradeCarriedPlanarLcExitReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID
    }
    pub const fn analytic_kernel_id(&self) -> &'static str {
        PLANAR_LC_ANALYTIC_KERNEL_V1_ID
    }
    pub const fn mass_kernel_id(&self) -> &'static str {
        PLANAR_LC_MASS_KERNEL_ID
    }
    pub const fn parent_invariant_id(&self) -> &'static str {
        PARENT_CARRIED_ORDINARY_SOLUTION_INVARIANT_V1_ID
    }
    pub fn obligations(&self) -> &[ProofGradeCarriedPlanarLcExitObligation; 21] {
        &self.obligations
    }
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(ProofGradeCarriedPlanarLcExitObligation::satisfied)
    }
    pub fn entry_replay(&self) -> &ProofGradeCarriedPlanarLcEntryReplay {
        &self.entry_replay
    }
    pub fn lc_tube_replay(&self) -> &PlanarLcTubeReplay {
        &self.lc_tube_replay
    }
    pub fn target_tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.target_tube_replay
    }
    pub fn lifted_exit_slice(&self) -> Option<&crate::PlanarLcIntervalState> {
        self.lifted_exit_slice.as_ref()
    }
    pub fn exit_rho_interval(&self) -> Option<&RationalInterval> {
        self.exit_rho_interval.as_ref()
    }
    pub fn cartesian_projection(&self) -> Option<&PlanarLcCartesianStateProjection> {
        self.cartesian_projection.as_ref()
    }
    pub fn target_anchor_centers(&self) -> Option<&[BigRational]> {
        self.target_anchor_centers.as_deref()
    }
    pub fn maximum_projected_anchor_gap(&self) -> Option<&BigRational> {
        self.maximum_projected_anchor_gap.as_ref()
    }
    pub fn exit_time_interval(&self) -> Option<&RationalInterval> {
        self.exit_time_interval.as_ref()
    }
    pub fn target_clock_origin(&self) -> Option<&RationalInterval> {
        self.target_clock_origin.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CarriedPlanarLcExitError {
    SegmentUnavailable { segment_index: usize },
    SegmentNotPlanarLc { segment_index: usize },
    SourceSemantic(OrdinarySemanticError),
    TargetSemantic(OrdinarySemanticError),
    LcSemantic(PlanarLcSemanticError),
    Entry(CarriedPlanarLcEntryError),
    LcTube(PlanarLcTubeReplayError),
    TargetTube(OrdinaryTubeReplayError),
    State(PlanarLcStateError),
    Projection(PlanarLcProjectionError),
    TargetPositionPolynomial(PolynomialError),
    TargetVelocityPolynomial(PolynomialError),
    Numeric(NumericError),
    InternalShapeInvariant,
}

impl fmt::Display for CarriedPlanarLcExitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "carried planar LC exit replay failed: {self:?}")
    }
}
impl std::error::Error for CarriedPlanarLcExitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceSemantic(source) | Self::TargetSemantic(source) => Some(source),
            Self::LcSemantic(source) => Some(source),
            Self::Entry(source) => Some(source),
            Self::LcTube(source) => Some(source),
            Self::TargetTube(source) => Some(source),
            Self::State(source) => Some(source),
            Self::Projection(source) => Some(source),
            Self::TargetPositionPolynomial(source) | Self::TargetVelocityPolynomial(source) => {
                Some(source)
            }
            Self::Numeric(source) => Some(source),
            Self::SegmentUnavailable { .. }
            | Self::SegmentNotPlanarLc { .. }
            | Self::InternalShapeInvariant => None,
        }
    }
}
impl From<NumericError> for CarriedPlanarLcExitError {
    fn from(value: NumericError) -> Self {
        Self::Numeric(value)
    }
}
impl From<PlanarLcStateError> for CarriedPlanarLcExitError {
    fn from(value: PlanarLcStateError) -> Self {
        Self::State(value)
    }
}

/// Admission-bound evidence that is sufficient to reconstruct the LC chart's
/// complete inflated right slice without reading the ordinary exit target.
///
/// This is crate-private because it is a retention primitive, not another
/// public certificate profile.  In particular, it does not establish the
/// Cartesian exit containment or a chain commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionBoundPlanarLcRightReplay {
    source_chart: OrdinaryChartInput,
    source_tube: OrdinaryTubeInput,
    entry: PlanarLcEntryInput,
    entry_replay: CarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    namespace: bool,
    entry_handoff: bool,
    entry_certified: bool,
    constrained_entry: bool,
    lc_certified: bool,
    right_parameter: BigRational,
    lifted_right_slice: Option<crate::PlanarLcIntervalState>,
}

impl AdmissionBoundPlanarLcRightReplay {
    pub(crate) fn lc_chart_id(&self) -> &str {
        self.entry.target_chart().chart_id()
    }

    pub(crate) fn pair(&self) -> [usize; 2] {
        self.entry.target_chart().pair()
    }

    pub(crate) fn right_parameter(&self) -> &BigRational {
        &self.right_parameter
    }

    pub(crate) fn lifted_right_slice(&self) -> Option<&crate::PlanarLcIntervalState> {
        self.lifted_right_slice.as_ref()
    }
}

/// Reconstruct only the independently supportable LC-right evidence for one
/// admission-bound passage.  The target ordinary chart and tube are never
/// parsed or consumed, so malformed later exit evidence cannot erase a
/// freshly justified LC frontier.
pub(crate) fn replay_admission_bound_planar_lc_right_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_source_clock_origin: &RationalInterval,
) -> Result<AdmissionBoundPlanarLcRightReplay, CarriedPlanarLcExitError> {
    let wire = admission.wire();
    match wire.segments.get(segment_index) {
        Some(SegmentWire::PlanarLcPassage(_)) => {}
        Some(SegmentWire::OrdinaryBridge(_)) => {
            return Err(CarriedPlanarLcExitError::SegmentNotPlanarLc { segment_index });
        }
        None => return Err(CarriedPlanarLcExitError::SegmentUnavailable { segment_index }),
    }
    let (source_chart_wire, source_tube_wire) = if segment_index == 0 {
        (&wire.initial_chart, &wire.initial_tube)
    } else {
        match &wire.segments[segment_index - 1] {
            SegmentWire::OrdinaryBridge(value) => (&value.target_chart, &value.target_tube),
            SegmentWire::PlanarLcPassage(value) => (&value.target_chart, &value.target_tube),
        }
    };
    let source_chart = ordinary_chart_input_from_wire(source_chart_wire)
        .map_err(CarriedPlanarLcExitError::SourceSemantic)?;
    let source_tube = ordinary_tube_input_from_wire(source_tube_wire);
    let entry =
        planar_lc_entry_input_from_admission(admission, segment_index, &source_chart, &source_tube)
            .map_err(CarriedPlanarLcExitError::LcSemantic)?;
    let entry_replay = replay_carried_planar_lc_entry_exact_rational_v04(
        admission,
        segment_index,
        &source_chart,
        &source_tube,
        parent_source_clock_origin,
    )
    .map_err(CarriedPlanarLcExitError::Entry)?;
    let lc_tube_replay =
        replay_planar_lc_tube_exact_rational_v04(entry.target_chart(), entry.target_tube())
            .map_err(CarriedPlanarLcExitError::LcTube)?;
    let namespace =
        raw_v1_defining_namespace_unique(admission).map_err(CarriedPlanarLcExitError::Entry)?;
    let entry_handoff = entry.source_right_parameter() == source_chart.parameter_interval().upper()
        && source_tube.anchor_parameter() == source_chart.parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_chart().parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_tube().anchor_parameter();
    let right_parameter = entry.target_chart().parameter_interval().upper().clone();
    let entry_certified = entry_replay.conditional_profile_satisfied();
    let constrained_entry = entry_certified
        && entry_replay.selected_assignment().is_some()
        && entry_replay
            .selected_transformed_patches()
            .is_some_and(|patches| !patches.is_empty())
        && entry_replay.analytic_kernel_id()
            == Some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID);
    let lc_certified = lc_tube_replay.certified();
    let slice_gate = entry_certified && entry_handoff && constrained_entry && lc_certified;
    let lifted_right_slice = if slice_gate {
        let right = RationalInterval::try_point(right_parameter.clone())?;
        let polynomial = PlanarLcStatePolynomial::from_chart(entry.target_chart())?;
        let state = polynomial.evaluate(&right)?.inflate(
            lc_tube_replay
                .gronwall_upper()
                .ok_or(CarriedPlanarLcExitError::InternalShapeInvariant)?,
        )?;
        (state.components().len() == PLANAR_LC_LIFTED_STATE_DIMENSION).then_some(state)
    } else {
        None
    };

    Ok(AdmissionBoundPlanarLcRightReplay {
        source_chart,
        source_tube,
        entry,
        entry_replay,
        lc_tube_replay,
        namespace,
        entry_handoff,
        entry_certified,
        constrained_entry,
        lc_certified,
        right_parameter,
        lifted_right_slice,
    })
}

/// The proof-grade counterpart of [`AdmissionBoundPlanarLcRightReplay`].
/// It differs only in which nested entry ledger is decisive: the proof-grade
/// entry's claimed-tail chart outcomes remain diagnostics and cannot erase
/// independently reconstructed LC-right evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofGradeAdmissionBoundPlanarLcRightReplay {
    source_chart: OrdinaryChartInput,
    entry: PlanarLcEntryInput,
    entry_replay: ProofGradeCarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    namespace: bool,
    entry_handoff: bool,
    entry_certified: bool,
    constrained_entry: bool,
    lc_certified: bool,
    right_parameter: BigRational,
    lifted_right_slice: Option<crate::PlanarLcIntervalState>,
}

impl ProofGradeAdmissionBoundPlanarLcRightReplay {
    pub(crate) fn lc_chart_id(&self) -> &str {
        self.entry.target_chart().chart_id()
    }

    pub(crate) fn pair(&self) -> [usize; 2] {
        self.entry.target_chart().pair()
    }

    pub(crate) fn right_parameter(&self) -> &BigRational {
        &self.right_parameter
    }

    pub(crate) fn lifted_right_slice(&self) -> Option<&crate::PlanarLcIntervalState> {
        self.lifted_right_slice.as_ref()
    }
}

pub(crate) fn replay_proof_grade_admission_bound_planar_lc_right_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_source_clock_origin: &RationalInterval,
) -> Result<ProofGradeAdmissionBoundPlanarLcRightReplay, CarriedPlanarLcExitError> {
    let wire = admission.wire();
    match wire.segments.get(segment_index) {
        Some(SegmentWire::PlanarLcPassage(_)) => {}
        Some(SegmentWire::OrdinaryBridge(_)) => {
            return Err(CarriedPlanarLcExitError::SegmentNotPlanarLc { segment_index });
        }
        None => return Err(CarriedPlanarLcExitError::SegmentUnavailable { segment_index }),
    }
    let (source_chart_wire, source_tube_wire) = if segment_index == 0 {
        (&wire.initial_chart, &wire.initial_tube)
    } else {
        match &wire.segments[segment_index - 1] {
            SegmentWire::OrdinaryBridge(value) => (&value.target_chart, &value.target_tube),
            SegmentWire::PlanarLcPassage(value) => (&value.target_chart, &value.target_tube),
        }
    };
    let source_chart = ordinary_chart_input_from_wire(source_chart_wire)
        .map_err(CarriedPlanarLcExitError::SourceSemantic)?;
    let source_tube = ordinary_tube_input_from_wire(source_tube_wire);
    let entry =
        planar_lc_entry_input_from_admission(admission, segment_index, &source_chart, &source_tube)
            .map_err(CarriedPlanarLcExitError::LcSemantic)?;
    let entry_replay = replay_proof_grade_carried_planar_lc_entry_exact_rational_v04(
        admission,
        segment_index,
        &source_chart,
        &source_tube,
        parent_source_clock_origin,
    )
    .map_err(CarriedPlanarLcExitError::Entry)?;
    let lc_tube_replay =
        replay_planar_lc_tube_exact_rational_v04(entry.target_chart(), entry.target_tube())
            .map_err(CarriedPlanarLcExitError::LcTube)?;
    let namespace =
        raw_v1_defining_namespace_unique(admission).map_err(CarriedPlanarLcExitError::Entry)?;
    let entry_handoff = entry.source_right_parameter() == source_chart.parameter_interval().upper()
        && source_tube.anchor_parameter() == source_chart.parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_chart().parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_tube().anchor_parameter();
    let right_parameter = entry.target_chart().parameter_interval().upper().clone();
    let entry_certified = entry_replay.conditional_profile_satisfied();
    let constrained_entry = entry_certified
        && entry_replay.selected_assignment().is_some()
        && entry_replay
            .selected_transformed_patches()
            .is_some_and(|patches| !patches.is_empty())
        && entry_replay.analytic_kernel_id()
            == Some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID);
    let lc_certified = lc_tube_replay.certified();
    let slice_gate = entry_certified && entry_handoff && constrained_entry && lc_certified;
    let lifted_right_slice = if slice_gate {
        let right = RationalInterval::try_point(right_parameter.clone())?;
        let polynomial = PlanarLcStatePolynomial::from_chart(entry.target_chart())?;
        let state = polynomial.evaluate(&right)?.inflate(
            lc_tube_replay
                .gronwall_upper()
                .ok_or(CarriedPlanarLcExitError::InternalShapeInvariant)?,
        )?;
        (state.components().len() == PLANAR_LC_LIFTED_STATE_DIMENSION).then_some(state)
    } else {
        None
    };

    Ok(ProofGradeAdmissionBoundPlanarLcRightReplay {
        source_chart,
        entry,
        entry_replay,
        lc_tube_replay,
        namespace,
        entry_handoff,
        entry_certified,
        constrained_entry,
        lc_certified,
        right_parameter,
        lifted_right_slice,
    })
}

/// Replay one admission-bound `N -> LC -> N` passage.  The parent clock is a
/// conditional premise, not certificate evidence.
pub fn replay_carried_planar_lc_exit_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_source_clock_origin: &RationalInterval,
) -> Result<CarriedPlanarLcExitReplay, CarriedPlanarLcExitError> {
    let wire = admission.wire();
    let passage = match wire.segments.get(segment_index) {
        Some(SegmentWire::PlanarLcPassage(value)) => value,
        Some(SegmentWire::OrdinaryBridge(_)) => {
            return Err(CarriedPlanarLcExitError::SegmentNotPlanarLc { segment_index });
        }
        None => return Err(CarriedPlanarLcExitError::SegmentUnavailable { segment_index }),
    };
    let target_chart = ordinary_chart_input_from_wire(&passage.target_chart)
        .map_err(CarriedPlanarLcExitError::TargetSemantic)?;
    let target_tube = ordinary_tube_input_from_wire(&passage.target_tube);
    let right_replay = replay_admission_bound_planar_lc_right_exact_rational_v04(
        admission,
        segment_index,
        parent_source_clock_origin,
    )?;
    let target_tube_replay = replay_ordinary_tube_exact_rational_v04(&target_chart, &target_tube)
        .map_err(CarriedPlanarLcExitError::TargetTube)?;
    let AdmissionBoundPlanarLcRightReplay {
        source_chart,
        source_tube: _,
        entry,
        entry_replay,
        lc_tube_replay,
        namespace,
        entry_handoff,
        entry_certified,
        constrained_entry,
        lc_certified,
        right_parameter: _,
        lifted_right_slice,
    } = right_replay;
    let exit = &passage.exit_transition;
    let lc_right = exit.source_parameter.binary64_rational()
        == entry.target_chart().parameter_interval().upper();
    let raw_schemas = passage.segment_type == "planar_lc_passage_v1"
        && !exit.source.is_empty()
        && !exit.transition_id.is_empty()
        && !exit.source_chart_id.is_empty()
        && !exit.target_chart_id.is_empty()
        && !target_tube.tube_id().is_empty()
        && !target_tube.chart_id().is_empty()
        && target_tube_replay.obligations()[1].satisfied();
    let identifiers = namespace
        && exit.source_chart_id == entry.target_chart().chart_id()
        && exit.target_chart_id == target_chart.chart_id()
        && entry.target_tube().chart_id() == entry.target_chart().chart_id()
        && target_tube.chart_id() == target_chart.chart_id()
        && target_tube_replay.obligations()[0].satisfied();
    let common_problem = source_chart.masses() == entry.target_chart().masses()
        && source_chart.masses() == target_chart.masses();
    let pair = entry.target_chart().pair();
    let canonical_pair = pair[0] < pair[1] && pair[1] < 3;
    let mass_arithmetic = entry_replay.mass_kernel_id() == Some(PLANAR_LC_MASS_KERNEL_ID);
    let target_left = exit.target_parameter.binary64_rational()
        == target_chart.parameter_interval().lower()
        && exit.target_parameter.binary64_rational() == target_tube.anchor_parameter();
    let endpoints = entry_handoff && lc_right && target_left;
    let constraint_kernel = constrained_entry && lc_certified;
    let third_body_separated = lc_certified
        && lc_tube_replay
            .third_body_squared_distance_floors()
            .is_some_and(|values| values.iter().all(|value| value.is_positive()));
    let target_certified = target_tube_replay.certified();
    let strict_clock =
        endpoints && constrained_entry && constraint_kernel && lc_certified && third_body_separated;

    let preliminary = [
        raw_schemas,
        identifiers,
        true,
        entry_certified,
        common_problem,
        canonical_pair,
        mass_arithmetic,
        endpoints,
        constrained_entry,
        constraint_kernel,
        lc_certified,
        third_body_separated,
        target_certified,
        strict_clock,
    ];
    // Reconstruct the LC right slice whenever its actual dependencies pass.
    // In particular, a false target-tube result suppresses only containment;
    // it must not discard independent LC slice, projection, or clock evidence.
    if lifted_right_slice.is_none() || !lc_right {
        return Ok(build_replay(
            preliminary,
            entry_replay,
            lc_tube_replay,
            target_tube_replay,
            [false; 7],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
    }

    let state = lifted_right_slice.ok_or(CarriedPlanarLcExitError::InternalShapeInvariant)?;
    let slice_arithmetic = state.components().len() == PLANAR_LC_LIFTED_STATE_DIMENSION;
    let slice_reconstructed = entry_certified
        && entry_handoff
        && lc_right
        && constrained_entry
        && lc_certified
        && slice_arithmetic;
    let rho = state.rho()?;
    let rho_positive = rho.lower().is_positive();
    let projection = if rho_positive {
        Some(
            project_planar_lc_full_state_exact_rational(entry.target_chart(), &state)
                .map_err(CarriedPlanarLcExitError::Projection)?,
        )
    } else {
        None
    };
    let projection_reconstructed =
        mass_arithmetic && slice_reconstructed && rho_positive && projection.is_some();
    let deck_kernel =
        constraint_kernel && rho_positive && projection_reconstructed && mass_arithmetic;
    let anchor = RationalInterval::try_point(exit.target_parameter.binary64_rational().clone())?;
    let mut centers = target_chart
        .position_polynomial()
        .evaluate(&anchor)
        .map_err(CarriedPlanarLcExitError::TargetPositionPolynomial)?;
    centers.extend(
        target_chart
            .velocity_polynomial()
            .evaluate(&anchor)
            .map_err(CarriedPlanarLcExitError::TargetVelocityPolynomial)?,
    );
    if centers.len() != 12 || centers.iter().any(|value| !value.is_point()) {
        return Err(CarriedPlanarLcExitError::InternalShapeInvariant);
    }
    let center_values = centers
        .iter()
        .map(|value| value.lower().clone())
        .collect::<Vec<_>>();
    let maximum_gap = projection.as_ref().map(|value| {
        value
            .positions()
            .iter()
            .flatten()
            .chain(value.velocities().iter().flatten())
            .zip(&center_values)
            .fold(BigRational::zero(), |maximum, (interval, center)| {
                maximum
                    .max((interval.lower() - center).abs())
                    .max((interval.upper() - center).abs())
            })
    });
    let contains_arithmetic = maximum_gap
        .as_ref()
        .is_some_and(|gap| gap <= target_tube.initial_error_bound());
    let contains = common_problem
        && endpoints
        && target_certified
        && projection_reconstructed
        && deck_kernel
        && contains_arithmetic;
    let exit_time = state.physical_time().clone();
    let target_clock = exit_time.subtract(&anchor)?;
    let time_derived = slice_reconstructed;
    let clock_derived = endpoints && time_derived;
    let derived = [
        slice_reconstructed,
        slice_reconstructed && rho_positive,
        projection_reconstructed,
        deck_kernel,
        contains,
        time_derived,
        clock_derived,
    ];
    Ok(build_replay(
        preliminary,
        entry_replay,
        lc_tube_replay,
        target_tube_replay,
        derived,
        Some(state),
        Some(rho),
        projection,
        Some(center_values),
        maximum_gap,
        Some(exit_time),
        Some(target_clock),
    ))
}

/// Replay one admission-bound `N -> LC -> N` passage under the proof-grade
/// carried-entry profile.  The parent clock remains a conditional premise.
/// Unlike the compatibility profile, source and LC chart claimed-tail replays
/// are visible only through the nested entry diagnostics and never gate this
/// exit's direct evidence.
pub fn replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    parent_source_clock_origin: &RationalInterval,
) -> Result<ProofGradeCarriedPlanarLcExitReplay, CarriedPlanarLcExitError> {
    let wire = admission.wire();
    let passage = match wire.segments.get(segment_index) {
        Some(SegmentWire::PlanarLcPassage(value)) => value,
        Some(SegmentWire::OrdinaryBridge(_)) => {
            return Err(CarriedPlanarLcExitError::SegmentNotPlanarLc { segment_index });
        }
        None => return Err(CarriedPlanarLcExitError::SegmentUnavailable { segment_index }),
    };
    // Keep the compatibility replay's target parse and target-tube order.
    let target_chart = ordinary_chart_input_from_wire(&passage.target_chart)
        .map_err(CarriedPlanarLcExitError::TargetSemantic)?;
    let target_tube = ordinary_tube_input_from_wire(&passage.target_tube);
    let right_replay = replay_proof_grade_admission_bound_planar_lc_right_exact_rational_v04(
        admission,
        segment_index,
        parent_source_clock_origin,
    )?;
    let target_tube_replay = replay_ordinary_tube_exact_rational_v04(&target_chart, &target_tube)
        .map_err(CarriedPlanarLcExitError::TargetTube)?;
    let lc_chart_id = right_replay.lc_chart_id().to_owned();
    let pair = right_replay.pair();
    let right_parameter = right_replay.right_parameter().clone();
    let lifted_right_slice = right_replay.lifted_right_slice().cloned();
    let ProofGradeAdmissionBoundPlanarLcRightReplay {
        source_chart,
        entry,
        entry_replay,
        lc_tube_replay,
        namespace,
        entry_handoff,
        entry_certified,
        constrained_entry,
        lc_certified,
        right_parameter: _,
        lifted_right_slice: _,
    } = right_replay;
    let exit = &passage.exit_transition;
    let lc_right = exit.source_parameter.binary64_rational() == &right_parameter;
    let raw_schemas = passage.segment_type == "planar_lc_passage_v1"
        && !exit.source.is_empty()
        && !exit.transition_id.is_empty()
        && !exit.source_chart_id.is_empty()
        && !exit.target_chart_id.is_empty()
        && !target_tube.tube_id().is_empty()
        && !target_tube.chart_id().is_empty()
        && target_tube_replay.obligations()[1].satisfied();
    let identifiers = namespace
        && exit.source_chart_id == lc_chart_id
        && exit.target_chart_id == target_chart.chart_id()
        && entry.target_tube().chart_id() == entry.target_chart().chart_id()
        && target_tube.chart_id() == target_chart.chart_id()
        && target_tube_replay.obligations()[0].satisfied();
    let common_problem = source_chart.masses() == entry.target_chart().masses()
        && source_chart.masses() == target_chart.masses();
    let canonical_pair = pair[0] < pair[1] && pair[1] < 3;
    let mass_arithmetic = entry_replay.mass_kernel_id() == Some(PLANAR_LC_MASS_KERNEL_ID);
    let target_left = exit.target_parameter.binary64_rational()
        == target_chart.parameter_interval().lower()
        && exit.target_parameter.binary64_rational() == target_tube.anchor_parameter();
    let endpoints = entry_handoff && lc_right && target_left;
    let constraint_kernel = constrained_entry && lc_certified;
    let third_body_separated = lc_certified
        && lc_tube_replay
            .third_body_squared_distance_floors()
            .is_some_and(|values| values.iter().all(|value| value.is_positive()));
    let target_certified = target_tube_replay.certified();
    let strict_clock =
        endpoints && constrained_entry && constraint_kernel && lc_certified && third_body_separated;

    let preliminary = [
        raw_schemas,
        identifiers,
        true,
        entry_certified,
        common_problem,
        canonical_pair,
        mass_arithmetic,
        endpoints,
        constrained_entry,
        constraint_kernel,
        lc_certified,
        third_body_separated,
        target_certified,
        strict_clock,
    ];
    // A failed target tube can suppress only containment.  The direct
    // LC-right evidence is retained whenever its own dependencies pass.
    if lifted_right_slice.is_none() || !lc_right {
        return Ok(build_proof_grade_replay(
            preliminary,
            entry_replay,
            lc_tube_replay,
            target_tube_replay,
            [false; 7],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
    }

    let state = lifted_right_slice.ok_or(CarriedPlanarLcExitError::InternalShapeInvariant)?;
    let slice_arithmetic = state.components().len() == PLANAR_LC_LIFTED_STATE_DIMENSION;
    let slice_reconstructed = entry_certified
        && entry_handoff
        && lc_right
        && constrained_entry
        && lc_certified
        && slice_arithmetic;
    let rho = state.rho()?;
    let rho_positive = rho.lower().is_positive();
    let projection = if rho_positive {
        Some(
            project_planar_lc_full_state_exact_rational(entry.target_chart(), &state)
                .map_err(CarriedPlanarLcExitError::Projection)?,
        )
    } else {
        None
    };
    let projection_reconstructed =
        mass_arithmetic && slice_reconstructed && rho_positive && projection.is_some();
    let deck_kernel =
        constraint_kernel && rho_positive && projection_reconstructed && mass_arithmetic;
    let anchor = RationalInterval::try_point(exit.target_parameter.binary64_rational().clone())?;
    let mut centers = target_chart
        .position_polynomial()
        .evaluate(&anchor)
        .map_err(CarriedPlanarLcExitError::TargetPositionPolynomial)?;
    centers.extend(
        target_chart
            .velocity_polynomial()
            .evaluate(&anchor)
            .map_err(CarriedPlanarLcExitError::TargetVelocityPolynomial)?,
    );
    if centers.len() != 12 || centers.iter().any(|value| !value.is_point()) {
        return Err(CarriedPlanarLcExitError::InternalShapeInvariant);
    }
    let center_values = centers
        .iter()
        .map(|value| value.lower().clone())
        .collect::<Vec<_>>();
    let maximum_gap = projection.as_ref().map(|value| {
        value
            .positions()
            .iter()
            .flatten()
            .chain(value.velocities().iter().flatten())
            .zip(&center_values)
            .fold(BigRational::zero(), |maximum, (interval, center)| {
                maximum
                    .max((interval.lower() - center).abs())
                    .max((interval.upper() - center).abs())
            })
    });
    let contains_arithmetic = maximum_gap
        .as_ref()
        .is_some_and(|gap| gap <= target_tube.initial_error_bound());
    let contains = common_problem
        && endpoints
        && target_certified
        && projection_reconstructed
        && deck_kernel
        && contains_arithmetic;
    let exit_time = state.physical_time().clone();
    let target_clock = exit_time.subtract(&anchor)?;
    let time_derived = slice_reconstructed;
    let clock_derived = endpoints && time_derived;
    let derived = [
        slice_reconstructed,
        slice_reconstructed && rho_positive,
        projection_reconstructed,
        deck_kernel,
        contains,
        time_derived,
        clock_derived,
    ];
    Ok(build_proof_grade_replay(
        preliminary,
        entry_replay,
        lc_tube_replay,
        target_tube_replay,
        derived,
        Some(state),
        Some(rho),
        projection,
        Some(center_values),
        maximum_gap,
        Some(exit_time),
        Some(target_clock),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_replay(
    preliminary: [bool; 14],
    entry_replay: CarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    target_tube_replay: OrdinaryTubeReplay,
    derived: [bool; 7],
    lifted_exit_slice: Option<crate::PlanarLcIntervalState>,
    exit_rho_interval: Option<RationalInterval>,
    cartesian_projection: Option<PlanarLcCartesianStateProjection>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_projected_anchor_gap: Option<BigRational>,
    exit_time_interval: Option<RationalInterval>,
    target_clock_origin: Option<RationalInterval>,
) -> CarriedPlanarLcExitReplay {
    let satisfaction: [bool; 21] = std::array::from_fn(|index| {
        if index < 14 {
            preliminary[index]
        } else {
            derived[index - 14]
        }
    });
    CarriedPlanarLcExitReplay {
        obligations: std::array::from_fn(|index| CarriedPlanarLcExitObligation {
            id: CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        entry_replay,
        lc_tube_replay,
        target_tube_replay,
        lifted_exit_slice,
        exit_rho_interval,
        cartesian_projection,
        target_anchor_centers,
        maximum_projected_anchor_gap,
        exit_time_interval,
        target_clock_origin,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_proof_grade_replay(
    preliminary: [bool; 14],
    entry_replay: ProofGradeCarriedPlanarLcEntryReplay,
    lc_tube_replay: PlanarLcTubeReplay,
    target_tube_replay: OrdinaryTubeReplay,
    derived: [bool; 7],
    lifted_exit_slice: Option<crate::PlanarLcIntervalState>,
    exit_rho_interval: Option<RationalInterval>,
    cartesian_projection: Option<PlanarLcCartesianStateProjection>,
    target_anchor_centers: Option<Vec<BigRational>>,
    maximum_projected_anchor_gap: Option<BigRational>,
    exit_time_interval: Option<RationalInterval>,
    target_clock_origin: Option<RationalInterval>,
) -> ProofGradeCarriedPlanarLcExitReplay {
    let satisfaction: [bool; 21] = std::array::from_fn(|index| {
        if index < 14 {
            preliminary[index]
        } else {
            derived[index - 14]
        }
    });
    ProofGradeCarriedPlanarLcExitReplay {
        obligations: std::array::from_fn(|index| ProofGradeCarriedPlanarLcExitObligation {
            id: PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        entry_replay,
        lc_tube_replay,
        target_tube_replay,
        lifted_exit_slice,
        exit_rho_interval,
        cartesian_projection,
        target_anchor_centers,
        maximum_projected_anchor_gap,
        exit_time_interval,
        target_clock_origin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));
    const FAILED_REVISIT: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
    ));

    fn canonical_parent() -> RationalInterval {
        RationalInterval::try_point(BigRational::new(BigInt::from(1), BigInt::from(1_u64 << 40)))
            .unwrap()
    }

    fn replay_changed(changed: &str) -> CarriedPlanarLcExitReplay {
        let admission = CanonicalRawV1Admission::admit(changed.as_bytes()).unwrap();
        replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &canonical_parent())
            .unwrap()
    }

    fn proof_grade_truth(replay: &ProofGradeCarriedPlanarLcExitReplay) -> [bool; 21] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    fn replay_proof_grade_changed(changed: &str) -> ProofGradeCarriedPlanarLcExitReplay {
        let admission = CanonicalRawV1Admission::admit(changed.as_bytes()).unwrap();
        replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
            &admission,
            1,
            &canonical_parent(),
        )
        .unwrap()
    }

    fn assert_direct_exit_artifacts_equal(
        proof_grade: &ProofGradeCarriedPlanarLcExitReplay,
        compatibility: &CarriedPlanarLcExitReplay,
    ) {
        assert_eq!(proof_grade.lc_tube_replay(), compatibility.lc_tube_replay());
        assert_eq!(
            proof_grade.target_tube_replay(),
            compatibility.target_tube_replay()
        );
        assert_eq!(
            proof_grade.lifted_exit_slice(),
            compatibility.lifted_exit_slice()
        );
        assert_eq!(
            proof_grade.exit_rho_interval(),
            compatibility.exit_rho_interval()
        );
        assert_eq!(
            proof_grade.cartesian_projection(),
            compatibility.cartesian_projection()
        );
        assert_eq!(
            proof_grade.target_anchor_centers(),
            compatibility.target_anchor_centers()
        );
        assert_eq!(
            proof_grade.maximum_projected_anchor_gap(),
            compatibility.maximum_projected_anchor_gap()
        );
        assert_eq!(
            proof_grade.exit_time_interval(),
            compatibility.exit_time_interval()
        );
        assert_eq!(
            proof_grade.target_clock_origin(),
            compatibility.target_clock_origin()
        );
        assert_eq!(
            proof_grade.analytic_kernel_id(),
            compatibility.analytic_kernel_id()
        );
        assert_eq!(proof_grade.mass_kernel_id(), compatibility.mass_kernel_id());
        assert_eq!(
            proof_grade.parent_invariant_id(),
            compatibility.parent_invariant_id()
        );
    }

    fn replace_scalar_after(raw: &str, anchor: &str, field: &str, replacement: &str) -> String {
        let anchor_start = raw.find(anchor).expect("missing record anchor");
        let marker = format!("\"{field}\":");
        let value_start = anchor_start
            + raw[anchor_start..]
                .find(&marker)
                .expect("missing anchored scalar")
            + marker.len();
        let value_end = raw[value_start..]
            .find([',', '}'])
            .map(|offset| value_start + offset)
            .expect("unterminated scalar");
        format!(
            "{}{}{}",
            &raw[..value_start],
            replacement,
            &raw[value_end..]
        )
    }

    fn matching_array_end(raw: &str, start: usize) -> usize {
        let bytes = raw.as_bytes();
        assert_eq!(bytes[start], b'[');
        let mut depth = 0_usize;
        let mut in_string = false;
        let mut escaped = false;
        for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'\"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'\"' => in_string = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return index;
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated array")
    }

    fn containing_object_start(raw: &str, position: usize) -> usize {
        let mut object_stack = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        for (index, byte) in raw.as_bytes().iter().copied().enumerate().take(position) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'\"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'\"' => in_string = true,
                b'{' => object_stack.push(index),
                b'}' => {
                    object_stack.pop().expect("unbalanced object");
                }
                _ => {}
            }
        }
        *object_stack.last().expect("anchor outside object")
    }

    fn extend_planar_lc_coefficients_to(raw: &str, certificate_id: &str, count: usize) -> String {
        let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
        let mut result = raw.to_owned();
        for (field, zero) in [
            ("z_coefficients", "[0.0,0.0]"),
            ("z_velocity_coefficients", "[0.0,0.0]"),
            ("pair_energy_coefficients", "0.0"),
            ("binary_center_coefficients", "[0.0,0.0]"),
            ("binary_center_velocity_coefficients", "[0.0,0.0]"),
            ("third_offset_coefficients", "[0.0,0.0]"),
            ("third_offset_velocity_coefficients", "[0.0,0.0]"),
            ("physical_time_coefficients", "0.0"),
        ] {
            let anchor_start = result.find(&anchor).expect("missing record anchor");
            let object_start = containing_object_start(&result, anchor_start);
            let marker = format!("\"{field}\":[");
            let relative = result[object_start..]
                .find(&marker)
                .expect("missing anchored coefficient array");
            let array_start = object_start + relative + marker.len() - 1;
            let array_end = matching_array_end(&result, array_start);
            let inner = &result[array_start + 1..array_end];
            let mut depth = 0_usize;
            let mut existing = usize::from(!inner.is_empty());
            for byte in inner.as_bytes() {
                match byte {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    b',' if depth == 0 => existing += 1,
                    _ => {}
                }
            }
            assert!(existing <= count);
            let padding = vec![zero; count - existing].join(",");
            let replacement = if padding.is_empty() {
                format!("[{inner}]")
            } else {
                format!("[{inner},{padding}]")
            };
            result = format!(
                "{}{}{}",
                &result[..array_start],
                replacement,
                &result[array_end + 1..]
            );
        }
        result
    }

    #[test]
    fn proof_grade_first_canonical_exit_has_ordered_ledger_and_identical_direct_artifacts() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let compatibility =
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &canonical_parent())
                .unwrap();
        let proof_grade = replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
            &admission,
            1,
            &canonical_parent(),
        )
        .unwrap();

        assert_eq!(
            proof_grade.profile_id(),
            EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID
        );
        assert_eq!(
            proof_grade
                .obligations()
                .iter()
                .map(ProofGradeCarriedPlanarLcExitObligation::id)
                .collect::<Vec<_>>(),
            PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS
        );
        assert_eq!(proof_grade_truth(&proof_grade), [true; 21]);
        assert!(proof_grade.conditional_profile_satisfied());
        assert_eq!(proof_grade.entry_replay().obligations().len(), 19);
        assert!(proof_grade.entry_replay().conditional_profile_satisfied());
        assert!(proof_grade
            .entry_replay()
            .source_claimed_tail_chart_diagnostic()
            .unwrap()
            .conditional_profile_satisfied());
        assert!(proof_grade
            .entry_replay()
            .target_claimed_tail_chart_diagnostic()
            .unwrap()
            .conditional_profile_satisfied());
        assert_direct_exit_artifacts_equal(&proof_grade, &compatibility);
    }

    #[test]
    fn proof_grade_exit_retains_claimed_tail_failures_as_nested_diagnostics() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let canonical_admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let canonical = replay_carried_planar_lc_exit_exact_rational_v04(
            &canonical_admission,
            1,
            &canonical_parent(),
        )
        .unwrap();

        for (certificate_id, source_diagnostic_fails) in [
            ("review-v03:n:certificate:1", true),
            ("review-v03:lc:certificate:0", false),
        ] {
            let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
            let mutated = replace_scalar_after(raw, &anchor, "tail_bound", "-1.0");
            let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
            let compatibility = replay_carried_planar_lc_exit_exact_rational_v04(
                &admission,
                1,
                &canonical_parent(),
            )
            .unwrap();
            assert!(!compatibility.conditional_profile_satisfied());

            let proof_grade = replay_proof_grade_changed(&mutated);
            assert_eq!(proof_grade_truth(&proof_grade), [true; 21]);
            assert!(proof_grade.conditional_profile_satisfied());
            assert_direct_exit_artifacts_equal(&proof_grade, &canonical);

            let source_diagnostic = proof_grade
                .entry_replay()
                .source_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied();
            let target_diagnostic = proof_grade
                .entry_replay()
                .target_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied();
            assert_eq!(source_diagnostic, !source_diagnostic_fails);
            assert_eq!(target_diagnostic, source_diagnostic_fails);
        }
    }

    #[test]
    fn proof_grade_target_tube_failure_retains_lc_slice_projection_time_and_clock() {
        let changed = std::str::from_utf8(SUCCESS)
            .unwrap()
            .replacen(
                "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06,\"max_defect_bound\":1.0",
                "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06,\"max_defect_bound\":0.0",
                1,
            );
        let replay = replay_proof_grade_changed(&changed);
        assert!(!replay.obligations()[12].satisfied());
        assert!(!replay.obligations()[18].satisfied());
        for index in [14, 15, 16, 17, 19, 20] {
            assert!(replay.obligations()[index].satisfied(), "row {}", index + 1);
        }
        assert!(replay.lifted_exit_slice().is_some());
        assert!(replay.exit_rho_interval().is_some());
        assert!(replay.cartesian_projection().is_some());
        assert!(replay.exit_time_interval().is_some());
        assert!(replay.target_clock_origin().is_some());
    }

    #[test]
    fn proof_grade_exit_retains_target_lc_chart_resource_error_without_gating_direct_evidence() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let mutated = extend_planar_lc_coefficients_to(raw, "review-v03:lc:certificate:0", 89);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        assert!(matches!(
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &canonical_parent(),),
            Err(CarriedPlanarLcExitError::Entry(
                CarriedPlanarLcEntryError::TargetChart(crate::PlanarLcChartReplayError::Series(
                    crate::PlanarLcSeriesError::WorkLimitExceeded {
                        required: 4_055_552,
                        limit: 4_000_000,
                    }
                ))
            ))
        ));

        let proof_grade = replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
            &admission,
            1,
            &canonical_parent(),
        )
        .unwrap();
        assert!(matches!(
            proof_grade
                .entry_replay()
                .target_claimed_tail_chart_diagnostic(),
            Err(crate::PlanarLcChartReplayError::Series(
                crate::PlanarLcSeriesError::WorkLimitExceeded {
                    required: 4_055_552,
                    limit: 4_000_000,
                }
            ))
        ));
        assert_eq!(proof_grade_truth(&proof_grade), [true; 21]);
        assert!(proof_grade.conditional_profile_satisfied());
        let canonical_admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let canonical = replay_carried_planar_lc_exit_exact_rational_v04(
            &canonical_admission,
            1,
            &canonical_parent(),
        )
        .unwrap();
        assert_direct_exit_artifacts_equal(&proof_grade, &canonical);
    }

    #[test]
    fn proof_grade_canonical_files_have_the_compatibility_lc_prefix_behavior() {
        for (bytes, expected_passes) in [(SUCCESS, 4_usize), (FAILED_REVISIT, 3_usize)] {
            let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
            let mut parent = canonical_parent();
            let lc_indices = admission
                .wire()
                .segments
                .iter()
                .enumerate()
                .filter_map(|(index, segment)| {
                    matches!(segment, SegmentWire::PlanarLcPassage(_)).then_some(index)
                })
                .collect::<Vec<_>>();
            for (ordinal, index) in lc_indices.iter().copied().enumerate() {
                let replay = replay_proof_grade_carried_planar_lc_exit_exact_rational_v04(
                    &admission, index, &parent,
                )
                .unwrap();
                assert_eq!(
                    replay.conditional_profile_satisfied(),
                    ordinal < expected_passes,
                    "LC passage {ordinal}"
                );
                if ordinal >= expected_passes {
                    assert!(!replay.obligations()[18].satisfied());
                    assert!(replay.obligations()[14..18]
                        .iter()
                        .all(ProofGradeCarriedPlanarLcExitObligation::satisfied));
                    assert!(replay.obligations()[19..]
                        .iter()
                        .all(ProofGradeCarriedPlanarLcExitObligation::satisfied));
                    break;
                }
                parent = replay.target_clock_origin().unwrap().clone();
            }
        }
    }

    #[test]
    fn first_canonical_exit_passes_all_twenty_one_ordered_obligations() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();
        let replay =
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &parent).unwrap();
        assert!(
            replay.conditional_profile_satisfied(),
            "{:?}",
            replay.obligations()
        );
        assert_eq!(
            replay
                .obligations()
                .iter()
                .map(CarriedPlanarLcExitObligation::id)
                .collect::<Vec<_>>(),
            CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS
        );
        assert_eq!(
            replay.profile_id(),
            "exact_rational_carried_planar_lc_exit_v04"
        );
        assert_eq!(replay.analytic_kernel_id(), "planar_lc_analytic_kernel_v1");
        assert_eq!(
            replay.parent_invariant_id(),
            "parent_carried_ordinary_solution_invariant_v1"
        );
        assert_eq!(replay.lifted_exit_slice().unwrap().components().len(), 14);
        assert_eq!(replay.target_anchor_centers().unwrap().len(), 12);
        assert!(replay.maximum_projected_anchor_gap().is_some());
        assert_eq!(replay.exit_time_interval(), replay.target_clock_origin());

        // Differential evidence only: the Rust verifier does not consume the
        // archived Python transcript or attempt to mimic its interval path.
        let archived = RationalInterval::new(
            BigRational::new(
                BigInt::parse_bytes(b"4533271374569969", 10).unwrap(),
                BigInt::parse_bytes(b"2361183241434822606848", 10).unwrap(),
            ),
            BigRational::new(
                BigInt::parse_bytes(b"5005509942997599", 10).unwrap(),
                BigInt::parse_bytes(b"2361183241434822606848", 10).unwrap(),
            ),
        )
        .unwrap();
        let rust = replay.exit_time_interval().unwrap();
        assert_ne!(rust, &archived);
        assert!(rust.lower() <= archived.lower());
        assert!(rust.upper() >= archived.upper());
        assert!(rust.lower() < archived.lower() || rust.upper() > archived.upper());
    }

    #[test]
    fn exit_source_label_is_unpinned_but_must_remain_nonempty() {
        let changed = std::str::from_utf8(SUCCESS).unwrap().replacen(
            "serialized_planar_lc_to_ordinary_enclosure_transition",
            "another_nonempty_exit_producer",
            1,
        );
        let admission = CanonicalRawV1Admission::admit(changed.as_bytes()).unwrap();
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();
        let replay =
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &parent).unwrap();
        assert!(replay.obligations()[0].satisfied());
        assert!(replay.conditional_profile_satisfied());
    }

    #[test]
    fn exit_schema_identity_and_segment_selection_fail_independently() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let empty_source = replay_changed(&raw.replacen(
            "\"source\":\"serialized_planar_lc_to_ordinary_enclosure_transition\"",
            "\"source\":\"\"",
            1,
        ));
        assert!(!empty_source.obligations()[0].satisfied());

        let empty_target_tube_source = replay_changed(&raw.replacen(
            "\"source\":\"serialized_ordinary_aposteriori_tube\",\"tube_id\":\"review-v03:n:tube:2\"",
            "\"source\":\"\",\"tube_id\":\"review-v03:n:tube:2\"",
            1,
        ));
        assert!(!empty_target_tube_source.obligations()[0].satisfied());

        let empty_target_tube_id = replay_changed(&raw.replacen(
            "\"tube_id\":\"review-v03:n:tube:2\",\"tube_radius\":0.0001",
            "\"tube_id\":\"\",\"tube_radius\":0.0001",
            1,
        ));
        assert!(!empty_target_tube_id.obligations()[0].satisfied());
        assert!(!empty_target_tube_id.obligations()[1].satisfied());

        let wrong_reference = replay_changed(&raw.replacen(
            "\"source_chart_id\":\"review-v03:lc:chart:0\",\"source_parameter\":9.5367431640625e-07",
            "\"source_chart_id\":\"distinct-wrong-lc-chart\",\"source_parameter\":9.5367431640625e-07",
            1,
        ));
        assert!(!wrong_reference.obligations()[1].satisfied());
        assert!(wrong_reference.obligations()[14..]
            .iter()
            .all(CarriedPlanarLcExitObligation::satisfied));

        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        assert!(matches!(
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 0, &canonical_parent()),
            Err(CarriedPlanarLcExitError::SegmentNotPlanarLc { segment_index: 0 })
        ));
        assert!(matches!(
            replay_carried_planar_lc_exit_exact_rational_v04(
                &admission,
                usize::MAX,
                &canonical_parent()
            ),
            Err(CarriedPlanarLcExitError::SegmentUnavailable { .. })
        ));
    }

    #[test]
    fn target_tube_failure_retains_lc_slice_projection_time_and_clock() {
        let changed = std::str::from_utf8(SUCCESS).unwrap().replacen(
            "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06,\"max_defect_bound\":1.0",
            "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06,\"max_defect_bound\":0.0",
            1,
        );
        assert_ne!(changed.as_bytes(), SUCCESS);
        let admission = CanonicalRawV1Admission::admit(changed.as_bytes()).unwrap();
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();
        let replay =
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &parent).unwrap();
        assert!(!replay.obligations()[12].satisfied());
        assert!(!replay.obligations()[18].satisfied());
        for index in [14, 15, 16, 17, 19, 20] {
            assert!(replay.obligations()[index].satisfied(), "row {}", index + 1);
        }
        assert!(replay.lifted_exit_slice().is_some());
        assert!(replay.exit_rho_interval().is_some());
        assert!(replay.cartesian_projection().is_some());
        assert!(replay.exit_time_interval().is_some());
        assert!(replay.target_clock_origin().is_some());
    }

    #[test]
    fn too_small_target_allowance_fails_only_containment_among_derived_rows() {
        let changed = std::str::from_utf8(SUCCESS).unwrap().replacen(
            "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06",
            "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":0.0",
            1,
        );
        let replay = replay_changed(&changed);
        assert!(replay.obligations()[12].satisfied());
        for index in 14..21 {
            assert_eq!(
                replay.obligations()[index].satisfied(),
                index != 18,
                "row {}",
                index + 1
            );
        }
    }

    #[test]
    fn nonpoint_parent_clock_is_an_explicit_valid_condition() {
        let center = BigRational::new(BigInt::from(1), BigInt::from(1_u64 << 40));
        let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1_u128 << 80));
        let parent = RationalInterval::new(&center - &epsilon, &center + &epsilon).unwrap();
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let replay =
            replay_carried_planar_lc_exit_exact_rational_v04(&admission, 1, &parent).unwrap();
        assert!(replay.obligations()[2].satisfied());
        assert!(replay.conditional_profile_satisfied());
    }

    #[test]
    fn endpoint_failures_distinguish_lc_right_from_target_left_evidence() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let wrong_target = raw.replacen(
            "\"target_chart_id\":\"review-v03:n:chart:2\",\"target_parameter\":0.0",
            "\"target_chart_id\":\"review-v03:n:chart:2\",\"target_parameter\":9.094947017729282e-13",
            1,
        );
        let target = replay_changed(&wrong_target);
        assert!(!target.obligations()[7].satisfied());
        for index in [14, 15, 16, 17, 19] {
            assert!(target.obligations()[index].satisfied(), "row {}", index + 1);
        }
        assert!(!target.obligations()[18].satisfied());
        assert!(!target.obligations()[20].satisfied());

        let wrong_right = raw.replacen(
            "\"source_chart_id\":\"review-v03:lc:chart:0\",\"source_parameter\":9.5367431640625e-07",
            "\"source_chart_id\":\"review-v03:lc:chart:0\",\"source_parameter\":4.76837158203125e-07",
            1,
        );
        let right = replay_changed(&wrong_right);
        assert!(!right.obligations()[7].satisfied());
        assert!(right.obligations()[14..].iter().all(|row| !row.satisfied()));
        assert!(right.lifted_exit_slice().is_none());
    }

    #[test]
    fn namespace_entry_and_lc_tube_failures_are_fail_closed() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let namespace = replay_changed(&raw.replacen(
            "\"transition_id\":\"review-v03:lc:exit:0\"",
            "\"transition_id\":\"review-v03:lc:entry:0\"",
            1,
        ));
        assert!(!namespace.obligations()[1].satisfied());
        assert!(!namespace.obligations()[3].satisfied());

        let entry_changed = raw.replacen(
            "\"source_right_parameter\":9.5367431640625e-07",
            "\"source_right_parameter\":0.0",
            1,
        );
        let entry_admission = CanonicalRawV1Admission::admit(entry_changed.as_bytes()).unwrap();
        assert!(matches!(
            replay_carried_planar_lc_exit_exact_rational_v04(
                &entry_admission,
                1,
                &canonical_parent()
            ),
            Err(CarriedPlanarLcExitError::LcSemantic(_))
        ));

        let lc_tube = replay_changed(&raw.replacen(
            "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":1e-07,\"max_defect_bound\":1.0",
            "\"chart_id\":\"review-v03:lc:chart:0\",\"initial_error_bound\":1e-07,\"max_defect_bound\":0.0",
            1,
        ));
        assert!(!lc_tube.obligations()[3].satisfied());
        assert!(!lc_tube.obligations()[10].satisfied());
        assert!(lc_tube.lifted_exit_slice().is_none());
    }

    #[test]
    fn target_physical_time_metadata_is_irrelevant() {
        let raw = std::str::from_utf8(SUCCESS).unwrap();
        let baseline = replay_changed(raw);
        let marker = "\"certificate_id\":\"review-v03:n:certificate:2\"";
        let marker_index = raw.find(marker).unwrap();
        let (prefix, suffix) = raw.split_at(marker_index);
        let changed_suffix = suffix.replacen(
            "\"physical_time_interval\":[2.019915513158379e-06,2.973589829564629e-06]",
            "\"physical_time_interval\":[100.0,101.0]",
            1,
        );
        let changed = format!("{prefix}{changed_suffix}");
        assert_eq!(replay_changed(&changed), baseline);
    }

    #[test]
    fn canonical_files_have_expected_sequential_lc_prefix_behavior() {
        for (bytes, expected_passes) in [(SUCCESS, 4_usize), (FAILED_REVISIT, 3_usize)] {
            let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
            let mut parent = canonical_parent();
            let lc_indices = admission
                .wire()
                .segments
                .iter()
                .enumerate()
                .filter_map(|(index, segment)| {
                    matches!(segment, SegmentWire::PlanarLcPassage(_)).then_some(index)
                })
                .collect::<Vec<_>>();
            for (ordinal, index) in lc_indices.iter().copied().enumerate() {
                let replay =
                    replay_carried_planar_lc_exit_exact_rational_v04(&admission, index, &parent)
                        .unwrap();
                assert_eq!(
                    replay.conditional_profile_satisfied(),
                    ordinal < expected_passes
                );
                if ordinal >= expected_passes {
                    assert!(!replay.obligations()[18].satisfied());
                    assert!(replay.obligations()[14..18]
                        .iter()
                        .all(CarriedPlanarLcExitObligation::satisfied));
                    assert!(replay.obligations()[19..]
                        .iter()
                        .all(CarriedPlanarLcExitObligation::satisfied));
                    break;
                }
                parent = replay.target_clock_origin().unwrap().clone();
            }
        }
    }
}
