//! Conditional exact-rational carried ordinary-to-planar-LC entry composer.
//!
//! The chart ledgers below are compatibility gates only.  The decisive local
//! evidence is the fresh source tube, direct endpoint box, canonical lift
//! cover, and target-ball containment.  The parent must already have carried
//! the actual branch inside the source tube.

use core::fmt;
use std::collections::HashSet;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire,
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError, PLANAR_LC_MASS_KERNEL_ID},
    planar_lc_entry_input_from_admission,
    raw_schema::SegmentWire,
    replay_ordinary_chart_exact_rational_claimed_tail_v04, replay_ordinary_tube_exact_rational_v04,
    replay_planar_lc_chart_exact_rational_claimed_tail_v04,
    replay_planar_lc_lift_cover_exact_rational_v04, replay_planar_lc_tube_exact_rational_v04,
    CanonicalRawV1Admission, ExactBinary64, NumericError, OrdinaryChartInput, OrdinaryChartReplay,
    OrdinaryChartReplayError, OrdinarySemanticError, OrdinaryTubeInput, OrdinaryTubeReplay,
    OrdinaryTubeReplayError, PlanarLcChartReplay, PlanarLcChartReplayError, PlanarLcEntryInput,
    PlanarLcLiftError, PlanarLcLiftPatch, PlanarLcLiftReplay, PlanarLcSemanticError,
    PlanarLcStateError, PlanarLcStatePolynomial, PlanarLcTubeReplay, PlanarLcTubeReplayError,
    PolynomialError, RationalInterval, PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID,
    PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION,
};

pub const EXACT_RATIONAL_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID: &str =
    "exact_rational_carried_planar_lc_entry_v04";
pub const EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID: &str =
    "exact_rational_proof_grade_carried_planar_lc_entry_v04";
pub const HARD_MAX_RAW_V1_NAMESPACE_SEGMENTS: usize = 256;

pub const CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS: [&str; 21] = [
    "carried_lc_entry_exact_raw_schemas",
    "carried_lc_entry_transition_canonical_round_trip",
    "carried_lc_entry_identifiers_match_and_are_unique",
    "carried_lc_entry_parent_clock_origin_is_exact_interval",
    "carried_lc_entry_source_ordinary_chart_freshly_certified",
    "carried_lc_entry_source_ordinary_tube_freshly_certified",
    "carried_lc_entry_target_lc_chart_freshly_certified",
    "carried_lc_entry_target_lc_tube_freshly_certified",
    "carried_lc_entry_common_planar_mass_problem",
    "carried_lc_entry_pair_is_canonical_ascending",
    "carried_lc_entry_outward_mass_arithmetic_certified",
    "carried_lc_entry_exact_source_right_to_lc_left_anchor",
    "carried_lc_entry_complete_source_endpoint_box_reconstructed",
    "carried_lc_entry_selected_pair_collision_free",
    "carried_lc_entry_canonical_square_root_atlas_reconstructed",
    "carried_lc_entry_derived_parity_graph_certified",
    "carried_lc_entry_physical_time_interval_exactly_derived",
    "carried_lc_entry_all_lift_patches_have_positive_rho",
    "carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
    "carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
    "carried_lc_entry_one_global_complement_contains_all_complete_patches",
];

pub const PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS: [&str; 19] = [
    "proof_grade_carried_lc_entry_exact_raw_schemas",
    "proof_grade_carried_lc_entry_transition_canonical_round_trip",
    "proof_grade_carried_lc_entry_identifiers_match_and_are_unique",
    "proof_grade_carried_lc_entry_parent_clock_origin_is_exact_interval",
    "proof_grade_carried_lc_entry_source_ordinary_tube_freshly_certified",
    "proof_grade_carried_lc_entry_target_lc_tube_freshly_certified",
    "proof_grade_carried_lc_entry_common_planar_mass_problem",
    "proof_grade_carried_lc_entry_pair_is_canonical_ascending",
    "proof_grade_carried_lc_entry_outward_mass_arithmetic_certified",
    "proof_grade_carried_lc_entry_exact_source_right_to_lc_left_anchor",
    "proof_grade_carried_lc_entry_complete_source_endpoint_box_reconstructed",
    "proof_grade_carried_lc_entry_selected_pair_collision_free",
    "proof_grade_carried_lc_entry_canonical_square_root_atlas_reconstructed",
    "proof_grade_carried_lc_entry_derived_parity_graph_certified",
    "proof_grade_carried_lc_entry_physical_time_interval_exactly_derived",
    "proof_grade_carried_lc_entry_all_lift_patches_have_positive_rho",
    "proof_grade_carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
    "proof_grade_carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
    "proof_grade_carried_lc_entry_one_global_complement_contains_all_complete_patches",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CarriedPlanarLcEntryObligation {
    id: &'static str,
    satisfied: bool,
}

impl CarriedPlanarLcEntryObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CarriedPlanarLcEntryReplay {
    obligations: [CarriedPlanarLcEntryObligation; 21],
    source_chart_replay: OrdinaryChartReplay,
    source_tube_replay: OrdinaryTubeReplay,
    target_chart_replay: PlanarLcChartReplay,
    target_tube_replay: PlanarLcTubeReplay,
    source_endpoint_state_box: Option<Vec<RationalInterval>>,
    lift_replay: Option<PlanarLcLiftReplay>,
    entry_time_interval: Option<RationalInterval>,
    target_anchor: Option<[BigRational; 14]>,
    gauge_assignments: Option<[Vec<u8>; 2]>,
    assignment_maximum_gaps: Option<[BigRational; 2]>,
    assignment_containments: Option<[bool; 2]>,
    selected_assignment_index: Option<usize>,
    selected_assignment: Option<Vec<u8>>,
    selected_transformed_patches: Option<Vec<[RationalInterval; 14]>>,
    mass_kernel_id: Option<&'static str>,
    analytic_kernel_id: Option<&'static str>,
}

impl CarriedPlanarLcEntryReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[CarriedPlanarLcEntryObligation; 21] {
        &self.obligations
    }

    /// Conditional on the parent chain induction premise that its actual
    /// branch lies in the freshly replayed source tube.
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(CarriedPlanarLcEntryObligation::satisfied)
    }

    pub fn source_chart_replay(&self) -> &OrdinaryChartReplay {
        &self.source_chart_replay
    }
    pub fn source_tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.source_tube_replay
    }
    pub fn target_chart_replay(&self) -> &PlanarLcChartReplay {
        &self.target_chart_replay
    }
    pub fn target_tube_replay(&self) -> &PlanarLcTubeReplay {
        &self.target_tube_replay
    }
    pub fn source_endpoint_state_box(&self) -> Option<&[RationalInterval]> {
        self.source_endpoint_state_box.as_deref()
    }
    pub fn lift_replay(&self) -> Option<&PlanarLcLiftReplay> {
        self.lift_replay.as_ref()
    }
    pub fn entry_time_interval(&self) -> Option<&RationalInterval> {
        self.entry_time_interval.as_ref()
    }
    pub fn target_anchor(&self) -> Option<&[BigRational; 14]> {
        self.target_anchor.as_ref()
    }
    pub fn gauge_assignments(&self) -> Option<&[Vec<u8>; 2]> {
        self.gauge_assignments.as_ref()
    }
    pub fn assignment_maximum_gaps(&self) -> Option<&[BigRational; 2]> {
        self.assignment_maximum_gaps.as_ref()
    }
    pub fn assignment_containments(&self) -> Option<&[bool; 2]> {
        self.assignment_containments.as_ref()
    }
    pub const fn selected_assignment_index(&self) -> Option<usize> {
        self.selected_assignment_index
    }
    pub fn selected_assignment(&self) -> Option<&[u8]> {
        self.selected_assignment.as_deref()
    }
    pub fn selected_transformed_patches(&self) -> Option<&[[RationalInterval; 14]]> {
        self.selected_transformed_patches.as_deref()
    }
    pub const fn mass_kernel_id(&self) -> Option<&'static str> {
        self.mass_kernel_id
    }
    pub const fn analytic_kernel_id(&self) -> Option<&'static str> {
        self.analytic_kernel_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeCarriedPlanarLcEntryObligation {
    id: &'static str,
    satisfied: bool,
}

impl ProofGradeCarriedPlanarLcEntryObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

/// Proof-grade carried entry evidence.  Claimed-tail chart replays are
/// retained as diagnostics and are intentionally absent from the decisive
/// obligation ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeCarriedPlanarLcEntryReplay {
    obligations: [ProofGradeCarriedPlanarLcEntryObligation; 19],
    source_claimed_tail_chart_diagnostic: Result<OrdinaryChartReplay, OrdinaryChartReplayError>,
    target_claimed_tail_chart_diagnostic: Result<PlanarLcChartReplay, PlanarLcChartReplayError>,
    source_tube_replay: OrdinaryTubeReplay,
    target_tube_replay: PlanarLcTubeReplay,
    direct_evidence: Option<CarriedPlanarLcEntryDirectEvidence>,
    mass_kernel_id: Option<&'static str>,
}

impl ProofGradeCarriedPlanarLcEntryReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID
    }

    pub fn obligations(&self) -> &[ProofGradeCarriedPlanarLcEntryObligation; 19] {
        &self.obligations
    }

    /// Conditional on the parent chain induction premise that its actual
    /// branch lies in the freshly replayed source tube.
    pub fn conditional_profile_satisfied(&self) -> bool {
        self.obligations
            .iter()
            .all(ProofGradeCarriedPlanarLcEntryObligation::satisfied)
    }

    pub fn source_claimed_tail_chart_diagnostic(
        &self,
    ) -> Result<&OrdinaryChartReplay, &OrdinaryChartReplayError> {
        self.source_claimed_tail_chart_diagnostic.as_ref()
    }

    pub fn target_claimed_tail_chart_diagnostic(
        &self,
    ) -> Result<&PlanarLcChartReplay, &PlanarLcChartReplayError> {
        self.target_claimed_tail_chart_diagnostic.as_ref()
    }

    pub fn source_tube_replay(&self) -> &OrdinaryTubeReplay {
        &self.source_tube_replay
    }

    pub fn target_tube_replay(&self) -> &PlanarLcTubeReplay {
        &self.target_tube_replay
    }

    pub fn source_endpoint_state_box(&self) -> Option<&[RationalInterval]> {
        self.direct_evidence
            .as_ref()
            .map(|evidence| evidence.source_endpoint_state_box.as_slice())
    }

    pub fn lift_replay(&self) -> Option<&PlanarLcLiftReplay> {
        self.direct_evidence
            .as_ref()
            .map(|evidence| &evidence.lift_replay)
    }

    pub fn entry_time_interval(&self) -> Option<&RationalInterval> {
        self.direct_evidence
            .as_ref()
            .map(|evidence| &evidence.entry_time_interval)
    }

    pub fn target_anchor(&self) -> Option<&[BigRational; 14]> {
        self.direct_evidence
            .as_ref()
            .map(|evidence| &evidence.target_anchor)
    }

    pub fn gauge_assignments(&self) -> Option<&[Vec<u8>; 2]> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.gauge_assignments.as_ref())
    }

    pub fn assignment_maximum_gaps(&self) -> Option<&[BigRational; 2]> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.assignment_maximum_gaps.as_ref())
    }

    pub fn assignment_containments(&self) -> Option<&[bool; 2]> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.assignment_containments.as_ref())
    }

    pub fn selected_assignment_index(&self) -> Option<usize> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.selected_assignment_index)
    }

    pub fn selected_assignment(&self) -> Option<&[u8]> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.selected_assignment.as_deref())
    }

    pub fn selected_transformed_patches(&self) -> Option<&[[RationalInterval; 14]]> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.selected_transformed_patches.as_deref())
    }

    pub const fn mass_kernel_id(&self) -> Option<&'static str> {
        self.mass_kernel_id
    }

    pub fn analytic_kernel_id(&self) -> Option<&'static str> {
        self.direct_evidence
            .as_ref()
            .and_then(|evidence| evidence.analytic_kernel_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CarriedPlanarLcEntryDirectEvidence {
    source_endpoint_state_box: Vec<RationalInterval>,
    lift_replay: PlanarLcLiftReplay,
    entry_time_interval: RationalInterval,
    target_anchor: [BigRational; 14],
    gauge_assignments: Option<[Vec<u8>; 2]>,
    assignment_maximum_gaps: Option<[BigRational; 2]>,
    assignment_containments: Option<[bool; 2]>,
    selected_assignment_index: Option<usize>,
    selected_assignment: Option<Vec<u8>>,
    selected_transformed_patches: Option<Vec<[RationalInterval; 14]>>,
    analytic_kernel_id: Option<&'static str>,
    derived_satisfaction: [bool; 9],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CarriedPlanarLcEntryError {
    Semantic(PlanarLcSemanticError),
    SourceProvenanceSemantic(OrdinarySemanticError),
    SourceProvenanceMismatch { field: &'static str },
    NamespaceSegmentLimit { actual: usize, limit: usize },
    SourceChart(OrdinaryChartReplayError),
    SourceTube(OrdinaryTubeReplayError),
    TargetChart(PlanarLcChartReplayError),
    TargetTube(PlanarLcTubeReplayError),
    Lift(PlanarLcLiftError),
    TargetState(PlanarLcStateError),
    SourcePositionPolynomial(PolynomialError),
    SourceVelocityPolynomial(PolynomialError),
    MassDecode { index: usize, source: NumericError },
    MassProfile(MassProfileError),
    InternalShapeInvariant { detail: &'static str },
    Numeric(NumericError),
}

impl fmt::Display for CarriedPlanarLcEntryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "carried planar LC entry replay failed: {self:?}")
    }
}

impl std::error::Error for CarriedPlanarLcEntryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Semantic(source) => Some(source),
            Self::SourceProvenanceSemantic(source) => Some(source),
            Self::SourceChart(source) => Some(source),
            Self::SourceTube(source) => Some(source),
            Self::TargetChart(source) => Some(source),
            Self::TargetTube(source) => Some(source),
            Self::Lift(source) => Some(source),
            Self::TargetState(source) => Some(source),
            Self::SourcePositionPolynomial(source) | Self::SourceVelocityPolynomial(source) => {
                Some(source)
            }
            Self::MassDecode { source, .. } | Self::Numeric(source) => Some(source),
            Self::MassProfile(source) => Some(source),
            Self::NamespaceSegmentLimit { .. }
            | Self::SourceProvenanceMismatch { .. }
            | Self::InternalShapeInvariant { .. } => None,
        }
    }
}

impl From<NumericError> for CarriedPlanarLcEntryError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}
impl From<PlanarLcSemanticError> for CarriedPlanarLcEntryError {
    fn from(source: PlanarLcSemanticError) -> Self {
        Self::Semantic(source)
    }
}
impl From<MassProfileError> for CarriedPlanarLcEntryError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

pub fn replay_carried_planar_lc_entry_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
    parent_clock_origin: &RationalInterval,
) -> Result<CarriedPlanarLcEntryReplay, CarriedPlanarLcEntryError> {
    require_admission_bound_source(admission, segment_index, source_chart, source_tube)?;
    let entry =
        planar_lc_entry_input_from_admission(admission, segment_index, source_chart, source_tube)?;
    let namespace_unique = raw_v1_defining_namespace_unique(admission)?;
    let parent_clock = RationalInterval::new(
        parent_clock_origin.lower().clone(),
        parent_clock_origin.upper().clone(),
    )?;
    let source_chart_replay = replay_ordinary_chart_exact_rational_claimed_tail_v04(source_chart)
        .map_err(CarriedPlanarLcEntryError::SourceChart)?;
    let source_tube_replay = replay_ordinary_tube_exact_rational_v04(source_chart, source_tube)
        .map_err(CarriedPlanarLcEntryError::SourceTube)?;
    let target_chart_replay =
        replay_planar_lc_chart_exact_rational_claimed_tail_v04(entry.target_chart())
            .map_err(CarriedPlanarLcEntryError::TargetChart)?;
    let target_tube_replay =
        replay_planar_lc_tube_exact_rational_v04(entry.target_chart(), entry.target_tube())
            .map_err(CarriedPlanarLcEntryError::TargetTube)?;
    let common_masses = source_chart.masses() == entry.target_chart().masses();
    let pair = entry.target_chart().pair();
    let canonical_pair = pair[0] < pair[1] && pair[1] < 3;
    let mass_kernel_id = revalidate_mass_kernel(entry.target_chart())?;
    let endpoints = entry.source_right_parameter() == source_chart.parameter_interval().upper()
        && entry.target_left_parameter() == entry.target_chart().parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_tube().anchor_parameter();
    let base = [
        true,
        true,
        namespace_unique,
        true,
        source_chart_replay.conditional_profile_satisfied(),
        source_tube_replay.certified(),
        target_chart_replay.conditional_profile_satisfied(),
        target_tube_replay.certified(),
        common_masses,
        canonical_pair,
        mass_kernel_id == PLANAR_LC_MASS_KERNEL_ID,
        endpoints,
    ];
    if !base.iter().all(|value| *value) {
        return Ok(build_replay(
            base,
            source_chart_replay,
            source_tube_replay,
            target_chart_replay,
            target_tube_replay,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(mass_kernel_id),
            None,
        ));
    }

    let CarriedPlanarLcEntryDirectEvidence {
        source_endpoint_state_box,
        lift_replay,
        entry_time_interval,
        target_anchor,
        gauge_assignments,
        assignment_maximum_gaps,
        assignment_containments,
        selected_assignment_index,
        selected_assignment,
        selected_transformed_patches,
        analytic_kernel_id,
        derived_satisfaction,
    } = replay_carried_planar_lc_entry_direct_evidence(
        &entry,
        source_chart,
        &source_tube_replay,
        &parent_clock,
    )?;
    Ok(build_replay(
        base,
        source_chart_replay,
        source_tube_replay,
        target_chart_replay,
        target_tube_replay,
        Some(source_endpoint_state_box),
        Some(lift_replay),
        Some(entry_time_interval),
        Some(target_anchor),
        gauge_assignments,
        assignment_maximum_gaps,
        assignment_containments,
        selected_assignment_index,
        selected_assignment,
        Some(mass_kernel_id),
        analytic_kernel_id,
    )
    .with_derived(derived_satisfaction, selected_transformed_patches))
}

/// Replay the proof-grade carried ordinary-to-LC entry profile.
///
/// The ordinary and LC claimed-tail chart profiles are retained only as
/// diagnostics.  Their false obligations and typed resource/errors cannot
/// gate or erase the direct tube, lift-cover, clock, gauge, and containment
/// evidence in this result.
pub fn replay_proof_grade_carried_planar_lc_entry_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
    parent_clock_origin: &RationalInterval,
) -> Result<ProofGradeCarriedPlanarLcEntryReplay, CarriedPlanarLcEntryError> {
    require_admission_bound_source(admission, segment_index, source_chart, source_tube)?;
    let entry =
        planar_lc_entry_input_from_admission(admission, segment_index, source_chart, source_tube)?;
    let namespace_unique = raw_v1_defining_namespace_unique(admission)?;
    let parent_clock = RationalInterval::new(
        parent_clock_origin.lower().clone(),
        parent_clock_origin.upper().clone(),
    )?;

    let source_claimed_tail_chart_diagnostic =
        replay_ordinary_chart_exact_rational_claimed_tail_v04(source_chart);
    let target_claimed_tail_chart_diagnostic =
        replay_planar_lc_chart_exact_rational_claimed_tail_v04(entry.target_chart());
    let source_tube_replay = replay_ordinary_tube_exact_rational_v04(source_chart, source_tube)
        .map_err(CarriedPlanarLcEntryError::SourceTube)?;
    let target_tube_replay =
        replay_planar_lc_tube_exact_rational_v04(entry.target_chart(), entry.target_tube())
            .map_err(CarriedPlanarLcEntryError::TargetTube)?;

    let common_masses = source_chart.masses() == entry.target_chart().masses();
    let pair = entry.target_chart().pair();
    let canonical_pair = pair[0] < pair[1] && pair[1] < 3;
    let mass_kernel_id = revalidate_mass_kernel(entry.target_chart())?;
    let endpoints = entry.source_right_parameter() == source_chart.parameter_interval().upper()
        && entry.target_left_parameter() == entry.target_chart().parameter_interval().lower()
        && entry.target_left_parameter() == entry.target_tube().anchor_parameter();
    let base = [
        true,
        true,
        namespace_unique,
        true,
        source_tube_replay.certified(),
        target_tube_replay.certified(),
        common_masses,
        canonical_pair,
        mass_kernel_id == PLANAR_LC_MASS_KERNEL_ID,
        endpoints,
    ];
    let direct_evidence = if base.iter().all(|value| *value) {
        Some(replay_carried_planar_lc_entry_direct_evidence(
            &entry,
            source_chart,
            &source_tube_replay,
            &parent_clock,
        )?)
    } else {
        None
    };

    Ok(build_proof_grade_replay(
        base,
        source_claimed_tail_chart_diagnostic,
        target_claimed_tail_chart_diagnostic,
        source_tube_replay,
        target_tube_replay,
        Some(mass_kernel_id),
        direct_evidence,
    ))
}

fn replay_carried_planar_lc_entry_direct_evidence(
    entry: &PlanarLcEntryInput,
    source_chart: &OrdinaryChartInput,
    source_tube_replay: &OrdinaryTubeReplay,
    parent_clock: &RationalInterval,
) -> Result<CarriedPlanarLcEntryDirectEvidence, CarriedPlanarLcEntryError> {
    let source_endpoint_state_box = reconstruct_source_endpoint(
        source_chart,
        entry.source_right_parameter(),
        source_tube_replay,
    )?;
    let endpoint_reconstructed = source_endpoint_state_box.len() == 12;
    let lift_replay = replay_planar_lc_lift_cover_exact_rational_v04(
        entry.target_chart(),
        &source_endpoint_state_box,
    )
    .map_err(CarriedPlanarLcEntryError::Lift)?;
    let collision_free = lift_replay.collision_free();
    let atlas_reconstructed =
        lift_replay.complete_cover() && matches!(lift_replay.patches().len(), 1 | 2);
    let gauge_assignments = canonical_gauge_assignments(&lift_replay);
    let graph_certified = gauge_assignments.is_some();
    let entry_time_interval = derive_entry_time(parent_clock, entry.source_right_parameter())?;
    let positive_rho = !lift_replay.patches().is_empty()
        && lift_replay
            .patches()
            .iter()
            .all(|patch| patch.rho_lower_bound().numer().sign() == Sign::Plus);
    let target_anchor = reconstruct_target_anchor(entry)?;
    let target_anchor_reconstructed = true;
    let analytic_kernel_id = lift_replay.analytic_kernel_id();
    let analytic_kernel =
        analytic_kernel_id == Some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID);

    let (
        assignment_maximum_gaps,
        assignment_containments,
        selected_assignment_index,
        selected_transformed_patches,
    ) = if let Some(assignments) = &gauge_assignments {
        let (gaps, containments, selected, patches) = evaluate_global_assignments(
            lift_replay.patches(),
            assignments,
            &entry_time_interval,
            &target_anchor,
            entry.target_tube().initial_error_bound(),
        )?;
        (Some(gaps), Some(containments), selected, patches)
    } else {
        (None, None, None, None)
    };
    let containment = selected_assignment_index.is_some();
    let selected_assignment = selected_assignment_index.and_then(|index| {
        gauge_assignments
            .as_ref()
            .map(|values| values[index].clone())
    });
    let derived_satisfaction = derived_satisfaction(
        endpoint_reconstructed,
        collision_free,
        atlas_reconstructed,
        graph_certified,
        true,
        positive_rho,
        target_anchor_reconstructed,
        analytic_kernel,
        containment,
    );
    Ok(CarriedPlanarLcEntryDirectEvidence {
        source_endpoint_state_box,
        lift_replay,
        entry_time_interval,
        target_anchor,
        gauge_assignments,
        assignment_maximum_gaps,
        assignment_containments,
        selected_assignment_index,
        selected_assignment,
        selected_transformed_patches,
        analytic_kernel_id,
        derived_satisfaction,
    })
}

fn require_admission_bound_source(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
) -> Result<(), CarriedPlanarLcEntryError> {
    let wire = admission.wire();
    let (chart_wire, tube_wire) = if segment_index == 0 {
        (&wire.initial_chart, &wire.initial_tube)
    } else {
        match wire.segments.get(segment_index - 1) {
            Some(SegmentWire::OrdinaryBridge(segment)) => {
                (&segment.target_chart, &segment.target_tube)
            }
            Some(SegmentWire::PlanarLcPassage(segment)) => {
                (&segment.target_chart, &segment.target_tube)
            }
            None => {
                return Err(CarriedPlanarLcEntryError::Semantic(
                    PlanarLcSemanticError::SegmentUnavailable { segment_index },
                ));
            }
        }
    };
    let expected_chart = ordinary_chart_input_from_wire(chart_wire)
        .map_err(CarriedPlanarLcEntryError::SourceProvenanceSemantic)?;
    let expected_tube = ordinary_tube_input_from_wire(tube_wire);
    if source_chart != &expected_chart {
        return Err(CarriedPlanarLcEntryError::SourceProvenanceMismatch {
            field: "source_chart",
        });
    }
    if source_tube != &expected_tube {
        return Err(CarriedPlanarLcEntryError::SourceProvenanceMismatch {
            field: "source_tube",
        });
    }
    Ok(())
}

trait WithDerived {
    fn with_derived(
        self,
        derived: [bool; 9],
        selected_patches: Option<Vec<[RationalInterval; 14]>>,
    ) -> Self;
}

#[allow(clippy::too_many_arguments)]
fn derived_satisfaction(
    endpoint: bool,
    collision_free: bool,
    atlas: bool,
    graph: bool,
    time: bool,
    positive_rho: bool,
    anchor: bool,
    analytic_kernel: bool,
    containment: bool,
) -> [bool; 9] {
    let collision = endpoint && collision_free;
    let atlas = collision && atlas;
    let graph = atlas && graph;
    let time = endpoint && time;
    let rho = atlas && positive_rho;
    let anchor = endpoint && anchor;
    let analytic = graph && rho && analytic_kernel;
    let containment = graph && time && rho && anchor && analytic && containment;
    [
        endpoint,
        collision,
        atlas,
        graph,
        time,
        rho,
        anchor,
        analytic,
        containment,
    ]
}

impl WithDerived for CarriedPlanarLcEntryReplay {
    fn with_derived(
        mut self,
        derived: [bool; 9],
        selected_patches: Option<Vec<[RationalInterval; 14]>>,
    ) -> Self {
        for (offset, satisfied) in derived.into_iter().enumerate() {
            self.obligations[12 + offset].satisfied = satisfied;
        }
        self.selected_transformed_patches = selected_patches;
        self
    }
}

/// Bounded global defining-ID namespace check, including all six deterministic
/// derived identifiers reserved by every LC entry.
pub fn raw_v1_defining_namespace_unique(
    admission: &CanonicalRawV1Admission,
) -> Result<bool, CarriedPlanarLcEntryError> {
    defining_namespace_unique_wire(admission.wire())
}

fn defining_namespace_unique_wire(
    wire: &crate::raw_schema::RawPlanarChainWire,
) -> Result<bool, CarriedPlanarLcEntryError> {
    if wire.segments.len() > HARD_MAX_RAW_V1_NAMESPACE_SEGMENTS {
        return Err(CarriedPlanarLcEntryError::NamespaceSegmentLimit {
            actual: wire.segments.len(),
            limit: HARD_MAX_RAW_V1_NAMESPACE_SEGMENTS,
        });
    }
    let mut identifiers = vec![
        wire.certificate_id.as_str(),
        wire.root_binding.binding_id.as_str(),
        wire.initial_chart.certificate_id.as_str(),
        wire.initial_chart.chart_id.as_str(),
        wire.initial_tube.tube_id.as_str(),
    ];
    let mut owned_reserved = Vec::new();
    for segment in &wire.segments {
        match segment {
            SegmentWire::OrdinaryBridge(segment) => identifiers.extend([
                segment.transition.transition_id.as_str(),
                segment.target_chart.certificate_id.as_str(),
                segment.target_chart.chart_id.as_str(),
                segment.target_tube.tube_id.as_str(),
            ]),
            SegmentWire::PlanarLcPassage(segment) => {
                identifiers.extend([
                    segment.entry_transition.transition_id.as_str(),
                    segment.lc_chart.certificate_id.as_str(),
                    segment.lc_chart.chart_id.as_str(),
                    segment.lc_tube.tube_id.as_str(),
                    segment.exit_transition.transition_id.as_str(),
                    segment.target_chart.certificate_id.as_str(),
                    segment.target_chart.chart_id.as_str(),
                    segment.target_tube.tube_id.as_str(),
                ]);
                for suffix in [
                    "derived-gauge-cover",
                    "patch:0-upper",
                    "patch:0-lower",
                    "patch:0-right",
                    "patch:1-lower",
                    "negative-axis-overlap",
                ] {
                    owned_reserved.push(format!(
                        "{}:{suffix}",
                        segment.entry_transition.transition_id
                    ));
                }
            }
        }
    }
    let mut seen = HashSet::with_capacity(identifiers.len() + owned_reserved.len());
    for identifier in identifiers
        .into_iter()
        .chain(owned_reserved.iter().map(String::as_str))
    {
        if identifier.is_empty() || !seen.insert(identifier) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reconstruct_source_endpoint(
    source_chart: &OrdinaryChartInput,
    parameter: &BigRational,
    source_tube_replay: &OrdinaryTubeReplay,
) -> Result<Vec<RationalInterval>, CarriedPlanarLcEntryError> {
    let argument = RationalInterval::try_point(parameter.clone())?;
    let mut centers = source_chart
        .position_polynomial()
        .evaluate(&argument)
        .map_err(CarriedPlanarLcEntryError::SourcePositionPolynomial)?;
    centers.extend(
        source_chart
            .velocity_polynomial()
            .evaluate(&argument)
            .map_err(CarriedPlanarLcEntryError::SourceVelocityPolynomial)?,
    );
    if centers.len() != 12 {
        return Err(CarriedPlanarLcEntryError::InternalShapeInvariant {
            detail: "source endpoint state is not 12-dimensional",
        });
    }
    let radius = source_tube_replay.gronwall_upper().ok_or(
        CarriedPlanarLcEntryError::InternalShapeInvariant {
            detail: "freshly certified source tube lacks Gronwall bound",
        },
    )?;
    let inflation = RationalInterval::new(-radius.clone(), radius.clone())?;
    centers
        .into_iter()
        .map(|center| center.add(&inflation).map_err(Into::into))
        .collect()
}

fn derive_entry_time(
    parent_clock: &RationalInterval,
    source_right: &BigRational,
) -> Result<RationalInterval, CarriedPlanarLcEntryError> {
    Ok(parent_clock.add(&RationalInterval::try_point(source_right.clone())?)?)
}

fn reconstruct_target_anchor(
    entry: &PlanarLcEntryInput,
) -> Result<[BigRational; 14], CarriedPlanarLcEntryError> {
    let polynomial = PlanarLcStatePolynomial::from_chart(entry.target_chart())
        .map_err(CarriedPlanarLcEntryError::TargetState)?;
    let point = polynomial
        .evaluate_anchor_point(entry.target_left_parameter())
        .map_err(CarriedPlanarLcEntryError::TargetState)?;
    let mut values = point.lifted_without_time().to_vec();
    values.push(point.physical_time().clone());
    values
        .try_into()
        .map_err(|_| CarriedPlanarLcEntryError::InternalShapeInvariant {
            detail: "target anchor is not 14-dimensional",
        })
}

fn canonical_gauge_assignments(lift: &PlanarLcLiftReplay) -> Option<[Vec<u8>; 2]> {
    let edge = match lift.parity_edges() {
        [] => None,
        [edge] => Some((edge.source_patch(), edge.target_patch(), edge.parity())),
        _ => return None,
    };
    canonical_gauge_assignments_from_graph(lift.patches().len(), edge)
}

fn canonical_gauge_assignments_from_graph(
    patch_count: usize,
    edge: Option<(usize, usize, u8)>,
) -> Option<[Vec<u8>; 2]> {
    match (patch_count, edge) {
        (1, None) => Some([vec![0], vec![1]]),
        (2, Some((0, 1, 1))) => Some([vec![0, 1], vec![1, 0]]),
        _ => None,
    }
}

type AssignmentEvaluation = (
    [BigRational; 2],
    [bool; 2],
    Option<usize>,
    Option<Vec<[RationalInterval; 14]>>,
);

fn evaluate_global_assignments(
    patches: &[PlanarLcLiftPatch],
    assignments: &[Vec<u8>; 2],
    entry_time: &RationalInterval,
    target_anchor: &[BigRational; 14],
    allowance: &BigRational,
) -> Result<AssignmentEvaluation, CarriedPlanarLcEntryError> {
    let mut gaps = [BigRational::zero(), BigRational::zero()];
    let mut containments = [false, false];
    let mut transformed_by_assignment = [Vec::new(), Vec::new()];
    for assignment_index in 0..2 {
        if assignments[assignment_index].len() != patches.len() {
            return Err(CarriedPlanarLcEntryError::InternalShapeInvariant {
                detail: "gauge assignment dimension mismatch",
            });
        }
        let mut maximum = BigRational::zero();
        let mut contained = true;
        for (patch, bit) in patches.iter().zip(&assignments[assignment_index]) {
            let transformed = match bit {
                0 => patch.clone(),
                1 => patch
                    .deck_transform()
                    .map_err(CarriedPlanarLcEntryError::Lift)?,
                _ => {
                    return Err(CarriedPlanarLcEntryError::InternalShapeInvariant {
                        detail: "non-binary gauge assignment",
                    })
                }
            };
            let timed = append_time(&transformed, entry_time)?;
            for (component, center) in timed.iter().zip(target_anchor) {
                let lower_gap = (component.lower() - center).abs();
                let upper_gap = (component.upper() - center).abs();
                maximum = maximum.max(lower_gap).max(upper_gap);
                let target = RationalInterval::new(center - allowance, center + allowance)?;
                contained &= target.contains_interval(component);
            }
            transformed_by_assignment[assignment_index].push(timed);
        }
        gaps[assignment_index] = maximum;
        containments[assignment_index] = contained;
    }
    let selected = containments.iter().position(|value| *value);
    let selected_patches = selected.map(|index| transformed_by_assignment[index].clone());
    Ok((gaps, containments, selected, selected_patches))
}

fn append_time(
    patch: &PlanarLcLiftPatch,
    time: &RationalInterval,
) -> Result<[RationalInterval; 14], CarriedPlanarLcEntryError> {
    let mut components = patch.components().to_vec();
    if components.len() != PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION {
        return Err(CarriedPlanarLcEntryError::InternalShapeInvariant {
            detail: "lift patch is not 13-dimensional",
        });
    }
    components.push(RationalInterval::new(
        time.lower().clone(),
        time.upper().clone(),
    )?);
    components
        .try_into()
        .map_err(|_| CarriedPlanarLcEntryError::InternalShapeInvariant {
            detail: "timed lift patch is not 14-dimensional",
        })
}

fn revalidate_mass_kernel(
    chart: &crate::PlanarLcChartInput,
) -> Result<&'static str, CarriedPlanarLcEntryError> {
    let mut masses = Vec::with_capacity(3);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        masses.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| CarriedPlanarLcEntryError::MassDecode { index, source })?,
        );
    }
    let masses: [ExactBinary64; 3] =
        masses
            .try_into()
            .map_err(|_| CarriedPlanarLcEntryError::InternalShapeInvariant {
                detail: "mass vector is not three-dimensional",
            })?;
    let profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    profile.validate_against(&masses, chart.pair())?;
    Ok(profile.kernel_id())
}

#[allow(clippy::too_many_arguments)]
fn build_proof_grade_replay(
    base: [bool; 10],
    source_claimed_tail_chart_diagnostic: Result<OrdinaryChartReplay, OrdinaryChartReplayError>,
    target_claimed_tail_chart_diagnostic: Result<PlanarLcChartReplay, PlanarLcChartReplayError>,
    source_tube_replay: OrdinaryTubeReplay,
    target_tube_replay: PlanarLcTubeReplay,
    mass_kernel_id: Option<&'static str>,
    direct_evidence: Option<CarriedPlanarLcEntryDirectEvidence>,
) -> ProofGradeCarriedPlanarLcEntryReplay {
    let mut satisfaction = [false; 19];
    satisfaction[..10].copy_from_slice(&base);
    if let Some(evidence) = &direct_evidence {
        satisfaction[10..].copy_from_slice(&evidence.derived_satisfaction);
    }
    ProofGradeCarriedPlanarLcEntryReplay {
        obligations: std::array::from_fn(|index| ProofGradeCarriedPlanarLcEntryObligation {
            id: PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        source_claimed_tail_chart_diagnostic,
        target_claimed_tail_chart_diagnostic,
        source_tube_replay,
        target_tube_replay,
        direct_evidence,
        mass_kernel_id,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_replay(
    base: [bool; 12],
    source_chart_replay: OrdinaryChartReplay,
    source_tube_replay: OrdinaryTubeReplay,
    target_chart_replay: PlanarLcChartReplay,
    target_tube_replay: PlanarLcTubeReplay,
    source_endpoint_state_box: Option<Vec<RationalInterval>>,
    lift_replay: Option<PlanarLcLiftReplay>,
    entry_time_interval: Option<RationalInterval>,
    target_anchor: Option<[BigRational; 14]>,
    gauge_assignments: Option<[Vec<u8>; 2]>,
    assignment_maximum_gaps: Option<[BigRational; 2]>,
    assignment_containments: Option<[bool; 2]>,
    selected_assignment_index: Option<usize>,
    selected_assignment: Option<Vec<u8>>,
    mass_kernel_id: Option<&'static str>,
    analytic_kernel_id: Option<&'static str>,
) -> CarriedPlanarLcEntryReplay {
    let mut satisfaction = [false; 21];
    satisfaction[..12].copy_from_slice(&base);
    CarriedPlanarLcEntryReplay {
        obligations: std::array::from_fn(|index| CarriedPlanarLcEntryObligation {
            id: CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS[index],
            satisfied: satisfaction[index],
        }),
        source_chart_replay,
        source_tube_replay,
        target_chart_replay,
        target_tube_replay,
        source_endpoint_state_box,
        lift_replay,
        entry_time_interval,
        target_anchor,
        gauge_assignments,
        assignment_maximum_gaps,
        assignment_containments,
        selected_assignment_index,
        selected_assignment,
        selected_transformed_patches: None,
        mass_kernel_id,
        analytic_kernel_id,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use num_bigint::BigInt;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, PlanarLcSeriesError, DEFAULT_JSON_NUMBER_LIMITS,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
    ));
    const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/failed-revisit.raw.json"
    ));

    fn replay(bytes: &[u8]) -> CarriedPlanarLcEntryReplay {
        let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
        replay_admission(
            &admission,
            &RationalInterval::try_point(BigRational::new(
                BigInt::from(1),
                BigInt::from(1_u64 << 40),
            ))
            .unwrap(),
        )
    }

    fn replay_admission(
        admission: &CanonicalRawV1Admission,
        parent: &RationalInterval,
    ) -> CarriedPlanarLcEntryReplay {
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
        replay_carried_planar_lc_entry_exact_rational_v04(
            admission,
            1,
            &source_chart,
            &source_tube,
            parent,
        )
        .unwrap()
    }

    fn canonical_replays() -> &'static [CarriedPlanarLcEntryReplay; 2] {
        static REPLAYS: OnceLock<[CarriedPlanarLcEntryReplay; 2]> = OnceLock::new();
        REPLAYS.get_or_init(|| [replay(SUCCESS_RAW), replay(FAILED_REVISIT_RAW)])
    }

    fn proof_grade_replay(bytes: &[u8]) -> ProofGradeCarriedPlanarLcEntryReplay {
        let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
        proof_grade_replay_admission(
            &admission,
            &RationalInterval::try_point(BigRational::new(
                BigInt::from(1),
                BigInt::from(1_u64 << 40),
            ))
            .unwrap(),
        )
    }

    fn proof_grade_replay_admission(
        admission: &CanonicalRawV1Admission,
        parent: &RationalInterval,
    ) -> ProofGradeCarriedPlanarLcEntryReplay {
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
        replay_proof_grade_carried_planar_lc_entry_exact_rational_v04(
            admission,
            1,
            &source_chart,
            &source_tube,
            parent,
        )
        .unwrap()
    }

    fn proof_grade_canonical_replays() -> &'static [ProofGradeCarriedPlanarLcEntryReplay; 2] {
        static REPLAYS: OnceLock<[ProofGradeCarriedPlanarLcEntryReplay; 2]> = OnceLock::new();
        REPLAYS.get_or_init(|| {
            [
                proof_grade_replay(SUCCESS_RAW),
                proof_grade_replay(FAILED_REVISIT_RAW),
            ]
        })
    }

    fn truth(replay: &CarriedPlanarLcEntryReplay) -> [bool; 21] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    fn proof_grade_truth(replay: &ProofGradeCarriedPlanarLcEntryReplay) -> [bool; 19] {
        std::array::from_fn(|index| replay.obligations()[index].satisfied())
    }

    fn assert_direct_artifacts_equal(
        left: &ProofGradeCarriedPlanarLcEntryReplay,
        right: &ProofGradeCarriedPlanarLcEntryReplay,
    ) {
        assert_eq!(left.source_tube_replay(), right.source_tube_replay());
        assert_eq!(left.target_tube_replay(), right.target_tube_replay());
        assert_eq!(
            left.source_endpoint_state_box(),
            right.source_endpoint_state_box()
        );
        assert_eq!(left.lift_replay(), right.lift_replay());
        assert_eq!(left.entry_time_interval(), right.entry_time_interval());
        assert_eq!(left.target_anchor(), right.target_anchor());
        assert_eq!(left.gauge_assignments(), right.gauge_assignments());
        assert_eq!(
            left.assignment_maximum_gaps(),
            right.assignment_maximum_gaps()
        );
        assert_eq!(
            left.assignment_containments(),
            right.assignment_containments()
        );
        assert_eq!(
            left.selected_assignment_index(),
            right.selected_assignment_index()
        );
        assert_eq!(left.selected_assignment(), right.selected_assignment());
        assert_eq!(
            left.selected_transformed_patches(),
            right.selected_transformed_patches()
        );
        assert_eq!(left.mass_kernel_id(), right.mass_kernel_id());
        assert_eq!(left.analytic_kernel_id(), right.analytic_kernel_id());
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
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
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
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' => object_stack.push(index),
                b'}' => {
                    object_stack.pop().expect("unbalanced object");
                }
                _ => {}
            }
        }
        *object_stack.last().expect("anchor outside object")
    }

    fn extend_ordinary_coefficients_to(raw: &str, certificate_id: &str, count: usize) -> String {
        let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
        let zero = "[[0.0,0.0],[0.0,0.0],[0.0,0.0]]";
        let mut result = raw.to_owned();
        for field in ["position_coefficients", "velocity_coefficients"] {
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
    fn proof_grade_canonical_entries_have_distinct_all_true_direct_ledgers() {
        for (compatibility, proof_grade) in canonical_replays()
            .iter()
            .zip(proof_grade_canonical_replays())
        {
            assert_eq!(
                compatibility.profile_id(),
                EXACT_RATIONAL_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID
            );
            assert_eq!(
                proof_grade.profile_id(),
                EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_V04_PROFILE_ID
            );
            assert_eq!(
                proof_grade
                    .obligations()
                    .iter()
                    .map(ProofGradeCarriedPlanarLcEntryObligation::id)
                    .collect::<Vec<_>>(),
                PROOF_GRADE_CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS
            );
            assert_eq!(proof_grade_truth(proof_grade), [true; 19]);
            assert!(proof_grade.conditional_profile_satisfied());
            assert!(proof_grade
                .source_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied());
            assert!(proof_grade
                .target_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied());
            assert_eq!(
                proof_grade.source_endpoint_state_box(),
                compatibility.source_endpoint_state_box()
            );
            assert_eq!(proof_grade.lift_replay(), compatibility.lift_replay());
            assert_eq!(
                proof_grade.entry_time_interval(),
                compatibility.entry_time_interval()
            );
            assert_eq!(proof_grade.target_anchor(), compatibility.target_anchor());
            assert_eq!(
                proof_grade.assignment_containments(),
                compatibility.assignment_containments()
            );
            assert_eq!(
                proof_grade.selected_transformed_patches(),
                compatibility.selected_transformed_patches()
            );
        }
    }

    #[test]
    fn proof_grade_claimed_tail_failures_are_diagnostic_only_and_preserve_direct_artifacts() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let canonical = &proof_grade_canonical_replays()[0];
        for (certificate_id, source_diagnostic_fails) in [
            ("review-v03:n:certificate:1", true),
            ("review-v03:lc:certificate:0", false),
        ] {
            let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
            let mutated = replace_scalar_after(raw, &anchor, "tail_bound", "-1.0");
            assert_ne!(mutated, raw);
            let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
            let parent = RationalInterval::try_point(BigRational::new(
                BigInt::from(1),
                BigInt::from(1_u64 << 40),
            ))
            .unwrap();

            let compatibility = replay_admission(&admission, &parent);
            assert!(!compatibility.conditional_profile_satisfied());
            let proof_grade = proof_grade_replay_admission(&admission, &parent);
            assert_eq!(proof_grade_truth(&proof_grade), [true; 19]);
            assert!(proof_grade.conditional_profile_satisfied());
            assert_direct_artifacts_equal(&proof_grade, canonical);

            let source_diagnostic = proof_grade
                .source_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied();
            let target_diagnostic = proof_grade
                .target_claimed_tail_chart_diagnostic()
                .unwrap()
                .conditional_profile_satisfied();
            assert_eq!(source_diagnostic, !source_diagnostic_fails);
            assert_eq!(target_diagnostic, source_diagnostic_fails);
        }
    }

    #[test]
    fn proof_grade_source_chart_resource_exhaustion_is_retained_without_gating_direct_entry() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let mutated = extend_ordinary_coefficients_to(raw, "review-v03:n:certificate:1", 146);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();

        assert!(matches!(
            replay_carried_planar_lc_entry_exact_rational_v04(
                &admission,
                1,
                &source_chart,
                &source_tube,
                &parent,
            ),
            Err(CarriedPlanarLcEntryError::SourceChart(
                OrdinaryChartReplayError::WorkLimitExceeded {
                    required: 2_018_400,
                    limit: 2_000_000,
                }
            ))
        ));

        let proof_grade = replay_proof_grade_carried_planar_lc_entry_exact_rational_v04(
            &admission,
            1,
            &source_chart,
            &source_tube,
            &parent,
        )
        .unwrap();
        assert!(matches!(
            proof_grade.source_claimed_tail_chart_diagnostic(),
            Err(OrdinaryChartReplayError::WorkLimitExceeded {
                required: 2_018_400,
                limit: 2_000_000,
            })
        ));
        assert_eq!(proof_grade_truth(&proof_grade), [true; 19]);
        assert!(proof_grade.conditional_profile_satisfied());
        assert!(proof_grade.source_tube_replay().certified());
        assert!(proof_grade.target_tube_replay().certified());
        assert!(proof_grade.source_endpoint_state_box().is_some());
        assert!(proof_grade.lift_replay().is_some());
        assert!(proof_grade.target_anchor().is_some());
    }

    #[test]
    fn proof_grade_target_chart_resource_exhaustion_is_retained_without_gating_direct_entry() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let mutated = extend_planar_lc_coefficients_to(raw, "review-v03:lc:certificate:0", 89);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();

        assert!(matches!(
            replay_carried_planar_lc_entry_exact_rational_v04(
                &admission,
                1,
                &source_chart,
                &source_tube,
                &parent,
            ),
            Err(CarriedPlanarLcEntryError::TargetChart(
                PlanarLcChartReplayError::Series(PlanarLcSeriesError::WorkLimitExceeded {
                    required: 4_055_552,
                    limit: 4_000_000,
                })
            ))
        ));

        let proof_grade = replay_proof_grade_carried_planar_lc_entry_exact_rational_v04(
            &admission,
            1,
            &source_chart,
            &source_tube,
            &parent,
        )
        .unwrap();
        assert!(matches!(
            proof_grade.target_claimed_tail_chart_diagnostic(),
            Err(PlanarLcChartReplayError::Series(
                PlanarLcSeriesError::WorkLimitExceeded {
                    required: 4_055_552,
                    limit: 4_000_000,
                }
            ))
        ));
        assert_eq!(proof_grade_truth(&proof_grade), [true; 19]);
        assert!(proof_grade.conditional_profile_satisfied());
        assert!(proof_grade.target_tube_replay().certified());
        assert_direct_artifacts_equal(&proof_grade, &proof_grade_canonical_replays()[0]);
    }

    #[test]
    fn both_canonical_first_entries_have_exact_ordered_all_true_ledgers() {
        let mut selected = Vec::new();
        for replay in canonical_replays() {
            assert_eq!(
                replay
                    .obligations()
                    .iter()
                    .map(CarriedPlanarLcEntryObligation::id)
                    .collect::<Vec<_>>(),
                CARRIED_PLANAR_LC_ENTRY_OBLIGATION_IDS
            );
            assert_eq!(
                truth(replay),
                [true; 21],
                "gaps={:?}, containment={:?}",
                replay.assignment_maximum_gaps(),
                replay.assignment_containments()
            );
            assert!(replay.conditional_profile_satisfied());
            assert_eq!(replay.source_endpoint_state_box().unwrap().len(), 12);
            assert_eq!(replay.target_anchor().unwrap().len(), 14);
            assert_eq!(replay.mass_kernel_id(), Some(PLANAR_LC_MASS_KERNEL_ID));
            assert_eq!(
                replay.analytic_kernel_id(),
                Some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID)
            );
            let expected_time =
                BigRational::new(BigInt::from(1_048_577), BigInt::from(1_u64 << 40));
            assert_eq!(
                replay.entry_time_interval(),
                Some(&RationalInterval::try_point(expected_time).unwrap())
            );
            assert_eq!(replay.selected_assignment_index(), Some(0));
            assert_eq!(replay.selected_assignment(), Some(&[0][..]));
            selected.push((
                replay.entry_time_interval().unwrap().clone(),
                replay.selected_assignment_index(),
                replay.selected_assignment().unwrap().to_vec(),
            ));
        }
        assert_eq!(selected[0], selected[1]);
    }

    #[test]
    fn canonical_derived_quantities_equal_independent_recomputations() {
        for (bytes, replay) in [SUCCESS_RAW, FAILED_REVISIT_RAW]
            .into_iter()
            .zip(canonical_replays())
        {
            let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
            let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
                panic!()
            };
            let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
            let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
            let entry =
                planar_lc_entry_input_from_admission(&admission, 1, &source_chart, &source_tube)
                    .unwrap();
            assert_eq!(
                replay.source_endpoint_state_box(),
                Some(
                    reconstruct_source_endpoint(
                        &source_chart,
                        entry.source_right_parameter(),
                        replay.source_tube_replay(),
                    )
                    .unwrap()
                    .as_slice()
                )
            );
            assert_eq!(
                replay.target_anchor(),
                Some(&reconstruct_target_anchor(&entry).unwrap())
            );
            let assignments = canonical_gauge_assignments(replay.lift_replay().unwrap()).unwrap();
            let mut independent_gaps = [BigRational::zero(), BigRational::zero()];
            let mut independent_containments = [true, true];
            let mut independently_transformed = [Vec::new(), Vec::new()];
            for assignment_index in 0..2 {
                for (patch, bit) in replay
                    .lift_replay()
                    .unwrap()
                    .patches()
                    .iter()
                    .zip(&assignments[assignment_index])
                {
                    let patch = if *bit == 0 {
                        patch.clone()
                    } else {
                        patch.deck_transform().unwrap()
                    };
                    let timed = append_time(&patch, replay.entry_time_interval().unwrap()).unwrap();
                    for (component, center) in timed.iter().zip(replay.target_anchor().unwrap()) {
                        independent_gaps[assignment_index] = independent_gaps[assignment_index]
                            .clone()
                            .max((component.lower() - center).abs())
                            .max((component.upper() - center).abs());
                        let target = RationalInterval::new(
                            center - entry.target_tube().initial_error_bound(),
                            center + entry.target_tube().initial_error_bound(),
                        )
                        .unwrap();
                        independent_containments[assignment_index] &=
                            target.contains_interval(component);
                    }
                    independently_transformed[assignment_index].push(timed);
                }
            }
            let independent_selected = independent_containments.iter().position(|value| *value);
            assert_eq!(replay.gauge_assignments(), Some(&assignments));
            assert_eq!(replay.assignment_maximum_gaps(), Some(&independent_gaps));
            assert_eq!(
                replay.assignment_containments(),
                Some(&independent_containments)
            );
            assert_eq!(replay.selected_assignment_index(), independent_selected);
            assert_eq!(
                replay.selected_transformed_patches(),
                independent_selected.map(|index| independently_transformed[index].as_slice())
            );
        }
    }

    #[test]
    fn nonpoint_parent_clock_translation_is_exact() {
        let parent = RationalInterval::new(
            BigRational::new(BigInt::from(-3), BigInt::from(2)),
            BigRational::new(BigInt::from(7), BigInt::from(3)),
        )
        .unwrap();
        let delta = BigRational::new(BigInt::from(5), BigInt::from(11));
        assert_eq!(
            derive_entry_time(&parent, &delta).unwrap(),
            RationalInterval::new(
                BigRational::new(BigInt::from(-23), BigInt::from(22)),
                BigRational::new(BigInt::from(92), BigInt::from(33)),
            )
            .unwrap()
        );
    }

    #[test]
    fn too_small_allowance_fails_containment_without_erasing_prior_evidence() {
        for replay in canonical_replays() {
            let evaluation = evaluate_global_assignments(
                replay.lift_replay().unwrap().patches(),
                replay.gauge_assignments().unwrap(),
                replay.entry_time_interval().unwrap(),
                replay.target_anchor().unwrap(),
                &BigRational::zero(),
            )
            .unwrap();
            assert_eq!(evaluation.1, [false, false]);
            assert_eq!(evaluation.2, None);
            assert!(replay.obligations()[12..20]
                .iter()
                .all(CarriedPlanarLcEntryObligation::satisfied));
        }
    }

    #[test]
    fn singleton_and_parity_one_graph_assignments_are_exact() {
        assert_eq!(
            canonical_gauge_assignments_from_graph(1, None),
            Some([vec![0], vec![1]])
        );
        assert_eq!(
            canonical_gauge_assignments_from_graph(2, Some((0, 1, 1))),
            Some([vec![0, 1], vec![1, 0]])
        );
        for invalid in [
            canonical_gauge_assignments_from_graph(2, None),
            canonical_gauge_assignments_from_graph(2, Some((1, 0, 1))),
            canonical_gauge_assignments_from_graph(2, Some((0, 1, 0))),
            canonical_gauge_assignments_from_graph(3, Some((0, 1, 1))),
        ] {
            assert_eq!(invalid, None);
        }
    }

    #[test]
    fn global_and_reserved_namespace_duplicates_fail_closed() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
        assert!(defining_namespace_unique_wire(admission.wire()).unwrap());

        let mut global_duplicate = admission.wire().clone();
        global_duplicate.initial_tube.tube_id = global_duplicate.certificate_id.clone();
        assert!(!defining_namespace_unique_wire(&global_duplicate).unwrap());

        let mut reserved_duplicate = admission.wire().clone();
        let SegmentWire::PlanarLcPassage(passage) = &reserved_duplicate.segments[1] else {
            panic!()
        };
        reserved_duplicate.initial_chart.chart_id =
            format!("{}:patch:0-upper", passage.entry_transition.transition_id);
        assert!(!defining_namespace_unique_wire(&reserved_duplicate).unwrap());
    }

    #[test]
    fn dependency_rows_fail_closed_while_independent_time_and_anchor_survive() {
        assert_eq!(
            derived_satisfaction(true, false, true, true, true, true, true, true, true),
            [true, false, false, false, true, false, true, false, false]
        );
        assert_eq!(
            derived_satisfaction(true, true, true, false, true, true, true, true, true),
            [true, true, true, false, true, true, true, false, false]
        );
        assert_eq!(
            derived_satisfaction(false, true, true, true, true, true, true, true, true),
            [false; 9]
        );
    }

    #[test]
    fn source_or_target_tube_gate_failure_retains_nested_replays_and_blocks_containment() {
        let canonical = &canonical_replays()[0];
        for failed_gate in [5_usize, 7] {
            let mut base = [true; 12];
            base[failed_gate] = false;
            let gated = build_replay(
                base,
                canonical.source_chart_replay().clone(),
                canonical.source_tube_replay().clone(),
                canonical.target_chart_replay().clone(),
                canonical.target_tube_replay().clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(PLANAR_LC_MASS_KERNEL_ID),
                None,
            );
            assert!(!gated.obligations()[failed_gate].satisfied());
            assert!(gated.obligations()[12..]
                .iter()
                .all(|obligation| !obligation.satisfied()));
            assert_eq!(gated.source_tube_replay(), canonical.source_tube_replay());
            assert_eq!(gated.target_tube_replay(), canonical.target_tube_replay());
            assert_eq!(gated.assignment_maximum_gaps(), None);
            assert_eq!(gated.assignment_containments(), None);
        }
    }

    #[test]
    fn same_id_forged_source_records_are_rejected_by_opaque_provenance_binding() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let expected_tube = ordinary_tube_input_from_wire(&bridge.target_tube);

        let mut forged_chart_wire = bridge.target_chart.clone();
        forged_chart_wire.position_coefficients[0][0][0] =
            forged_chart_wire.position_coefficients[1][0][0].clone();
        let forged_chart = ordinary_chart_input_from_wire(&forged_chart_wire).unwrap();
        assert_eq!(forged_chart.chart_id(), bridge.target_chart.chart_id);
        assert!(matches!(
            replay_carried_planar_lc_entry_exact_rational_v04(
                &admission,
                1,
                &forged_chart,
                &expected_tube,
                &RationalInterval::try_point(BigRational::zero()).unwrap(),
            ),
            Err(CarriedPlanarLcEntryError::SourceProvenanceMismatch {
                field: "source_chart"
            })
        ));

        let expected_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let mut forged_tube_wire = bridge.target_tube.clone();
        forged_tube_wire.source = "same-id-forged-source".into();
        let forged_tube = ordinary_tube_input_from_wire(&forged_tube_wire);
        assert_eq!(forged_tube.tube_id(), bridge.target_tube.tube_id);
        assert!(matches!(
            replay_carried_planar_lc_entry_exact_rational_v04(
                &admission,
                1,
                &expected_chart,
                &forged_tube,
                &RationalInterval::try_point(BigRational::zero()).unwrap(),
            ),
            Err(CarriedPlanarLcEntryError::SourceProvenanceMismatch {
                field: "source_tube"
            })
        ));
    }

    #[test]
    fn pair_mass_and_endpoint_mismatches_keep_typed_semantic_attribution() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let SegmentWire::PlanarLcPassage(passage) = &admission.wire().segments[1] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);

        let mut pair = passage.lc_chart.clone();
        pair.pair.swap(0, 1);
        assert!(matches!(
            crate::planar_lc_semantic::planar_lc_entry_input_from_wire(
                &passage.entry_transition,
                &source_chart,
                &source_tube,
                &pair,
                &passage.lc_tube,
            ),
            Err(PlanarLcSemanticError::PairNotCanonical { .. })
        ));

        let mut mass = passage.lc_chart.clone();
        mass.masses[0] =
            checked_real_binary64_from_json("2.0", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        assert!(matches!(
            crate::planar_lc_semantic::planar_lc_entry_input_from_wire(
                &passage.entry_transition,
                &source_chart,
                &source_tube,
                &mass,
                &passage.lc_tube,
            ),
            Err(PlanarLcSemanticError::MassBinding { index: 0 })
        ));

        let mut endpoint = passage.entry_transition.clone();
        endpoint.source_right_parameter = endpoint.target_left_parameter.clone();
        assert_eq!(
            crate::planar_lc_semantic::planar_lc_entry_input_from_wire(
                &endpoint,
                &source_chart,
                &source_tube,
                &passage.lc_chart,
                &passage.lc_tube,
            ),
            Err(PlanarLcSemanticError::EndpointBinding {
                field: "source_right_parameter"
            })
        );
    }

    #[test]
    fn admitted_zero_target_allowance_retains_rows_thirteen_through_twenty() {
        let text = String::from_utf8(SUCCESS_RAW.to_vec()).unwrap();
        let mutated = text.replacen(
            "\"initial_error_bound\":1e-07",
            "\"initial_error_bound\":0.0",
            1,
        );
        assert_ne!(mutated, text);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let replay = replay_admission(
            &admission,
            &RationalInterval::try_point(BigRational::new(
                BigInt::from(1),
                BigInt::from(1_u64 << 40),
            ))
            .unwrap(),
        );
        assert!(replay.obligations()[..20]
            .iter()
            .all(CarriedPlanarLcEntryObligation::satisfied));
        assert!(!replay.obligations()[20].satisfied());
        assert_eq!(replay.assignment_containments(), Some(&[false, false]));
        assert!(replay.source_endpoint_state_box().is_some());
        assert!(replay.lift_replay().is_some());
        assert!(replay.target_anchor().is_some());
    }

    #[test]
    fn admitted_nonpoint_parent_clock_propagates_exactly_through_replay() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
        let center = BigRational::new(BigInt::from(1), BigInt::from(1_u64 << 40));
        let epsilon = BigRational::new(BigInt::from(1), BigInt::from(1_u128 << 100));
        let parent = RationalInterval::new(&center - &epsilon, &center + &epsilon).unwrap();
        let replay = replay_admission(&admission, &parent);
        let source_right = BigRational::new(BigInt::from(1), BigInt::from(1_u64 << 20));
        assert_eq!(
            replay.entry_time_interval(),
            Some(
                &RationalInterval::new(
                    &center + &source_right - &epsilon,
                    &center + &source_right + &epsilon,
                )
                .unwrap()
            )
        );
        assert!(replay.obligations()[16].satisfied());
    }

    #[test]
    fn namespace_segment_cap_is_typed() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS_RAW).unwrap();
        let mut oversized = admission.wire().clone();
        oversized.segments = vec![oversized.segments[0].clone(); 257];
        assert_eq!(
            defining_namespace_unique_wire(&oversized),
            Err(CarriedPlanarLcEntryError::NamespaceSegmentLimit {
                actual: 257,
                limit: HARD_MAX_RAW_V1_NAMESPACE_SEGMENTS,
            })
        );
    }
}
