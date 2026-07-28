//! Proof-grade 13-row semantic outcome for one already admitted raw-v1
//! payload.
//!
//! This profile is intentionally separate from the compatibility admitted
//! outcome.  It consumes only [`CanonicalRawV1Admission`], owns the proof-grade
//! mixed replay, and treats claimed-tail chart replays as retained diagnostics
//! rather than decisive rows.  It is neither a parser/CLI envelope nor an
//! independence claim.

use core::fmt;

use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    raw_schema::SegmentWire, raw_v1_defining_namespace_unique,
    replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04, CanonicalRawV1Admission,
    CarriedPlanarLcEntryError, MixedChainRetainedRegion, OrdinaryChainFinalEnclosure,
    ProofGradeRawMixedPlanarChainReplay, ProofGradeRawMixedPlanarChainReplayError,
    RationalInterval, EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
    EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID,
    PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES, PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS,
};

pub const EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID: &str =
    "exact_rational_proof_grade_admitted_raw_v1_outcome_v04";
pub const RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID: &str =
    "raw-v1-rust-proof-grade-semantic-outcome-v1";

pub const PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS: [&str; 13] = [
    "proof_grade_raw_planar_chain_outer_schema_exact",
    "proof_grade_raw_planar_chain_global_identifier_namespace_unique",
    "proof_grade_raw_planar_chain_canonical_evidence_serializable",
    "proof_grade_raw_planar_chain_requested_target_finite",
    "proof_grade_raw_planar_chain_requested_width_admissible",
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[0],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[1],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[2],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[3],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[4],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[5],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[6],
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[7],
];

const CARTESIAN_COMPONENT_NAMES: [&str; 12] = [
    "q1x", "q1y", "q2x", "q2y", "q3x", "q3y", "v1x", "v1y", "v2x", "v2y", "v3x", "v3y",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeRawV1OutcomeObligation {
    id: &'static str,
    satisfied: bool,
}

impl ProofGradeRawV1OutcomeObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1OutcomeStatus {
    CertifiedToT,
    Unresolved,
}

impl ProofGradeRawV1OutcomeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CertifiedToT => "CERTIFIED_TO_T",
            Self::Unresolved => "UNRESOLVED",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1OutcomeSegmentKind {
    OrdinaryBridge,
    PlanarLcPassage,
}

impl ProofGradeRawV1OutcomeSegmentKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryBridge => "ordinary_bridge",
            Self::PlanarLcPassage => "planar_lc_passage",
        }
    }

    pub const fn profile_id(self) -> &'static str {
        match self {
            Self::OrdinaryBridge => EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
            Self::PlanarLcPassage => {
                EXACT_RATIONAL_PROOF_GRADE_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID
            }
        }
    }
}

/// Source-word metadata.  The authoritative certification evidence remains
/// private in the proof-grade mixed replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeRawV1OutcomeSegmentProfile {
    segment_index: usize,
    kind: ProofGradeRawV1OutcomeSegmentKind,
    pair: Option<[usize; 2]>,
}

impl ProofGradeRawV1OutcomeSegmentProfile {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    pub const fn kind(&self) -> ProofGradeRawV1OutcomeSegmentKind {
        self.kind
    }

    pub const fn profile_id(&self) -> &'static str {
        self.kind.profile_id()
    }

    pub const fn pair(&self) -> Option<[usize; 2]> {
        self.pair
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1OutcomeError {
    Namespace(CarriedPlanarLcEntryError),
    MixedReplay(ProofGradeRawMixedPlanarChainReplayError),
}

impl fmt::Display for ProofGradeRawV1OutcomeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Namespace(source) => {
                write!(
                    formatter,
                    "proof-grade raw-v1 outcome namespace replay failed: {source}"
                )
            }
            Self::MixedReplay(source) => {
                write!(
                    formatter,
                    "proof-grade raw-v1 outcome mixed replay failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for ProofGradeRawV1OutcomeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Namespace(source) => Some(source),
            Self::MixedReplay(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub struct ProofGradeRawV1OutcomeSerializationError {
    source: serde_json::Error,
}

impl fmt::Display for ProofGradeRawV1OutcomeSerializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "proof-grade raw-v1 outcome JSON serialization failed: {}",
            self.source
        )
    }
}

impl std::error::Error for ProofGradeRawV1OutcomeSerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// A profile-local 13-row semantic outcome from one immutable admission.
/// Private fields ensure every portable field remains admission/replay derived.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofGradeRawV1Outcome {
    obligations: [ProofGradeRawV1OutcomeObligation; 13],
    status: ProofGradeRawV1OutcomeStatus,
    certificate_id: String,
    evidence_sha256: String,
    requested_target_time: BigRational,
    requested_maximum_component_width: BigRational,
    first_failed_obligation: Option<String>,
    segment_profiles: Vec<ProofGradeRawV1OutcomeSegmentProfile>,
    mixed_replay: ProofGradeRawMixedPlanarChainReplay,
}

impl ProofGradeRawV1Outcome {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    }

    pub const fn result_schema_id(&self) -> &'static str {
        RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID
    }

    pub fn obligations(&self) -> &[ProofGradeRawV1OutcomeObligation; 13] {
        &self.obligations
    }

    pub const fn status(&self) -> ProofGradeRawV1OutcomeStatus {
        self.status
    }

    pub const fn certified_to_t(&self) -> bool {
        matches!(self.status, ProofGradeRawV1OutcomeStatus::CertifiedToT)
    }

    pub fn certificate_id(&self) -> &str {
        &self.certificate_id
    }

    pub fn evidence_sha256(&self) -> &str {
        &self.evidence_sha256
    }

    pub fn requested_target_time(&self) -> &BigRational {
        &self.requested_target_time
    }

    pub fn requested_maximum_component_width(&self) -> &BigRational {
        &self.requested_maximum_component_width
    }

    pub fn first_failed_obligation(&self) -> Option<&str> {
        self.first_failed_obligation.as_deref()
    }

    pub const fn certified_segment_count(&self) -> usize {
        self.mixed_replay.certified_segment_count()
    }

    pub const fn failed_segment_index(&self) -> Option<usize> {
        self.mixed_replay.failed_segment_index()
    }

    pub fn failed_local_obligation_ids(&self) -> &[&'static str] {
        self.mixed_replay.failed_local_obligation_ids()
    }

    pub fn segment_profiles(&self) -> &[ProofGradeRawV1OutcomeSegmentProfile] {
        &self.segment_profiles
    }

    pub fn mixed_replay(&self) -> &ProofGradeRawMixedPlanarChainReplay {
        &self.mixed_replay
    }

    /// Deterministic compact UTF-8 JSON. No diagnostic, typed-error, tail, or
    /// resource field is projected; callers needing diagnostics inspect the
    /// owned mixed replay explicitly.
    pub fn to_portable_json_bytes(
        &self,
    ) -> Result<Vec<u8>, ProofGradeRawV1OutcomeSerializationError> {
        serde_json::to_vec(&self.portable_projection())
            .map_err(|source| ProofGradeRawV1OutcomeSerializationError { source })
    }

    pub(crate) fn portable_projection(&self) -> ProofGradePortableOutcomeProjection<'_> {
        let replay = &self.mixed_replay;
        ProofGradePortableOutcomeProjection {
            schema: RAW_V1_RUST_PROOF_GRADE_SEMANTIC_OUTCOME_V1_SCHEMA_ID,
            profile: EXACT_RATIONAL_PROOF_GRADE_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID,
            status: self.status.as_str(),
            certificate_id: &self.certificate_id,
            evidence_sha256: &self.evidence_sha256,
            request: ProofGradePortableRequestProjection {
                target_physical_time: rational_projection(&self.requested_target_time),
                maximum_component_width: rational_projection(
                    &self.requested_maximum_component_width,
                ),
            },
            top_level_obligations: self
                .obligations
                .iter()
                .map(|row| ProofGradePortableObligationProjection {
                    obligation: row.id,
                    certified: row.satisfied,
                })
                .collect(),
            first_failed_obligation: self.first_failed_obligation.as_deref(),
            certified_segment_count: replay.certified_segment_count(),
            failed_segment_index: replay.failed_segment_index(),
            failed_segment_missing_obligations: replay.failed_local_obligation_ids(),
            segment_profiles: self
                .segment_profiles
                .iter()
                .map(|segment| ProofGradePortableSegmentProjection {
                    segment_index: segment.segment_index,
                    kind: segment.kind.as_str(),
                    profile: segment.kind.profile_id(),
                    pair: segment.pair,
                })
                .collect(),
            clock_ledger: replay
                .clock_ledger()
                .iter()
                .map(|entry| ProofGradePortableClockProjection {
                    vertex_index: entry.vertex_index(),
                    chart_id: entry.chart_id(),
                    clock_origin: interval_projection(entry.clock_origin()),
                })
                .collect(),
            current_chart: replay
                .current_chart()
                .map(|chart| ProofGradePortableChartProjection {
                    certificate_id: chart.certificate_id(),
                    chart_id: chart.chart_id(),
                    parameter_interval: interval_projection(chart.parameter_interval()),
                    physical_time_interval: interval_projection(chart.physical_time_interval()),
                }),
            current_clock_origin: replay.current_clock_origin().map(interval_projection),
            covered_physical_time_interval: replay
                .covered_physical_interval()
                .map(interval_projection),
            target_parameter_preimage_interval: replay.target_preimage().map(interval_projection),
            maximum_final_component_width: replay
                .maximum_component_width()
                .map(rational_projection),
            final_enclosure: replay.final_enclosure().map(final_enclosure_projection),
            retained_region: retained_region_projection(replay),
        }
    }
}

/// Produce the proof-grade semantic outcome after strict opaque admission.
/// Parser rejection remains intentionally outside this API.
pub fn replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
) -> Result<ProofGradeRawV1Outcome, ProofGradeRawV1OutcomeError> {
    let namespace_unique = raw_v1_defining_namespace_unique(admission)
        .map_err(ProofGradeRawV1OutcomeError::Namespace)?;
    let requested_width = admission
        .wire()
        .requested_maximum_component_width
        .binary64_rational();
    let mixed_replay = replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04(admission)
        .map_err(|source| {
            let _failure_disposition = source.disposition();
            ProofGradeRawV1OutcomeError::MixedReplay(source)
        })?;

    let outer_satisfaction = [
        true,
        namespace_unique,
        true,
        true,
        !requested_width.is_negative(),
    ];
    let satisfaction: [bool; 13] = std::array::from_fn(|index| {
        if index < outer_satisfaction.len() {
            outer_satisfaction[index]
        } else {
            mixed_replay.obligations()[index - outer_satisfaction.len()].satisfied()
        }
    });
    let obligations = std::array::from_fn(|index| ProofGradeRawV1OutcomeObligation {
        id: PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[index],
        satisfied: satisfaction[index],
    });
    let status = if satisfaction.iter().all(|value| *value) {
        ProofGradeRawV1OutcomeStatus::CertifiedToT
    } else {
        ProofGradeRawV1OutcomeStatus::Unresolved
    };

    Ok(ProofGradeRawV1Outcome {
        obligations,
        status,
        certificate_id: admission.wire().certificate_id.clone(),
        evidence_sha256: sha256_lower_hex(admission.canonical_bytes()),
        requested_target_time: admission
            .wire()
            .requested_target_time
            .binary64_rational()
            .clone(),
        requested_maximum_component_width: requested_width.clone(),
        first_failed_obligation: first_failed_obligation(&satisfaction, &mixed_replay),
        segment_profiles: source_segment_profiles(admission),
        mixed_replay,
    })
}

fn sha256_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn source_segment_profiles(
    admission: &CanonicalRawV1Admission,
) -> Vec<ProofGradeRawV1OutcomeSegmentProfile> {
    admission
        .wire()
        .segments
        .iter()
        .enumerate()
        .map(|(segment_index, segment)| match segment {
            SegmentWire::OrdinaryBridge(_) => ProofGradeRawV1OutcomeSegmentProfile {
                segment_index,
                kind: ProofGradeRawV1OutcomeSegmentKind::OrdinaryBridge,
                pair: None,
            },
            SegmentWire::PlanarLcPassage(segment) => ProofGradeRawV1OutcomeSegmentProfile {
                segment_index,
                kind: ProofGradeRawV1OutcomeSegmentKind::PlanarLcPassage,
                pair: segment.lc_chart.pair[0]
                    .value()
                    .to_usize()
                    .zip(segment.lc_chart.pair[1].value().to_usize())
                    .map(|(left, right)| [left, right]),
            },
        })
        .collect()
}

fn first_failed_obligation(
    satisfaction: &[bool; 13],
    replay: &ProofGradeRawMixedPlanarChainReplay,
) -> Option<String> {
    for index in 0..5 {
        if !satisfaction[index] {
            return Some(PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[index].to_owned());
        }
    }
    if !satisfaction[5] {
        return Some(PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[5].to_owned());
    }
    if !satisfaction[6] {
        if let Some(root) = replay.root_replay() {
            if let Some(row) = root.obligations().iter().find(|row| !row.satisfied()) {
                return Some(format!("root:{}", row.id()));
            }
        }
        return Some(PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[6].to_owned());
    }
    if !satisfaction[7] {
        if let (Some(segment_index), Some(local)) = (
            replay.failed_segment_index(),
            replay.failed_local_obligation_ids().first(),
        ) {
            return Some(format!("segment[{segment_index}]:{local}"));
        }
        return Some(PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[7].to_owned());
    }
    satisfaction
        .iter()
        .enumerate()
        .skip(8)
        .find_map(|(index, satisfied)| {
            (!*satisfied).then(|| PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[index].to_owned())
        })
}

#[derive(Serialize)]
pub(crate) struct ProofGradePortableOutcomeProjection<'a> {
    schema: &'static str,
    profile: &'static str,
    status: &'static str,
    certificate_id: &'a str,
    evidence_sha256: &'a str,
    request: ProofGradePortableRequestProjection,
    top_level_obligations: Vec<ProofGradePortableObligationProjection>,
    first_failed_obligation: Option<&'a str>,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    failed_segment_missing_obligations: &'a [&'static str],
    segment_profiles: Vec<ProofGradePortableSegmentProjection>,
    clock_ledger: Vec<ProofGradePortableClockProjection<'a>>,
    current_chart: Option<ProofGradePortableChartProjection<'a>>,
    current_clock_origin: Option<ProofGradePortableIntervalProjection>,
    covered_physical_time_interval: Option<ProofGradePortableIntervalProjection>,
    target_parameter_preimage_interval: Option<ProofGradePortableIntervalProjection>,
    maximum_final_component_width: Option<ProofGradePortableRationalProjection>,
    final_enclosure: Option<ProofGradePortableEnclosureProjection>,
    retained_region: Option<ProofGradePortableRetainedRegionProjection<'a>>,
}

#[derive(Serialize)]
struct ProofGradePortableRequestProjection {
    target_physical_time: ProofGradePortableRationalProjection,
    maximum_component_width: ProofGradePortableRationalProjection,
}

#[derive(Serialize)]
struct ProofGradePortableObligationProjection {
    obligation: &'static str,
    certified: bool,
}

#[derive(Serialize)]
struct ProofGradePortableSegmentProjection {
    segment_index: usize,
    kind: &'static str,
    profile: &'static str,
    pair: Option<[usize; 2]>,
}

#[derive(Serialize)]
struct ProofGradePortableClockProjection<'a> {
    vertex_index: usize,
    chart_id: &'a str,
    clock_origin: ProofGradePortableIntervalProjection,
}

#[derive(Serialize)]
struct ProofGradePortableChartProjection<'a> {
    certificate_id: &'a str,
    chart_id: &'a str,
    parameter_interval: ProofGradePortableIntervalProjection,
    physical_time_interval: ProofGradePortableIntervalProjection,
}

#[derive(Serialize)]
struct ProofGradePortableRationalProjection {
    numerator: String,
    denominator: String,
}

type ProofGradePortableIntervalProjection = [ProofGradePortableRationalProjection; 2];

#[derive(Serialize)]
struct ProofGradePortableNamedIntervalProjection {
    component: &'static str,
    interval: ProofGradePortableIntervalProjection,
}

#[derive(Serialize)]
struct ProofGradePortableEnclosureProjection {
    enclosure_type: &'static str,
    coordinate_system: &'static str,
    physical_time_interval: ProofGradePortableIntervalProjection,
    parameter_interval: ProofGradePortableIntervalProjection,
    components: Vec<ProofGradePortableNamedIntervalProjection>,
}

#[derive(Serialize)]
struct ProofGradePortableRetainedRegionProjection<'a> {
    region_type: &'static str,
    coordinate_system: &'static str,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    chart_id: Option<&'a str>,
    pair: Option<[usize; 2]>,
    parameter_interval: ProofGradePortableIntervalProjection,
    physical_time_interval: ProofGradePortableIntervalProjection,
    components: Vec<ProofGradePortableNamedIntervalProjection>,
}

fn rational_projection(value: &BigRational) -> ProofGradePortableRationalProjection {
    ProofGradePortableRationalProjection {
        numerator: value.numer().to_string(),
        denominator: value.denom().to_string(),
    }
}

fn interval_projection(value: &RationalInterval) -> ProofGradePortableIntervalProjection {
    [
        rational_projection(value.lower()),
        rational_projection(value.upper()),
    ]
}

fn point_interval_projection(value: &BigRational) -> ProofGradePortableIntervalProjection {
    [rational_projection(value), rational_projection(value)]
}

fn cartesian_components(
    position: &[RationalInterval],
    velocity: &[RationalInterval],
) -> Vec<ProofGradePortableNamedIntervalProjection> {
    CARTESIAN_COMPONENT_NAMES
        .into_iter()
        .zip(position.iter().chain(velocity))
        .map(
            |(component, interval)| ProofGradePortableNamedIntervalProjection {
                component,
                interval: interval_projection(interval),
            },
        )
        .collect()
}

fn final_enclosure_projection(
    enclosure: &OrdinaryChainFinalEnclosure,
) -> ProofGradePortableEnclosureProjection {
    ProofGradePortableEnclosureProjection {
        enclosure_type: "validated_planar_chain_fixed_physical_time",
        coordinate_system: "planar_cartesian_12",
        physical_time_interval: point_interval_projection(enclosure.physical_time()),
        parameter_interval: interval_projection(enclosure.parameter_preimage()),
        components: cartesian_components(
            enclosure.position_intervals(),
            enclosure.velocity_intervals(),
        ),
    }
}

fn retained_region_projection(
    replay: &ProofGradeRawMixedPlanarChainReplay,
) -> Option<ProofGradePortableRetainedRegionProjection<'_>> {
    match replay.retained_region()? {
        MixedChainRetainedRegion::OrdinaryFixedTime(enclosure) => {
            Some(ProofGradePortableRetainedRegionProjection {
                region_type: "certified_ordinary_fixed_time_enclosure",
                coordinate_system: "planar_cartesian_12",
                certified_segment_count: replay.certified_segment_count(),
                failed_segment_index: replay.failed_segment_index(),
                chart_id: replay.current_chart().map(|chart| chart.chart_id()),
                pair: None,
                parameter_interval: interval_projection(enclosure.parameter_preimage()),
                physical_time_interval: point_interval_projection(enclosure.physical_time()),
                components: cartesian_components(
                    enclosure.position_intervals(),
                    enclosure.velocity_intervals(),
                ),
            })
        }
        MixedChainRetainedRegion::LiftedPlanarLcRight(frontier) => {
            Some(ProofGradePortableRetainedRegionProjection {
                region_type: "certified_lifted_lc_right_frontier",
                coordinate_system: "planar_lc_lifted_14",
                certified_segment_count: replay.certified_segment_count(),
                failed_segment_index: Some(frontier.failed_segment_index()),
                chart_id: Some(frontier.lc_chart_id()),
                pair: Some(frontier.pair()),
                parameter_interval: point_interval_projection(frontier.right_parameter()),
                physical_time_interval: interval_projection(frontier.physical_time_interval()),
                components: frontier
                    .lifted_state()
                    .components()
                    .iter()
                    .enumerate()
                    .map(
                        |(index, interval)| ProofGradePortableNamedIntervalProjection {
                            component: PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES[index],
                            interval: interval_projection(interval),
                        },
                    )
                    .collect(),
            })
        }
        MixedChainRetainedRegion::CurrentOrdinaryRight(frontier) => {
            Some(ProofGradePortableRetainedRegionProjection {
                region_type: "certified_current_ordinary_right_frontier",
                coordinate_system: "planar_cartesian_12",
                certified_segment_count: replay.certified_segment_count(),
                failed_segment_index: replay.failed_segment_index(),
                chart_id: Some(frontier.chart_id()),
                pair: None,
                parameter_interval: point_interval_projection(frontier.right_parameter()),
                physical_time_interval: interval_projection(frontier.physical_time_interval()),
                components: cartesian_components(
                    frontier.position_intervals(),
                    frontier.velocity_intervals(),
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_ids_append_the_exported_proof_mixed_rows_exactly() {
        assert_eq!(
            &PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS[5..],
            PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS
        );
        assert_eq!(PROOF_GRADE_RAW_V1_OUTCOME_OBLIGATION_IDS.len(), 13);
    }
}
