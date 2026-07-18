//! Profile-local 13-row semantic outcome for one already admitted raw-v1
//! payload when the bounded replay returns Boolean evidence.
//!
//! This module begins strictly after [`CanonicalRawV1Admission`]: it reports
//! no parser outcome and provides no CLI or environment boundary.  Rows 1,
//! 3, and 4 are true because the opaque admission already owns canonical
//! bytes and their typed finite-binary64 decode; this layer freshly checks the
//! namespace, hashes those exact bytes, checks the requested width sign, and
//! owns the exact-rational mixed replay for rows 6--13.
//!
//! The arithmetic profile intentionally differs from the frozen binary64
//! verifier.  A terminal label produced here is local to this admitted-payload
//! profile: it is not cross-verifier agreement and does not establish v0.4
//! verifier independence.

use core::fmt;

use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    raw_schema::SegmentWire, raw_v1_defining_namespace_unique,
    replay_raw_mixed_planar_chain_exact_rational_v04, CanonicalRawV1Admission,
    CarriedPlanarLcEntryError, MixedChainRetainedRegion, OrdinaryChainFinalEnclosure,
    RationalInterval, RawMixedPlanarChainReplay, RawMixedPlanarChainReplayError,
    EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
    EXACT_RATIONAL_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID, PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES,
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS,
};

pub const EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID: &str =
    "exact_rational_admitted_raw_v1_outcome_v04";
pub const RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID: &str = "raw-v1-rust-semantic-outcome-v1";

pub const RAW_V1_OUTCOME_OBLIGATION_IDS: [&str; 13] = [
    "raw_planar_chain_outer_schema_exact",
    "raw_planar_chain_global_identifier_namespace_unique",
    "raw_planar_chain_canonical_evidence_serializable",
    "raw_planar_chain_requested_target_finite",
    "raw_planar_chain_requested_width_admissible",
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[0],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[1],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[2],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[3],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[4],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[5],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[6],
    RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS[7],
];

const CARTESIAN_COMPONENT_NAMES: [&str; 12] = [
    "q1x", "q1y", "q2x", "q2y", "q3x", "q3y", "v1x", "v1y", "v2x", "v2y", "v3x", "v3y",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawV1OutcomeObligation {
    id: &'static str,
    satisfied: bool,
}

impl RawV1OutcomeObligation {
    pub const fn id(&self) -> &'static str {
        self.id
    }

    pub const fn satisfied(&self) -> bool {
        self.satisfied
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1OutcomeStatus {
    CertifiedToT,
    Unresolved,
}

impl RawV1OutcomeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CertifiedToT => "CERTIFIED_TO_T",
            Self::Unresolved => "UNRESOLVED",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1OutcomeSegmentKind {
    OrdinaryBridge,
    PlanarLcPassage,
}

impl RawV1OutcomeSegmentKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryBridge => "ordinary_bridge",
            Self::PlanarLcPassage => "planar_lc_passage",
        }
    }

    pub const fn profile_id(self) -> &'static str {
        match self {
            Self::OrdinaryBridge => EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
            Self::PlanarLcPassage => EXACT_RATIONAL_CARRIED_PLANAR_LC_EXIT_V04_PROFILE_ID,
        }
    }
}

/// Source-word metadata only.  Certification remains exclusively in the
/// owned mixed replay; this record cannot be supplied by a caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawV1OutcomeSegmentProfile {
    segment_index: usize,
    kind: RawV1OutcomeSegmentKind,
    pair: Option<[usize; 2]>,
}

impl RawV1OutcomeSegmentProfile {
    pub const fn segment_index(&self) -> usize {
        self.segment_index
    }

    pub const fn kind(&self) -> RawV1OutcomeSegmentKind {
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
pub enum RawV1OutcomeError {
    Namespace(CarriedPlanarLcEntryError),
    MixedReplay(RawMixedPlanarChainReplayError),
}

impl fmt::Display for RawV1OutcomeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Namespace(source) => {
                write!(
                    formatter,
                    "raw-v1 outcome namespace replay failed: {source}"
                )
            }
            Self::MixedReplay(source) => {
                write!(formatter, "raw-v1 outcome mixed replay failed: {source}")
            }
        }
    }
}

impl std::error::Error for RawV1OutcomeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Namespace(source) => Some(source),
            Self::MixedReplay(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub struct RawV1OutcomeSerializationError {
    source: serde_json::Error,
}

impl fmt::Display for RawV1OutcomeSerializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "raw-v1 outcome JSON serialization failed: {}",
            self.source
        )
    }
}

impl std::error::Error for RawV1OutcomeSerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// A 13-row outcome under the admitted-payload exact-rational profile after
/// its bounded replay has returned Boolean evidence. All fields are private so
/// callers cannot replace admission-derived data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawV1Outcome {
    obligations: [RawV1OutcomeObligation; 13],
    status: RawV1OutcomeStatus,
    certificate_id: String,
    evidence_sha256: String,
    requested_target_time: BigRational,
    requested_maximum_component_width: BigRational,
    first_failed_obligation: Option<String>,
    segment_profiles: Vec<RawV1OutcomeSegmentProfile>,
    mixed_replay: RawMixedPlanarChainReplay,
}

impl RawV1Outcome {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID
    }

    pub const fn result_schema_id(&self) -> &'static str {
        RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID
    }

    pub fn obligations(&self) -> &[RawV1OutcomeObligation; 13] {
        &self.obligations
    }

    pub const fn status(&self) -> RawV1OutcomeStatus {
        self.status
    }

    pub const fn certified_to_t(&self) -> bool {
        matches!(self.status, RawV1OutcomeStatus::CertifiedToT)
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

    pub fn segment_profiles(&self) -> &[RawV1OutcomeSegmentProfile] {
        &self.segment_profiles
    }

    pub fn mixed_replay(&self) -> &RawMixedPlanarChainReplay {
        &self.mixed_replay
    }

    /// Deterministic compact UTF-8 JSON under the profile-local result schema.
    /// `serde_json::to_vec` appends no terminal newline.
    pub fn to_portable_json_bytes(&self) -> Result<Vec<u8>, RawV1OutcomeSerializationError> {
        serde_json::to_vec(&self.portable_projection())
            .map_err(|source| RawV1OutcomeSerializationError { source })
    }

    pub(crate) fn portable_projection(&self) -> PortableOutcomeProjection<'_> {
        let replay = &self.mixed_replay;
        PortableOutcomeProjection {
            schema: RAW_V1_RUST_SEMANTIC_OUTCOME_V1_SCHEMA_ID,
            profile: EXACT_RATIONAL_ADMITTED_RAW_V1_OUTCOME_V04_PROFILE_ID,
            status: self.status.as_str(),
            certificate_id: &self.certificate_id,
            evidence_sha256: &self.evidence_sha256,
            request: PortableRequestProjection {
                target_physical_time: rational_projection(&self.requested_target_time),
                maximum_component_width: rational_projection(
                    &self.requested_maximum_component_width,
                ),
            },
            top_level_obligations: self
                .obligations
                .iter()
                .map(|row| PortableObligationProjection {
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
                .map(|segment| PortableSegmentProjection {
                    segment_index: segment.segment_index,
                    kind: segment.kind.as_str(),
                    profile: segment.kind.profile_id(),
                    pair: segment.pair,
                })
                .collect(),
            clock_ledger: replay
                .clock_ledger()
                .iter()
                .map(|entry| PortableClockProjection {
                    vertex_index: entry.vertex_index(),
                    chart_id: entry.chart_id(),
                    clock_origin: interval_projection(entry.clock_origin()),
                })
                .collect(),
            current_chart: replay.current_chart().map(|chart| PortableChartProjection {
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

/// Replay a 13-row admitted-payload result when every bounded stage returns
/// Boolean evidence. Parser rejection is intentionally outside this function:
/// callers must first obtain the opaque admission token. A structured
/// diagnostic carried by an `Ok` mixed replay remains a sound profile-local
/// `UNRESOLVED` outcome; only namespace replay failure or an actual mixed
/// replay `Err` escapes as [`RawV1OutcomeError`].
pub fn replay_admitted_raw_v1_outcome_exact_rational_v04(
    admission: &CanonicalRawV1Admission,
) -> Result<RawV1Outcome, RawV1OutcomeError> {
    // Row 1 follows only from possession of the opaque canonical+typed token.
    let outer_schema_exact = true;
    let namespace_unique =
        raw_v1_defining_namespace_unique(admission).map_err(RawV1OutcomeError::Namespace)?;

    // Row 3 follows because admission owns validated canonical bytes and this
    // layer hashes exactly that immutable byte slice.
    let evidence_sha256 = sha256_lower_hex(admission.canonical_bytes());
    let canonical_evidence_serializable = true;

    // Row 4 follows from the admitted finite binary64 real wrapper.
    let requested_target_finite = true;
    let requested_width = admission
        .wire()
        .requested_maximum_component_width
        .binary64_rational();
    let requested_width_admissible = !requested_width.is_negative();

    let mixed_replay = replay_raw_mixed_planar_chain_exact_rational_v04(admission)
        .map_err(RawV1OutcomeError::MixedReplay)?;

    let outer_satisfaction = [
        outer_schema_exact,
        namespace_unique,
        canonical_evidence_serializable,
        requested_target_finite,
        requested_width_admissible,
    ];
    let satisfaction: [bool; 13] = std::array::from_fn(|index| {
        if index < outer_satisfaction.len() {
            outer_satisfaction[index]
        } else {
            mixed_replay.obligations()[index - outer_satisfaction.len()].satisfied()
        }
    });
    let obligations = std::array::from_fn(|index| RawV1OutcomeObligation {
        id: RAW_V1_OUTCOME_OBLIGATION_IDS[index],
        satisfied: satisfaction[index],
    });
    let status = if satisfaction.iter().all(|value| *value) {
        RawV1OutcomeStatus::CertifiedToT
    } else {
        RawV1OutcomeStatus::Unresolved
    };
    let first_failed_obligation = first_failed_obligation(&satisfaction, &mixed_replay);

    Ok(RawV1Outcome {
        obligations,
        status,
        certificate_id: admission.wire().certificate_id.clone(),
        evidence_sha256,
        requested_target_time: admission
            .wire()
            .requested_target_time
            .binary64_rational()
            .clone(),
        requested_maximum_component_width: requested_width.clone(),
        first_failed_obligation,
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

fn source_segment_profiles(admission: &CanonicalRawV1Admission) -> Vec<RawV1OutcomeSegmentProfile> {
    admission
        .wire()
        .segments
        .iter()
        .enumerate()
        .map(|(segment_index, segment)| match segment {
            SegmentWire::OrdinaryBridge(_) => RawV1OutcomeSegmentProfile {
                segment_index,
                kind: RawV1OutcomeSegmentKind::OrdinaryBridge,
                pair: None,
            },
            SegmentWire::PlanarLcPassage(segment) => RawV1OutcomeSegmentProfile {
                segment_index,
                kind: RawV1OutcomeSegmentKind::PlanarLcPassage,
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
    replay: &RawMixedPlanarChainReplay,
) -> Option<String> {
    for index in 0..5 {
        if !satisfaction[index] {
            return Some(RAW_V1_OUTCOME_OBLIGATION_IDS[index].to_owned());
        }
    }
    if !satisfaction[5] {
        return Some(RAW_V1_OUTCOME_OBLIGATION_IDS[5].to_owned());
    }
    if !satisfaction[6] {
        if let Some(chart) = replay.initial_chart_replay() {
            if let Some(row) = chart.obligations().iter().find(|row| !row.satisfied()) {
                return Some(format!("root:{}", row.id()));
            }
        }
        if let Some(root) = replay.root_replay() {
            if let Some(row) = root.obligations().iter().find(|row| !row.satisfied()) {
                return Some(format!("root:{}", row.id()));
            }
        }
        return Some(RAW_V1_OUTCOME_OBLIGATION_IDS[6].to_owned());
    }
    if !satisfaction[7] {
        if let (Some(segment_index), Some(local)) = (
            replay.failed_segment_index(),
            replay.failed_local_obligation_ids().first(),
        ) {
            return Some(format!("segment[{segment_index}]:{local}"));
        }
        return Some(RAW_V1_OUTCOME_OBLIGATION_IDS[7].to_owned());
    }
    satisfaction
        .iter()
        .enumerate()
        .skip(8)
        .find_map(|(index, satisfied)| {
            (!*satisfied).then(|| RAW_V1_OUTCOME_OBLIGATION_IDS[index].to_owned())
        })
}

#[derive(Serialize)]
pub(crate) struct PortableOutcomeProjection<'a> {
    schema: &'static str,
    profile: &'static str,
    status: &'static str,
    certificate_id: &'a str,
    evidence_sha256: &'a str,
    request: PortableRequestProjection,
    top_level_obligations: Vec<PortableObligationProjection>,
    first_failed_obligation: Option<&'a str>,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    failed_segment_missing_obligations: &'a [&'static str],
    segment_profiles: Vec<PortableSegmentProjection>,
    clock_ledger: Vec<PortableClockProjection<'a>>,
    current_chart: Option<PortableChartProjection<'a>>,
    current_clock_origin: Option<PortableIntervalProjection>,
    covered_physical_time_interval: Option<PortableIntervalProjection>,
    target_parameter_preimage_interval: Option<PortableIntervalProjection>,
    maximum_final_component_width: Option<PortableRationalProjection>,
    final_enclosure: Option<PortableEnclosureProjection>,
    retained_region: Option<PortableRetainedRegionProjection<'a>>,
}

#[derive(Serialize)]
struct PortableRequestProjection {
    target_physical_time: PortableRationalProjection,
    maximum_component_width: PortableRationalProjection,
}

#[derive(Serialize)]
struct PortableObligationProjection {
    obligation: &'static str,
    certified: bool,
}

#[derive(Serialize)]
struct PortableSegmentProjection {
    segment_index: usize,
    kind: &'static str,
    profile: &'static str,
    pair: Option<[usize; 2]>,
}

#[derive(Serialize)]
struct PortableClockProjection<'a> {
    vertex_index: usize,
    chart_id: &'a str,
    clock_origin: PortableIntervalProjection,
}

#[derive(Serialize)]
struct PortableChartProjection<'a> {
    certificate_id: &'a str,
    chart_id: &'a str,
    parameter_interval: PortableIntervalProjection,
    physical_time_interval: PortableIntervalProjection,
}

#[derive(Serialize)]
struct PortableRationalProjection {
    numerator: String,
    denominator: String,
}

type PortableIntervalProjection = [PortableRationalProjection; 2];

#[derive(Serialize)]
struct PortableNamedIntervalProjection {
    component: &'static str,
    interval: PortableIntervalProjection,
}

#[derive(Serialize)]
struct PortableEnclosureProjection {
    enclosure_type: &'static str,
    coordinate_system: &'static str,
    physical_time_interval: PortableIntervalProjection,
    parameter_interval: PortableIntervalProjection,
    components: Vec<PortableNamedIntervalProjection>,
}

#[derive(Serialize)]
struct PortableRetainedRegionProjection<'a> {
    region_type: &'static str,
    coordinate_system: &'static str,
    certified_segment_count: usize,
    failed_segment_index: Option<usize>,
    chart_id: Option<&'a str>,
    pair: Option<[usize; 2]>,
    parameter_interval: PortableIntervalProjection,
    physical_time_interval: PortableIntervalProjection,
    components: Vec<PortableNamedIntervalProjection>,
}

fn rational_projection(value: &BigRational) -> PortableRationalProjection {
    PortableRationalProjection {
        numerator: value.numer().to_string(),
        denominator: value.denom().to_string(),
    }
}

fn interval_projection(value: &RationalInterval) -> PortableIntervalProjection {
    [
        rational_projection(value.lower()),
        rational_projection(value.upper()),
    ]
}

fn point_interval_projection(value: &BigRational) -> PortableIntervalProjection {
    [rational_projection(value), rational_projection(value)]
}

fn cartesian_components(
    position: &[RationalInterval],
    velocity: &[RationalInterval],
) -> Vec<PortableNamedIntervalProjection> {
    CARTESIAN_COMPONENT_NAMES
        .into_iter()
        .zip(position.iter().chain(velocity))
        .map(|(component, interval)| PortableNamedIntervalProjection {
            component,
            interval: interval_projection(interval),
        })
        .collect()
}

fn final_enclosure_projection(
    enclosure: &OrdinaryChainFinalEnclosure,
) -> PortableEnclosureProjection {
    PortableEnclosureProjection {
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
    replay: &RawMixedPlanarChainReplay,
) -> Option<PortableRetainedRegionProjection<'_>> {
    match replay.retained_region()? {
        MixedChainRetainedRegion::OrdinaryFixedTime(enclosure) => {
            Some(PortableRetainedRegionProjection {
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
            Some(PortableRetainedRegionProjection {
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
                    .map(|(index, interval)| PortableNamedIntervalProjection {
                        component: PLANAR_LC_LIFTED_STATE_COMPONENT_NAMES[index],
                        interval: interval_projection(interval),
                    })
                    .collect(),
            })
        }
        MixedChainRetainedRegion::CurrentOrdinaryRight(frontier) => {
            Some(PortableRetainedRegionProjection {
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
    use crate::CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS;

    #[test]
    fn top_level_ids_append_the_existing_mixed_rows_exactly() {
        assert_eq!(
            &RAW_V1_OUTCOME_OBLIGATION_IDS[5..],
            RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS
        );
        assert_eq!(RAW_V1_OUTCOME_OBLIGATION_IDS.len(), 13);
    }

    #[test]
    fn carried_exit_row_nineteen_is_the_expected_stable_identifier() {
        assert_eq!(
            CARRIED_PLANAR_LC_EXIT_OBLIGATION_IDS[18],
            "carried_lc_exit_target_initial_ball_contains_complete_projection"
        );
    }
}
