//! Total byte-level execution envelope for the proof-grade admitted outcome.
//!
//! This profile strictly admits raw-v1 bytes, evaluates the separate
//! proof-grade semantic outcome, and exposes only stable classification
//! stages.  It does not alter the compatibility execution envelope or claim
//! verifier independence.

use core::fmt;

use serde::Serialize;

use crate::{
    raw_v1_proof_grade_outcome::ProofGradePortableOutcomeProjection,
    replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04, CanonicalRawV1Admission,
    CanonicalRawV1AdmissionError, ProofGradeRawV1Outcome, ProofGradeRawV1OutcomeError,
};

pub const EXACT_RATIONAL_PROOF_GRADE_RAW_V1_EXECUTION_V04_PROFILE_ID: &str =
    "exact_rational_proof_grade_raw_v1_execution_v04";
pub const RAW_V1_RUST_PROOF_GRADE_EXECUTION_V1_SCHEMA_ID: &str =
    "raw-v1-rust-proof-grade-execution-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1ParseOutcome {
    Accept,
    Reject,
}

impl ProofGradeRawV1ParseOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "ACCEPT",
            Self::Reject => "REJECT",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1RejectionStage {
    CanonicalWire,
    Schema,
}

impl ProofGradeRawV1RejectionStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalWire => "CANONICAL_WIRE",
            Self::Schema => "SCHEMA",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1EvaluationOutcome {
    NotRun,
    Result,
    Error,
}

impl ProofGradeRawV1EvaluationOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "NOT_RUN",
            Self::Result => "RESULT",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofGradeRawV1EvaluationErrorStage {
    Namespace,
    MixedReplay,
}

impl ProofGradeRawV1EvaluationErrorStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Namespace => "NAMESPACE",
            Self::MixedReplay => "MIXED_REPLAY",
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
enum ProofGradeRawV1ExecutionInner {
    Rejected(CanonicalRawV1AdmissionError),
    SemanticResult(Box<ProofGradeRawV1Outcome>),
    EvaluationError(ProofGradeRawV1OutcomeError),
}

/// Owned total classification of one supplied byte slice. Source errors are
/// intentionally private to this object; portable JSON contains only stages.
#[derive(Clone, Eq, PartialEq)]
pub struct ProofGradeRawV1Execution {
    inner: ProofGradeRawV1ExecutionInner,
}

impl fmt::Debug for ProofGradeRawV1Execution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProofGradeRawV1Execution")
            .field("parse_outcome", &self.parse_outcome())
            .field("rejection_stage", &self.rejection_stage())
            .field("evaluation_outcome", &self.evaluation_outcome())
            .field("evaluation_error_stage", &self.evaluation_error_stage())
            .finish_non_exhaustive()
    }
}

impl ProofGradeRawV1Execution {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PROOF_GRADE_RAW_V1_EXECUTION_V04_PROFILE_ID
    }

    pub const fn result_schema_id(&self) -> &'static str {
        RAW_V1_RUST_PROOF_GRADE_EXECUTION_V1_SCHEMA_ID
    }

    pub const fn parse_outcome(&self) -> ProofGradeRawV1ParseOutcome {
        match self.inner {
            ProofGradeRawV1ExecutionInner::Rejected(_) => ProofGradeRawV1ParseOutcome::Reject,
            ProofGradeRawV1ExecutionInner::SemanticResult(_)
            | ProofGradeRawV1ExecutionInner::EvaluationError(_) => {
                ProofGradeRawV1ParseOutcome::Accept
            }
        }
    }

    pub const fn rejection_stage(&self) -> Option<ProofGradeRawV1RejectionStage> {
        match &self.inner {
            ProofGradeRawV1ExecutionInner::Rejected(
                CanonicalRawV1AdmissionError::CanonicalWire(_),
            ) => Some(ProofGradeRawV1RejectionStage::CanonicalWire),
            ProofGradeRawV1ExecutionInner::Rejected(CanonicalRawV1AdmissionError::Schema(_)) => {
                Some(ProofGradeRawV1RejectionStage::Schema)
            }
            ProofGradeRawV1ExecutionInner::SemanticResult(_)
            | ProofGradeRawV1ExecutionInner::EvaluationError(_) => None,
        }
    }

    pub const fn evaluation_outcome(&self) -> ProofGradeRawV1EvaluationOutcome {
        match self.inner {
            ProofGradeRawV1ExecutionInner::Rejected(_) => ProofGradeRawV1EvaluationOutcome::NotRun,
            ProofGradeRawV1ExecutionInner::SemanticResult(_) => {
                ProofGradeRawV1EvaluationOutcome::Result
            }
            ProofGradeRawV1ExecutionInner::EvaluationError(_) => {
                ProofGradeRawV1EvaluationOutcome::Error
            }
        }
    }

    pub const fn evaluation_error_stage(&self) -> Option<ProofGradeRawV1EvaluationErrorStage> {
        match &self.inner {
            ProofGradeRawV1ExecutionInner::EvaluationError(
                ProofGradeRawV1OutcomeError::Namespace(_),
            ) => Some(ProofGradeRawV1EvaluationErrorStage::Namespace),
            ProofGradeRawV1ExecutionInner::EvaluationError(
                ProofGradeRawV1OutcomeError::MixedReplay(_),
            ) => Some(ProofGradeRawV1EvaluationErrorStage::MixedReplay),
            ProofGradeRawV1ExecutionInner::Rejected(_)
            | ProofGradeRawV1ExecutionInner::SemanticResult(_) => None,
        }
    }

    pub fn semantic_result(&self) -> Option<&ProofGradeRawV1Outcome> {
        match &self.inner {
            ProofGradeRawV1ExecutionInner::SemanticResult(result) => Some(result.as_ref()),
            ProofGradeRawV1ExecutionInner::Rejected(_)
            | ProofGradeRawV1ExecutionInner::EvaluationError(_) => None,
        }
    }

    pub fn source_error(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.inner {
            ProofGradeRawV1ExecutionInner::Rejected(source) => Some(source),
            ProofGradeRawV1ExecutionInner::EvaluationError(source) => Some(source),
            ProofGradeRawV1ExecutionInner::SemanticResult(_) => None,
        }
    }

    /// Deterministic compact UTF-8 JSON with no terminal newline. The nested
    /// semantic record is serialized through the exact same projection as the
    /// proof-grade semantic API.
    pub fn to_portable_json_bytes(
        &self,
    ) -> Result<Vec<u8>, ProofGradeRawV1ExecutionSerializationError> {
        serde_json::to_vec(&self.portable_projection())
            .map_err(|source| ProofGradeRawV1ExecutionSerializationError { source })
    }

    fn portable_projection(&self) -> ProofGradeRawV1ExecutionProjection<'_> {
        ProofGradeRawV1ExecutionProjection {
            schema: RAW_V1_RUST_PROOF_GRADE_EXECUTION_V1_SCHEMA_ID,
            profile: EXACT_RATIONAL_PROOF_GRADE_RAW_V1_EXECUTION_V04_PROFILE_ID,
            parse_outcome: self.parse_outcome().as_str(),
            rejection_stage: self
                .rejection_stage()
                .map(ProofGradeRawV1RejectionStage::as_str),
            evaluation_outcome: self.evaluation_outcome().as_str(),
            evaluation_error_stage: self
                .evaluation_error_stage()
                .map(ProofGradeRawV1EvaluationErrorStage::as_str),
            semantic_result: self
                .semantic_result()
                .map(ProofGradeRawV1Outcome::portable_projection),
        }
    }
}

#[derive(Debug)]
pub struct ProofGradeRawV1ExecutionSerializationError {
    source: serde_json::Error,
}

impl fmt::Display for ProofGradeRawV1ExecutionSerializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "proof-grade raw-v1 execution JSON serialization failed: {}",
            self.source
        )
    }
}

impl std::error::Error for ProofGradeRawV1ExecutionSerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Serialize)]
struct ProofGradeRawV1ExecutionProjection<'a> {
    schema: &'static str,
    profile: &'static str,
    parse_outcome: &'static str,
    rejection_stage: Option<&'static str>,
    evaluation_outcome: &'static str,
    evaluation_error_stage: Option<&'static str>,
    semantic_result: Option<ProofGradePortableOutcomeProjection<'a>>,
}

/// Execute strict admission and then the proof-grade semantic outcome. Every
/// admission/evaluation failure is totalized into an owned classification.
pub fn execute_proof_grade_raw_v1_bytes_exact_rational_v04(
    bytes: &[u8],
) -> ProofGradeRawV1Execution {
    let admission = match CanonicalRawV1Admission::admit(bytes) {
        Ok(admission) => admission,
        Err(source) => {
            return ProofGradeRawV1Execution {
                inner: ProofGradeRawV1ExecutionInner::Rejected(source),
            };
        }
    };
    match replay_admitted_proof_grade_raw_v1_outcome_exact_rational_v04(&admission) {
        Ok(result) => ProofGradeRawV1Execution {
            inner: ProofGradeRawV1ExecutionInner::SemanticResult(Box::new(result)),
        },
        Err(source) => ProofGradeRawV1Execution {
            inner: ProofGradeRawV1ExecutionInner::EvaluationError(source),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CarriedPlanarLcEntryError, ProofGradeRawMixedPlanarChainReplayError,
        ProofGradeRawV1OutcomeError,
    };

    fn typed_error_execution(error: ProofGradeRawV1OutcomeError) -> ProofGradeRawV1Execution {
        ProofGradeRawV1Execution {
            inner: ProofGradeRawV1ExecutionInner::EvaluationError(error),
        }
    }

    #[test]
    fn typed_namespace_and_mixed_replay_errors_map_to_distinct_stable_stages() {
        let namespace = typed_error_execution(ProofGradeRawV1OutcomeError::Namespace(
            CarriedPlanarLcEntryError::InternalShapeInvariant {
                detail: "unit-only namespace stage",
            },
        ));
        let mixed = typed_error_execution(ProofGradeRawV1OutcomeError::MixedReplay(
            ProofGradeRawMixedPlanarChainReplayError::InternalInvariant {
                detail: "unit-only mixed stage",
            },
        ));

        for (execution, expected_stage) in [
            (namespace, ProofGradeRawV1EvaluationErrorStage::Namespace),
            (mixed, ProofGradeRawV1EvaluationErrorStage::MixedReplay),
        ] {
            assert_eq!(
                execution.parse_outcome(),
                ProofGradeRawV1ParseOutcome::Accept
            );
            assert_eq!(execution.rejection_stage(), None);
            assert_eq!(
                execution.evaluation_outcome(),
                ProofGradeRawV1EvaluationOutcome::Error
            );
            assert_eq!(execution.evaluation_error_stage(), Some(expected_stage));
            assert!(execution.semantic_result().is_none());
            assert!(execution.source_error().is_some());
            let json = execution.to_portable_json_bytes().unwrap();
            assert!(!json
                .windows(b"InternalInvariant".len())
                .any(|window| window == b"InternalInvariant"));
            assert!(!json
                .windows(b"unit-only".len())
                .any(|window| window == b"unit-only"));
        }
    }
}
