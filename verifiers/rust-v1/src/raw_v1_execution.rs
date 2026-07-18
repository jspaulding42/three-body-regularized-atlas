//! Total byte-level execution envelope for the exact-rational raw-v1 profile.
//!
//! This layer classifies canonical admission, admitted semantic outcome, and
//! evaluation failure without converting any failure into certification. Its
//! deterministic JSON is a profile artifact, not a neutral cross-verifier
//! agreement transcript. It claims neither frozen binary64 parity nor v0.4
//! verifier independence.

use core::fmt;

use serde::Serialize;

use crate::{
    raw_v1_outcome::PortableOutcomeProjection, replay_admitted_raw_v1_outcome_exact_rational_v04,
    CanonicalRawV1Admission, CanonicalRawV1AdmissionError, RawV1Outcome, RawV1OutcomeError,
};

pub const EXACT_RATIONAL_RAW_V1_EXECUTION_V04_PROFILE_ID: &str =
    "exact_rational_raw_v1_execution_v04";
pub const RAW_V1_RUST_EXECUTION_V1_SCHEMA_ID: &str = "raw-v1-rust-execution-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1ParseOutcome {
    Accept,
    Reject,
}

impl RawV1ParseOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "ACCEPT",
            Self::Reject => "REJECT",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1RejectionStage {
    CanonicalWire,
    Schema,
}

impl RawV1RejectionStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalWire => "CANONICAL_WIRE",
            Self::Schema => "SCHEMA",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1EvaluationOutcome {
    NotRun,
    Result,
    Error,
}

impl RawV1EvaluationOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "NOT_RUN",
            Self::Result => "RESULT",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawV1EvaluationErrorStage {
    Namespace,
    MixedReplay,
}

impl RawV1EvaluationErrorStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Namespace => "NAMESPACE",
            Self::MixedReplay => "MIXED_REPLAY",
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
enum RawV1ExecutionInner {
    Rejected(CanonicalRawV1AdmissionError),
    SemanticResult(Box<RawV1Outcome>),
    EvaluationError(RawV1OutcomeError),
}

/// Owned total classification of one supplied byte slice. Typed source
/// failures remain private; callers receive stable stage enums and may follow
/// the source chain through [`RawV1Execution::source_error`].
#[derive(Clone, Eq, PartialEq)]
pub struct RawV1Execution {
    inner: RawV1ExecutionInner,
}

impl fmt::Debug for RawV1Execution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawV1Execution")
            .field("parse_outcome", &self.parse_outcome())
            .field("rejection_stage", &self.rejection_stage())
            .field("evaluation_outcome", &self.evaluation_outcome())
            .field("evaluation_error_stage", &self.evaluation_error_stage())
            .finish_non_exhaustive()
    }
}

impl RawV1Execution {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_RAW_V1_EXECUTION_V04_PROFILE_ID
    }

    pub const fn result_schema_id(&self) -> &'static str {
        RAW_V1_RUST_EXECUTION_V1_SCHEMA_ID
    }

    pub const fn parse_outcome(&self) -> RawV1ParseOutcome {
        match self.inner {
            RawV1ExecutionInner::Rejected(_) => RawV1ParseOutcome::Reject,
            RawV1ExecutionInner::SemanticResult(_) | RawV1ExecutionInner::EvaluationError(_) => {
                RawV1ParseOutcome::Accept
            }
        }
    }

    pub const fn rejection_stage(&self) -> Option<RawV1RejectionStage> {
        match &self.inner {
            RawV1ExecutionInner::Rejected(CanonicalRawV1AdmissionError::CanonicalWire(_)) => {
                Some(RawV1RejectionStage::CanonicalWire)
            }
            RawV1ExecutionInner::Rejected(CanonicalRawV1AdmissionError::Schema(_)) => {
                Some(RawV1RejectionStage::Schema)
            }
            RawV1ExecutionInner::SemanticResult(_) | RawV1ExecutionInner::EvaluationError(_) => {
                None
            }
        }
    }

    pub const fn evaluation_outcome(&self) -> RawV1EvaluationOutcome {
        match self.inner {
            RawV1ExecutionInner::Rejected(_) => RawV1EvaluationOutcome::NotRun,
            RawV1ExecutionInner::SemanticResult(_) => RawV1EvaluationOutcome::Result,
            RawV1ExecutionInner::EvaluationError(_) => RawV1EvaluationOutcome::Error,
        }
    }

    pub const fn evaluation_error_stage(&self) -> Option<RawV1EvaluationErrorStage> {
        match &self.inner {
            RawV1ExecutionInner::EvaluationError(RawV1OutcomeError::Namespace(_)) => {
                Some(RawV1EvaluationErrorStage::Namespace)
            }
            RawV1ExecutionInner::EvaluationError(RawV1OutcomeError::MixedReplay(_)) => {
                Some(RawV1EvaluationErrorStage::MixedReplay)
            }
            RawV1ExecutionInner::Rejected(_) | RawV1ExecutionInner::SemanticResult(_) => None,
        }
    }

    pub fn semantic_result(&self) -> Option<&RawV1Outcome> {
        match &self.inner {
            RawV1ExecutionInner::SemanticResult(result) => Some(result.as_ref()),
            RawV1ExecutionInner::Rejected(_) | RawV1ExecutionInner::EvaluationError(_) => None,
        }
    }

    pub fn source_error(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.inner {
            RawV1ExecutionInner::Rejected(source) => Some(source),
            RawV1ExecutionInner::EvaluationError(source) => Some(source),
            RawV1ExecutionInner::SemanticResult(_) => None,
        }
    }

    /// Deterministic compact UTF-8 JSON with no terminal newline.
    pub fn to_portable_json_bytes(&self) -> Result<Vec<u8>, RawV1ExecutionSerializationError> {
        serde_json::to_vec(&self.portable_projection())
            .map_err(|source| RawV1ExecutionSerializationError { source })
    }

    fn portable_projection(&self) -> RawV1ExecutionProjection<'_> {
        RawV1ExecutionProjection {
            schema: RAW_V1_RUST_EXECUTION_V1_SCHEMA_ID,
            profile: EXACT_RATIONAL_RAW_V1_EXECUTION_V04_PROFILE_ID,
            parse_outcome: self.parse_outcome().as_str(),
            rejection_stage: self.rejection_stage().map(RawV1RejectionStage::as_str),
            evaluation_outcome: self.evaluation_outcome().as_str(),
            evaluation_error_stage: self
                .evaluation_error_stage()
                .map(RawV1EvaluationErrorStage::as_str),
            semantic_result: self
                .semantic_result()
                .map(RawV1Outcome::portable_projection),
        }
    }
}

#[derive(Debug)]
pub struct RawV1ExecutionSerializationError {
    source: serde_json::Error,
}

impl fmt::Display for RawV1ExecutionSerializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "raw-v1 execution JSON serialization failed: {}",
            self.source
        )
    }
}

impl std::error::Error for RawV1ExecutionSerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Serialize)]
struct RawV1ExecutionProjection<'a> {
    schema: &'static str,
    profile: &'static str,
    parse_outcome: &'static str,
    rejection_stage: Option<&'static str>,
    evaluation_outcome: &'static str,
    evaluation_error_stage: Option<&'static str>,
    semantic_result: Option<PortableOutcomeProjection<'a>>,
}

/// Execute the complete byte/admission/evaluation envelope without panicking
/// or returning a failure. Every failure is retained in exactly one private
/// typed branch and exposed only through stable stage classification.
pub fn execute_raw_v1_bytes_exact_rational_v04(bytes: &[u8]) -> RawV1Execution {
    let admission = match CanonicalRawV1Admission::admit(bytes) {
        Ok(admission) => admission,
        Err(source) => {
            return RawV1Execution {
                inner: RawV1ExecutionInner::Rejected(source),
            };
        }
    };
    match replay_admitted_raw_v1_outcome_exact_rational_v04(&admission) {
        Ok(result) => RawV1Execution {
            inner: RawV1ExecutionInner::SemanticResult(Box::new(result)),
        },
        Err(source) => RawV1Execution {
            inner: RawV1ExecutionInner::EvaluationError(source),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NumericError, OrdinaryCoefficientKind, OrdinarySemanticError, OrdinarySemanticResource,
        PolynomialError, RawMixedPlanarChainReplayError,
    };

    #[test]
    fn empty_bytes_are_total_canonical_wire_rejection() {
        let execution = execute_raw_v1_bytes_exact_rational_v04(&[]);
        assert_eq!(execution.parse_outcome(), RawV1ParseOutcome::Reject);
        assert_eq!(
            execution.rejection_stage(),
            Some(RawV1RejectionStage::CanonicalWire)
        );
        assert_eq!(
            execution.evaluation_outcome(),
            RawV1EvaluationOutcome::NotRun
        );
        assert_eq!(execution.evaluation_error_stage(), None);
        assert!(execution.semantic_result().is_none());
        assert!(execution.source_error().is_some());
    }

    #[test]
    fn non_certificate_initial_semantic_failures_remain_evaluation_errors() {
        let sources = [
            OrdinarySemanticError::ResourceExhausted {
                resource: OrdinarySemanticResource::CoefficientCount,
                required: 2,
                limit: 1,
            },
            OrdinarySemanticError::Polynomial {
                coefficients: OrdinaryCoefficientKind::Position,
                source: PolynomialError::EmptyCoefficientTable,
            },
            OrdinarySemanticError::Numeric(NumericError::InvalidIntervalBounds),
        ];

        for source in sources {
            assert!(!source.is_certificate_defect());
            let execution = RawV1Execution {
                inner: RawV1ExecutionInner::EvaluationError(RawV1OutcomeError::MixedReplay(
                    RawMixedPlanarChainReplayError::InitialChartSemantic(source),
                )),
            };
            assert_eq!(execution.parse_outcome(), RawV1ParseOutcome::Accept);
            assert_eq!(execution.rejection_stage(), None);
            assert_eq!(
                execution.evaluation_outcome(),
                RawV1EvaluationOutcome::Error
            );
            assert_eq!(
                execution.evaluation_error_stage(),
                Some(RawV1EvaluationErrorStage::MixedReplay)
            );
            assert!(execution.semantic_result().is_none());
            assert!(execution.source_error().is_some());
            assert_eq!(
                execution.to_portable_json_bytes().unwrap(),
                b"{\"schema\":\"raw-v1-rust-execution-v1\",\"profile\":\"exact_rational_raw_v1_execution_v04\",\"parse_outcome\":\"ACCEPT\",\"rejection_stage\":null,\"evaluation_outcome\":\"ERROR\",\"evaluation_error_stage\":\"MIXED_REPLAY\",\"semantic_result\":null}"
            );
        }
    }
}
