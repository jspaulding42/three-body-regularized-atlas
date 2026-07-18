//! Opaque ownership of one canonically admitted raw-v1 chain.
//!
//! Construction preserves the syntax-versus-schema failure stage. This layer
//! owns the exact canonical bytes and decoded wire value but performs no
//! semantic replay, hashing, or certification.

use core::fmt;

use crate::{
    raw_schema::{
        decode_raw_chain, PlanarLcPassageSegmentWire, RawPlanarChainWire, SchemaDecodeError,
        SchemaProfile, SegmentWire,
    },
    validate_canonical_wire_json, WireJsonError, WireJsonLimits, DEFAULT_WIRE_JSON_LIMITS,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalRawV1AdmissionError {
    CanonicalWire(WireJsonError),
    Schema(SchemaDecodeError),
}

impl fmt::Display for CanonicalRawV1AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CanonicalWire(source) => {
                write!(formatter, "raw-v1 wire admission failed: {source}")
            }
            Self::Schema(source) => write!(formatter, "raw-v1 schema admission failed: {source}"),
        }
    }
}

impl std::error::Error for CanonicalRawV1AdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CanonicalWire(source) => Some(source),
            Self::Schema(source) => Some(source),
        }
    }
}

/// Canonical raw bytes bound immutably to their strict v0.3-compatible wire
/// decode. Private fields prevent callers from replacing either half.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalRawV1Admission {
    canonical_bytes: Box<[u8]>,
    wire: RawPlanarChainWire,
}

impl CanonicalRawV1Admission {
    /// Admit with the fixed public raw-v1 resource profile.
    pub fn admit(bytes: &[u8]) -> Result<Self, CanonicalRawV1AdmissionError> {
        Self::admit_with_limits(bytes, DEFAULT_WIRE_JSON_LIMITS)
    }

    fn admit_with_limits(
        bytes: &[u8],
        limits: WireJsonLimits,
    ) -> Result<Self, CanonicalRawV1AdmissionError> {
        let syntax = validate_canonical_wire_json(bytes, limits)
            .map_err(CanonicalRawV1AdmissionError::CanonicalWire)?;
        let wire = decode_raw_chain(syntax, SchemaProfile::V03Compatible)
            .map_err(CanonicalRawV1AdmissionError::Schema)?;
        Ok(Self {
            canonical_bytes: bytes.to_vec().into_boxed_slice(),
            wire,
        })
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub fn wire(&self) -> &RawPlanarChainWire {
        &self.wire
    }

    /// Borrow an LC passage from the bytes-bound decoded chain. `None` means
    /// either an out-of-range index or a different segment tag.
    pub fn planar_lc_passage(&self, segment_index: usize) -> Option<&PlanarLcPassageSegmentWire> {
        match self.wire.segments.get(segment_index) {
            Some(SegmentWire::PlanarLcPassage(segment)) => Some(segment),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{raw_schema::SchemaDecodeErrorKind, DEFAULT_WIRE_JSON_LIMITS};

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    #[test]
    fn canonical_bytes_and_wire_are_owned_and_immutably_bound() {
        let mut source = SUCCESS.to_vec();
        let admitted = CanonicalRawV1Admission::admit(&source).unwrap();
        let expected_id = admitted.wire().certificate_id.clone();
        source[0] = b'[';
        assert_eq!(admitted.canonical_bytes(), SUCCESS);
        assert_eq!(admitted.wire().certificate_id, expected_id);
    }

    #[test]
    fn noncanonical_input_preserves_the_wire_stage() {
        let mut bytes = SUCCESS.to_vec();
        bytes.push(b'\n');
        assert!(matches!(
            CanonicalRawV1Admission::admit(&bytes),
            Err(CanonicalRawV1AdmissionError::CanonicalWire(
                WireJsonError::NonCanonicalBytes { .. }
            ))
        ));
    }

    #[test]
    fn canonical_but_wrong_schema_preserves_the_schema_stage() {
        let error = CanonicalRawV1Admission::admit(b"{}").unwrap_err();
        let CanonicalRawV1AdmissionError::Schema(source) = error else {
            panic!("wrong failure stage");
        };
        assert_eq!(source.path(), "$.certificate_id");
        assert_eq!(source.kind(), &SchemaDecodeErrorKind::MissingField);
    }

    #[test]
    fn fixed_public_input_limit_fails_before_parsing() {
        let oversized = vec![b' '; DEFAULT_WIRE_JSON_LIMITS.max_input_bytes + 1];
        assert!(matches!(
            CanonicalRawV1Admission::admit(&oversized),
            Err(CanonicalRawV1AdmissionError::CanonicalWire(
                WireJsonError::InputByteLimitExceeded
            ))
        ));
    }
}
