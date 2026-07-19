//! Internal, execution-schema-neutral classification of mixed replay failures.
//!
//! This module is deliberately crate-private.  It gives the future executor a
//! total typed distinction between certificate defects, bounded-resource
//! exhaustion, and verifier invariants without changing the public v0.4
//! replay or its serialized execution envelope.

use crate::{
    outward_mass::{MassProfileError, OutwardBinary64Error},
    CarriedPlanarLcEntryError, CarriedPlanarLcExitError, ExpEnclosureError,
    InitialValueBindingReplayError, MixedChainReplayFailure, NumericError,
    OrdinaryBridgeReplayError, OrdinaryChartReplayError, OrdinaryFieldError,
    OrdinaryPolynomialDefectError, OrdinarySemanticError, OrdinarySemanticResource,
    OrdinaryTubeReplayError, PlanarLcChartReplayError, PlanarLcFieldError, PlanarLcLiftError,
    PlanarLcPolynomialDefectError, PlanarLcProjectionError, PlanarLcSemanticError,
    PlanarLcSemanticResource, PlanarLcSeriesError, PlanarLcSeriesKind, PlanarLcStateError,
    PlanarLcTubeReplayError, PolynomialError, RawMixedPlanarChainReplayError,
    RawOrdinaryOnlyChainReplayError, ValidatedOrdinaryRootReplayError,
};

/// The three disposition classes consumed by the planned v2 executor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MixedChainFailureDisposition {
    CertificateDefect,
    ResourceLimit(MixedChainResourceLimit),
    InternalInvariant,
}

/// A normalized bounded-resource record.
///
/// Amounts are decimal strings so that later serialization cannot narrow a
/// Rust integer.  They are optional because several leaf errors identify a
/// resource class but do not carry a meaningful required/limit measurement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MixedChainResourceLimit {
    pub(crate) stage: MixedChainResourceStage,
    pub(crate) resource: MixedChainResource,
    pub(crate) required: Option<String>,
    pub(crate) limit: Option<String>,
}

impl MixedChainResourceLimit {
    fn new(
        stage: MixedChainResourceStage,
        resource: MixedChainResource,
        required: Option<String>,
        limit: Option<String>,
    ) -> Self {
        // Exercise and guard the explicit stable-code tables even before the
        // v2 execution schema begins serializing them.
        debug_assert!(!stage.code().is_empty());
        debug_assert!(!resource.code().is_empty());
        Self {
            stage,
            resource,
            required,
            limit,
        }
    }
}

/// Stable mixed-execution location for a bounded resource failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MixedChainResourceStage {
    MixedChain,
    InitialChartSemantic,
    InitialChartReplay,
    Root,
    OrdinaryShared,
    MixedArithmetic,
    OrdinaryTargetSemantic,
    OrdinaryBridge,
    PlanarLcExit,
    FinalTube,
}

impl MixedChainResourceStage {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::MixedChain => "mixed_chain",
            Self::InitialChartSemantic => "initial_chart_semantic",
            Self::InitialChartReplay => "initial_chart_replay",
            Self::Root => "root",
            Self::OrdinaryShared => "ordinary_shared",
            Self::MixedArithmetic => "mixed_arithmetic",
            Self::OrdinaryTargetSemantic => "ordinary_target_semantic",
            Self::OrdinaryBridge => "ordinary_bridge",
            Self::PlanarLcExit => "planar_lc_exit",
            Self::FinalTube => "final_tube",
        }
    }
}

/// Stable resource identity independent of any `Display` or `Debug` text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MixedChainResource {
    SegmentCount,
    OrdinaryCoefficientCount,
    OrdinaryPositionPolynomialWork,
    OrdinaryVelocityPolynomialWork,
    OrdinaryChartSeriesWork,
    OrdinaryChartPositiveSqrtResolution,
    JsonSignificandDigits,
    JsonExponentDigits,
    JsonExponentMagnitude,
    JsonEffectiveExponent,
    RationalComponentBits,
    PolynomialCoefficientCount,
    DualDimension,
    SquareRootPrecision,
    PolynomialDimension,
    PolynomialDegree,
    PolynomialWork,
    ExponentialInputComponentBits,
    ExponentialIntermediateComponentBits,
    ExponentialRangeReductions,
    ExponentialTaylorCutoff,
    ExponentialInsufficientTaylorCutoff,
    ExponentialWork,
    ExponentialWitnessStorage,
    PlanarLcCoefficientCount,
    PlanarLcAggregatePolynomialWork,
    PlanarLcPolynomialWork(PlanarLcSeriesKind),
    PlanarLcSeriesWork,
    MassCoefficientComponentBits,
    OutwardRationalComponentBits,
    OutwardFiniteEnclosure,
}

impl MixedChainResource {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::SegmentCount => "segment_count",
            Self::OrdinaryCoefficientCount => "ordinary_coefficient_count",
            Self::OrdinaryPositionPolynomialWork => "ordinary_position_polynomial_work",
            Self::OrdinaryVelocityPolynomialWork => "ordinary_velocity_polynomial_work",
            Self::OrdinaryChartSeriesWork => "ordinary_chart_series_work",
            Self::OrdinaryChartPositiveSqrtResolution => "ordinary_chart_positive_sqrt_resolution",
            Self::JsonSignificandDigits => "json_significand_digits",
            Self::JsonExponentDigits => "json_exponent_digits",
            Self::JsonExponentMagnitude => "json_exponent_magnitude",
            Self::JsonEffectiveExponent => "json_effective_exponent",
            Self::RationalComponentBits => "rational_component_bits",
            Self::PolynomialCoefficientCount => "polynomial_coefficient_count",
            Self::DualDimension => "dual_dimension",
            Self::SquareRootPrecision => "square_root_precision",
            Self::PolynomialDimension => "polynomial_dimension",
            Self::PolynomialDegree => "polynomial_degree",
            Self::PolynomialWork => "polynomial_work",
            Self::ExponentialInputComponentBits => "exponential_input_component_bits",
            Self::ExponentialIntermediateComponentBits => "exponential_intermediate_component_bits",
            Self::ExponentialRangeReductions => "exponential_range_reductions",
            Self::ExponentialTaylorCutoff => "exponential_taylor_cutoff",
            Self::ExponentialInsufficientTaylorCutoff => "exponential_insufficient_taylor_cutoff",
            Self::ExponentialWork => "exponential_work",
            Self::ExponentialWitnessStorage => "exponential_witness_storage",
            Self::PlanarLcCoefficientCount => "planar_lc_coefficient_count",
            Self::PlanarLcAggregatePolynomialWork => "planar_lc_aggregate_polynomial_work",
            Self::PlanarLcPolynomialWork(kind) => match kind {
                PlanarLcSeriesKind::Z => "planar_lc_z_polynomial_work",
                PlanarLcSeriesKind::ZVelocity => "planar_lc_z_velocity_polynomial_work",
                PlanarLcSeriesKind::PairEnergy => "planar_lc_pair_energy_polynomial_work",
                PlanarLcSeriesKind::BinaryCenter => "planar_lc_binary_center_polynomial_work",
                PlanarLcSeriesKind::BinaryCenterVelocity => {
                    "planar_lc_binary_center_velocity_polynomial_work"
                }
                PlanarLcSeriesKind::ThirdOffset => "planar_lc_third_offset_polynomial_work",
                PlanarLcSeriesKind::ThirdOffsetVelocity => {
                    "planar_lc_third_offset_velocity_polynomial_work"
                }
                PlanarLcSeriesKind::PhysicalTime => "planar_lc_physical_time_polynomial_work",
            },
            Self::PlanarLcSeriesWork => "planar_lc_series_work",
            Self::MassCoefficientComponentBits => "mass_coefficient_component_bits",
            Self::OutwardRationalComponentBits => "outward_rational_component_bits",
            Self::OutwardFiniteEnclosure => "outward_finite_enclosure",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LeafDisposition {
    CertificateDefect,
    ResourceLimit(ResourceEvidence),
    InternalInvariant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResourceEvidence {
    resource: MixedChainResource,
    required: Option<String>,
    limit: Option<String>,
}

impl LeafDisposition {
    fn at(self, stage: MixedChainResourceStage) -> MixedChainFailureDisposition {
        match self {
            Self::CertificateDefect => MixedChainFailureDisposition::CertificateDefect,
            Self::ResourceLimit(evidence) => {
                MixedChainFailureDisposition::ResourceLimit(MixedChainResourceLimit::new(
                    stage,
                    evidence.resource,
                    evidence.required,
                    evidence.limit,
                ))
            }
            Self::InternalInvariant => MixedChainFailureDisposition::InternalInvariant,
        }
    }
}

fn resource_limit(
    resource: MixedChainResource,
    required: Option<String>,
    limit: Option<String>,
) -> LeafDisposition {
    LeafDisposition::ResourceLimit(ResourceEvidence {
        resource,
        required,
        limit,
    })
}

fn bounded(
    resource: MixedChainResource,
    required: impl ToString,
    limit: impl ToString,
) -> LeafDisposition {
    resource_limit(
        resource,
        Some(required.to_string()),
        Some(limit.to_string()),
    )
}

fn unmeasured(resource: MixedChainResource) -> LeafDisposition {
    resource_limit(resource, None, None)
}

fn exhausted_at(resource: MixedChainResource, limit: impl ToString) -> LeafDisposition {
    resource_limit(resource, None, Some(limit.to_string()))
}

fn component_bits(
    resource: MixedChainResource,
    numerator_bits: u64,
    denominator_bits: u64,
    limit: u64,
) -> LeafDisposition {
    bounded(resource, numerator_bits.max(denominator_bits), limit)
}

fn numeric(error: &NumericError) -> LeafDisposition {
    match error {
        NumericError::JsonNumberSignificandDigitLimitExceeded => {
            unmeasured(MixedChainResource::JsonSignificandDigits)
        }
        NumericError::JsonNumberExponentDigitLimitExceeded => {
            unmeasured(MixedChainResource::JsonExponentDigits)
        }
        NumericError::JsonNumberExponentMagnitudeLimitExceeded => {
            unmeasured(MixedChainResource::JsonExponentMagnitude)
        }
        NumericError::JsonNumberEffectiveExponentLimitExceeded => {
            unmeasured(MixedChainResource::JsonEffectiveExponent)
        }
        NumericError::RationalComponentBitLimitExceeded {
            numerator_bits,
            denominator_bits,
            limit,
        } => component_bits(
            MixedChainResource::RationalComponentBits,
            *numerator_bits,
            *denominator_bits,
            *limit,
        ),
        NumericError::PolynomialCoefficientCountLimitExceeded {
            coefficient_count,
            limit,
        } => bounded(
            MixedChainResource::PolynomialCoefficientCount,
            *coefficient_count,
            *limit,
        ),
        NumericError::DualDimensionLimitExceeded { dimension, limit } => {
            bounded(MixedChainResource::DualDimension, *dimension, *limit)
        }
        NumericError::SquareRootPrecisionBitLimitExceeded {
            precision_bits,
            limit,
        } => bounded(
            MixedChainResource::SquareRootPrecision,
            *precision_bits,
            *limit,
        ),
        NumericError::NonFiniteBinary64 { .. }
        | NumericError::MalformedJsonNumber
        | NumericError::JsonRealLexemeRequired
        | NumericError::HostBinary64ParseRejected
        | NumericError::Binary64CandidateNonFinite { .. }
        | NumericError::Binary64Overflow
        | NumericError::Binary64UnderflowSignMismatch
        | NumericError::Binary64CandidateSignMismatch
        | NumericError::Binary64NearestProofFailed
        | NumericError::InvalidRationalDenominator
        | NumericError::NonCanonicalRational
        | NumericError::DualDimensionMismatch { .. }
        | NumericError::DualIndexOutOfBounds { .. }
        | NumericError::DualSquareRootRequiresStrictlyPositiveValue
        | NumericError::InvalidIntervalBounds
        | NumericError::DivisionByZeroInterval
        | NumericError::NegativeSquareRoot
        | NumericError::PrecisionOverflow
        | NumericError::InternalSquareRootPostconditionFailure => {
            LeafDisposition::InternalInvariant
        }
    }
}

fn polynomial(error: &PolynomialError) -> LeafDisposition {
    match error {
        PolynomialError::Numeric(source) => numeric(source),
        PolynomialError::DimensionLimitExceeded { dimension, limit } => {
            bounded(MixedChainResource::PolynomialDimension, *dimension, *limit)
        }
        PolynomialError::DegreeLimitExceeded { degree, limit } => {
            bounded(MixedChainResource::PolynomialDegree, *degree, *limit)
        }
        PolynomialError::WorkLimitExceeded { required, limit } => {
            bounded(MixedChainResource::PolynomialWork, *required, *limit)
        }
        PolynomialError::EmptyCoefficientTable
        | PolynomialError::ZeroDimension
        | PolynomialError::CoefficientDimensionMismatch { .. } => {
            LeafDisposition::InternalInvariant
        }
    }
}

fn exponential(error: &ExpEnclosureError) -> LeafDisposition {
    match error {
        ExpEnclosureError::InputComponentBitLimitExceeded => {
            unmeasured(MixedChainResource::ExponentialInputComponentBits)
        }
        ExpEnclosureError::IntermediateComponentBitLimitExceeded => {
            unmeasured(MixedChainResource::ExponentialIntermediateComponentBits)
        }
        ExpEnclosureError::RangeReductionLimitExceeded => {
            unmeasured(MixedChainResource::ExponentialRangeReductions)
        }
        ExpEnclosureError::TaylorCutoffLimitExceeded => {
            unmeasured(MixedChainResource::ExponentialTaylorCutoff)
        }
        ExpEnclosureError::InsufficientTaylorCutoff { cutoff } => exhausted_at(
            MixedChainResource::ExponentialInsufficientTaylorCutoff,
            *cutoff,
        ),
        ExpEnclosureError::WorkBudgetExceeded => unmeasured(MixedChainResource::ExponentialWork),
        ExpEnclosureError::WitnessStorageBudgetExceeded => {
            unmeasured(MixedChainResource::ExponentialWitnessStorage)
        }
        ExpEnclosureError::NegativeExponent
        | ExpEnclosureError::NonpositiveMaxReducedTailUpper
        | ExpEnclosureError::InvalidResourceLimits
        | ExpEnclosureError::MalformedRationalDenominator
        | ExpEnclosureError::NoncanonicalRational
        | ExpEnclosureError::TaylorCutoffArithmeticOverflow
        | ExpEnclosureError::ExactDivisionByZero
        | ExpEnclosureError::WitnessPostconditionFailure => LeafDisposition::InternalInvariant,
    }
}

fn ordinary_semantic(error: &OrdinarySemanticError) -> LeafDisposition {
    match error {
        OrdinarySemanticError::EmptyString { .. }
        | OrdinarySemanticError::UnexpectedChartType { .. }
        | OrdinarySemanticError::MassCountMismatch { .. }
        | OrdinarySemanticError::NonPositiveMass { .. }
        | OrdinarySemanticError::NonPositiveSampleCount { .. }
        | OrdinarySemanticError::CoefficientCountTooSmall { .. }
        | OrdinarySemanticError::CoefficientCountMismatch { .. }
        | OrdinarySemanticError::CoefficientBodyCountMismatch { .. }
        | OrdinarySemanticError::CoefficientAxisCountMismatch { .. }
        | OrdinarySemanticError::IntervalNotStrictlyIncreasing { .. } => {
            LeafDisposition::CertificateDefect
        }
        OrdinarySemanticError::Polynomial { source, .. } => polynomial(source),
        OrdinarySemanticError::Numeric(source) => numeric(source),
        OrdinarySemanticError::ResourceExhausted {
            resource,
            required,
            limit,
        } => bounded(
            match resource {
                OrdinarySemanticResource::CoefficientCount => {
                    MixedChainResource::OrdinaryCoefficientCount
                }
                OrdinarySemanticResource::PositionPolynomialWork => {
                    MixedChainResource::OrdinaryPositionPolynomialWork
                }
                OrdinarySemanticResource::VelocityPolynomialWork => {
                    MixedChainResource::OrdinaryVelocityPolynomialWork
                }
            },
            *required,
            *limit,
        ),
    }
}

fn ordinary_chart(error: &OrdinaryChartReplayError) -> LeafDisposition {
    match error {
        OrdinaryChartReplayError::Numeric(source) => numeric(source),
        OrdinaryChartReplayError::WorkLimitExceeded { required, limit } => bounded(
            MixedChainResource::OrdinaryChartSeriesWork,
            *required,
            *limit,
        ),
        OrdinaryChartReplayError::PositiveSquareRootResolutionLimitExceeded { precision_bits } => {
            exhausted_at(
                MixedChainResource::OrdinaryChartPositiveSqrtResolution,
                *precision_bits,
            )
        }
        OrdinaryChartReplayError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
    }
}

fn ordinary_field(error: &OrdinaryFieldError) -> LeafDisposition {
    match error {
        OrdinaryFieldError::Numeric(source) => numeric(source),
        OrdinaryFieldError::NonPositiveMass { .. }
        | OrdinaryFieldError::PairDistanceNotSeparated { .. } => LeafDisposition::CertificateDefect,
    }
}

fn ordinary_defect(error: &OrdinaryPolynomialDefectError) -> LeafDisposition {
    match error {
        OrdinaryPolynomialDefectError::PositionDimensionMismatch { .. }
        | OrdinaryPolynomialDefectError::VelocityDimensionMismatch { .. } => {
            LeafDisposition::InternalInvariant
        }
        OrdinaryPolynomialDefectError::PositionPolynomial(source)
        | OrdinaryPolynomialDefectError::VelocityPolynomial(source) => polynomial(source),
        OrdinaryPolynomialDefectError::OrdinaryField(source) => ordinary_field(source),
        OrdinaryPolynomialDefectError::Numeric(source) => numeric(source),
    }
}

fn ordinary_tube(error: &OrdinaryTubeReplayError) -> LeafDisposition {
    match error {
        OrdinaryTubeReplayError::Numeric(source) => numeric(source),
        OrdinaryTubeReplayError::Polynomial(source) => polynomial(source),
        OrdinaryTubeReplayError::Defect(source) => ordinary_defect(source),
        OrdinaryTubeReplayError::Exponential(source) => exponential(source),
    }
}

fn initial_binding(error: &InitialValueBindingReplayError) -> LeafDisposition {
    match error {
        InitialValueBindingReplayError::Numeric(source) => numeric(source),
        InitialValueBindingReplayError::PositionPolynomial(source)
        | InitialValueBindingReplayError::VelocityPolynomial(source) => polynomial(source),
        InitialValueBindingReplayError::InternalInvariant { .. } => {
            LeafDisposition::InternalInvariant
        }
    }
}

fn validated_root(error: &ValidatedOrdinaryRootReplayError) -> LeafDisposition {
    match error {
        ValidatedOrdinaryRootReplayError::Binding(source) => initial_binding(source),
        ValidatedOrdinaryRootReplayError::Tube(source) => ordinary_tube(source),
    }
}

fn ordinary_chain(error: &RawOrdinaryOnlyChainReplayError) -> LeafDisposition {
    match error {
        RawOrdinaryOnlyChainReplayError::InitialChartSemantic(source) => ordinary_semantic(source),
        RawOrdinaryOnlyChainReplayError::InitialChartReplay(source) => ordinary_chart(source),
        RawOrdinaryOnlyChainReplayError::Root(source) => validated_root(source),
        RawOrdinaryOnlyChainReplayError::FrontierTube(source) => ordinary_tube(source),
        RawOrdinaryOnlyChainReplayError::PositionPolynomial(source)
        | RawOrdinaryOnlyChainReplayError::VelocityPolynomial(source) => polynomial(source),
        RawOrdinaryOnlyChainReplayError::Numeric(source) => numeric(source),
        RawOrdinaryOnlyChainReplayError::InternalInvariant { .. } => {
            LeafDisposition::InternalInvariant
        }
    }
}

fn ordinary_bridge(error: &OrdinaryBridgeReplayError) -> LeafDisposition {
    match error {
        OrdinaryBridgeReplayError::Numeric(source) => numeric(source),
        OrdinaryBridgeReplayError::SourcePositionPolynomial(source)
        | OrdinaryBridgeReplayError::SourceVelocityPolynomial(source)
        | OrdinaryBridgeReplayError::TargetPositionPolynomial(source)
        | OrdinaryBridgeReplayError::TargetVelocityPolynomial(source) => polynomial(source),
        OrdinaryBridgeReplayError::SourceTube(source)
        | OrdinaryBridgeReplayError::TargetTube(source) => ordinary_tube(source),
        OrdinaryBridgeReplayError::InternalInvariant { .. } => LeafDisposition::InternalInvariant,
    }
}

fn planar_lc_semantic(error: &PlanarLcSemanticError) -> LeafDisposition {
    match error {
        PlanarLcSemanticError::EmptyString { .. }
        | PlanarLcSemanticError::UnexpectedConstant { .. }
        | PlanarLcSemanticError::SchemaVersion { .. }
        | PlanarLcSemanticError::MassCount { .. }
        | PlanarLcSemanticError::NonPositiveMass { .. }
        | PlanarLcSemanticError::PairNotCanonical { .. }
        | PlanarLcSemanticError::PairIndexConversion { .. }
        | PlanarLcSemanticError::NonPositiveSampleCount { .. }
        | PlanarLcSemanticError::CoefficientCount { .. }
        | PlanarLcSemanticError::CoefficientCountTooSmall { .. }
        | PlanarLcSemanticError::CoefficientAxisCount { .. }
        | PlanarLcSemanticError::IntervalNotStrict { .. }
        | PlanarLcSemanticError::TubeSign { .. }
        | PlanarLcSemanticError::TubeAnchorOutsideChart
        | PlanarLcSemanticError::IdentityBinding { .. }
        | PlanarLcSemanticError::MassBinding { .. }
        | PlanarLcSemanticError::EndpointBinding { .. } => LeafDisposition::CertificateDefect,
        PlanarLcSemanticError::SegmentUnavailable { .. } => LeafDisposition::InternalInvariant,
        PlanarLcSemanticError::ResourceExhausted {
            resource,
            required,
            limit,
        } => bounded(
            match resource {
                PlanarLcSemanticResource::CoefficientCount => {
                    MixedChainResource::PlanarLcCoefficientCount
                }
                PlanarLcSemanticResource::AggregatePolynomialWork => {
                    MixedChainResource::PlanarLcAggregatePolynomialWork
                }
                PlanarLcSemanticResource::PolynomialWork(kind) => {
                    MixedChainResource::PlanarLcPolynomialWork(*kind)
                }
            },
            *required,
            *limit,
        ),
        PlanarLcSemanticError::Polynomial { source, .. } => polynomial(source),
        PlanarLcSemanticError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_state(error: &PlanarLcStateError) -> LeafDisposition {
    match error {
        PlanarLcStateError::CoefficientCountMismatch
        | PlanarLcStateError::StateDimensionMismatch { .. }
        | PlanarLcStateError::NegativeInflation
        | PlanarLcStateError::NonPointAnchorComponent { .. } => LeafDisposition::InternalInvariant,
        PlanarLcStateError::Polynomial(source) => polynomial(source),
        PlanarLcStateError::Numeric(source) => numeric(source),
    }
}

fn outward(error: &OutwardBinary64Error) -> LeafDisposition {
    match error {
        OutwardBinary64Error::Numeric(source) => numeric(source),
        OutwardBinary64Error::RationalComponentBitLimitExceeded {
            numerator_bits,
            denominator_bits,
            limit,
        } => component_bits(
            MixedChainResource::OutwardRationalComponentBits,
            *numerator_bits,
            *denominator_bits,
            *limit,
        ),
        OutwardBinary64Error::NoFiniteEnclosure => {
            unmeasured(MixedChainResource::OutwardFiniteEnclosure)
        }
        OutwardBinary64Error::RationalZeroDenominator
        | OutwardBinary64Error::RationalNegativeDenominator
        | OutwardBinary64Error::RationalNotCanonical
        | OutwardBinary64Error::Postcondition(_) => LeafDisposition::InternalInvariant,
    }
}

fn mass_profile(error: &MassProfileError) -> LeafDisposition {
    match error {
        MassProfileError::MassNotPositive { .. }
        | MassProfileError::PairIndexOutOfRange { .. }
        | MassProfileError::PairIndicesNotDistinct { .. } => LeafDisposition::CertificateDefect,
        MassProfileError::MetadataMismatch | MassProfileError::ExactFormulaMismatch { .. } => {
            LeafDisposition::InternalInvariant
        }
        MassProfileError::DerivedCoefficientComponentBitBoundExceeded {
            numerator_bits,
            denominator_bits,
            limit,
            ..
        } => component_bits(
            MixedChainResource::MassCoefficientComponentBits,
            *numerator_bits,
            *denominator_bits,
            *limit,
        ),
        MassProfileError::CoefficientEnclosure { source, .. } => outward(source),
    }
}

fn planar_lc_series(error: &PlanarLcSeriesError) -> LeafDisposition {
    match error {
        PlanarLcSeriesError::State(source) => planar_lc_state(source),
        PlanarLcSeriesError::MassDecode { source, .. } => numeric(source),
        PlanarLcSeriesError::MassProfile(source) => mass_profile(source),
        PlanarLcSeriesError::DegreeTooSmall { .. }
        | PlanarLcSeriesError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
        PlanarLcSeriesError::WorkLimitExceeded { required, limit } => {
            bounded(MixedChainResource::PlanarLcSeriesWork, *required, *limit)
        }
        PlanarLcSeriesError::SqrtPrecisionExceeded { requested, limit } => {
            bounded(MixedChainResource::SquareRootPrecision, *requested, *limit)
        }
        PlanarLcSeriesError::ThirdBodyCollision { .. } => LeafDisposition::CertificateDefect,
        // Exact positivity is established before the dyadic lower enclosure
        // can remain zero, so this is precision exhaustion, not collision.
        PlanarLcSeriesError::PositiveSquareRootUnresolved { .. } => {
            unmeasured(MixedChainResource::SquareRootPrecision)
        }
        PlanarLcSeriesError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_field(error: &PlanarLcFieldError) -> LeafDisposition {
    match error {
        PlanarLcFieldError::MassDecode { source, .. } => numeric(source),
        PlanarLcFieldError::MassProfile(source) => mass_profile(source),
        PlanarLcFieldError::ThirdBodyCollision { .. } => LeafDisposition::CertificateDefect,
        PlanarLcFieldError::SqrtPrecisionExceeded { requested, limit } => {
            bounded(MixedChainResource::SquareRootPrecision, *requested, *limit)
        }
        PlanarLcFieldError::InternalDimension { .. } => LeafDisposition::InternalInvariant,
        PlanarLcFieldError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_projection(error: &PlanarLcProjectionError) -> LeafDisposition {
    match error {
        PlanarLcProjectionError::State(source) => planar_lc_state(source),
        PlanarLcProjectionError::Field(source) => planar_lc_field(source),
        PlanarLcProjectionError::MassDecode { source, .. } => numeric(source),
        PlanarLcProjectionError::MassProfile(source) => mass_profile(source),
        PlanarLcProjectionError::NegativeProjectionFloor
        | PlanarLcProjectionError::RhoFloorNotStrict
        | PlanarLcProjectionError::RhoNotStrictlyPositive
        | PlanarLcProjectionError::NewtonCollision { .. } => LeafDisposition::CertificateDefect,
        // The exact norm-square lower bound is already strictly positive at
        // this branch; only the configured dyadic resolution is insufficient.
        PlanarLcProjectionError::PositiveSquareRootUnresolved { .. } => {
            unmeasured(MixedChainResource::SquareRootPrecision)
        }
        PlanarLcProjectionError::SqrtPrecisionExceeded { requested, limit } => {
            bounded(MixedChainResource::SquareRootPrecision, *requested, *limit)
        }
        PlanarLcProjectionError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
        PlanarLcProjectionError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_lift(error: &PlanarLcLiftError) -> LeafDisposition {
    match error {
        PlanarLcLiftError::CartesianStateDimension { .. }
        | PlanarLcLiftError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
        PlanarLcLiftError::SqrtPrecisionExceeded { requested, limit } => {
            bounded(MixedChainResource::SquareRootPrecision, *requested, *limit)
        }
        PlanarLcLiftError::MassDecode { source, .. } => numeric(source),
        PlanarLcLiftError::MassProfile(source) => mass_profile(source),
        PlanarLcLiftError::SquareRootDomain { .. } => LeafDisposition::CertificateDefect,
        PlanarLcLiftError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_defect(error: &PlanarLcPolynomialDefectError) -> LeafDisposition {
    match error {
        PlanarLcPolynomialDefectError::State(source) => planar_lc_state(source),
        PlanarLcPolynomialDefectError::Field(source) => planar_lc_field(source),
        PlanarLcPolynomialDefectError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_tube(error: &PlanarLcTubeReplayError) -> LeafDisposition {
    match error {
        PlanarLcTubeReplayError::State(source) => planar_lc_state(source),
        PlanarLcTubeReplayError::Defect(source) => planar_lc_defect(source),
        PlanarLcTubeReplayError::Field(source) => planar_lc_field(source),
        PlanarLcTubeReplayError::MassDecode { source, .. } => numeric(source),
        PlanarLcTubeReplayError::MassProfile(source) => mass_profile(source),
        PlanarLcTubeReplayError::Exponential(source) => exponential(source),
        PlanarLcTubeReplayError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
        PlanarLcTubeReplayError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_chart(error: &PlanarLcChartReplayError) -> LeafDisposition {
    match error {
        PlanarLcChartReplayError::State(source) => planar_lc_state(source),
        PlanarLcChartReplayError::Series(source) => planar_lc_series(source),
        PlanarLcChartReplayError::Projection(source) => planar_lc_projection(source),
        PlanarLcChartReplayError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
        PlanarLcChartReplayError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_entry(error: &CarriedPlanarLcEntryError) -> LeafDisposition {
    match error {
        CarriedPlanarLcEntryError::Semantic(source) => planar_lc_semantic(source),
        CarriedPlanarLcEntryError::SourceProvenanceSemantic(source) => ordinary_semantic(source),
        CarriedPlanarLcEntryError::SourceProvenanceMismatch { .. } => {
            LeafDisposition::CertificateDefect
        }
        CarriedPlanarLcEntryError::NamespaceSegmentLimit { actual, limit } => {
            bounded(MixedChainResource::SegmentCount, *actual, *limit)
        }
        CarriedPlanarLcEntryError::SourceChart(source) => ordinary_chart(source),
        CarriedPlanarLcEntryError::SourceTube(source) => ordinary_tube(source),
        CarriedPlanarLcEntryError::TargetChart(source) => planar_lc_chart(source),
        CarriedPlanarLcEntryError::TargetTube(source) => planar_lc_tube(source),
        CarriedPlanarLcEntryError::Lift(source) => planar_lc_lift(source),
        CarriedPlanarLcEntryError::TargetState(source) => planar_lc_state(source),
        CarriedPlanarLcEntryError::SourcePositionPolynomial(source)
        | CarriedPlanarLcEntryError::SourceVelocityPolynomial(source) => polynomial(source),
        CarriedPlanarLcEntryError::MassDecode { source, .. } => numeric(source),
        CarriedPlanarLcEntryError::MassProfile(source) => mass_profile(source),
        CarriedPlanarLcEntryError::InternalShapeInvariant { .. } => {
            LeafDisposition::InternalInvariant
        }
        CarriedPlanarLcEntryError::Numeric(source) => numeric(source),
    }
}

fn planar_lc_exit(error: &CarriedPlanarLcExitError) -> LeafDisposition {
    match error {
        CarriedPlanarLcExitError::SegmentUnavailable { .. }
        | CarriedPlanarLcExitError::SegmentNotPlanarLc { .. } => LeafDisposition::InternalInvariant,
        CarriedPlanarLcExitError::SourceSemantic(source)
        | CarriedPlanarLcExitError::TargetSemantic(source) => ordinary_semantic(source),
        CarriedPlanarLcExitError::LcSemantic(source) => planar_lc_semantic(source),
        CarriedPlanarLcExitError::Entry(source) => planar_lc_entry(source),
        CarriedPlanarLcExitError::LcTube(source) => planar_lc_tube(source),
        CarriedPlanarLcExitError::TargetTube(source) => ordinary_tube(source),
        CarriedPlanarLcExitError::State(source) => planar_lc_state(source),
        CarriedPlanarLcExitError::Projection(source) => planar_lc_projection(source),
        CarriedPlanarLcExitError::TargetPositionPolynomial(source)
        | CarriedPlanarLcExitError::TargetVelocityPolynomial(source) => polynomial(source),
        CarriedPlanarLcExitError::Numeric(source) => numeric(source),
        CarriedPlanarLcExitError::InternalShapeInvariant => LeafDisposition::InternalInvariant,
    }
}

impl MixedChainReplayFailure {
    /// Classify every structured failure retained by a successful v0.4 mixed
    /// replay.  No formatted error text participates in this decision.
    pub(crate) fn disposition(&self) -> MixedChainFailureDisposition {
        match self {
            Self::InitialChartSemantic { source } => {
                ordinary_semantic(source).at(MixedChainResourceStage::InitialChartSemantic)
            }
            Self::SegmentLimitExceeded { actual, limit } => {
                bounded(MixedChainResource::SegmentCount, *actual, *limit)
                    .at(MixedChainResourceStage::MixedChain)
            }
            Self::LocalObligations { .. } => MixedChainFailureDisposition::CertificateDefect,
            Self::OrdinaryTargetSemantic { source, .. } => {
                ordinary_semantic(source).at(MixedChainResourceStage::OrdinaryTargetSemantic)
            }
            Self::OrdinaryBridgeKernel { source, .. } => {
                ordinary_bridge(source).at(MixedChainResourceStage::OrdinaryBridge)
            }
            Self::PlanarLcExitKernel { source, .. } => {
                planar_lc_exit(source).at(MixedChainResourceStage::PlanarLcExit)
            }
            Self::FinalTube { source } => {
                ordinary_tube(source).at(MixedChainResourceStage::FinalTube)
            }
        }
    }
}

impl RawMixedPlanarChainReplayError {
    /// Classify every top-level mixed replay error for the future v2 wrapper.
    pub(crate) fn disposition(&self) -> MixedChainFailureDisposition {
        match self {
            Self::InitialChartSemantic(source) => {
                ordinary_semantic(source).at(MixedChainResourceStage::InitialChartSemantic)
            }
            Self::InitialChartReplay(source) => {
                ordinary_chart(source).at(MixedChainResourceStage::InitialChartReplay)
            }
            Self::Root(source) => validated_root(source).at(MixedChainResourceStage::Root),
            Self::OrdinaryShared(source) => {
                ordinary_chain(source).at(MixedChainResourceStage::OrdinaryShared)
            }
            Self::PositionPolynomial(source) | Self::VelocityPolynomial(source) => {
                polynomial(source).at(MixedChainResourceStage::MixedArithmetic)
            }
            Self::Numeric(source) => numeric(source).at(MixedChainResourceStage::MixedArithmetic),
            Self::InternalInvariant { .. } => MixedChainFailureDisposition::InternalInvariant,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        replay_raw_mixed_planar_chain_exact_rational_v04, CanonicalRawV1Admission,
        MixedPlanarSegmentKind, PlanarLcProjectionPair, PlanarLcSeriesDenominator,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn assert_resource(
        disposition: MixedChainFailureDisposition,
        stage: MixedChainResourceStage,
        resource: MixedChainResource,
        required: Option<&str>,
        limit: Option<&str>,
    ) {
        let actual = match disposition {
            MixedChainFailureDisposition::ResourceLimit(actual) => actual,
            other => panic!("expected resource disposition, got {other:?}"),
        };
        assert_eq!(actual.stage, stage);
        assert_eq!(actual.resource, resource);
        assert_eq!(actual.required.as_deref(), required);
        assert_eq!(actual.limit.as_deref(), limit);
        assert!(!actual.stage.code().is_empty());
        assert!(!actual.resource.code().is_empty());
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

    fn replace_array_after(raw: &str, anchor: &str, field: &str, replacement: &str) -> String {
        let anchor_start = raw.find(anchor).expect("missing record anchor");
        let marker = format!("\"{field}\":[");
        let relative = raw[anchor_start..]
            .find(&marker)
            .expect("missing anchored array");
        let array_start = anchor_start + relative + marker.len() - 1;
        let array_end = matching_array_end(raw, array_start);
        format!(
            "{}{}{}",
            &raw[..array_start],
            replacement,
            &raw[array_end + 1..]
        )
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

    fn replace_array_in_anchored_object(
        raw: &str,
        anchor: &str,
        field: &str,
        replacement: &str,
    ) -> String {
        let anchor_start = raw.find(anchor).expect("missing record anchor");
        let object_start = containing_object_start(raw, anchor_start);
        let marker = format!("\"{field}\":[");
        let relative = raw[object_start..]
            .find(&marker)
            .expect("missing array in anchored object");
        let array_start = object_start + relative + marker.len() - 1;
        let array_end = matching_array_end(raw, array_start);
        format!(
            "{}{}{}",
            &raw[..array_start],
            replacement,
            &raw[array_end + 1..]
        )
    }

    fn repeated_array(row: &str, count: usize) -> String {
        format!("[{}]", vec![row; count].join(","))
    }

    fn resize_ordinary(raw: &str, certificate_id: &str, count: usize) -> String {
        let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
        let row = repeated_array("[[0.0,0.0],[0.0,0.0],[0.0,0.0]]", count);
        let with_positions = replace_array_after(raw, &anchor, "position_coefficients", &row);
        replace_array_after(&with_positions, &anchor, "velocity_coefficients", &row)
    }

    fn resize_ordinary_preserving_first(raw: &str, certificate_id: &str, count: usize) -> String {
        assert!(count > 0);
        let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
        let zero = "[[0.0,0.0],[0.0,0.0],[0.0,0.0]]";
        let mut result = raw.to_owned();
        for field in ["position_coefficients", "velocity_coefficients"] {
            let marker = format!("\"{field}\":[");
            let anchor_start = result.find(&anchor).expect("missing record anchor");
            let relative = result[anchor_start..]
                .find(&marker)
                .expect("missing anchored array");
            let array_start = anchor_start + relative + marker.len() - 1;
            let first_start = array_start + 1;
            let first_end = matching_array_end(&result, first_start);
            let first = result[first_start..=first_end].to_owned();
            let replacement = if count == 1 {
                format!("[{first}]")
            } else {
                format!("[{first},{}]", vec![zero; count - 1].join(","))
            };
            result = replace_array_after(&result, &anchor, field, &replacement);
        }
        result
    }

    fn resize_planar_lc(raw: &str, certificate_id: &str, count: usize) -> String {
        let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
        let vector_rows = repeated_array("[0.0,0.0]", count);
        let scalar_rows = repeated_array("0.0", count);
        let mut result = raw.to_owned();
        for field in [
            "z_coefficients",
            "z_velocity_coefficients",
            "binary_center_coefficients",
            "binary_center_velocity_coefficients",
            "third_offset_coefficients",
            "third_offset_velocity_coefficients",
        ] {
            result = replace_array_in_anchored_object(&result, &anchor, field, &vector_rows);
        }
        for field in ["pair_energy_coefficients", "physical_time_coefficients"] {
            result = replace_array_in_anchored_object(&result, &anchor, field, &scalar_rows);
        }
        result
    }

    fn first_segment(raw: &str) -> &str {
        let marker = "\"segments\":[";
        let start = raw.find(marker).unwrap() + marker.len();
        let bytes = raw.as_bytes();
        assert_eq!(bytes[start], b'{');
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
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &raw[start..=index];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated segment")
    }

    fn repeat_first_segment(raw: &str, count: usize) -> String {
        let marker = "\"segments\":[";
        let start = raw.find(marker).unwrap() + marker.len();
        let end = raw.rfind("],\"source\":").unwrap();
        let segments = vec![first_segment(raw); count].join(",");
        format!("{}{}{}", &raw[..start], segments, &raw[end..])
    }

    #[test]
    fn stable_stage_and_resource_codes_are_unique() {
        let stages = [
            MixedChainResourceStage::MixedChain,
            MixedChainResourceStage::InitialChartSemantic,
            MixedChainResourceStage::InitialChartReplay,
            MixedChainResourceStage::Root,
            MixedChainResourceStage::OrdinaryShared,
            MixedChainResourceStage::MixedArithmetic,
            MixedChainResourceStage::OrdinaryTargetSemantic,
            MixedChainResourceStage::OrdinaryBridge,
            MixedChainResourceStage::PlanarLcExit,
            MixedChainResourceStage::FinalTube,
        ];
        let stage_codes = stages.map(MixedChainResourceStage::code);
        assert_eq!(
            stage_codes.into_iter().collect::<BTreeSet<_>>().len(),
            stages.len()
        );

        let resources = [
            MixedChainResource::SegmentCount,
            MixedChainResource::OrdinaryCoefficientCount,
            MixedChainResource::OrdinaryPositionPolynomialWork,
            MixedChainResource::OrdinaryVelocityPolynomialWork,
            MixedChainResource::OrdinaryChartSeriesWork,
            MixedChainResource::OrdinaryChartPositiveSqrtResolution,
            MixedChainResource::JsonSignificandDigits,
            MixedChainResource::JsonExponentDigits,
            MixedChainResource::JsonExponentMagnitude,
            MixedChainResource::JsonEffectiveExponent,
            MixedChainResource::RationalComponentBits,
            MixedChainResource::PolynomialCoefficientCount,
            MixedChainResource::DualDimension,
            MixedChainResource::SquareRootPrecision,
            MixedChainResource::PolynomialDimension,
            MixedChainResource::PolynomialDegree,
            MixedChainResource::PolynomialWork,
            MixedChainResource::ExponentialInputComponentBits,
            MixedChainResource::ExponentialIntermediateComponentBits,
            MixedChainResource::ExponentialRangeReductions,
            MixedChainResource::ExponentialTaylorCutoff,
            MixedChainResource::ExponentialInsufficientTaylorCutoff,
            MixedChainResource::ExponentialWork,
            MixedChainResource::ExponentialWitnessStorage,
            MixedChainResource::PlanarLcCoefficientCount,
            MixedChainResource::PlanarLcAggregatePolynomialWork,
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::Z),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::ZVelocity),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::PairEnergy),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::BinaryCenter),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::BinaryCenterVelocity),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::ThirdOffset),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::ThirdOffsetVelocity),
            MixedChainResource::PlanarLcPolynomialWork(PlanarLcSeriesKind::PhysicalTime),
            MixedChainResource::PlanarLcSeriesWork,
            MixedChainResource::MassCoefficientComponentBits,
            MixedChainResource::OutwardRationalComponentBits,
            MixedChainResource::OutwardFiniteEnclosure,
        ];
        let resource_codes = resources.map(MixedChainResource::code);
        assert_eq!(
            resource_codes.into_iter().collect::<BTreeSet<_>>().len(),
            resources.len()
        );
    }

    #[test]
    fn every_retained_mixed_failure_variant_has_a_typed_disposition() {
        let certificate_failures = [
            MixedChainReplayFailure::InitialChartSemantic {
                source: OrdinarySemanticError::UnexpectedChartType {
                    actual: "bad".to_owned(),
                },
            },
            MixedChainReplayFailure::LocalObligations {
                segment_index: 0,
                kind: MixedPlanarSegmentKind::OrdinaryBridge,
            },
        ];
        for failure in certificate_failures {
            assert_eq!(
                failure.disposition(),
                MixedChainFailureDisposition::CertificateDefect
            );
        }

        assert_resource(
            MixedChainReplayFailure::SegmentLimitExceeded {
                actual: 257,
                limit: 256,
            }
            .disposition(),
            MixedChainResourceStage::MixedChain,
            MixedChainResource::SegmentCount,
            Some("257"),
            Some("256"),
        );
        assert_resource(
            MixedChainReplayFailure::OrdinaryTargetSemantic {
                segment_index: 3,
                source: OrdinarySemanticError::ResourceExhausted {
                    resource: OrdinarySemanticResource::PositionPolynomialWork,
                    required: 65_544,
                    limit: 65_536,
                },
            }
            .disposition(),
            MixedChainResourceStage::OrdinaryTargetSemantic,
            MixedChainResource::OrdinaryPositionPolynomialWork,
            Some("65544"),
            Some("65536"),
        );
        assert_eq!(
            MixedChainReplayFailure::OrdinaryBridgeKernel {
                segment_index: 2,
                source: OrdinaryBridgeReplayError::InternalInvariant { detail: "probe" },
            }
            .disposition(),
            MixedChainFailureDisposition::InternalInvariant
        );
        assert_eq!(
            MixedChainReplayFailure::PlanarLcExitKernel {
                segment_index: 1,
                source: CarriedPlanarLcExitError::InternalShapeInvariant,
            }
            .disposition(),
            MixedChainFailureDisposition::InternalInvariant
        );
        assert_resource(
            MixedChainReplayFailure::FinalTube {
                source: OrdinaryTubeReplayError::Exponential(ExpEnclosureError::WorkBudgetExceeded),
            }
            .disposition(),
            MixedChainResourceStage::FinalTube,
            MixedChainResource::ExponentialWork,
            None,
            None,
        );
    }

    #[test]
    fn direct_top_level_internal_and_component_bound_variants_are_typed() {
        assert_eq!(
            RawMixedPlanarChainReplayError::InternalInvariant { detail: "probe" }.disposition(),
            MixedChainFailureDisposition::InternalInvariant
        );
        assert_resource(
            RawMixedPlanarChainReplayError::Numeric(
                NumericError::RationalComponentBitLimitExceeded {
                    numerator_bits: 513,
                    denominator_bits: 500,
                    limit: 512,
                },
            )
            .disposition(),
            MixedChainResourceStage::MixedArithmetic,
            MixedChainResource::RationalComponentBits,
            Some("513"),
            Some("512"),
        );
    }

    #[test]
    fn positive_sqrt_unresolved_is_precision_exhaustion_not_a_domain_defect() {
        assert_resource(
            planar_lc_series(&PlanarLcSeriesError::PositiveSquareRootUnresolved {
                denominator: PlanarLcSeriesDenominator::First,
            })
            .at(MixedChainResourceStage::PlanarLcExit),
            MixedChainResourceStage::PlanarLcExit,
            MixedChainResource::SquareRootPrecision,
            None,
            None,
        );
        assert_resource(
            planar_lc_projection(&PlanarLcProjectionError::PositiveSquareRootUnresolved {
                pair: PlanarLcProjectionPair::for_failure_disposition_test(0, 1),
            })
            .at(MixedChainResourceStage::PlanarLcExit),
            MixedChainResourceStage::PlanarLcExit,
            MixedChainResource::SquareRootPrecision,
            None,
            None,
        );

        assert_eq!(
            planar_lc_series(&PlanarLcSeriesError::ThirdBodyCollision {
                denominator: PlanarLcSeriesDenominator::Second,
            })
            .at(MixedChainResourceStage::PlanarLcExit),
            MixedChainFailureDisposition::CertificateDefect
        );
        assert_eq!(
            planar_lc_projection(&PlanarLcProjectionError::NewtonCollision {
                pair: PlanarLcProjectionPair::for_failure_disposition_test(1, 2),
            })
            .at(MixedChainResourceStage::PlanarLcExit),
            MixedChainFailureDisposition::CertificateDefect
        );
    }

    #[test]
    fn initial_chart_146_rows_normalizes_exact_series_work() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let mutated = resize_ordinary_preserving_first(raw, "review-v03:n:certificate:0", 146);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let error = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap_err();
        assert!(matches!(
            &error,
            RawMixedPlanarChainReplayError::InitialChartReplay(
                OrdinaryChartReplayError::WorkLimitExceeded { .. }
            )
        ));
        assert_resource(
            error.disposition(),
            MixedChainResourceStage::InitialChartReplay,
            MixedChainResource::OrdinaryChartSeriesWork,
            Some("2018400"),
            Some("2000000"),
        );
    }

    #[test]
    fn retained_segment_cap_and_target_2731_rows_have_exact_limits() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let repeated = repeat_first_segment(raw, 257);
        let admission = CanonicalRawV1Admission::admit(repeated.as_bytes()).unwrap();
        let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();
        assert_resource(
            replay.replay_failure().unwrap().disposition(),
            MixedChainResourceStage::MixedChain,
            MixedChainResource::SegmentCount,
            Some("257"),
            Some("256"),
        );

        let mutated = resize_ordinary(raw, "review-v03:n:certificate:1", 2_731);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();
        assert_resource(
            replay.replay_failure().unwrap().disposition(),
            MixedChainResourceStage::OrdinaryTargetSemantic,
            MixedChainResource::OrdinaryPositionPolynomialWork,
            Some("65544"),
            Some("65536"),
        );
    }

    #[test]
    fn retained_planar_lc_89_rows_normalizes_exact_series_work() {
        let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
        let mutated = resize_planar_lc(raw, "review-v03:lc:certificate:0", 89);
        let admission = CanonicalRawV1Admission::admit(mutated.as_bytes()).unwrap();
        let replay = replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap();
        let failure = replay.replay_failure().unwrap();
        assert!(
            matches!(failure, MixedChainReplayFailure::PlanarLcExitKernel { .. }),
            "unexpected retained failure: {failure:#?}"
        );
        assert_resource(
            failure.disposition(),
            MixedChainResourceStage::PlanarLcExit,
            MixedChainResource::PlanarLcSeriesWork,
            Some("4055552"),
            Some("4000000"),
        );
    }
}
