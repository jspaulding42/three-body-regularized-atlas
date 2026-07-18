//! Strict, non-certifying semantic admission for planar Levi-Civita records.
//!
//! Every finite real is preserved as its proved binary64 dyadic. This module
//! validates composition shapes, signs, identifiers, intervals, pair order,
//! and entry bindings only. It neither checks an LC differential equation nor
//! certifies any chart, tube, entry, gauge, projection, or continuation claim.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::{
    ordinary_tube_binding_status,
    raw_schema::{LcEntryTransitionWire, PlanarLcChartWire, PlanarLcTubeWire, RealVectorSeries},
    CanonicalRawV1Admission, ExactRationalPolynomial, NumericError, OrdinaryChartInput,
    OrdinaryTubeInput, PolynomialError, RationalInterval, HARD_MAX_POLYNOMIAL_DEGREE,
    HARD_MAX_POLYNOMIAL_WORK_UNITS,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const MINIMUM_COEFFICIENT_COUNT: usize = 2;
const LC_CHART_TYPE: &str = "planar_levi_civita_binary";
const ENTRY_RECORD_TYPE: &str = "carried_planar_lc_entry_transition";
const ENTRY_SOURCE: &str = "private_carried_planar_lc_entry_v1";

pub const HARD_MAX_PLANAR_LC_SEMANTIC_COEFFICIENT_COUNT: usize = HARD_MAX_POLYNOMIAL_DEGREE + 1;
pub const HARD_MAX_PLANAR_LC_SEMANTIC_TOTAL_WORK_UNITS: usize = HARD_MAX_POLYNOMIAL_WORK_UNITS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcSemanticResource {
    CoefficientCount,
    AggregatePolynomialWork,
    PolynomialWork(PlanarLcSeriesKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcSeriesKind {
    Z,
    ZVelocity,
    PairEnergy,
    BinaryCenter,
    BinaryCenterVelocity,
    ThirdOffset,
    ThirdOffsetVelocity,
    PhysicalTime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcIntervalKind {
    Parameter,
    PhysicalTime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcSemanticError {
    EmptyString {
        field: &'static str,
    },
    UnexpectedConstant {
        field: &'static str,
        actual: String,
    },
    SchemaVersion {
        actual: BigInt,
    },
    MassCount {
        actual: usize,
    },
    NonPositiveMass {
        index: usize,
    },
    PairNotCanonical {
        left: BigInt,
        right: BigInt,
    },
    PairIndexConversion {
        field: &'static str,
        value: BigInt,
    },
    NonPositiveSampleCount {
        value: BigInt,
    },
    CoefficientCount {
        kind: PlanarLcSeriesKind,
        expected: usize,
        actual: usize,
    },
    CoefficientCountTooSmall {
        actual: usize,
    },
    CoefficientAxisCount {
        kind: PlanarLcSeriesKind,
        degree: usize,
        actual: usize,
    },
    IntervalNotStrict {
        kind: PlanarLcIntervalKind,
    },
    TubeSign {
        field: &'static str,
    },
    TubeAnchorOutsideChart,
    IdentityBinding {
        field: &'static str,
    },
    MassBinding {
        index: usize,
    },
    EndpointBinding {
        field: &'static str,
    },
    SegmentUnavailable {
        segment_index: usize,
    },
    ResourceExhausted {
        resource: PlanarLcSemanticResource,
        required: usize,
        limit: usize,
    },
    Polynomial {
        kind: PlanarLcSeriesKind,
        source: PolynomialError,
    },
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC semantic admission failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcSemanticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial { source, .. } => Some(source),
            Self::Numeric(source) => Some(source),
            _ => None,
        }
    }
}

impl From<NumericError> for PlanarLcSemanticError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcChartInput {
    certificate_id: String,
    chart_id: String,
    source: String,
    masses: [BigRational; BODY_COUNT],
    pair: [usize; 2],
    z: ExactRationalPolynomial,
    z_velocity: ExactRationalPolynomial,
    pair_energy: ExactRationalPolynomial,
    binary_center: ExactRationalPolynomial,
    binary_center_velocity: ExactRationalPolynomial,
    third_offset: ExactRationalPolynomial,
    third_offset_velocity: ExactRationalPolynomial,
    physical_time: ExactRationalPolynomial,
    parameter_interval: RationalInterval,
    physical_time_interval: RationalInterval,
    coefficient_tolerance: BigRational,
    regularized_residual_tolerance: BigRational,
    projected_residual_tolerance: BigRational,
    tail_bound: BigRational,
    projection_rho_lower_bound: BigRational,
    sample_count: BigInt,
}

impl PlanarLcChartInput {
    pub fn certificate_id(&self) -> &str {
        &self.certificate_id
    }
    pub fn chart_id(&self) -> &str {
        &self.chart_id
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn masses(&self) -> &[BigRational; BODY_COUNT] {
        &self.masses
    }
    pub const fn pair(&self) -> [usize; 2] {
        self.pair
    }
    pub fn z_polynomial(&self) -> &ExactRationalPolynomial {
        &self.z
    }
    pub fn z_velocity_polynomial(&self) -> &ExactRationalPolynomial {
        &self.z_velocity
    }
    pub fn pair_energy_polynomial(&self) -> &ExactRationalPolynomial {
        &self.pair_energy
    }
    pub fn binary_center_polynomial(&self) -> &ExactRationalPolynomial {
        &self.binary_center
    }
    pub fn binary_center_velocity_polynomial(&self) -> &ExactRationalPolynomial {
        &self.binary_center_velocity
    }
    pub fn third_offset_polynomial(&self) -> &ExactRationalPolynomial {
        &self.third_offset
    }
    pub fn third_offset_velocity_polynomial(&self) -> &ExactRationalPolynomial {
        &self.third_offset_velocity
    }
    pub fn physical_time_polynomial(&self) -> &ExactRationalPolynomial {
        &self.physical_time
    }
    pub fn parameter_interval(&self) -> &RationalInterval {
        &self.parameter_interval
    }
    pub fn physical_time_interval(&self) -> &RationalInterval {
        &self.physical_time_interval
    }
    pub fn coefficient_tolerance(&self) -> &BigRational {
        &self.coefficient_tolerance
    }
    pub fn regularized_residual_tolerance(&self) -> &BigRational {
        &self.regularized_residual_tolerance
    }
    pub fn projected_residual_tolerance(&self) -> &BigRational {
        &self.projected_residual_tolerance
    }
    pub fn tail_bound(&self) -> &BigRational {
        &self.tail_bound
    }
    pub fn projection_rho_lower_bound(&self) -> &BigRational {
        &self.projection_rho_lower_bound
    }
    pub fn sample_count(&self) -> &BigInt {
        &self.sample_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcTubeInput {
    tube_id: String,
    chart_id: String,
    source: String,
    anchor_parameter: BigRational,
    initial_error_bound: BigRational,
    tube_radius: BigRational,
    maximum_defect_bound: BigRational,
    maximum_lipschitz_bound: BigRational,
    require_pair_energy_constraint: bool,
}

impl PlanarLcTubeInput {
    pub fn tube_id(&self) -> &str {
        &self.tube_id
    }
    pub fn chart_id(&self) -> &str {
        &self.chart_id
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn anchor_parameter(&self) -> &BigRational {
        &self.anchor_parameter
    }
    pub fn initial_error_bound(&self) -> &BigRational {
        &self.initial_error_bound
    }
    pub fn tube_radius(&self) -> &BigRational {
        &self.tube_radius
    }
    pub fn maximum_defect_bound(&self) -> &BigRational {
        &self.maximum_defect_bound
    }
    pub fn maximum_lipschitz_bound(&self) -> &BigRational {
        &self.maximum_lipschitz_bound
    }
    pub const fn require_pair_energy_constraint(&self) -> bool {
        self.require_pair_energy_constraint
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcEntryInput {
    transition_id: String,
    source_chart_id: String,
    source_tube_id: String,
    source_right_parameter: BigRational,
    target_left_parameter: BigRational,
    target_chart: PlanarLcChartInput,
    target_tube: PlanarLcTubeInput,
}

impl PlanarLcEntryInput {
    pub fn transition_id(&self) -> &str {
        &self.transition_id
    }
    pub fn source_chart_id(&self) -> &str {
        &self.source_chart_id
    }
    pub fn source_tube_id(&self) -> &str {
        &self.source_tube_id
    }
    pub fn source_right_parameter(&self) -> &BigRational {
        &self.source_right_parameter
    }
    pub fn target_left_parameter(&self) -> &BigRational {
        &self.target_left_parameter
    }
    pub fn target_chart(&self) -> &PlanarLcChartInput {
        &self.target_chart
    }
    pub fn target_tube(&self) -> &PlanarLcTubeInput {
        &self.target_tube
    }
}

pub fn planar_lc_chart_input_from_wire(
    wire: &PlanarLcChartWire,
) -> Result<PlanarLcChartInput, PlanarLcSemanticError> {
    for (value, field) in [
        (&wire.certificate_id, "certificate_id"),
        (&wire.chart_id, "chart_id"),
        (&wire.source, "source"),
    ] {
        require_nonempty(value, field)?;
    }
    if wire.chart_type != LC_CHART_TYPE {
        return Err(PlanarLcSemanticError::UnexpectedConstant {
            field: "chart_type",
            actual: wire.chart_type.clone(),
        });
    }
    if wire.masses.len() != BODY_COUNT {
        return Err(PlanarLcSemanticError::MassCount {
            actual: wire.masses.len(),
        });
    }
    let masses = std::array::from_fn(|index| wire.masses[index].binary64_rational().clone());
    for (index, mass) in masses.iter().enumerate() {
        if mass.numer().sign() != Sign::Plus {
            return Err(PlanarLcSemanticError::NonPositiveMass { index });
        }
    }
    let left = wire.pair[0].value().clone();
    let right = wire.pair[1].value().clone();
    if left.sign() == Sign::Minus
        || right.sign() == Sign::Minus
        || left >= right
        || right >= BigInt::from(BODY_COUNT)
    {
        return Err(PlanarLcSemanticError::PairNotCanonical { left, right });
    }
    let left_index = left
        .to_usize()
        .ok_or_else(|| PlanarLcSemanticError::PairIndexConversion {
            field: "pair[0]",
            value: left.clone(),
        })?;
    let right_index =
        right
            .to_usize()
            .ok_or_else(|| PlanarLcSemanticError::PairIndexConversion {
                field: "pair[1]",
                value: right.clone(),
            })?;
    let pair = [left_index, right_index];
    let count = wire.z_coefficients.len();
    if count < MINIMUM_COEFFICIENT_COUNT {
        return Err(PlanarLcSemanticError::CoefficientCountTooSmall { actual: count });
    }
    if count > HARD_MAX_PLANAR_LC_SEMANTIC_COEFFICIENT_COUNT {
        return Err(PlanarLcSemanticError::ResourceExhausted {
            resource: PlanarLcSemanticResource::CoefficientCount,
            required: count,
            limit: HARD_MAX_PLANAR_LC_SEMANTIC_COEFFICIENT_COUNT,
        });
    }
    let aggregate_work = count
        .checked_mul(6 * PLANE_DIMENSION + 2)
        .and_then(|value| value.checked_mul(4))
        .unwrap_or(usize::MAX);
    if aggregate_work > HARD_MAX_PLANAR_LC_SEMANTIC_TOTAL_WORK_UNITS {
        return Err(PlanarLcSemanticError::ResourceExhausted {
            resource: PlanarLcSemanticResource::AggregatePolynomialWork,
            required: aggregate_work,
            limit: HARD_MAX_PLANAR_LC_SEMANTIC_TOTAL_WORK_UNITS,
        });
    }
    let vectors = [
        (PlanarLcSeriesKind::Z, &wire.z_coefficients),
        (PlanarLcSeriesKind::ZVelocity, &wire.z_velocity_coefficients),
        (
            PlanarLcSeriesKind::BinaryCenter,
            &wire.binary_center_coefficients,
        ),
        (
            PlanarLcSeriesKind::BinaryCenterVelocity,
            &wire.binary_center_velocity_coefficients,
        ),
        (
            PlanarLcSeriesKind::ThirdOffset,
            &wire.third_offset_coefficients,
        ),
        (
            PlanarLcSeriesKind::ThirdOffsetVelocity,
            &wire.third_offset_velocity_coefficients,
        ),
    ];
    for (kind, series) in vectors {
        validate_vector_series(series, kind, count)?;
    }
    for (kind, actual) in [
        (
            PlanarLcSeriesKind::PairEnergy,
            wire.pair_energy_coefficients.len(),
        ),
        (
            PlanarLcSeriesKind::PhysicalTime,
            wire.physical_time_coefficients.len(),
        ),
    ] {
        if actual != count {
            return Err(PlanarLcSemanticError::CoefficientCount {
                kind,
                expected: count,
                actual,
            });
        }
    }
    let sample_count = wire.sample_count.value().clone();
    if sample_count.sign() != Sign::Plus {
        return Err(PlanarLcSemanticError::NonPositiveSampleCount {
            value: sample_count,
        });
    }
    Ok(PlanarLcChartInput {
        certificate_id: wire.certificate_id.clone(),
        chart_id: wire.chart_id.clone(),
        source: wire.source.clone(),
        masses,
        pair,
        z: vector_polynomial(&wire.z_coefficients, PlanarLcSeriesKind::Z)?,
        z_velocity: vector_polynomial(
            &wire.z_velocity_coefficients,
            PlanarLcSeriesKind::ZVelocity,
        )?,
        pair_energy: scalar_polynomial(
            &wire.pair_energy_coefficients,
            PlanarLcSeriesKind::PairEnergy,
        )?,
        binary_center: vector_polynomial(
            &wire.binary_center_coefficients,
            PlanarLcSeriesKind::BinaryCenter,
        )?,
        binary_center_velocity: vector_polynomial(
            &wire.binary_center_velocity_coefficients,
            PlanarLcSeriesKind::BinaryCenterVelocity,
        )?,
        third_offset: vector_polynomial(
            &wire.third_offset_coefficients,
            PlanarLcSeriesKind::ThirdOffset,
        )?,
        third_offset_velocity: vector_polynomial(
            &wire.third_offset_velocity_coefficients,
            PlanarLcSeriesKind::ThirdOffsetVelocity,
        )?,
        physical_time: scalar_polynomial(
            &wire.physical_time_coefficients,
            PlanarLcSeriesKind::PhysicalTime,
        )?,
        parameter_interval: strict_interval(
            &wire.parameter_interval,
            PlanarLcIntervalKind::Parameter,
        )?,
        physical_time_interval: strict_interval(
            &wire.physical_time_interval,
            PlanarLcIntervalKind::PhysicalTime,
        )?,
        coefficient_tolerance: wire.coefficient_tolerance.binary64_rational().clone(),
        regularized_residual_tolerance: wire
            .regularized_residual_tolerance
            .binary64_rational()
            .clone(),
        projected_residual_tolerance: wire
            .projected_residual_tolerance
            .binary64_rational()
            .clone(),
        tail_bound: wire.tail_bound.binary64_rational().clone(),
        projection_rho_lower_bound: wire.projection_rho_lower_bound.binary64_rational().clone(),
        sample_count,
    })
}

pub fn planar_lc_tube_input_from_wire(
    wire: &PlanarLcTubeWire,
    chart: &PlanarLcChartInput,
) -> Result<PlanarLcTubeInput, PlanarLcSemanticError> {
    for (value, field) in [
        (&wire.tube_id, "tube_id"),
        (&wire.chart_id, "chart_id"),
        (&wire.source, "source"),
    ] {
        require_nonempty(value, field)?;
    }
    if wire.chart_id != chart.chart_id {
        return Err(PlanarLcSemanticError::IdentityBinding {
            field: "tube.chart_id",
        });
    }
    let anchor = wire.anchor_parameter.binary64_rational().clone();
    if anchor < *chart.parameter_interval.lower() || anchor > *chart.parameter_interval.upper() {
        return Err(PlanarLcSemanticError::TubeAnchorOutsideChart);
    }
    for (value, field, strictly_positive) in [
        (
            wire.initial_error_bound.binary64_rational(),
            "initial_error_bound",
            false,
        ),
        (wire.tube_radius.binary64_rational(), "tube_radius", true),
        (
            wire.max_defect_bound.binary64_rational(),
            "max_defect_bound",
            false,
        ),
        (
            wire.max_lipschitz_bound.binary64_rational(),
            "max_lipschitz_bound",
            false,
        ),
    ] {
        if value.numer().sign() == Sign::Minus
            || (strictly_positive && value.numer().sign() != Sign::Plus)
        {
            return Err(PlanarLcSemanticError::TubeSign { field });
        }
    }
    Ok(PlanarLcTubeInput {
        tube_id: wire.tube_id.clone(),
        chart_id: wire.chart_id.clone(),
        source: wire.source.clone(),
        anchor_parameter: anchor,
        initial_error_bound: wire.initial_error_bound.binary64_rational().clone(),
        tube_radius: wire.tube_radius.binary64_rational().clone(),
        maximum_defect_bound: wire.max_defect_bound.binary64_rational().clone(),
        maximum_lipschitz_bound: wire.max_lipschitz_bound.binary64_rational().clone(),
        require_pair_energy_constraint: wire.require_pair_energy_constraint,
    })
}

fn planar_lc_entry_input_from_wire(
    transition: &LcEntryTransitionWire,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
    target_chart_wire: &PlanarLcChartWire,
    target_tube_wire: &PlanarLcTubeWire,
) -> Result<PlanarLcEntryInput, PlanarLcSemanticError> {
    for (value, field) in [
        (&transition.transition_id, "transition_id"),
        (&transition.source_chart_id, "source_chart_id"),
        (&transition.source_tube_id, "source_tube_id"),
        (&transition.target_chart_id, "target_chart_id"),
        (&transition.target_tube_id, "target_tube_id"),
        (&transition.source, "transition.source"),
    ] {
        require_nonempty(value, field)?;
    }
    if transition.record_type != ENTRY_RECORD_TYPE {
        return Err(PlanarLcSemanticError::UnexpectedConstant {
            field: "record_type",
            actual: transition.record_type.clone(),
        });
    }
    if transition.source != ENTRY_SOURCE {
        return Err(PlanarLcSemanticError::UnexpectedConstant {
            field: "transition.source",
            actual: transition.source.clone(),
        });
    }
    if transition.schema_version.value() != &BigInt::from(1) {
        return Err(PlanarLcSemanticError::SchemaVersion {
            actual: transition.schema_version.value().clone(),
        });
    }
    let target_chart = planar_lc_chart_input_from_wire(target_chart_wire)?;
    let target_tube = planar_lc_tube_input_from_wire(target_tube_wire, &target_chart)?;
    if !ordinary_tube_binding_status(source_tube, source_chart).bound() {
        return Err(PlanarLcSemanticError::IdentityBinding {
            field: "source_tube",
        });
    }
    for (matches, field) in [
        (
            transition.source_chart_id == source_chart.chart_id(),
            "source_chart_id",
        ),
        (
            transition.source_tube_id == source_tube.tube_id(),
            "source_tube_id",
        ),
        (
            transition.target_chart_id == target_chart.chart_id(),
            "target_chart_id",
        ),
        (
            transition.target_tube_id == target_tube.tube_id(),
            "target_tube_id",
        ),
    ] {
        if !matches {
            return Err(PlanarLcSemanticError::IdentityBinding { field });
        }
    }
    for index in 0..BODY_COUNT {
        if source_chart.masses()[index] != target_chart.masses[index] {
            return Err(PlanarLcSemanticError::MassBinding { index });
        }
    }
    let source_right = transition
        .source_right_parameter
        .binary64_rational()
        .clone();
    let target_left = transition.target_left_parameter.binary64_rational().clone();
    if source_right != *source_chart.parameter_interval().upper() {
        return Err(PlanarLcSemanticError::EndpointBinding {
            field: "source_right_parameter",
        });
    }
    if target_left != *target_chart.parameter_interval.lower()
        || target_left != target_tube.anchor_parameter
    {
        return Err(PlanarLcSemanticError::EndpointBinding {
            field: "target_left_parameter",
        });
    }
    Ok(PlanarLcEntryInput {
        transition_id: transition.transition_id.clone(),
        source_chart_id: transition.source_chart_id.clone(),
        source_tube_id: transition.source_tube_id.clone(),
        source_right_parameter: source_right,
        target_left_parameter: target_left,
        target_chart,
        target_tube,
    })
}

/// Bind semantic LC entry admission to a segment borrowed from an opaque
/// canonical-byte/schema admission. This still proves no top-level replay
/// obligation and performs no hash or certificate check.
pub fn planar_lc_entry_input_from_admission(
    admission: &CanonicalRawV1Admission,
    segment_index: usize,
    source_chart: &OrdinaryChartInput,
    source_tube: &OrdinaryTubeInput,
) -> Result<PlanarLcEntryInput, PlanarLcSemanticError> {
    let segment = admission
        .planar_lc_passage(segment_index)
        .ok_or(PlanarLcSemanticError::SegmentUnavailable { segment_index })?;
    planar_lc_entry_input_from_wire(
        &segment.entry_transition,
        source_chart,
        source_tube,
        &segment.lc_chart,
        &segment.lc_tube,
    )
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), PlanarLcSemanticError> {
    if value.is_empty() {
        return Err(PlanarLcSemanticError::EmptyString { field });
    }
    Ok(())
}

fn validate_vector_series(
    series: &RealVectorSeries,
    kind: PlanarLcSeriesKind,
    expected: usize,
) -> Result<(), PlanarLcSemanticError> {
    if series.len() != expected {
        return Err(PlanarLcSemanticError::CoefficientCount {
            kind,
            expected,
            actual: series.len(),
        });
    }
    for (degree, row) in series.iter().enumerate() {
        if row.len() != PLANE_DIMENSION {
            return Err(PlanarLcSemanticError::CoefficientAxisCount {
                kind,
                degree,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

fn vector_polynomial(
    series: &RealVectorSeries,
    kind: PlanarLcSeriesKind,
) -> Result<ExactRationalPolynomial, PlanarLcSemanticError> {
    let coefficients = series
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.binary64_rational().clone())
                .collect()
        })
        .collect();
    ExactRationalPolynomial::from_degree_major_coefficients(coefficients)
        .map_err(|source| map_polynomial(kind, source))
}

fn scalar_polynomial(
    series: &[crate::raw_schema::Real],
    kind: PlanarLcSeriesKind,
) -> Result<ExactRationalPolynomial, PlanarLcSemanticError> {
    let coefficients = series
        .iter()
        .map(|value| vec![value.binary64_rational().clone()])
        .collect();
    ExactRationalPolynomial::from_degree_major_coefficients(coefficients)
        .map_err(|source| map_polynomial(kind, source))
}

fn map_polynomial(kind: PlanarLcSeriesKind, source: PolynomialError) -> PlanarLcSemanticError {
    match source {
        PolynomialError::DegreeLimitExceeded { degree, limit } => {
            PlanarLcSemanticError::ResourceExhausted {
                resource: PlanarLcSemanticResource::CoefficientCount,
                required: degree.saturating_add(1),
                limit: limit.saturating_add(1),
            }
        }
        PolynomialError::WorkLimitExceeded { required, limit } => {
            PlanarLcSemanticError::ResourceExhausted {
                resource: PlanarLcSemanticResource::PolynomialWork(kind),
                required,
                limit,
            }
        }
        _ => PlanarLcSemanticError::Polynomial { kind, source },
    }
}

fn strict_interval(
    wire: &[crate::raw_schema::Real; 2],
    kind: PlanarLcIntervalKind,
) -> Result<RationalInterval, PlanarLcSemanticError> {
    let lower = wire[0].binary64_rational().clone();
    let upper = wire[1].binary64_rational().clone();
    if lower >= upper {
        return Err(PlanarLcSemanticError::IntervalNotStrict { kind });
    }
    Ok(RationalInterval::new(lower, upper)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, parse_wire_json,
        raw_schema::{decode_raw_chain, SchemaProfile, SegmentWire},
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn first_entry() -> (
        crate::raw_schema::OrdinaryBridgeSegmentWire,
        crate::raw_schema::PlanarLcPassageSegmentWire,
    ) {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let SegmentWire::OrdinaryBridge(ordinary) = &chain.segments[0] else {
            panic!()
        };
        let SegmentWire::PlanarLcPassage(lc) = &chain.segments[1] else {
            panic!()
        };
        ((**ordinary).clone(), (**lc).clone())
    }

    #[test]
    fn canonical_first_entry_is_strictly_admitted_without_certification() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let SegmentWire::OrdinaryBridge(ordinary) = &admission.wire().segments[0] else {
            panic!()
        };
        let lc = admission.planar_lc_passage(1).unwrap();
        let source_chart = ordinary_chart_input_from_wire(&ordinary.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&ordinary.target_tube);
        let admitted =
            planar_lc_entry_input_from_admission(&admission, 1, &source_chart, &source_tube)
                .unwrap();
        assert_eq!(admitted.target_chart().pair(), [0, 1]);
        assert_eq!(
            admitted.target_chart().z_polynomial().coefficient_count(),
            lc.lc_chart.z_coefficients.len()
        );
        assert_eq!(admitted.target_chart().z_polynomial().dimension(), 2);
        assert_eq!(
            &admitted
                .target_chart()
                .z_polynomial()
                .coefficients_by_degree()[0][0],
            lc.lc_chart.z_coefficients[0][0].binary64_rational()
        );
        assert_eq!(
            admitted.target_chart().coefficient_tolerance(),
            lc.lc_chart.coefficient_tolerance.binary64_rational()
        );
        assert_eq!(
            admitted.target_chart().pair_energy_polynomial().dimension(),
            1
        );
        assert_eq!(admitted.transition_id(), lc.entry_transition.transition_id);
    }

    #[test]
    fn every_canonical_pair_is_admitted_from_its_carried_source() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let chain = admission.wire();
        let mut current_chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
        let mut current_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
        let mut pairs = Vec::new();
        for (index, segment) in chain.segments.iter().enumerate() {
            match segment {
                SegmentWire::OrdinaryBridge(segment) => {
                    current_chart = ordinary_chart_input_from_wire(&segment.target_chart).unwrap();
                    current_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                }
                SegmentWire::PlanarLcPassage(segment) => {
                    let entry = planar_lc_entry_input_from_admission(
                        &admission,
                        index,
                        &current_chart,
                        &current_tube,
                    )
                    .unwrap();
                    pairs.push(entry.target_chart().pair());
                    current_chart = ordinary_chart_input_from_wire(&segment.target_chart).unwrap();
                    current_tube = ordinary_tube_input_from_wire(&segment.target_tube);
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs, vec![[0, 1], [0, 2], [1, 2]]);
    }

    #[test]
    fn finite_untrusted_chart_tolerances_preserve_their_signs() {
        let (_, lc) = first_entry();
        let negative = checked_real_binary64_from_json("-1.0", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        let mut admitted = Vec::new();
        for field in 0..5 {
            let mut chart = lc.lc_chart.clone();
            match field {
                0 => chart.coefficient_tolerance = negative.clone(),
                1 => chart.regularized_residual_tolerance = negative.clone(),
                2 => chart.projected_residual_tolerance = negative.clone(),
                3 => chart.tail_bound = negative.clone(),
                4 => chart.projection_rho_lower_bound = negative.clone(),
                _ => unreachable!(),
            }
            admitted.push(planar_lc_chart_input_from_wire(&chart).unwrap());
        }
        assert!(admitted[0].coefficient_tolerance().numer().sign() == Sign::Minus);
        assert!(admitted[1].regularized_residual_tolerance().numer().sign() == Sign::Minus);
        assert!(admitted[2].projected_residual_tolerance().numer().sign() == Sign::Minus);
        assert!(admitted[3].tail_bound().numer().sign() == Sign::Minus);
        assert!(admitted[4].projection_rho_lower_bound().numer().sign() == Sign::Minus);
    }

    #[test]
    fn pair_shape_tube_sign_and_entry_identity_fail_before_claims() {
        let (ordinary, lc) = first_entry();
        let mut reversed = lc.lc_chart.clone();
        reversed.pair.swap(0, 1);
        assert!(matches!(
            planar_lc_chart_input_from_wire(&reversed),
            Err(PlanarLcSemanticError::PairNotCanonical { .. })
        ));

        let chart = planar_lc_chart_input_from_wire(&lc.lc_chart).unwrap();
        let mut bad_tube = lc.lc_tube.clone();
        bad_tube.tube_radius =
            checked_real_binary64_from_json("0.0", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        assert!(matches!(
            planar_lc_tube_input_from_wire(&bad_tube, &chart),
            Err(PlanarLcSemanticError::TubeSign {
                field: "tube_radius"
            })
        ));

        let source_chart = ordinary_chart_input_from_wire(&ordinary.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&ordinary.target_tube);
        let mut transition = lc.entry_transition.clone();
        transition.target_chart_id.push_str(":wrong");
        assert!(matches!(
            planar_lc_entry_input_from_wire(
                &transition,
                &source_chart,
                &source_tube,
                &lc.lc_chart,
                &lc.lc_tube
            ),
            Err(PlanarLcSemanticError::IdentityBinding {
                field: "target_chart_id"
            })
        ));
    }

    #[test]
    fn mismatched_vector_shapes_and_strict_intervals_are_rejected() {
        let (_, lc) = first_entry();
        let mut shape = lc.lc_chart.clone();
        shape.z_coefficients[0].pop();
        assert!(matches!(
            planar_lc_chart_input_from_wire(&shape),
            Err(PlanarLcSemanticError::CoefficientAxisCount { .. })
        ));
        let mut interval = lc.lc_chart.clone();
        interval.parameter_interval[1] = interval.parameter_interval[0].clone();
        assert!(matches!(
            planar_lc_chart_input_from_wire(&interval),
            Err(PlanarLcSemanticError::IntervalNotStrict {
                kind: PlanarLcIntervalKind::Parameter
            })
        ));
        let mut vector_count = lc.lc_chart.clone();
        vector_count.z_velocity_coefficients.pop();
        assert!(matches!(
            planar_lc_chart_input_from_wire(&vector_count),
            Err(PlanarLcSemanticError::CoefficientCount {
                kind: PlanarLcSeriesKind::ZVelocity,
                ..
            })
        ));
        let mut scalar_count = lc.lc_chart.clone();
        scalar_count.pair_energy_coefficients.pop();
        assert!(matches!(
            planar_lc_chart_input_from_wire(&scalar_count),
            Err(PlanarLcSemanticError::CoefficientCount {
                kind: PlanarLcSeriesKind::PairEnergy,
                ..
            })
        ));
    }

    #[test]
    fn transition_constants_masses_endpoints_and_resources_fail_closed() {
        let (ordinary, lc) = first_entry();
        let source_chart = ordinary_chart_input_from_wire(&ordinary.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&ordinary.target_tube);
        for (record_type, source, expected_field) in [
            ("wrong", ENTRY_SOURCE, "record_type"),
            (ENTRY_RECORD_TYPE, "wrong", "transition.source"),
        ] {
            let mut transition = lc.entry_transition.clone();
            transition.record_type = record_type.to_owned();
            transition.source = source.to_owned();
            assert!(matches!(
                planar_lc_entry_input_from_wire(
                    &transition,
                    &source_chart,
                    &source_tube,
                    &lc.lc_chart,
                    &lc.lc_tube
                ),
                Err(PlanarLcSemanticError::UnexpectedConstant { field, .. }) if field == expected_field
            ));
        }

        let changed = std::str::from_utf8(SUCCESS).unwrap().replacen(
            "\"record_type\":\"carried_planar_lc_entry_transition\",\"schema_version\":1",
            "\"record_type\":\"carried_planar_lc_entry_transition\",\"schema_version\":2",
            1,
        );
        let syntax = parse_wire_json(changed.as_bytes(), DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let changed_chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let SegmentWire::PlanarLcPassage(changed_lc) = &changed_chain.segments[1] else {
            panic!()
        };
        assert!(matches!(
            planar_lc_entry_input_from_wire(
                &changed_lc.entry_transition,
                &source_chart,
                &source_tube,
                &changed_lc.lc_chart,
                &changed_lc.lc_tube
            ),
            Err(PlanarLcSemanticError::SchemaVersion { .. })
        ));

        let mut mass = lc.lc_chart.clone();
        mass.masses[0] =
            checked_real_binary64_from_json("0.11", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        assert!(matches!(
            planar_lc_entry_input_from_wire(
                &lc.entry_transition,
                &source_chart,
                &source_tube,
                &mass,
                &lc.lc_tube
            ),
            Err(PlanarLcSemanticError::MassBinding { index: 0 })
        ));

        let mut endpoint = lc.entry_transition.clone();
        endpoint.source_right_parameter =
            checked_real_binary64_from_json("0.0", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        assert!(matches!(
            planar_lc_entry_input_from_wire(
                &endpoint,
                &source_chart,
                &source_tube,
                &lc.lc_chart,
                &lc.lc_tube
            ),
            Err(PlanarLcSemanticError::EndpointBinding {
                field: "source_right_parameter"
            })
        ));

        let mut resource = lc.lc_chart.clone();
        let row = resource.z_coefficients[0].clone();
        resource
            .z_coefficients
            .resize(HARD_MAX_PLANAR_LC_SEMANTIC_TOTAL_WORK_UNITS / 56 + 1, row);
        assert!(matches!(
            planar_lc_chart_input_from_wire(&resource),
            Err(PlanarLcSemanticError::ResourceExhausted {
                resource: PlanarLcSemanticResource::AggregatePolynomialWork,
                ..
            })
        ));
    }
}
