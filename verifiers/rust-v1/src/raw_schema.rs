//! Typed decoding for the raw-v1 planar continuation-chain wire schema.
//!
//! This module intentionally stops at the archived v0.3 parser boundary. It
//! validates record fields, JSON scalar/container classes, the segment union,
//! and the historically fixed length-two pair/interval arrays. Mathematical
//! shape, identity, sign, and value obligations belong to later replay code.

use core::fmt;
use std::collections::BTreeMap;

use num_bigint::BigInt;

use crate::{CheckedJsonBinary64, JsonNumberKind, ParsedJsonNumber, WireJsonValue};

/// Typed-schema conformance profile selected by a verifier entry point.
///
/// The raw-v1 release has no implicit default: callers must name the archived
/// v0.3-compatible parser boundary explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchemaProfile {
    V03Compatible,
}

/// A deterministic typed-schema failure at a JSON-path-like location.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaDecodeError {
    path: String,
    kind: SchemaDecodeErrorKind,
}

impl SchemaDecodeError {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub const fn kind(&self) -> &SchemaDecodeErrorKind {
        &self.kind
    }

    fn new(path: String, kind: SchemaDecodeErrorKind) -> Self {
        Self { path, kind }
    }
}

/// The class of a typed-schema failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaDecodeErrorKind {
    ExpectedObject,
    ExpectedArray,
    ExpectedString,
    ExpectedReal,
    ExpectedInteger,
    ExpectedBoolean,
    MissingField,
    UnknownField,
    WrongArrayLength { expected: usize, actual: usize },
    UnknownSegmentType { tag: String },
}

impl fmt::Display for SchemaDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "raw-v1 schema error at {}: ", self.path)?;
        match &self.kind {
            SchemaDecodeErrorKind::ExpectedObject => formatter.write_str("expected object"),
            SchemaDecodeErrorKind::ExpectedArray => formatter.write_str("expected array"),
            SchemaDecodeErrorKind::ExpectedString => formatter.write_str("expected string"),
            SchemaDecodeErrorKind::ExpectedReal => {
                formatter.write_str("expected real-number token")
            }
            SchemaDecodeErrorKind::ExpectedInteger => formatter.write_str("expected integer token"),
            SchemaDecodeErrorKind::ExpectedBoolean => formatter.write_str("expected Boolean"),
            SchemaDecodeErrorKind::MissingField => formatter.write_str("missing field"),
            SchemaDecodeErrorKind::UnknownField => formatter.write_str("unknown field"),
            SchemaDecodeErrorKind::WrongArrayLength { expected, actual } => {
                write!(
                    formatter,
                    "expected array length {expected}, found length {actual}"
                )
            }
            SchemaDecodeErrorKind::UnknownSegmentType { tag } => {
                write!(formatter, "unknown segment_type {tag:?}")
            }
        }
    }
}

impl std::error::Error for SchemaDecodeError {}

pub type Real = CheckedJsonBinary64;
pub type RealPair = [Real; 2];
pub type IntegerPair = [RawInteger; 2];
pub type RealSeries = Vec<Real>;
pub type RealVectorSeries = Vec<Vec<Real>>;
pub type RealMatrix = Vec<Vec<Real>>;
pub type RealTensor3 = Vec<Vec<Vec<Real>>>;

/// An arbitrary-precision JSON integer retaining its parsed wire proof.
///
/// The fields are private so semantic code cannot construct a value that
/// disagrees with the retained [`ParsedJsonNumber`]. No machine-size
/// narrowing occurs at the schema boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawInteger {
    parsed: ParsedJsonNumber,
    value: BigInt,
}

impl RawInteger {
    pub fn parsed(&self) -> &ParsedJsonNumber {
        &self.parsed
    }

    pub fn value(&self) -> &BigInt {
        &self.value
    }

    pub fn lexeme(&self) -> &str {
        self.parsed.lexeme()
    }

    fn from_parsed(parsed: ParsedJsonNumber) -> Self {
        let value = parsed.exact_decimal().to_integer();
        Self { parsed, value }
    }

    #[cfg(test)]
    pub(crate) fn from_parsed_for_test(parsed: ParsedJsonNumber) -> Self {
        Self::from_parsed(parsed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawPlanarChainWire {
    pub certificate_id: String,
    pub root_binding: RootBindingWire,
    pub initial_chart: OrdinaryChartWire,
    pub initial_tube: OrdinaryTubeWire,
    pub segments: Vec<SegmentWire>,
    pub requested_target_time: Real,
    pub requested_maximum_component_width: Real,
    pub schema_version: RawInteger,
    pub certificate_type: String,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentWire {
    OrdinaryBridge(Box<OrdinaryBridgeSegmentWire>),
    PlanarLcPassage(Box<PlanarLcPassageSegmentWire>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryBridgeSegmentWire {
    pub transition: OrdinaryBridgeTransitionWire,
    pub target_chart: OrdinaryChartWire,
    pub target_tube: OrdinaryTubeWire,
    pub segment_type: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcPassageSegmentWire {
    pub entry_transition: LcEntryTransitionWire,
    pub lc_chart: PlanarLcChartWire,
    pub lc_tube: PlanarLcTubeWire,
    pub exit_transition: LcExitTransitionWire,
    pub target_chart: OrdinaryChartWire,
    pub target_tube: OrdinaryTubeWire,
    pub segment_type: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootBindingWire {
    pub binding_id: String,
    pub chart_id: String,
    pub masses: Vec<Real>,
    pub initial_time: Real,
    pub chart_parameter: Real,
    pub positions: RealMatrix,
    pub velocities: RealMatrix,
    pub time_tolerance: Real,
    pub position_tolerance: Real,
    pub velocity_tolerance: Real,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChartWire {
    pub certificate_id: String,
    pub chart_id: String,
    pub chart_type: String,
    pub masses: Vec<Real>,
    pub position_coefficients: RealTensor3,
    pub velocity_coefficients: RealTensor3,
    pub parameter_interval: RealPair,
    pub physical_time_interval: RealPair,
    pub coefficient_tolerance: Real,
    pub residual_tolerance: Real,
    pub tail_bound: Real,
    pub sample_count: RawInteger,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryTubeWire {
    pub tube_id: String,
    pub chart_id: String,
    pub anchor_parameter: Real,
    pub initial_error_bound: Real,
    pub tube_radius: Real,
    pub max_defect_bound: Real,
    pub max_lipschitz_bound: Real,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryBridgeTransitionWire {
    pub transition_id: String,
    pub source_chart_id: String,
    pub source_tube_id: String,
    pub target_chart_id: String,
    pub target_tube_id: String,
    pub source_parameter: Real,
    pub target_parameter: Real,
    pub schema_version: RawInteger,
    pub record_type: String,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LcEntryTransitionWire {
    pub transition_id: String,
    pub source_chart_id: String,
    pub source_tube_id: String,
    pub target_chart_id: String,
    pub target_tube_id: String,
    pub source_right_parameter: Real,
    pub target_left_parameter: Real,
    pub schema_version: RawInteger,
    pub record_type: String,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcChartWire {
    pub certificate_id: String,
    pub chart_id: String,
    pub chart_type: String,
    pub masses: Vec<Real>,
    pub pair: IntegerPair,
    pub z_coefficients: RealVectorSeries,
    pub z_velocity_coefficients: RealVectorSeries,
    pub pair_energy_coefficients: RealSeries,
    pub binary_center_coefficients: RealVectorSeries,
    pub binary_center_velocity_coefficients: RealVectorSeries,
    pub third_offset_coefficients: RealVectorSeries,
    pub third_offset_velocity_coefficients: RealVectorSeries,
    pub physical_time_coefficients: RealSeries,
    pub parameter_interval: RealPair,
    pub physical_time_interval: RealPair,
    pub coefficient_tolerance: Real,
    pub regularized_residual_tolerance: Real,
    pub projected_residual_tolerance: Real,
    pub tail_bound: Real,
    pub sample_count: RawInteger,
    pub projection_rho_lower_bound: Real,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcTubeWire {
    pub tube_id: String,
    pub chart_id: String,
    pub anchor_parameter: Real,
    pub initial_error_bound: Real,
    pub tube_radius: Real,
    pub max_defect_bound: Real,
    pub max_lipschitz_bound: Real,
    pub require_pair_energy_constraint: bool,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LcExitTransitionWire {
    pub transition_id: String,
    pub source_chart_id: String,
    pub target_chart_id: String,
    pub source_parameter: Real,
    pub target_parameter: Real,
    pub source: String,
}

/// Decode a strict JSON AST into the archived raw-v1 typed wire model.
///
/// This function does not establish any theorem/replay obligation. In
/// particular, nonempty identifiers, constant record values, signs, array
/// replay shapes, and identifier relationships are deliberately preserved for
/// later semantic validation.
pub fn decode_raw_chain(
    value: WireJsonValue,
    profile: SchemaProfile,
) -> Result<RawPlanarChainWire, SchemaDecodeError> {
    match profile {
        SchemaProfile::V03Compatible => decode_raw_chain_v03(value, "$"),
    }
}

fn decode_raw_chain_v03(
    value: WireJsonValue,
    path: &str,
) -> Result<RawPlanarChainWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "certificate_id",
        "root_binding",
        "initial_chart",
        "initial_tube",
        "segments",
        "requested_target_time",
        "requested_maximum_component_width",
        "schema_version",
        "certificate_type",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(RawPlanarChainWire {
        certificate_id: string_field(&mut object, path, "certificate_id")?,
        root_binding: nested_field(&mut object, path, "root_binding", decode_root_binding)?,
        initial_chart: nested_field(&mut object, path, "initial_chart", decode_ordinary_chart)?,
        initial_tube: nested_field(&mut object, path, "initial_tube", decode_ordinary_tube)?,
        segments: vector_field(&mut object, path, "segments", decode_segment)?,
        requested_target_time: real_field(&mut object, path, "requested_target_time")?,
        requested_maximum_component_width: real_field(
            &mut object,
            path,
            "requested_maximum_component_width",
        )?,
        schema_version: integer_field(&mut object, path, "schema_version")?,
        certificate_type: string_field(&mut object, path, "certificate_type")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_segment(value: WireJsonValue, path: &str) -> Result<SegmentWire, SchemaDecodeError> {
    let tag = match &value {
        WireJsonValue::Object(object) => match object.get("segment_type") {
            Some(WireJsonValue::String(tag)) => tag.clone(),
            Some(_) => {
                return Err(error(
                    child(path, "segment_type"),
                    SchemaDecodeErrorKind::ExpectedString,
                ));
            }
            None => {
                return Err(error(
                    child(path, "segment_type"),
                    SchemaDecodeErrorKind::MissingField,
                ));
            }
        },
        _ => {
            return Err(error(
                path.to_owned(),
                SchemaDecodeErrorKind::ExpectedObject,
            ))
        }
    };
    match tag.as_str() {
        "ordinary_bridge_v1" => decode_ordinary_bridge_segment(value, path)
            .map(Box::new)
            .map(SegmentWire::OrdinaryBridge),
        "planar_lc_passage_v1" => decode_planar_lc_passage_segment(value, path)
            .map(Box::new)
            .map(SegmentWire::PlanarLcPassage),
        _ => Err(error(
            child(path, "segment_type"),
            SchemaDecodeErrorKind::UnknownSegmentType { tag },
        )),
    }
}

fn decode_ordinary_bridge_segment(
    value: WireJsonValue,
    path: &str,
) -> Result<OrdinaryBridgeSegmentWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &["transition", "target_chart", "target_tube", "segment_type"];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(OrdinaryBridgeSegmentWire {
        transition: nested_field(
            &mut object,
            path,
            "transition",
            decode_ordinary_bridge_transition,
        )?,
        target_chart: nested_field(&mut object, path, "target_chart", decode_ordinary_chart)?,
        target_tube: nested_field(&mut object, path, "target_tube", decode_ordinary_tube)?,
        segment_type: string_field(&mut object, path, "segment_type")?,
    })
}

fn decode_planar_lc_passage_segment(
    value: WireJsonValue,
    path: &str,
) -> Result<PlanarLcPassageSegmentWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "entry_transition",
        "lc_chart",
        "lc_tube",
        "exit_transition",
        "target_chart",
        "target_tube",
        "segment_type",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(PlanarLcPassageSegmentWire {
        entry_transition: nested_field(
            &mut object,
            path,
            "entry_transition",
            decode_lc_entry_transition,
        )?,
        lc_chart: nested_field(&mut object, path, "lc_chart", decode_planar_lc_chart)?,
        lc_tube: nested_field(&mut object, path, "lc_tube", decode_planar_lc_tube)?,
        exit_transition: nested_field(
            &mut object,
            path,
            "exit_transition",
            decode_lc_exit_transition,
        )?,
        target_chart: nested_field(&mut object, path, "target_chart", decode_ordinary_chart)?,
        target_tube: nested_field(&mut object, path, "target_tube", decode_ordinary_tube)?,
        segment_type: string_field(&mut object, path, "segment_type")?,
    })
}

fn decode_root_binding(
    value: WireJsonValue,
    path: &str,
) -> Result<RootBindingWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "binding_id",
        "chart_id",
        "masses",
        "initial_time",
        "chart_parameter",
        "positions",
        "velocities",
        "time_tolerance",
        "position_tolerance",
        "velocity_tolerance",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(RootBindingWire {
        binding_id: string_field(&mut object, path, "binding_id")?,
        chart_id: string_field(&mut object, path, "chart_id")?,
        masses: real_vector_field(&mut object, path, "masses")?,
        initial_time: real_field(&mut object, path, "initial_time")?,
        chart_parameter: real_field(&mut object, path, "chart_parameter")?,
        positions: real_matrix_field(&mut object, path, "positions")?,
        velocities: real_matrix_field(&mut object, path, "velocities")?,
        time_tolerance: real_field(&mut object, path, "time_tolerance")?,
        position_tolerance: real_field(&mut object, path, "position_tolerance")?,
        velocity_tolerance: real_field(&mut object, path, "velocity_tolerance")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_ordinary_chart(
    value: WireJsonValue,
    path: &str,
) -> Result<OrdinaryChartWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "certificate_id",
        "chart_id",
        "chart_type",
        "masses",
        "position_coefficients",
        "velocity_coefficients",
        "parameter_interval",
        "physical_time_interval",
        "coefficient_tolerance",
        "residual_tolerance",
        "tail_bound",
        "sample_count",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(OrdinaryChartWire {
        certificate_id: string_field(&mut object, path, "certificate_id")?,
        chart_id: string_field(&mut object, path, "chart_id")?,
        chart_type: string_field(&mut object, path, "chart_type")?,
        masses: real_vector_field(&mut object, path, "masses")?,
        position_coefficients: real_tensor3_field(&mut object, path, "position_coefficients")?,
        velocity_coefficients: real_tensor3_field(&mut object, path, "velocity_coefficients")?,
        parameter_interval: real_pair_field(&mut object, path, "parameter_interval")?,
        physical_time_interval: real_pair_field(&mut object, path, "physical_time_interval")?,
        coefficient_tolerance: real_field(&mut object, path, "coefficient_tolerance")?,
        residual_tolerance: real_field(&mut object, path, "residual_tolerance")?,
        tail_bound: real_field(&mut object, path, "tail_bound")?,
        sample_count: integer_field(&mut object, path, "sample_count")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_ordinary_tube(
    value: WireJsonValue,
    path: &str,
) -> Result<OrdinaryTubeWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "tube_id",
        "chart_id",
        "anchor_parameter",
        "initial_error_bound",
        "tube_radius",
        "max_defect_bound",
        "max_lipschitz_bound",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(OrdinaryTubeWire {
        tube_id: string_field(&mut object, path, "tube_id")?,
        chart_id: string_field(&mut object, path, "chart_id")?,
        anchor_parameter: real_field(&mut object, path, "anchor_parameter")?,
        initial_error_bound: real_field(&mut object, path, "initial_error_bound")?,
        tube_radius: real_field(&mut object, path, "tube_radius")?,
        max_defect_bound: real_field(&mut object, path, "max_defect_bound")?,
        max_lipschitz_bound: real_field(&mut object, path, "max_lipschitz_bound")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_ordinary_bridge_transition(
    value: WireJsonValue,
    path: &str,
) -> Result<OrdinaryBridgeTransitionWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "transition_id",
        "source_chart_id",
        "source_tube_id",
        "target_chart_id",
        "target_tube_id",
        "source_parameter",
        "target_parameter",
        "schema_version",
        "record_type",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(OrdinaryBridgeTransitionWire {
        transition_id: string_field(&mut object, path, "transition_id")?,
        source_chart_id: string_field(&mut object, path, "source_chart_id")?,
        source_tube_id: string_field(&mut object, path, "source_tube_id")?,
        target_chart_id: string_field(&mut object, path, "target_chart_id")?,
        target_tube_id: string_field(&mut object, path, "target_tube_id")?,
        source_parameter: real_field(&mut object, path, "source_parameter")?,
        target_parameter: real_field(&mut object, path, "target_parameter")?,
        schema_version: integer_field(&mut object, path, "schema_version")?,
        record_type: string_field(&mut object, path, "record_type")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_lc_entry_transition(
    value: WireJsonValue,
    path: &str,
) -> Result<LcEntryTransitionWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "transition_id",
        "source_chart_id",
        "source_tube_id",
        "target_chart_id",
        "target_tube_id",
        "source_right_parameter",
        "target_left_parameter",
        "schema_version",
        "record_type",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(LcEntryTransitionWire {
        transition_id: string_field(&mut object, path, "transition_id")?,
        source_chart_id: string_field(&mut object, path, "source_chart_id")?,
        source_tube_id: string_field(&mut object, path, "source_tube_id")?,
        target_chart_id: string_field(&mut object, path, "target_chart_id")?,
        target_tube_id: string_field(&mut object, path, "target_tube_id")?,
        source_right_parameter: real_field(&mut object, path, "source_right_parameter")?,
        target_left_parameter: real_field(&mut object, path, "target_left_parameter")?,
        schema_version: integer_field(&mut object, path, "schema_version")?,
        record_type: string_field(&mut object, path, "record_type")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_planar_lc_chart(
    value: WireJsonValue,
    path: &str,
) -> Result<PlanarLcChartWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "certificate_id",
        "chart_id",
        "chart_type",
        "masses",
        "pair",
        "z_coefficients",
        "z_velocity_coefficients",
        "pair_energy_coefficients",
        "binary_center_coefficients",
        "binary_center_velocity_coefficients",
        "third_offset_coefficients",
        "third_offset_velocity_coefficients",
        "physical_time_coefficients",
        "parameter_interval",
        "physical_time_interval",
        "coefficient_tolerance",
        "regularized_residual_tolerance",
        "projected_residual_tolerance",
        "tail_bound",
        "sample_count",
        "projection_rho_lower_bound",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(PlanarLcChartWire {
        certificate_id: string_field(&mut object, path, "certificate_id")?,
        chart_id: string_field(&mut object, path, "chart_id")?,
        chart_type: string_field(&mut object, path, "chart_type")?,
        masses: real_vector_field(&mut object, path, "masses")?,
        pair: integer_pair_field(&mut object, path, "pair")?,
        z_coefficients: real_matrix_field(&mut object, path, "z_coefficients")?,
        z_velocity_coefficients: real_matrix_field(&mut object, path, "z_velocity_coefficients")?,
        pair_energy_coefficients: real_vector_field(&mut object, path, "pair_energy_coefficients")?,
        binary_center_coefficients: real_matrix_field(
            &mut object,
            path,
            "binary_center_coefficients",
        )?,
        binary_center_velocity_coefficients: real_matrix_field(
            &mut object,
            path,
            "binary_center_velocity_coefficients",
        )?,
        third_offset_coefficients: real_matrix_field(
            &mut object,
            path,
            "third_offset_coefficients",
        )?,
        third_offset_velocity_coefficients: real_matrix_field(
            &mut object,
            path,
            "third_offset_velocity_coefficients",
        )?,
        physical_time_coefficients: real_vector_field(
            &mut object,
            path,
            "physical_time_coefficients",
        )?,
        parameter_interval: real_pair_field(&mut object, path, "parameter_interval")?,
        physical_time_interval: real_pair_field(&mut object, path, "physical_time_interval")?,
        coefficient_tolerance: real_field(&mut object, path, "coefficient_tolerance")?,
        regularized_residual_tolerance: real_field(
            &mut object,
            path,
            "regularized_residual_tolerance",
        )?,
        projected_residual_tolerance: real_field(
            &mut object,
            path,
            "projected_residual_tolerance",
        )?,
        tail_bound: real_field(&mut object, path, "tail_bound")?,
        sample_count: integer_field(&mut object, path, "sample_count")?,
        projection_rho_lower_bound: real_field(&mut object, path, "projection_rho_lower_bound")?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_planar_lc_tube(
    value: WireJsonValue,
    path: &str,
) -> Result<PlanarLcTubeWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "tube_id",
        "chart_id",
        "anchor_parameter",
        "initial_error_bound",
        "tube_radius",
        "max_defect_bound",
        "max_lipschitz_bound",
        "require_pair_energy_constraint",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(PlanarLcTubeWire {
        tube_id: string_field(&mut object, path, "tube_id")?,
        chart_id: string_field(&mut object, path, "chart_id")?,
        anchor_parameter: real_field(&mut object, path, "anchor_parameter")?,
        initial_error_bound: real_field(&mut object, path, "initial_error_bound")?,
        tube_radius: real_field(&mut object, path, "tube_radius")?,
        max_defect_bound: real_field(&mut object, path, "max_defect_bound")?,
        max_lipschitz_bound: real_field(&mut object, path, "max_lipschitz_bound")?,
        require_pair_energy_constraint: boolean_field(
            &mut object,
            path,
            "require_pair_energy_constraint",
        )?,
        source: string_field(&mut object, path, "source")?,
    })
}

fn decode_lc_exit_transition(
    value: WireJsonValue,
    path: &str,
) -> Result<LcExitTransitionWire, SchemaDecodeError> {
    const FIELDS: &[&str] = &[
        "transition_id",
        "source_chart_id",
        "target_chart_id",
        "source_parameter",
        "target_parameter",
        "source",
    ];
    let mut object = exact_object(value, path, FIELDS)?;
    Ok(LcExitTransitionWire {
        transition_id: string_field(&mut object, path, "transition_id")?,
        source_chart_id: string_field(&mut object, path, "source_chart_id")?,
        target_chart_id: string_field(&mut object, path, "target_chart_id")?,
        source_parameter: real_field(&mut object, path, "source_parameter")?,
        target_parameter: real_field(&mut object, path, "target_parameter")?,
        source: string_field(&mut object, path, "source")?,
    })
}

type Object = BTreeMap<String, WireJsonValue>;

fn exact_object(
    value: WireJsonValue,
    path: &str,
    expected_fields: &[&str],
) -> Result<Object, SchemaDecodeError> {
    let WireJsonValue::Object(object) = value else {
        return Err(error(
            path.to_owned(),
            SchemaDecodeErrorKind::ExpectedObject,
        ));
    };
    if let Some(unknown) = object
        .keys()
        .find(|field| !expected_fields.contains(&field.as_str()))
    {
        return Err(error(
            child(path, unknown),
            SchemaDecodeErrorKind::UnknownField,
        ));
    }
    for field in expected_fields {
        if !object.contains_key(*field) {
            return Err(error(
                child(path, field),
                SchemaDecodeErrorKind::MissingField,
            ));
        }
    }
    Ok(object)
}

fn take_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<WireJsonValue, SchemaDecodeError> {
    object
        .remove(field)
        .ok_or_else(|| error(child(path, field), SchemaDecodeErrorKind::MissingField))
}

fn string_field(object: &mut Object, path: &str, field: &str) -> Result<String, SchemaDecodeError> {
    expect_string(take_field(object, path, field)?, &child(path, field))
}

fn real_field(object: &mut Object, path: &str, field: &str) -> Result<Real, SchemaDecodeError> {
    expect_real(take_field(object, path, field)?, &child(path, field))
}

fn integer_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<RawInteger, SchemaDecodeError> {
    expect_integer(take_field(object, path, field)?, &child(path, field))
}

fn boolean_field(object: &mut Object, path: &str, field: &str) -> Result<bool, SchemaDecodeError> {
    expect_boolean(take_field(object, path, field)?, &child(path, field))
}

fn nested_field<T>(
    object: &mut Object,
    path: &str,
    field: &str,
    decoder: fn(WireJsonValue, &str) -> Result<T, SchemaDecodeError>,
) -> Result<T, SchemaDecodeError> {
    decoder(take_field(object, path, field)?, &child(path, field))
}

fn vector_field<T>(
    object: &mut Object,
    path: &str,
    field: &str,
    decoder: fn(WireJsonValue, &str) -> Result<T, SchemaDecodeError>,
) -> Result<Vec<T>, SchemaDecodeError> {
    decode_vector(
        take_field(object, path, field)?,
        &child(path, field),
        decoder,
    )
}

fn real_vector_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<Vec<Real>, SchemaDecodeError> {
    vector_field(object, path, field, expect_real)
}

fn real_matrix_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<RealMatrix, SchemaDecodeError> {
    vector_field(object, path, field, decode_real_vector)
}

fn real_tensor3_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<RealTensor3, SchemaDecodeError> {
    vector_field(object, path, field, decode_real_matrix)
}

fn real_pair_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<RealPair, SchemaDecodeError> {
    decode_pair(
        take_field(object, path, field)?,
        &child(path, field),
        expect_real,
    )
}

fn integer_pair_field(
    object: &mut Object,
    path: &str,
    field: &str,
) -> Result<IntegerPair, SchemaDecodeError> {
    decode_pair(
        take_field(object, path, field)?,
        &child(path, field),
        expect_integer,
    )
}

fn decode_real_vector(value: WireJsonValue, path: &str) -> Result<Vec<Real>, SchemaDecodeError> {
    decode_vector(value, path, expect_real)
}

fn decode_real_matrix(value: WireJsonValue, path: &str) -> Result<RealMatrix, SchemaDecodeError> {
    decode_vector(value, path, decode_real_vector)
}

fn decode_vector<T>(
    value: WireJsonValue,
    path: &str,
    decoder: fn(WireJsonValue, &str) -> Result<T, SchemaDecodeError>,
) -> Result<Vec<T>, SchemaDecodeError> {
    let WireJsonValue::Array(values) = value else {
        return Err(error(path.to_owned(), SchemaDecodeErrorKind::ExpectedArray));
    };
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| decoder(value, &index_path(path, index)))
        .collect()
}

fn decode_pair<T>(
    value: WireJsonValue,
    path: &str,
    decoder: fn(WireJsonValue, &str) -> Result<T, SchemaDecodeError>,
) -> Result<[T; 2], SchemaDecodeError> {
    let WireJsonValue::Array(values) = value else {
        return Err(error(path.to_owned(), SchemaDecodeErrorKind::ExpectedArray));
    };
    if values.len() != 2 {
        return Err(error(
            path.to_owned(),
            SchemaDecodeErrorKind::WrongArrayLength {
                expected: 2,
                actual: values.len(),
            },
        ));
    }
    let mut values = values.into_iter();
    let first = decoder(values.next().expect("length checked"), &index_path(path, 0))?;
    let second = decoder(values.next().expect("length checked"), &index_path(path, 1))?;
    Ok([first, second])
}

fn expect_string(value: WireJsonValue, path: &str) -> Result<String, SchemaDecodeError> {
    match value {
        WireJsonValue::String(value) => Ok(value),
        _ => Err(error(
            path.to_owned(),
            SchemaDecodeErrorKind::ExpectedString,
        )),
    }
}

fn expect_real(value: WireJsonValue, path: &str) -> Result<Real, SchemaDecodeError> {
    match value {
        WireJsonValue::Real(value) => Ok(value),
        _ => Err(error(path.to_owned(), SchemaDecodeErrorKind::ExpectedReal)),
    }
}

fn expect_integer(value: WireJsonValue, path: &str) -> Result<RawInteger, SchemaDecodeError> {
    match value {
        WireJsonValue::Integer(value) if value.kind() == JsonNumberKind::Integer => {
            Ok(RawInteger::from_parsed(value))
        }
        _ => Err(error(
            path.to_owned(),
            SchemaDecodeErrorKind::ExpectedInteger,
        )),
    }
}

fn expect_boolean(value: WireJsonValue, path: &str) -> Result<bool, SchemaDecodeError> {
    match value {
        WireJsonValue::Bool(value) => Ok(value),
        _ => Err(error(
            path.to_owned(),
            SchemaDecodeErrorKind::ExpectedBoolean,
        )),
    }
}

fn error(path: String, kind: SchemaDecodeErrorKind) -> SchemaDecodeError {
    SchemaDecodeError::new(path, kind)
}

fn child(path: &str, field: &str) -> String {
    let mut bytes = field.bytes();
    let identifier = matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
        && bytes.all(|byte| matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'));
    if identifier {
        format!("{path}.{field}")
    } else {
        format!("{path}[{field:?}]")
    }
}

fn index_path(path: &str, index: usize) -> String {
    format!("{path}[{index}]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parse_json_number_lexeme, parse_wire_json, validate_canonical_wire_json,
        DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const CONFORMANCE_SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));
    const CONFORMANCE_FAILED_REVISIT: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
    ));

    fn canonical_ast(input: &[u8]) -> WireJsonValue {
        validate_canonical_wire_json(input, DEFAULT_WIRE_JSON_LIMITS).unwrap()
    }

    fn decode(input: &[u8]) -> RawPlanarChainWire {
        decode_raw_chain(canonical_ast(input), SchemaProfile::V03Compatible).unwrap()
    }

    fn scalar(input: &[u8]) -> WireJsonValue {
        parse_wire_json(input, DEFAULT_WIRE_JSON_LIMITS).unwrap()
    }

    fn object(value: &WireJsonValue) -> &Object {
        let WireJsonValue::Object(object) = value else {
            panic!("test fixture expected object")
        };
        object
    }

    fn array(value: &WireJsonValue) -> &[WireJsonValue] {
        let WireJsonValue::Array(array) = value else {
            panic!("test fixture expected array")
        };
        array
    }

    fn field<'a>(value: &'a WireJsonValue, field: &str) -> &'a WireJsonValue {
        object(value).get(field).unwrap()
    }

    fn object_mut(value: &mut WireJsonValue) -> &mut Object {
        let WireJsonValue::Object(object) = value else {
            panic!("test fixture expected object")
        };
        object
    }

    fn array_mut(value: &mut WireJsonValue) -> &mut Vec<WireJsonValue> {
        let WireJsonValue::Array(array) = value else {
            panic!("test fixture expected array")
        };
        array
    }

    fn root_field_mut<'a>(value: &'a mut WireJsonValue, field: &str) -> &'a mut WireJsonValue {
        object_mut(value).get_mut(field).unwrap()
    }

    fn nested_field_mut<'a>(
        value: &'a mut WireJsonValue,
        record: &str,
        field: &str,
    ) -> &'a mut WireJsonValue {
        object_mut(root_field_mut(value, record))
            .get_mut(field)
            .unwrap()
    }

    fn segment_mut(value: &mut WireJsonValue, index: usize) -> &mut Object {
        object_mut(&mut array_mut(root_field_mut(value, "segments"))[index])
    }

    fn segment_nested_field_mut<'a>(
        value: &'a mut WireJsonValue,
        index: usize,
        record: &str,
        field: &str,
    ) -> &'a mut WireJsonValue {
        object_mut(segment_mut(value, index).get_mut(record).unwrap())
            .get_mut(field)
            .unwrap()
    }

    fn assert_error(
        result: Result<RawPlanarChainWire, SchemaDecodeError>,
        path: &str,
        kind: SchemaDecodeErrorKind,
    ) {
        let error = result.unwrap_err();
        assert_eq!(error.path(), path);
        assert_eq!(error.kind(), &kind);
        assert!(error.to_string().contains(path));
    }

    fn assert_conformance_record_field_counts(ast: &WireJsonValue, name: &str) {
        assert_eq!(object(ast).len(), 10, "outer record in {name}");
        assert_eq!(object(field(ast, "root_binding")).len(), 11, "{name}");
        assert_eq!(object(field(ast, "initial_chart")).len(), 13, "{name}");
        assert_eq!(object(field(ast, "initial_tube")).len(), 8, "{name}");

        let segments = array(field(ast, "segments"));
        assert_eq!(segments.len(), 5, "segment records in {name}");
        for segment in segments {
            let segment_object = object(segment);
            let WireJsonValue::String(tag) = segment_object.get("segment_type").unwrap() else {
                panic!("fixture segment_type must be a string")
            };
            assert_eq!(
                object(segment_object.get("target_chart").unwrap()).len(),
                13
            );
            assert_eq!(object(segment_object.get("target_tube").unwrap()).len(), 8);
            match tag.as_str() {
                "ordinary_bridge_v1" => {
                    assert_eq!(segment_object.len(), 4, "ordinary segment in {name}");
                    assert_eq!(
                        object(segment_object.get("transition").unwrap()).len(),
                        10,
                        "bridge transition in {name}"
                    );
                }
                "planar_lc_passage_v1" => {
                    assert_eq!(segment_object.len(), 7, "LC segment in {name}");
                    assert_eq!(
                        object(segment_object.get("entry_transition").unwrap()).len(),
                        10,
                        "LC entry in {name}"
                    );
                    assert_eq!(
                        object(segment_object.get("lc_chart").unwrap()).len(),
                        22,
                        "LC chart in {name}"
                    );
                    assert_eq!(
                        object(segment_object.get("lc_tube").unwrap()).len(),
                        9,
                        "LC tube in {name}"
                    );
                    assert_eq!(
                        object(segment_object.get("exit_transition").unwrap()).len(),
                        6,
                        "LC exit in {name}"
                    );
                }
                _ => panic!("unexpected frozen fixture tag {tag:?}"),
            }
        }
    }

    #[test]
    fn decodes_both_frozen_conformance_inputs_with_exact_record_counts_and_tags() {
        for (name, input) in [
            ("success", CONFORMANCE_SUCCESS),
            ("failed-revisit", CONFORMANCE_FAILED_REVISIT),
        ] {
            let ast = canonical_ast(input);
            assert_conformance_record_field_counts(&ast, name);
            let chain = decode_raw_chain(ast, SchemaProfile::V03Compatible).unwrap();
            assert_eq!(chain.segments.len(), 5, "{name}");

            let mut ordinary_bridges = 0;
            let mut lc_passages = 0;
            for segment in &chain.segments {
                match segment {
                    SegmentWire::OrdinaryBridge(segment) => {
                        ordinary_bridges += 1;
                        assert_eq!(segment.segment_type, "ordinary_bridge_v1", "{name}");
                    }
                    SegmentWire::PlanarLcPassage(segment) => {
                        lc_passages += 1;
                        assert_eq!(segment.segment_type, "planar_lc_passage_v1", "{name}");
                    }
                }
            }

            assert_eq!(ordinary_bridges, 1, "{name}");
            assert_eq!(lc_passages, 4, "{name}");
            // One initial chart/tube plus one target chart/tube per segment;
            // transition and LC-record counts follow their corresponding arm.
            assert_eq!(1 + ordinary_bridges + lc_passages, 6, "{name}");
            assert_eq!((ordinary_bridges, lc_passages), (1, 4), "{name}");
            assert_eq!(chain.schema_version.lexeme(), "1", "{name}");
            assert_eq!(chain.initial_chart.sample_count.lexeme(), "5", "{name}");
        }
    }

    #[test]
    fn rejects_unknown_and_missing_fields_with_deterministic_unambiguous_paths() {
        let mut unknown = canonical_ast(CONFORMANCE_SUCCESS);
        object_mut(&mut unknown).insert("bad.key[0]".into(), WireJsonValue::Null);
        assert_error(
            decode_raw_chain(unknown, SchemaProfile::V03Compatible),
            "$[\"bad.key[0]\"]",
            SchemaDecodeErrorKind::UnknownField,
        );

        let mut missing = canonical_ast(CONFORMANCE_SUCCESS);
        object_mut(&mut missing).remove("certificate_id");
        assert_error(
            decode_raw_chain(missing, SchemaProfile::V03Compatible),
            "$.certificate_id",
            SchemaDecodeErrorKind::MissingField,
        );

        let mut wrong_arm_fields = canonical_ast(CONFORMANCE_SUCCESS);
        segment_mut(&mut wrong_arm_fields, 0).insert("lc_chart".into(), WireJsonValue::Null);
        assert_error(
            decode_raw_chain(wrong_arm_fields, SchemaProfile::V03Compatible),
            "$.segments[0].lc_chart",
            SchemaDecodeErrorKind::UnknownField,
        );
    }

    #[test]
    fn rejects_wrong_scalar_and_nesting_classes_at_the_precise_path() {
        let mut wrong_string = canonical_ast(CONFORMANCE_SUCCESS);
        *root_field_mut(&mut wrong_string, "certificate_id") = WireJsonValue::Bool(false);
        assert_error(
            decode_raw_chain(wrong_string, SchemaProfile::V03Compatible),
            "$.certificate_id",
            SchemaDecodeErrorKind::ExpectedString,
        );

        let mut wrong_real = canonical_ast(CONFORMANCE_SUCCESS);
        *root_field_mut(&mut wrong_real, "requested_target_time") = scalar(b"1");
        assert_error(
            decode_raw_chain(wrong_real, SchemaProfile::V03Compatible),
            "$.requested_target_time",
            SchemaDecodeErrorKind::ExpectedReal,
        );

        let mut wrong_nested_row = canonical_ast(CONFORMANCE_SUCCESS);
        *nested_field_mut(&mut wrong_nested_row, "root_binding", "positions") =
            WireJsonValue::Array(vec![scalar(b"1.0")]);
        assert_error(
            decode_raw_chain(wrong_nested_row, SchemaProfile::V03Compatible),
            "$.root_binding.positions[0]",
            SchemaDecodeErrorKind::ExpectedArray,
        );

        let mut null_record = canonical_ast(CONFORMANCE_SUCCESS);
        *root_field_mut(&mut null_record, "initial_tube") = WireJsonValue::Null;
        assert_error(
            decode_raw_chain(null_record, SchemaProfile::V03Compatible),
            "$.initial_tube",
            SchemaDecodeErrorKind::ExpectedObject,
        );
    }

    #[test]
    fn segment_type_is_the_only_constant_string_enforced_by_schema_decoding() {
        let mut unknown_tag = canonical_ast(CONFORMANCE_SUCCESS);
        segment_mut(&mut unknown_tag, 0).insert(
            "segment_type".into(),
            WireJsonValue::String("future_segment".into()),
        );
        assert_error(
            decode_raw_chain(unknown_tag, SchemaProfile::V03Compatible),
            "$.segments[0].segment_type",
            SchemaDecodeErrorKind::UnknownSegmentType {
                tag: "future_segment".into(),
            },
        );

        let mut non_string_tag = canonical_ast(CONFORMANCE_SUCCESS);
        segment_mut(&mut non_string_tag, 0)
            .insert("segment_type".into(), WireJsonValue::Bool(true));
        assert_error(
            decode_raw_chain(non_string_tag, SchemaProfile::V03Compatible),
            "$.segments[0].segment_type",
            SchemaDecodeErrorKind::ExpectedString,
        );

        let mut semantic_mismatches = canonical_ast(CONFORMANCE_SUCCESS);
        *root_field_mut(&mut semantic_mismatches, "certificate_type") =
            WireJsonValue::String("not-the-required-type".into());
        *root_field_mut(&mut semantic_mismatches, "source") = WireJsonValue::String(String::new());
        *root_field_mut(&mut semantic_mismatches, "schema_version") = scalar(b"-99");
        *nested_field_mut(&mut semantic_mismatches, "initial_chart", "chart_type") =
            WireJsonValue::String("future_chart".into());
        *nested_field_mut(&mut semantic_mismatches, "initial_chart", "sample_count") =
            scalar(b"-5");
        *segment_nested_field_mut(&mut semantic_mismatches, 0, "transition", "record_type") =
            WireJsonValue::String("future_transition".into());
        *segment_nested_field_mut(&mut semantic_mismatches, 1, "lc_chart", "pair") =
            WireJsonValue::Array(vec![scalar(b"2"), scalar(b"0")]);

        let decoded = decode_raw_chain(semantic_mismatches, SchemaProfile::V03Compatible).unwrap();
        assert_eq!(decoded.certificate_type, "not-the-required-type");
        assert_eq!(decoded.schema_version.lexeme(), "-99");
        assert_eq!(decoded.initial_chart.chart_type, "future_chart");
        assert_eq!(decoded.initial_chart.sample_count.lexeme(), "-5");
    }

    #[test]
    fn fixed_pair_and_interval_lengths_reject_while_legacy_arrays_remain_dynamic_and_ragged() {
        let mut dynamic = canonical_ast(CONFORMANCE_SUCCESS);
        *nested_field_mut(&mut dynamic, "root_binding", "masses") =
            WireJsonValue::Array(Vec::new());
        *nested_field_mut(&mut dynamic, "root_binding", "positions") = WireJsonValue::Array(vec![
            WireJsonValue::Array(Vec::new()),
            WireJsonValue::Array(vec![scalar(b"1.0"), scalar(b"2.0"), scalar(b"3.0")]),
        ]);
        *nested_field_mut(&mut dynamic, "root_binding", "velocities") =
            WireJsonValue::Array(Vec::new());
        *nested_field_mut(&mut dynamic, "initial_chart", "position_coefficients") =
            WireJsonValue::Array(Vec::new());
        *nested_field_mut(&mut dynamic, "initial_chart", "velocity_coefficients") =
            WireJsonValue::Array(vec![
                WireJsonValue::Array(Vec::new()),
                WireJsonValue::Array(vec![WireJsonValue::Array(Vec::new())]),
            ]);
        *segment_nested_field_mut(&mut dynamic, 1, "lc_chart", "masses") =
            WireJsonValue::Array(Vec::new());
        *segment_nested_field_mut(&mut dynamic, 1, "lc_chart", "z_coefficients") =
            WireJsonValue::Array(vec![
                WireJsonValue::Array(Vec::new()),
                WireJsonValue::Array(vec![scalar(b"1.0"), scalar(b"2.0"), scalar(b"3.0")]),
            ]);
        *segment_nested_field_mut(&mut dynamic, 1, "lc_chart", "pair_energy_coefficients") =
            WireJsonValue::Array(Vec::new());

        let decoded = decode_raw_chain(dynamic, SchemaProfile::V03Compatible).unwrap();
        assert!(decoded.root_binding.masses.is_empty());
        assert_eq!(decoded.root_binding.positions[0].len(), 0);
        assert_eq!(decoded.root_binding.positions[1].len(), 3);
        assert!(decoded.initial_chart.position_coefficients.is_empty());
        assert_eq!(decoded.initial_chart.velocity_coefficients[1][0].len(), 0);

        let mut interval = canonical_ast(CONFORMANCE_SUCCESS);
        *nested_field_mut(&mut interval, "initial_chart", "parameter_interval") =
            WireJsonValue::Array(vec![scalar(b"0.0")]);
        assert_error(
            decode_raw_chain(interval, SchemaProfile::V03Compatible),
            "$.initial_chart.parameter_interval",
            SchemaDecodeErrorKind::WrongArrayLength {
                expected: 2,
                actual: 1,
            },
        );

        let mut pair = canonical_ast(CONFORMANCE_SUCCESS);
        *segment_nested_field_mut(&mut pair, 1, "lc_chart", "pair") =
            WireJsonValue::Array(vec![scalar(b"0"), scalar(b"1"), scalar(b"2")]);
        assert_error(
            decode_raw_chain(pair, SchemaProfile::V03Compatible),
            "$.segments[1].lc_chart.pair",
            SchemaDecodeErrorKind::WrongArrayLength {
                expected: 2,
                actual: 3,
            },
        );
    }

    #[test]
    fn boolean_fields_require_the_exact_json_boolean_class() {
        let chain = decode(CONFORMANCE_SUCCESS);
        let SegmentWire::PlanarLcPassage(first_lc) = &chain.segments[1] else {
            panic!("fixture expected LC passage")
        };
        assert!(!first_lc.lc_tube.require_pair_energy_constraint);

        for replacement in [scalar(b"0"), WireJsonValue::String("false".into())] {
            let mut ast = canonical_ast(CONFORMANCE_SUCCESS);
            *segment_nested_field_mut(&mut ast, 1, "lc_tube", "require_pair_energy_constraint") =
                replacement;
            assert_error(
                decode_raw_chain(ast, SchemaProfile::V03Compatible),
                "$.segments[1].lc_tube.require_pair_energy_constraint",
                SchemaDecodeErrorKind::ExpectedBoolean,
            );
        }
    }

    #[test]
    fn arbitrary_precision_integers_retain_lexeme_value_and_kind_invariant() {
        let huge = format!("-{}", "9".repeat(1000));
        let mut ast = canonical_ast(CONFORMANCE_SUCCESS);
        *root_field_mut(&mut ast, "schema_version") = scalar(huge.as_bytes());
        let decoded = decode_raw_chain(ast, SchemaProfile::V03Compatible).unwrap();
        assert_eq!(decoded.schema_version.lexeme(), huge);
        assert_eq!(decoded.schema_version.parsed().lexeme(), huge);
        assert_eq!(decoded.schema_version.value().to_string(), huge);

        let mut forged = canonical_ast(CONFORMANCE_SUCCESS);
        let misclassified = parse_json_number_lexeme("1.5", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        *root_field_mut(&mut forged, "schema_version") = WireJsonValue::Integer(misclassified);
        assert_error(
            decode_raw_chain(forged, SchemaProfile::V03Compatible),
            "$.schema_version",
            SchemaDecodeErrorKind::ExpectedInteger,
        );
    }
}
