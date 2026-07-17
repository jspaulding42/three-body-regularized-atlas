//! Non-certifying semantic inputs for ordinary planar chart and tube records.
//!
//! Every real field is imported from its proved binary64 bits as an exact
//! dyadic rational.  This layer validates only the finite chart/tube schema
//! used by ordinary bridge and tube arguments; it does not accept a primitive
//! Taylor recurrence, residual claim, or certificate status.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;

use crate::{
    raw_schema::{OrdinaryChartWire, OrdinaryTubeWire, RealTensor3},
    ExactRationalPolynomial, NumericError, PolynomialError, RationalInterval,
    HARD_MAX_POLYNOMIAL_DEGREE, HARD_MAX_POLYNOMIAL_WORK_UNITS,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
const CONFIGURATION_DIMENSION: usize = BODY_COUNT * PLANE_DIMENSION;
const MINIMUM_COEFFICIENT_COUNT: usize = 2;
const ORDINARY_CHART_TYPE: &str = "ordinary_taylor";

/// Maximum raw monomial coefficient count admitted before nested shape scans.
pub const HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT: usize = HARD_MAX_POLYNOMIAL_DEGREE + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinaryCoefficientKind {
    Position,
    Velocity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinaryIntervalKind {
    Parameter,
    PhysicalTime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinarySemanticResource {
    CoefficientCount,
    PositionPolynomialWork,
    VelocityPolynomialWork,
}

/// Typed semantic or implementation-resource failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrdinarySemanticError {
    EmptyString {
        field: &'static str,
    },
    UnexpectedChartType {
        actual: String,
    },
    MassCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonPositiveMass {
        index: usize,
    },
    NonPositiveSampleCount {
        value: BigInt,
    },
    CoefficientCountTooSmall {
        minimum: usize,
        actual: usize,
    },
    CoefficientCountMismatch {
        position: usize,
        velocity: usize,
    },
    CoefficientBodyCountMismatch {
        coefficients: OrdinaryCoefficientKind,
        degree_index: usize,
        expected: usize,
        actual: usize,
    },
    CoefficientAxisCountMismatch {
        coefficients: OrdinaryCoefficientKind,
        degree_index: usize,
        body_index: usize,
        expected: usize,
        actual: usize,
    },
    IntervalNotStrictlyIncreasing {
        interval: OrdinaryIntervalKind,
    },
    Polynomial {
        coefficients: OrdinaryCoefficientKind,
        source: PolynomialError,
    },
    Numeric(NumericError),
    ResourceExhausted {
        resource: OrdinarySemanticResource,
        required: usize,
        limit: usize,
    },
}

impl fmt::Display for OrdinarySemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyString { field } => {
                write!(formatter, "ordinary semantic field {field} is empty")
            }
            Self::UnexpectedChartType { actual } => write!(
                formatter,
                "ordinary chart type is {actual:?}, expected {ORDINARY_CHART_TYPE:?}"
            ),
            Self::MassCountMismatch { expected, actual } => write!(
                formatter,
                "ordinary chart has {actual} masses, expected {expected}"
            ),
            Self::NonPositiveMass { index } => {
                write!(formatter, "ordinary chart mass[{index}] is not strictly positive")
            }
            Self::NonPositiveSampleCount { value } => write!(
                formatter,
                "ordinary chart sample count {value} is not strictly positive"
            ),
            Self::CoefficientCountTooSmall { minimum, actual } => write!(
                formatter,
                "ordinary chart has {actual} coefficient rows, minimum is {minimum}"
            ),
            Self::CoefficientCountMismatch { position, velocity } => write!(
                formatter,
                "ordinary position/velocity coefficient counts differ: {position} versus {velocity}"
            ),
            Self::CoefficientBodyCountMismatch {
                coefficients,
                degree_index,
                expected,
                actual,
            } => write!(
                formatter,
                "ordinary {coefficients:?} coefficient row {degree_index} has {actual} bodies, expected {expected}"
            ),
            Self::CoefficientAxisCountMismatch {
                coefficients,
                degree_index,
                body_index,
                expected,
                actual,
            } => write!(
                formatter,
                "ordinary {coefficients:?} coefficient row {degree_index}, body {body_index} has {actual} axes, expected {expected}"
            ),
            Self::IntervalNotStrictlyIncreasing { interval } => write!(
                formatter,
                "ordinary {interval:?} interval is not strictly increasing"
            ),
            Self::Polynomial {
                coefficients,
                source,
            } => write!(
                formatter,
                "ordinary {coefficients:?} polynomial construction failed: {source}"
            ),
            Self::Numeric(source) => write!(formatter, "ordinary semantic numeric failure: {source}"),
            Self::ResourceExhausted {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "ordinary semantic {resource:?} resource requires {required}, limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for OrdinarySemanticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial { source, .. } => Some(source),
            Self::Numeric(source) => Some(source),
            Self::EmptyString { .. }
            | Self::UnexpectedChartType { .. }
            | Self::MassCountMismatch { .. }
            | Self::NonPositiveMass { .. }
            | Self::NonPositiveSampleCount { .. }
            | Self::CoefficientCountTooSmall { .. }
            | Self::CoefficientCountMismatch { .. }
            | Self::CoefficientBodyCountMismatch { .. }
            | Self::CoefficientAxisCountMismatch { .. }
            | Self::IntervalNotStrictlyIncreasing { .. }
            | Self::ResourceExhausted { .. } => None,
        }
    }
}

impl From<NumericError> for OrdinarySemanticError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

/// Validated finite ordinary chart input.  This type makes no certification
/// claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryChartInput {
    certificate_id: String,
    chart_id: String,
    source: String,
    masses: [BigRational; BODY_COUNT],
    position_polynomial: ExactRationalPolynomial,
    velocity_polynomial: ExactRationalPolynomial,
    parameter_interval: RationalInterval,
    physical_time_interval: RationalInterval,
    coefficient_tolerance: BigRational,
    residual_tolerance: BigRational,
    tail_bound: BigRational,
    sample_count: BigInt,
}

impl OrdinaryChartInput {
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

    pub fn position_polynomial(&self) -> &ExactRationalPolynomial {
        &self.position_polynomial
    }

    pub fn velocity_polynomial(&self) -> &ExactRationalPolynomial {
        &self.velocity_polynomial
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

    pub fn residual_tolerance(&self) -> &BigRational {
        &self.residual_tolerance
    }

    pub fn tail_bound(&self) -> &BigRational {
        &self.tail_bound
    }

    pub fn sample_count(&self) -> &BigInt {
        &self.sample_count
    }
}

/// Validated finite ordinary tube input, without any tube-math acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrdinaryTubeInput {
    tube_id: String,
    chart_id: String,
    source: String,
    anchor_parameter: BigRational,
    initial_error_bound: BigRational,
    tube_radius: BigRational,
    maximum_defect_bound: BigRational,
    maximum_lipschitz_bound: BigRational,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrdinaryTubeBindingStatus {
    identity_matches: bool,
    finite_inputs: bool,
}

impl OrdinaryTubeBindingStatus {
    pub const fn identity_matches(&self) -> bool {
        self.identity_matches
    }

    pub const fn finite_inputs(&self) -> bool {
        self.finite_inputs
    }

    pub const fn bound(&self) -> bool {
        self.identity_matches && self.finite_inputs
    }
}

impl OrdinaryTubeInput {
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
}

/// Convert and validate the finite ordinary-chart schema used by tube and
/// bridge replay.  Primitive recurrence obligations are out of scope for this
/// conversion.
pub fn ordinary_chart_input_from_wire(
    wire: &OrdinaryChartWire,
) -> Result<OrdinaryChartInput, OrdinarySemanticError> {
    require_nonempty(&wire.certificate_id, "certificate_id")?;
    require_nonempty(&wire.chart_id, "chart_id")?;
    require_nonempty(&wire.chart_type, "chart_type")?;
    require_nonempty(&wire.source, "source")?;
    if wire.chart_type != ORDINARY_CHART_TYPE {
        return Err(OrdinarySemanticError::UnexpectedChartType {
            actual: wire.chart_type.clone(),
        });
    }

    if wire.masses.len() != BODY_COUNT {
        return Err(OrdinarySemanticError::MassCountMismatch {
            expected: BODY_COUNT,
            actual: wire.masses.len(),
        });
    }
    let masses: [BigRational; BODY_COUNT] =
        std::array::from_fn(|index| wire.masses[index].binary64_rational().clone());
    for (index, mass) in masses.iter().enumerate() {
        if mass.numer().sign() != Sign::Plus {
            return Err(OrdinarySemanticError::NonPositiveMass { index });
        }
    }

    let position_count = wire.position_coefficients.len();
    let velocity_count = wire.velocity_coefficients.len();
    if position_count < MINIMUM_COEFFICIENT_COUNT {
        return Err(OrdinarySemanticError::CoefficientCountTooSmall {
            minimum: MINIMUM_COEFFICIENT_COUNT,
            actual: position_count,
        });
    }
    if position_count != velocity_count {
        return Err(OrdinarySemanticError::CoefficientCountMismatch {
            position: position_count,
            velocity: velocity_count,
        });
    }
    if position_count > HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT {
        return Err(OrdinarySemanticError::ResourceExhausted {
            resource: OrdinarySemanticResource::CoefficientCount,
            required: position_count,
            limit: HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT,
        });
    }
    let required_polynomial_work = CONFIGURATION_DIMENSION
        .checked_mul(position_count)
        .and_then(|work| work.checked_mul(4))
        .unwrap_or(usize::MAX);
    if required_polynomial_work > HARD_MAX_POLYNOMIAL_WORK_UNITS {
        return Err(OrdinarySemanticError::ResourceExhausted {
            resource: OrdinarySemanticResource::PositionPolynomialWork,
            required: required_polynomial_work,
            limit: HARD_MAX_POLYNOMIAL_WORK_UNITS,
        });
    }

    validate_coefficient_shape(
        &wire.position_coefficients,
        OrdinaryCoefficientKind::Position,
    )?;
    validate_coefficient_shape(
        &wire.velocity_coefficients,
        OrdinaryCoefficientKind::Velocity,
    )?;
    let position_polynomial = construct_polynomial(
        &wire.position_coefficients,
        OrdinaryCoefficientKind::Position,
    )?;
    let velocity_polynomial = construct_polynomial(
        &wire.velocity_coefficients,
        OrdinaryCoefficientKind::Velocity,
    )?;

    let parameter_interval =
        strictly_increasing_interval(&wire.parameter_interval, OrdinaryIntervalKind::Parameter)?;
    let physical_time_interval = strictly_increasing_interval(
        &wire.physical_time_interval,
        OrdinaryIntervalKind::PhysicalTime,
    )?;
    let sample_count = wire.sample_count.value().clone();
    if sample_count.sign() != Sign::Plus {
        return Err(OrdinarySemanticError::NonPositiveSampleCount {
            value: sample_count,
        });
    }

    Ok(OrdinaryChartInput {
        certificate_id: wire.certificate_id.clone(),
        chart_id: wire.chart_id.clone(),
        source: wire.source.clone(),
        masses,
        position_polynomial,
        velocity_polynomial,
        parameter_interval,
        physical_time_interval,
        coefficient_tolerance: wire.coefficient_tolerance.binary64_rational().clone(),
        residual_tolerance: wire.residual_tolerance.binary64_rational().clone(),
        tail_bound: wire.tail_bound.binary64_rational().clone(),
        sample_count,
    })
}

/// Capture an ordinary tube independently of any chart binding.
///
/// Every numeric field is already a proved finite binary64 value at the raw
/// schema boundary. Identity, interval membership, and sign obligations are
/// retained for ordered replay rather than rejected here.
pub fn ordinary_tube_input_from_wire(wire: &OrdinaryTubeWire) -> OrdinaryTubeInput {
    OrdinaryTubeInput {
        tube_id: wire.tube_id.clone(),
        chart_id: wire.chart_id.clone(),
        source: wire.source.clone(),
        anchor_parameter: wire.anchor_parameter.binary64_rational().clone(),
        initial_error_bound: wire.initial_error_bound.binary64_rational().clone(),
        tube_radius: wire.tube_radius.binary64_rational().clone(),
        maximum_defect_bound: wire.max_defect_bound.binary64_rational().clone(),
        maximum_lipschitz_bound: wire.max_lipschitz_bound.binary64_rational().clone(),
    }
}

/// Evaluate the first two tube obligations without suppressing either one.
pub fn ordinary_tube_binding_status(
    tube: &OrdinaryTubeInput,
    chart: &OrdinaryChartInput,
) -> OrdinaryTubeBindingStatus {
    let identity_matches =
        !tube.tube_id.is_empty() && !tube.chart_id.is_empty() && tube.chart_id == chart.chart_id;
    let finite_inputs = !tube.source.is_empty()
        && tube.anchor_parameter >= *chart.parameter_interval.lower()
        && tube.anchor_parameter <= *chart.parameter_interval.upper()
        && tube.initial_error_bound.numer().sign() != Sign::Minus
        && tube.tube_radius.numer().sign() == Sign::Plus
        && tube.maximum_defect_bound.numer().sign() != Sign::Minus
        && tube.maximum_lipschitz_bound.numer().sign() != Sign::Minus;
    OrdinaryTubeBindingStatus {
        identity_matches,
        finite_inputs,
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), OrdinarySemanticError> {
    if value.is_empty() {
        return Err(OrdinarySemanticError::EmptyString { field });
    }
    Ok(())
}

fn validate_coefficient_shape(
    coefficients: &RealTensor3,
    kind: OrdinaryCoefficientKind,
) -> Result<(), OrdinarySemanticError> {
    for (degree_index, row) in coefficients.iter().enumerate() {
        if row.len() != BODY_COUNT {
            return Err(OrdinarySemanticError::CoefficientBodyCountMismatch {
                coefficients: kind,
                degree_index,
                expected: BODY_COUNT,
                actual: row.len(),
            });
        }
        for (body_index, body) in row.iter().enumerate() {
            if body.len() != PLANE_DIMENSION {
                return Err(OrdinarySemanticError::CoefficientAxisCountMismatch {
                    coefficients: kind,
                    degree_index,
                    body_index,
                    expected: PLANE_DIMENSION,
                    actual: body.len(),
                });
            }
        }
    }
    Ok(())
}

fn construct_polynomial(
    coefficients: &RealTensor3,
    kind: OrdinaryCoefficientKind,
) -> Result<ExactRationalPolynomial, OrdinarySemanticError> {
    let mut flattened = Vec::with_capacity(coefficients.len());
    for degree_row in coefficients {
        let mut row = Vec::with_capacity(CONFIGURATION_DIMENSION);
        for body in degree_row {
            for component in body {
                row.push(component.binary64_rational().clone());
            }
        }
        flattened.push(row);
    }
    ExactRationalPolynomial::from_degree_major_coefficients(flattened)
        .map_err(|source| map_polynomial_error(kind, source))
}

fn map_polynomial_error(
    kind: OrdinaryCoefficientKind,
    source: PolynomialError,
) -> OrdinarySemanticError {
    match source {
        PolynomialError::DegreeLimitExceeded { degree, limit } => {
            OrdinarySemanticError::ResourceExhausted {
                resource: OrdinarySemanticResource::CoefficientCount,
                required: degree.saturating_add(1),
                limit: limit.saturating_add(1),
            }
        }
        PolynomialError::WorkLimitExceeded { required, limit } => {
            let resource = match kind {
                OrdinaryCoefficientKind::Position => {
                    OrdinarySemanticResource::PositionPolynomialWork
                }
                OrdinaryCoefficientKind::Velocity => {
                    OrdinarySemanticResource::VelocityPolynomialWork
                }
            };
            OrdinarySemanticError::ResourceExhausted {
                resource,
                required,
                limit,
            }
        }
        source => OrdinarySemanticError::Polynomial {
            coefficients: kind,
            source,
        },
    }
}

fn strictly_increasing_interval(
    wire: &[crate::raw_schema::Real; 2],
    kind: OrdinaryIntervalKind,
) -> Result<RationalInterval, OrdinarySemanticError> {
    let lower = wire[0].binary64_rational().clone();
    let upper = wire[1].binary64_rational().clone();
    if lower >= upper {
        return Err(OrdinarySemanticError::IntervalNotStrictlyIncreasing { interval: kind });
    }
    Ok(RationalInterval::new(lower, upper)?)
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use num_bigint::BigInt;
    use num_traits::Zero;

    use super::*;
    use crate::{
        checked_real_binary64_from_json, parse_json_number_lexeme, parse_wire_json,
        raw_schema::{decode_raw_chain, RawInteger, RawPlanarChainWire, Real, SchemaProfile},
        WireJsonValue, DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
        HARD_MAX_POLYNOMIAL_WORK_UNITS,
    };

    const CONFORMANCE_SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
    ));

    fn base_chain() -> &'static RawPlanarChainWire {
        static CHAIN: OnceLock<RawPlanarChainWire> = OnceLock::new();
        CHAIN.get_or_init(|| {
            let ast = parse_wire_json(CONFORMANCE_SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
            decode_raw_chain(ast, SchemaProfile::V03Compatible).unwrap()
        })
    }

    fn real(lexeme: &str) -> Real {
        checked_real_binary64_from_json(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn raw_sample_count(lexeme: &str) -> RawInteger {
        let mut ast = parse_wire_json(CONFORMANCE_SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let parsed = parse_json_number_lexeme(lexeme, DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        let WireJsonValue::Object(root) = &mut ast else {
            panic!("conformance fixture root must be an object");
        };
        let Some(WireJsonValue::Object(chart)) = root.get_mut("initial_chart") else {
            panic!("conformance fixture initial chart must be an object");
        };
        chart.insert("sample_count".into(), WireJsonValue::Integer(parsed));
        decode_raw_chain(ast, SchemaProfile::V03Compatible)
            .unwrap()
            .initial_chart
            .sample_count
    }

    fn row(values: [[&str; PLANE_DIMENSION]; BODY_COUNT]) -> Vec<Vec<Real>> {
        values
            .into_iter()
            .map(|body| body.into_iter().map(real).collect())
            .collect()
    }

    fn valid_chart_wire() -> OrdinaryChartWire {
        let mut wire = base_chain().initial_chart.clone();
        wire.certificate_id = "ordinary-certificate".into();
        wire.chart_id = "ordinary-chart".into();
        wire.chart_type = ORDINARY_CHART_TYPE.into();
        wire.masses = vec![real("1.0"), real("2.0"), real("3.0")];
        wire.position_coefficients = vec![
            row([["-1.0", "0.0"], ["0.0", "0.0"], ["1.0", "0.0"]]),
            row([["0.1", "0.0"], ["0.0", "0.1"], ["-0.1", "0.0"]]),
        ];
        wire.velocity_coefficients = vec![
            row([["0.1", "0.0"], ["0.0", "0.1"], ["-0.1", "0.0"]]),
            row([["0.0", "0.0"], ["0.0", "0.0"], ["0.0", "0.0"]]),
        ];
        wire.parameter_interval = [real("0.0"), real("1.0")];
        wire.physical_time_interval = [real("2.0"), real("3.0")];
        wire.coefficient_tolerance = real("0.0");
        wire.residual_tolerance = real("0.0");
        wire.tail_bound = real("0.0");
        wire.source = "ordinary-source".into();
        wire
    }

    fn valid_tube_wire() -> OrdinaryTubeWire {
        let mut wire = base_chain().initial_tube.clone();
        wire.tube_id = "ordinary-tube".into();
        wire.chart_id = "ordinary-chart".into();
        wire.anchor_parameter = real("0.5");
        wire.initial_error_bound = real("0.0");
        wire.tube_radius = real("0.1");
        wire.max_defect_bound = real("1.0");
        wire.max_lipschitz_bound = real("2.0");
        wire.source = "tube-source".into();
        wire
    }

    #[test]
    fn every_real_uses_the_exact_binary64_dyadic_not_the_decimal_lexeme() {
        let mut wire = valid_chart_wire();
        wire.masses[0] = real("0.1");
        wire.position_coefficients[0][0][0] = real("0.1");
        wire.coefficient_tolerance = real("0.1");
        assert_eq!(
            wire.coefficient_tolerance.exact_decimal(),
            &BigRational::new(BigInt::from(1), BigInt::from(10))
        );

        let chart = ordinary_chart_input_from_wire(&wire).unwrap();
        let exact_binary64 = BigRational::new(
            BigInt::from(3_602_879_701_896_397_u64),
            BigInt::from(36_028_797_018_963_968_u64),
        );
        assert_eq!(&chart.masses()[0], &exact_binary64);
        assert_eq!(
            &chart.position_polynomial().coefficients_by_degree()[0][0],
            &exact_binary64
        );
        assert_eq!(chart.coefficient_tolerance(), &exact_binary64);
        assert_ne!(
            exact_binary64,
            BigRational::new(BigInt::from(1), BigInt::from(10))
        );
    }

    #[test]
    fn coefficients_are_body_major_absolute_parameter_monomials() {
        let mut wire = valid_chart_wire();
        wire.position_coefficients = vec![
            row([["2.0", "20.0"], ["3.0", "30.0"], ["4.0", "40.0"]]),
            row([["3.0", "0.0"], ["5.0", "0.0"], ["7.0", "0.0"]]),
        ];
        wire.parameter_interval = [real("5.0"), real("6.0")];
        let chart = ordinary_chart_input_from_wire(&wire).unwrap();
        assert_eq!(
            chart.position_polynomial().coefficients_by_degree()[0],
            vec![
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(20)),
                BigRational::from_integer(BigInt::from(3)),
                BigRational::from_integer(BigInt::from(30)),
                BigRational::from_integer(BigInt::from(4)),
                BigRational::from_integer(BigInt::from(40)),
            ]
        );
        let at_absolute_five = chart
            .position_polynomial()
            .evaluate(
                &RationalInterval::try_point(BigRational::from_integer(BigInt::from(5))).unwrap(),
            )
            .unwrap();
        assert_eq!(
            at_absolute_five[0],
            RationalInterval::try_point(BigRational::from_integer(BigInt::from(17))).unwrap()
        );
    }

    #[test]
    fn degree_and_every_nested_planar_shape_are_exact() {
        let mut one_row = valid_chart_wire();
        one_row.position_coefficients.truncate(1);
        one_row.velocity_coefficients.truncate(1);
        assert_eq!(
            ordinary_chart_input_from_wire(&one_row),
            Err(OrdinarySemanticError::CoefficientCountTooSmall {
                minimum: 2,
                actual: 1,
            })
        );

        let mut unequal = valid_chart_wire();
        unequal
            .velocity_coefficients
            .push(unequal.velocity_coefficients[0].clone());
        assert_eq!(
            ordinary_chart_input_from_wire(&unequal),
            Err(OrdinarySemanticError::CoefficientCountMismatch {
                position: 2,
                velocity: 3,
            })
        );

        let mut wrong_body_count = valid_chart_wire();
        wrong_body_count.position_coefficients[1].pop();
        assert_eq!(
            ordinary_chart_input_from_wire(&wrong_body_count),
            Err(OrdinarySemanticError::CoefficientBodyCountMismatch {
                coefficients: OrdinaryCoefficientKind::Position,
                degree_index: 1,
                expected: 3,
                actual: 2,
            })
        );

        let mut ragged_axis = valid_chart_wire();
        ragged_axis.velocity_coefficients[1][2].pop();
        assert_eq!(
            ordinary_chart_input_from_wire(&ragged_axis),
            Err(OrdinarySemanticError::CoefficientAxisCountMismatch {
                coefficients: OrdinaryCoefficientKind::Velocity,
                degree_index: 1,
                body_index: 2,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn chart_identity_mass_and_interval_domains_fail_deterministically() {
        for (wire, field) in [
            (
                {
                    let mut wire = valid_chart_wire();
                    wire.certificate_id.clear();
                    wire
                },
                "certificate_id",
            ),
            (
                {
                    let mut wire = valid_chart_wire();
                    wire.chart_id.clear();
                    wire
                },
                "chart_id",
            ),
            (
                {
                    let mut wire = valid_chart_wire();
                    wire.chart_type.clear();
                    wire
                },
                "chart_type",
            ),
            (
                {
                    let mut wire = valid_chart_wire();
                    wire.source.clear();
                    wire
                },
                "source",
            ),
        ] {
            assert_eq!(
                ordinary_chart_input_from_wire(&wire),
                Err(OrdinarySemanticError::EmptyString { field })
            );
        }

        let mut wrong_type = valid_chart_wire();
        wrong_type.chart_type = "future_chart".into();
        assert_eq!(
            ordinary_chart_input_from_wire(&wrong_type),
            Err(OrdinarySemanticError::UnexpectedChartType {
                actual: "future_chart".into(),
            })
        );

        let mut wrong_mass_count = valid_chart_wire();
        wrong_mass_count.masses.pop();
        assert_eq!(
            ordinary_chart_input_from_wire(&wrong_mass_count),
            Err(OrdinarySemanticError::MassCountMismatch {
                expected: 3,
                actual: 2,
            })
        );
        let mut negative_mass = valid_chart_wire();
        negative_mass.masses[1] = real("-1.0");
        assert_eq!(
            ordinary_chart_input_from_wire(&negative_mass),
            Err(OrdinarySemanticError::NonPositiveMass { index: 1 })
        );

        let mut equal_parameter = valid_chart_wire();
        equal_parameter.parameter_interval = [real("1.0"), real("1.0")];
        assert_eq!(
            ordinary_chart_input_from_wire(&equal_parameter),
            Err(OrdinarySemanticError::IntervalNotStrictlyIncreasing {
                interval: OrdinaryIntervalKind::Parameter,
            })
        );
        let mut reversed_time = valid_chart_wire();
        reversed_time.physical_time_interval = [real("3.0"), real("2.0")];
        assert_eq!(
            ordinary_chart_input_from_wire(&reversed_time),
            Err(OrdinarySemanticError::IntervalNotStrictlyIncreasing {
                interval: OrdinaryIntervalKind::PhysicalTime,
            })
        );
    }

    #[test]
    fn finite_chart_input_preserves_negative_primitive_only_tolerances() {
        let mut wire = valid_chart_wire();
        wire.coefficient_tolerance = real("-0.1");
        wire.residual_tolerance = real("-2.0");
        wire.tail_bound = real("-3.0");
        let chart = ordinary_chart_input_from_wire(&wire).unwrap();
        assert!(chart.coefficient_tolerance() < &BigRational::zero());
        assert_eq!(
            chart.residual_tolerance(),
            &BigRational::from_integer(BigInt::from(-2))
        );
        assert_eq!(
            chart.tail_bound(),
            &BigRational::from_integer(BigInt::from(-3))
        );
    }

    #[test]
    fn sample_count_is_positive_exact_bigint_without_machine_narrowing() {
        for (lexeme, value) in [("0", BigInt::from(0)), ("-7", BigInt::from(-7))] {
            let mut wire = valid_chart_wire();
            wire.sample_count = raw_sample_count(lexeme);
            assert_eq!(
                ordinary_chart_input_from_wire(&wire),
                Err(OrdinarySemanticError::NonPositiveSampleCount { value })
            );
        }

        let huge_lexeme = format!("1{}", "0".repeat(300));
        let expected = BigInt::parse_bytes(huge_lexeme.as_bytes(), 10).unwrap();
        let mut wire = valid_chart_wire();
        wire.sample_count = raw_sample_count(&huge_lexeme);
        let chart = ordinary_chart_input_from_wire(&wire).unwrap();
        assert_eq!(chart.sample_count(), &expected);
        assert!(chart.sample_count().bits() > usize::BITS as u64);
    }

    #[test]
    fn tube_identity_anchor_and_sign_rules_are_exact() {
        let chart = ordinary_chart_input_from_wire(&valid_chart_wire()).unwrap();
        let accepted = ordinary_tube_input_from_wire(&valid_tube_wire());
        assert_eq!(accepted.chart_id(), chart.chart_id());
        assert_eq!(
            accepted.anchor_parameter(),
            &BigRational::new(BigInt::from(1), BigInt::from(2))
        );
        assert_eq!(
            ordinary_tube_binding_status(&accepted, &chart),
            OrdinaryTubeBindingStatus {
                identity_matches: true,
                finite_inputs: true,
            }
        );

        for wire in [
            {
                let mut wire = valid_tube_wire();
                wire.tube_id.clear();
                wire
            },
            {
                let mut wire = valid_tube_wire();
                wire.chart_id.clear();
                wire
            },
        ] {
            let status =
                ordinary_tube_binding_status(&ordinary_tube_input_from_wire(&wire), &chart);
            assert!(!status.identity_matches());
            assert!(status.finite_inputs());
        }

        let mut empty_source = valid_tube_wire();
        empty_source.source.clear();
        let status =
            ordinary_tube_binding_status(&ordinary_tube_input_from_wire(&empty_source), &chart);
        assert!(status.identity_matches());
        assert!(!status.finite_inputs());

        let mut mismatch = valid_tube_wire();
        mismatch.chart_id = "other-chart".into();
        let status =
            ordinary_tube_binding_status(&ordinary_tube_input_from_wire(&mismatch), &chart);
        assert!(!status.identity_matches());
        assert!(status.finite_inputs());

        let mut outside = valid_tube_wire();
        outside.anchor_parameter = real("2.0");
        let status = ordinary_tube_binding_status(&ordinary_tube_input_from_wire(&outside), &chart);
        assert!(status.identity_matches());
        assert!(!status.finite_inputs());

        for wire in [
            {
                let mut wire = valid_tube_wire();
                wire.initial_error_bound = real("-1.0");
                wire
            },
            {
                let mut wire = valid_tube_wire();
                wire.max_defect_bound = real("-1.0");
                wire
            },
            {
                let mut wire = valid_tube_wire();
                wire.max_lipschitz_bound = real("-1.0");
                wire
            },
            {
                let mut wire = valid_tube_wire();
                wire.tube_radius = real("0.0");
                wire
            },
        ] {
            let status =
                ordinary_tube_binding_status(&ordinary_tube_input_from_wire(&wire), &chart);
            assert!(status.identity_matches());
            assert!(!status.finite_inputs());
        }
    }

    #[test]
    fn signed_zero_follows_exact_rational_sign_domains() {
        let mut mass_negative_zero = valid_chart_wire();
        mass_negative_zero.masses[0] = real("-0.0");
        assert_eq!(
            ordinary_chart_input_from_wire(&mass_negative_zero),
            Err(OrdinarySemanticError::NonPositiveMass { index: 0 })
        );

        let chart = ordinary_chart_input_from_wire(&valid_chart_wire()).unwrap();
        let mut nonnegative_negative_zero = valid_tube_wire();
        nonnegative_negative_zero.anchor_parameter = real("-0.0");
        nonnegative_negative_zero.initial_error_bound = real("-0.0");
        nonnegative_negative_zero.max_defect_bound = real("-0.0");
        nonnegative_negative_zero.max_lipschitz_bound = real("-0.0");
        let tube = ordinary_tube_input_from_wire(&nonnegative_negative_zero);
        assert_eq!(tube.anchor_parameter(), &BigRational::zero());
        assert_eq!(tube.initial_error_bound(), &BigRational::zero());
        assert_eq!(tube.maximum_defect_bound(), &BigRational::zero());
        assert_eq!(tube.maximum_lipschitz_bound(), &BigRational::zero());

        let mut radius_negative_zero = valid_tube_wire();
        radius_negative_zero.tube_radius = real("-0.0");
        let status = ordinary_tube_binding_status(
            &ordinary_tube_input_from_wire(&radius_negative_zero),
            &chart,
        );
        assert!(status.identity_matches());
        assert!(!status.finite_inputs());
    }

    #[test]
    fn coefficient_count_and_work_exhaustion_are_distinct_and_never_panic() {
        let row = valid_chart_wire().position_coefficients[0].clone();
        let count = HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT + 1;
        let mut count_exhausted = valid_chart_wire();
        count_exhausted.position_coefficients = vec![row.clone(); count];
        count_exhausted.velocity_coefficients = vec![row.clone(); count];
        let count_result =
            std::panic::catch_unwind(|| ordinary_chart_input_from_wire(&count_exhausted));
        assert_eq!(
            count_result.unwrap(),
            Err(OrdinarySemanticError::ResourceExhausted {
                resource: OrdinarySemanticResource::CoefficientCount,
                required: count,
                limit: HARD_MAX_ORDINARY_SEMANTIC_COEFFICIENT_COUNT,
            })
        );

        let work_count = HARD_MAX_POLYNOMIAL_WORK_UNITS / (CONFIGURATION_DIMENSION * 4) + 1;
        let mut work_exhausted = valid_chart_wire();
        work_exhausted.position_coefficients = vec![row.clone(); work_count];
        work_exhausted.velocity_coefficients = vec![row; work_count];
        // Resource preflight must win before scanning an oversized nested
        // coefficient table or allocating its flattened representation.
        work_exhausted.position_coefficients[0].clear();
        let work_result =
            std::panic::catch_unwind(|| ordinary_chart_input_from_wire(&work_exhausted));
        assert_eq!(
            work_result.unwrap(),
            Err(OrdinarySemanticError::ResourceExhausted {
                resource: OrdinarySemanticResource::PositionPolynomialWork,
                required: CONFIGURATION_DIMENSION * work_count * 4,
                limit: HARD_MAX_POLYNOMIAL_WORK_UNITS,
            })
        );
    }
}
