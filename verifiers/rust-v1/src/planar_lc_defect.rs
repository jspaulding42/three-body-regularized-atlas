//! Direct exact-rational interval defect for an admitted planar LC chart.
//!
//! The chart polynomial is evaluated on its uninflated parameter interval and
//! compared directly with the independent LC field. Claimed chart-tail bounds
//! are deliberately not consumed. This arithmetic kernel alone certifies no
//! tube, entry, gauge, projection, exit, or chain obligation.

use core::fmt;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{
    evaluate_planar_lc_field, NumericError, PlanarLcChartInput, PlanarLcFieldError,
    PlanarLcStateError, PlanarLcStatePolynomial, RationalInterval,
    PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS, PLANAR_LC_LIFTED_STATE_DIMENSION,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcPolynomialDefectError {
    State(PlanarLcStateError),
    Field(PlanarLcFieldError),
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcPolynomialDefectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC polynomial defect failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcPolynomialDefectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::Field(source) => Some(source),
            Self::Numeric(source) => Some(source),
        }
    }
}

impl From<PlanarLcStateError> for PlanarLcPolynomialDefectError {
    fn from(source: PlanarLcStateError) -> Self {
        Self::State(source)
    }
}

impl From<PlanarLcFieldError> for PlanarLcPolynomialDefectError {
    fn from(source: PlanarLcFieldError) -> Self {
        Self::Field(source)
    }
}

impl From<NumericError> for PlanarLcPolynomialDefectError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcPolynomialDefectEnclosure {
    residuals: [RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
    maximum_absolute_endpoint_bound: BigRational,
}

impl PlanarLcPolynomialDefectEnclosure {
    /// Residual order is the canonical 14-component lifted-state order.
    pub fn residuals(&self) -> &[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION] {
        &self.residuals
    }

    pub fn maximum_absolute_endpoint_bound(&self) -> &BigRational {
        &self.maximum_absolute_endpoint_bound
    }
}

pub fn evaluate_planar_lc_polynomial_defect(
    chart: &PlanarLcChartInput,
    sqrt_precision_bits: usize,
) -> Result<PlanarLcPolynomialDefectEnclosure, PlanarLcPolynomialDefectError> {
    let polynomial = PlanarLcStatePolynomial::from_chart(chart)?;
    let state = polynomial.evaluate(chart.parameter_interval())?;
    let derivative = polynomial.evaluate_derivative(chart.parameter_interval())?;
    let field = evaluate_planar_lc_field(chart, &state, sqrt_precision_bits)?;
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut residuals = std::array::from_fn(|_| zero.clone());
    for (index, residual) in residuals.iter_mut().enumerate() {
        *residual = derivative.components()[index].subtract(&field.rhs()[index])?;
    }
    let maximum_absolute_endpoint_bound = maximum_absolute_endpoint(&residuals);
    Ok(PlanarLcPolynomialDefectEnclosure {
        residuals,
        maximum_absolute_endpoint_bound,
    })
}

pub fn evaluate_planar_lc_polynomial_defect_default(
    chart: &PlanarLcChartInput,
) -> Result<PlanarLcPolynomialDefectEnclosure, PlanarLcPolynomialDefectError> {
    evaluate_planar_lc_polynomial_defect(chart, PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS)
}

fn maximum_absolute_endpoint(
    residuals: &[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
) -> BigRational {
    let mut maximum = BigRational::zero();
    for residual in residuals {
        for endpoint in [residual.lower(), residual.upper()] {
            let magnitude = if endpoint.numer().sign() == Sign::Minus {
                -endpoint.clone()
            } else {
                endpoint.clone()
            };
            if magnitude > maximum {
                maximum = magnitude;
            }
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parse_wire_json, planar_lc_chart_input_from_wire,
        raw_schema::{decode_raw_chain, SchemaProfile, SegmentWire},
        DEFAULT_WIRE_JSON_LIMITS, HARD_MAX_SQRT_PRECISION_BITS,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn chart() -> PlanarLcChartInput {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap()
    }

    fn chart_with_changed_tail() -> PlanarLcChartInput {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        let mut wire = segment.lc_chart.clone();
        wire.tail_bound = wire.coefficient_tolerance.clone();
        planar_lc_chart_input_from_wire(&wire).unwrap()
    }

    #[test]
    fn direct_interval_defect_contains_independent_left_endpoint_residual() {
        let chart = chart();
        let enclosure = evaluate_planar_lc_polynomial_defect_default(&chart).unwrap();
        let point =
            RationalInterval::try_point(chart.parameter_interval().lower().clone()).unwrap();
        let polynomial = PlanarLcStatePolynomial::from_chart(&chart).unwrap();
        let state = polynomial.evaluate(&point).unwrap();
        let derivative = polynomial.evaluate_derivative(&point).unwrap();
        let field =
            evaluate_planar_lc_field(&chart, &state, PLANAR_LC_FIELD_DEFAULT_SQRT_PRECISION_BITS)
                .unwrap();
        for index in 0..PLANAR_LC_LIFTED_STATE_DIMENSION {
            let point_residual = derivative.components()[index]
                .subtract(&field.rhs()[index])
                .unwrap();
            assert!(enclosure.residuals()[index].contains_interval(&point_residual));
        }
    }

    #[test]
    fn reported_maximum_is_the_exact_maximum_of_residual_endpoints() {
        let enclosure = evaluate_planar_lc_polynomial_defect_default(&chart()).unwrap();
        assert_eq!(
            enclosure.maximum_absolute_endpoint_bound(),
            &maximum_absolute_endpoint(enclosure.residuals())
        );
    }

    #[test]
    fn direct_defect_is_independent_of_the_claimed_tail_bound() {
        let original = chart();
        let changed = chart_with_changed_tail();
        assert_ne!(original.tail_bound(), changed.tail_bound());
        assert_eq!(
            evaluate_planar_lc_polynomial_defect_default(&original).unwrap(),
            evaluate_planar_lc_polynomial_defect_default(&changed).unwrap()
        );
    }

    #[test]
    fn field_precision_exhaustion_is_preserved_as_a_typed_error() {
        assert_eq!(
            evaluate_planar_lc_polynomial_defect(&chart(), HARD_MAX_SQRT_PRECISION_BITS + 1),
            Err(PlanarLcPolynomialDefectError::Field(
                PlanarLcFieldError::SqrtPrecisionExceeded {
                    requested: HARD_MAX_SQRT_PRECISION_BITS + 1,
                    limit: HARD_MAX_SQRT_PRECISION_BITS,
                }
            ))
        );
    }
}
