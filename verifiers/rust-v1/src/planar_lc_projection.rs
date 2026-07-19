//! Exact-rational interval projection and Newton-residual arithmetic for one
//! admitted planar LC chart and lifted state.
//!
//! This fixed-dimensional kernel consumes no chart tail, tolerance, sample,
//! time interval, or serialized polynomial coefficient. It does not accept a
//! chart, prove a remainder, or replay a continuation step.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::{
    evaluate_planar_lc_field,
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError, PlanarLcMassProfile},
    sqrt_enclosure_dyadic, ExactBinary64, NumericError, PlanarLcChartInput, PlanarLcFieldError,
    PlanarLcIntervalState, PlanarLcStateError, RationalInterval, HARD_MAX_SQRT_PRECISION_BITS,
    LC_RX, LC_RY, LC_UX, LC_UY, LC_VX, LC_VY, LC_WX, LC_WY, LC_YX, LC_YY, LC_ZX, LC_ZY,
    PLANAR_LC_LIFTED_STATE_DIMENSION,
};

const BODY_COUNT: usize = 3;
const PLANE_DIMENSION: usize = 2;
pub const PLANAR_LC_PROJECTION_DEFAULT_SQRT_PRECISION_BITS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanarLcProjectionPair {
    first: usize,
    second: usize,
}

impl PlanarLcProjectionPair {
    pub const fn first(self) -> usize {
        self.first
    }

    pub const fn second(self) -> usize {
        self.second
    }

    #[cfg(test)]
    pub(crate) const fn for_failure_disposition_test(first: usize, second: usize) -> Self {
        Self { first, second }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcProjectionError {
    State(PlanarLcStateError),
    Field(PlanarLcFieldError),
    MassDecode { index: usize, source: NumericError },
    MassProfile(MassProfileError),
    NegativeProjectionFloor,
    RhoFloorNotStrict,
    RhoNotStrictlyPositive,
    NewtonCollision { pair: PlanarLcProjectionPair },
    PositiveSquareRootUnresolved { pair: PlanarLcProjectionPair },
    SqrtPrecisionExceeded { requested: usize, limit: usize },
    InternalShapeInvariant,
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "planar LC projection arithmetic failed: {self:?}"
        )
    }
}

impl std::error::Error for PlanarLcProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::Field(source) => Some(source),
            Self::MassDecode { source, .. } => Some(source),
            Self::MassProfile(source) => Some(source),
            Self::Numeric(source) => Some(source),
            _ => None,
        }
    }
}

impl From<PlanarLcStateError> for PlanarLcProjectionError {
    fn from(source: PlanarLcStateError) -> Self {
        Self::State(source)
    }
}

impl From<PlanarLcFieldError> for PlanarLcProjectionError {
    fn from(source: PlanarLcFieldError) -> Self {
        Self::Field(source)
    }
}

impl From<MassProfileError> for PlanarLcProjectionError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

impl From<NumericError> for PlanarLcProjectionError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcProjectionReplay {
    rho: RationalInterval,
    projection_rho_lower_bound: BigRational,
    positions: [[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    projected_accelerations: [[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    newton_accelerations: [[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    residuals: [RationalInterval; BODY_COUNT * PLANE_DIMENSION],
    maximum_absolute_residual_endpoint: BigRational,
    mass_profile: PlanarLcMassProfile,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcCartesianStateProjection {
    rho: RationalInterval,
    positions: [[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    velocities: [[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    mass_profile: PlanarLcMassProfile,
}

impl PlanarLcCartesianStateProjection {
    pub fn rho(&self) -> &RationalInterval {
        &self.rho
    }
    pub fn positions(&self) -> &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT] {
        &self.positions
    }
    pub fn velocities(&self) -> &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT] {
        &self.velocities
    }
    pub fn mass_profile(&self) -> &PlanarLcMassProfile {
        &self.mass_profile
    }
}

/// Project a complete lifted LC interval state to Cartesian `q,v` without
/// consuming the chart projection-floor claim or any residual/tail metadata.
pub fn project_planar_lc_full_state_exact_rational(
    chart: &PlanarLcChartInput,
    state: &PlanarLcIntervalState,
) -> Result<PlanarLcCartesianStateProjection, PlanarLcProjectionError> {
    let rho = state.rho()?;
    if rho.lower().numer().sign() != Sign::Plus {
        return Err(PlanarLcProjectionError::RhoNotStrictlyPositive);
    }
    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;
    let components = state.components();
    let square = state.lc_square()?;
    let two = BigRational::from_integer(BigInt::from(2));
    let relative_velocity = [
        components[LC_ZX]
            .multiply(&components[LC_WX])?
            .subtract(&components[LC_ZY].multiply(&components[LC_WY])?)?
            .scale(&two)?
            .divide(&rho)?,
        components[LC_ZY]
            .multiply(&components[LC_WX])?
            .add(&components[LC_ZX].multiply(&components[LC_WY])?)?
            .scale(&two)?
            .divide(&rho)?,
    ];
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut positions = std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    let mut velocities = std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    let [first, second] = chart.pair();
    let third = mass_profile.third_index();
    for axis in 0..PLANE_DIMENSION {
        let center = &components[LC_RX + axis];
        let center_velocity = &components[LC_UX + axis];
        positions[first][axis] =
            center.subtract(&square[axis].scale(mass_profile.alpha().exact())?)?;
        positions[second][axis] = center.add(&square[axis].scale(mass_profile.beta().exact())?)?;
        positions[third][axis] = center.add(&components[LC_YX + axis])?;
        velocities[first][axis] = center_velocity
            .subtract(&relative_velocity[axis].scale(mass_profile.alpha().exact())?)?;
        velocities[second][axis] =
            center_velocity.add(&relative_velocity[axis].scale(mass_profile.beta().exact())?)?;
        velocities[third][axis] = center_velocity.add(&components[LC_VX + axis])?;
    }
    Ok(PlanarLcCartesianStateProjection {
        rho,
        positions,
        velocities,
        mass_profile,
    })
}

impl PlanarLcProjectionReplay {
    pub fn rho(&self) -> &RationalInterval {
        &self.rho
    }

    pub fn projection_rho_lower_bound(&self) -> &BigRational {
        &self.projection_rho_lower_bound
    }

    pub fn positions(&self) -> &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT] {
        &self.positions
    }

    pub fn projected_accelerations(&self) -> &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT] {
        &self.projected_accelerations
    }

    pub fn newton_accelerations(&self) -> &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT] {
        &self.newton_accelerations
    }

    pub fn residuals(&self) -> &[RationalInterval; BODY_COUNT * PLANE_DIMENSION] {
        &self.residuals
    }

    pub fn maximum_absolute_residual_endpoint(&self) -> &BigRational {
        &self.maximum_absolute_residual_endpoint
    }

    pub fn mass_profile(&self) -> &PlanarLcMassProfile {
        &self.mass_profile
    }
}

pub fn replay_planar_lc_projection(
    chart: &PlanarLcChartInput,
    state: &PlanarLcIntervalState,
    sqrt_precision_bits: usize,
) -> Result<PlanarLcProjectionReplay, PlanarLcProjectionError> {
    if sqrt_precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(PlanarLcProjectionError::SqrtPrecisionExceeded {
            requested: sqrt_precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        });
    }
    let floor = chart.projection_rho_lower_bound();
    if floor.numer().sign() == Sign::Minus {
        return Err(PlanarLcProjectionError::NegativeProjectionFloor);
    }
    let rho = state.rho()?;
    if rho.lower() <= floor {
        return Err(PlanarLcProjectionError::RhoFloorNotStrict);
    }
    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;
    let components = state.components();
    let square = state.lc_square()?;
    let center = [&components[LC_RX], &components[LC_RY]];
    let offset = [&components[LC_YX], &components[LC_YY]];
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut positions = std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    for axis in 0..PLANE_DIMENSION {
        positions[chart.pair()[0]][axis] =
            center[axis].subtract(&square[axis].scale(mass_profile.alpha().exact())?)?;
        positions[chart.pair()[1]][axis] =
            center[axis].add(&square[axis].scale(mass_profile.beta().exact())?)?;
        positions[mass_profile.third_index()][axis] = center[axis].add(offset[axis])?;
    }
    let field = evaluate_planar_lc_field(chart, state, sqrt_precision_bits)?;
    let binary_acceleration = [
        field.rhs()[LC_UX].divide(&rho)?,
        field.rhs()[LC_UY].divide(&rho)?,
    ];
    let offset_acceleration = [
        field.rhs()[LC_VX].divide(&rho)?,
        field.rhs()[LC_VY].divide(&rho)?,
    ];
    let relative_acceleration = physical_relative_acceleration(state, field.rhs(), &rho)?;
    let mut projected_accelerations =
        std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    for axis in 0..PLANE_DIMENSION {
        projected_accelerations[chart.pair()[0]][axis] = binary_acceleration[axis]
            .subtract(&relative_acceleration[axis].scale(mass_profile.alpha().exact())?)?;
        projected_accelerations[chart.pair()[1]][axis] = binary_acceleration[axis]
            .add(&relative_acceleration[axis].scale(mass_profile.beta().exact())?)?;
        projected_accelerations[mass_profile.third_index()][axis] =
            binary_acceleration[axis].add(&offset_acceleration[axis])?;
    }
    let exact_masses = std::array::from_fn(|index| masses[index].rational().clone());
    let newton_accelerations =
        newton_accelerations(&positions, &exact_masses, sqrt_precision_bits)?;
    let mut residuals = std::array::from_fn(|_| zero.clone());
    for body in 0..BODY_COUNT {
        for axis in 0..PLANE_DIMENSION {
            residuals[body * PLANE_DIMENSION + axis] =
                projected_accelerations[body][axis].subtract(&newton_accelerations[body][axis])?;
        }
    }
    let maximum_absolute_residual_endpoint = residuals
        .iter()
        .fold(BigRational::zero(), |maximum, residual| {
            maximum.max(residual.lower().abs().max(residual.upper().abs()))
        });
    Ok(PlanarLcProjectionReplay {
        rho,
        projection_rho_lower_bound: floor.clone(),
        positions,
        projected_accelerations,
        newton_accelerations,
        residuals,
        maximum_absolute_residual_endpoint,
        mass_profile,
    })
}

pub fn replay_planar_lc_projection_default(
    chart: &PlanarLcChartInput,
    state: &PlanarLcIntervalState,
) -> Result<PlanarLcProjectionReplay, PlanarLcProjectionError> {
    replay_planar_lc_projection(
        chart,
        state,
        PLANAR_LC_PROJECTION_DEFAULT_SQRT_PRECISION_BITS,
    )
}

fn physical_relative_acceleration(
    state: &PlanarLcIntervalState,
    field_rhs: &[RationalInterval; PLANAR_LC_LIFTED_STATE_DIMENSION],
    rho: &RationalInterval,
) -> Result<[RationalInterval; PLANE_DIMENSION], PlanarLcProjectionError> {
    let state = state.components();
    let z = [&state[LC_ZX], &state[LC_ZY]];
    let w = [&state[LC_WX], &state[LC_WY]];
    let wprime = [&field_rhs[LC_WX], &field_rhs[LC_WY]];
    let lww = levi_civita_bilinear(w, w)?;
    let lzwprime = levi_civita_bilinear(z, wprime)?;
    let lzw = levi_civita_bilinear(z, w)?;
    let two = BigRational::from_integer(BigInt::from(2));
    let rho_prime = z[0]
        .multiply(w[0])?
        .add(&z[1].multiply(w[1])?)?
        .scale(&two)?;
    let rho_square = rho.multiply(rho)?;
    Ok([
        lww[0]
            .add(&lzwprime[0])?
            .divide(rho)?
            .subtract(&lzw[0].multiply(&rho_prime)?.divide(&rho_square)?)?
            .divide(rho)?,
        lww[1]
            .add(&lzwprime[1])?
            .divide(rho)?
            .subtract(&lzw[1].multiply(&rho_prime)?.divide(&rho_square)?)?
            .divide(rho)?,
    ])
}

fn levi_civita_bilinear(
    left: [&RationalInterval; PLANE_DIMENSION],
    right: [&RationalInterval; PLANE_DIMENSION],
) -> Result<[RationalInterval; PLANE_DIMENSION], PlanarLcProjectionError> {
    let two = BigRational::from_integer(BigInt::from(2));
    Ok([
        left[0]
            .multiply(right[0])?
            .subtract(&left[1].multiply(right[1])?)?
            .scale(&two)?,
        left[1]
            .multiply(right[0])?
            .add(&left[0].multiply(right[1])?)?
            .scale(&two)?,
    ])
}

fn newton_accelerations(
    positions: &[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT],
    masses: &[BigRational; BODY_COUNT],
    precision: usize,
) -> Result<[[RationalInterval; PLANE_DIMENSION]; BODY_COUNT], PlanarLcProjectionError> {
    let zero = RationalInterval::try_point(BigRational::zero())?;
    let mut accelerations = std::array::from_fn(|_| std::array::from_fn(|_| zero.clone()));
    for first in 0..BODY_COUNT {
        for second in first + 1..BODY_COUNT {
            let pair = PlanarLcProjectionPair { first, second };
            let delta = [
                positions[second][0].subtract(&positions[first][0])?,
                positions[second][1].subtract(&positions[first][1])?,
            ];
            let norm_square = interval_square(&delta[0])?.add(&interval_square(&delta[1])?)?;
            if norm_square.lower().numer().sign() != Sign::Plus {
                return Err(PlanarLcProjectionError::NewtonCollision { pair });
            }
            let lower_root = sqrt_enclosure_dyadic(norm_square.lower(), precision)?;
            let upper_root = sqrt_enclosure_dyadic(norm_square.upper(), precision)?;
            if lower_root.interval().lower().numer().sign() != Sign::Plus {
                return Err(PlanarLcProjectionError::PositiveSquareRootUnresolved { pair });
            }
            let root = RationalInterval::new(
                lower_root.interval().lower().clone(),
                upper_root.interval().upper().clone(),
            )?;
            let inverse_cube = norm_square.multiply(&root)?.reciprocal()?;
            for (axis, delta_component) in delta.iter().enumerate() {
                let direction = delta_component.multiply(&inverse_cube)?;
                accelerations[first][axis] =
                    accelerations[first][axis].add(&direction.scale(&masses[second])?)?;
                accelerations[second][axis] =
                    accelerations[second][axis].subtract(&direction.scale(&masses[first])?)?;
            }
        }
    }
    Ok(accelerations)
}

fn interval_square(value: &RationalInterval) -> Result<RationalInterval, PlanarLcProjectionError> {
    let lower_square = value.lower() * value.lower();
    let upper_square = value.upper() * value.upper();
    let upper = lower_square.clone().max(upper_square.clone());
    let lower = if value.contains(&BigRational::zero())? {
        BigRational::zero()
    } else {
        lower_square.min(upper_square)
    };
    Ok(RationalInterval::new(lower, upper)?)
}

fn reconstruct_masses(
    chart: &PlanarLcChartInput,
) -> Result<[ExactBinary64; BODY_COUNT], PlanarLcProjectionError> {
    let mut masses = Vec::with_capacity(BODY_COUNT);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        masses.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| PlanarLcProjectionError::MassDecode { index, source })?,
        );
    }
    masses
        .try_into()
        .map_err(|_| PlanarLcProjectionError::InternalShapeInvariant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checked_real_binary64_from_json, ordinary_chart_input_from_wire,
        ordinary_tube_input_from_wire, parse_json_number_lexeme, parse_wire_json,
        planar_lc_chart_input_from_wire, planar_lc_entry_input_from_admission,
        raw_schema::{decode_raw_chain, PlanarLcChartWire, SchemaProfile, SegmentWire},
        replay_carried_planar_lc_entry_exact_rational_v04, CanonicalRawV1Admission,
        PlanarLcStatePolynomial, DEFAULT_JSON_NUMBER_LIMITS, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));

    fn r(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn point(numerator: i64, denominator: i64) -> RationalInterval {
        RationalInterval::try_point(r(numerator, denominator)).unwrap()
    }

    fn real(value: &str) -> crate::raw_schema::Real {
        checked_real_binary64_from_json(value, DEFAULT_JSON_NUMBER_LIMITS).unwrap()
    }

    fn chain() -> crate::raw_schema::RawPlanarChainWire {
        let syntax = parse_wire_json(SUCCESS, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
    }

    fn first_wire() -> PlanarLcChartWire {
        let chain = chain();
        let SegmentWire::PlanarLcPassage(segment) = &chain.segments[1] else {
            panic!()
        };
        segment.lc_chart.clone()
    }

    fn unit_chart_with_floor(floor: &str) -> PlanarLcChartInput {
        let mut wire = first_wire();
        wire.masses = vec![real("1.0"), real("1.0"), real("1.0")];
        wire.projection_rho_lower_bound = real(floor);
        planar_lc_chart_input_from_wire(&wire).unwrap()
    }

    fn hand_state() -> PlanarLcIntervalState {
        let mut values = std::array::from_fn(|_| BigRational::zero());
        values[LC_ZX] = r(1, 1);
        values[LC_WX] = r(1, 1);
        values[crate::LC_H] = r(2, 1);
        values[LC_UX] = r(7, 1);
        values[LC_UY] = r(8, 1);
        values[LC_YX] = r(3, 1);
        values[LC_VX] = r(9, 1);
        values[LC_VY] = r(10, 1);
        PlanarLcIntervalState::from_exact_components(values).unwrap()
    }

    #[test]
    fn hand_unit_mass_projection_separates_binary_and_third_body_residuals() {
        let replay =
            replay_planar_lc_projection_default(&unit_chart_with_floor("0.0"), &hand_state())
                .unwrap();
        assert_eq!(replay.rho(), &point(1, 1));
        assert_eq!(
            replay.positions(),
            &[
                [point(-1, 2), point(0, 1)],
                [point(1, 2), point(0, 1)],
                [point(3, 1), point(0, 1)],
            ]
        );
        assert_eq!(
            replay.projected_accelerations(),
            &[
                [point(4, 49), point(0, 1)],
                [point(4, 25), point(0, 1)],
                [point(-296, 1225), point(0, 1)],
            ]
        );
        assert_eq!(
            replay.newton_accelerations(),
            &[
                [point(53, 49), point(0, 1)],
                [point(-21, 25), point(0, 1)],
                [point(-296, 1225), point(0, 1)],
            ]
        );
        assert_eq!(
            replay.residuals(),
            &[
                point(-1, 1),
                point(0, 1),
                point(1, 1),
                point(0, 1),
                point(0, 1),
                point(0, 1),
            ]
        );
        assert_eq!(replay.maximum_absolute_residual_endpoint(), &r(1, 1));
    }

    #[test]
    fn full_state_projection_pins_q_v_and_ignores_claimed_rho_floor() {
        let projection = project_planar_lc_full_state_exact_rational(
            &unit_chart_with_floor("2.0"),
            &hand_state(),
        )
        .unwrap();
        assert_eq!(projection.rho(), &point(1, 1));
        assert_eq!(
            projection.positions(),
            &[
                [point(-1, 2), point(0, 1)],
                [point(1, 2), point(0, 1)],
                [point(3, 1), point(0, 1)],
            ]
        );
        assert_eq!(
            projection.velocities(),
            &[
                [point(6, 1), point(8, 1)],
                [point(8, 1), point(8, 1)],
                [point(16, 1), point(18, 1)],
            ]
        );
        projection
            .mass_profile()
            .validate_against(
                &[
                    ExactBinary64::from_bits(1.0_f64.to_bits()).unwrap(),
                    ExactBinary64::from_bits(1.0_f64.to_bits()).unwrap(),
                    ExactBinary64::from_bits(1.0_f64.to_bits()).unwrap(),
                ],
                [0, 1],
            )
            .unwrap();
    }

    #[test]
    fn full_state_projection_is_deck_invariant_contains_points_and_rejects_zero_rho() {
        let chain = chain();
        let mut pairs = Vec::new();
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap();
            let point_projection =
                project_planar_lc_full_state_exact_rational(&chart, &hand_state()).unwrap();
            let deck = hand_state().deck_transform().unwrap();
            assert_eq!(
                project_planar_lc_full_state_exact_rational(&chart, &deck).unwrap(),
                point_projection
            );
            let inflated = hand_state().inflate(&r(1, 100)).unwrap();
            let interval_projection =
                project_planar_lc_full_state_exact_rational(&chart, &inflated).unwrap();
            for body in 0..BODY_COUNT {
                for axis in 0..PLANE_DIMENSION {
                    assert!(interval_projection.positions()[body][axis]
                        .contains_interval(&point_projection.positions()[body][axis]));
                    assert!(interval_projection.velocities()[body][axis]
                        .contains_interval(&point_projection.velocities()[body][axis]));
                }
            }
            pairs.push(chart.pair());
        }
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(pairs, vec![[0, 1], [0, 2], [1, 2]]);

        let zero = PlanarLcIntervalState::from_exact_components(std::array::from_fn(|_| {
            BigRational::zero()
        }))
        .unwrap();
        assert_eq!(
            project_planar_lc_full_state_exact_rational(&unit_chart_with_floor("0.0"), &zero),
            Err(PlanarLcProjectionError::RhoNotStrictlyPositive)
        );
    }

    #[test]
    fn first_canonical_exit_projection_is_feasible() {
        let admission = CanonicalRawV1Admission::admit(SUCCESS).unwrap();
        let SegmentWire::OrdinaryBridge(bridge) = &admission.wire().segments[0] else {
            panic!()
        };
        let SegmentWire::PlanarLcPassage(passage) = &admission.wire().segments[1] else {
            panic!()
        };
        let source_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
        let parent = RationalInterval::try_point(BigRational::new(
            BigInt::from(1),
            BigInt::from(1_u64 << 40),
        ))
        .unwrap();
        let entry_replay = replay_carried_planar_lc_entry_exact_rational_v04(
            &admission,
            1,
            &source_chart,
            &source_tube,
            &parent,
        )
        .unwrap();
        assert!(entry_replay.conditional_profile_satisfied());
        let entry =
            planar_lc_entry_input_from_admission(&admission, 1, &source_chart, &source_tube)
                .unwrap();
        let polynomial = PlanarLcStatePolynomial::from_chart(entry.target_chart()).unwrap();
        let right =
            RationalInterval::try_point(entry.target_chart().parameter_interval().upper().clone())
                .unwrap();
        let state = polynomial
            .evaluate(&right)
            .unwrap()
            .inflate(entry_replay.target_tube_replay().gronwall_upper().unwrap())
            .unwrap();
        let projected =
            project_planar_lc_full_state_exact_rational(entry.target_chart(), &state).unwrap();
        let target_chart = ordinary_chart_input_from_wire(&passage.target_chart).unwrap();
        let target_tube = ordinary_tube_input_from_wire(&passage.target_tube);
        let anchor =
            RationalInterval::try_point(target_chart.parameter_interval().lower().clone()).unwrap();
        let mut centers = target_chart
            .position_polynomial()
            .evaluate(&anchor)
            .unwrap();
        centers.extend(
            target_chart
                .velocity_polynomial()
                .evaluate(&anchor)
                .unwrap(),
        );
        let projected_flat = projected
            .positions()
            .iter()
            .flatten()
            .chain(projected.velocities().iter().flatten())
            .collect::<Vec<_>>();
        let maximum =
            projected_flat
                .iter()
                .zip(&centers)
                .fold(BigRational::zero(), |m, (value, center)| {
                    m.max((value.lower() - center.lower()).abs())
                        .max((value.upper() - center.lower()).abs())
                });
        assert!(maximum <= *target_tube.initial_error_bound());
        let d_out = state.physical_time().clone();
        let b_prime = d_out.subtract(&anchor).unwrap();
        assert_eq!(d_out, b_prime);
    }

    #[test]
    fn projection_is_exactly_invariant_under_the_lc_deck_transform() {
        let chart = unit_chart_with_floor("0.0");
        let state = hand_state();
        let original = replay_planar_lc_projection_default(&chart, &state).unwrap();
        let deck =
            replay_planar_lc_projection_default(&chart, &state.deck_transform().unwrap()).unwrap();
        assert_eq!(deck, original);
    }

    #[test]
    fn every_canonical_chart_interval_projects_within_its_claimed_allowance() {
        let chain = chain();
        let mut count = 0;
        for segment in &chain.segments {
            let SegmentWire::PlanarLcPassage(segment) = segment else {
                continue;
            };
            let chart = planar_lc_chart_input_from_wire(&segment.lc_chart).unwrap();
            let state = PlanarLcStatePolynomial::from_chart(&chart)
                .unwrap()
                .evaluate(chart.parameter_interval())
                .unwrap();
            let replay = replay_planar_lc_projection_default(&chart, &state).unwrap();
            assert!(
                replay.maximum_absolute_residual_endpoint() + chart.tail_bound()
                    < *chart.projected_residual_tolerance()
            );
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn strict_floor_equality_crossing_and_negative_claim_fail_as_domain_errors() {
        let equality = unit_chart_with_floor("1.0");
        assert!(matches!(
            replay_planar_lc_projection_default(&equality, &hand_state()),
            Err(PlanarLcProjectionError::RhoFloorNotStrict)
        ));
        let crossing_chart = unit_chart_with_floor("0.25");
        let mut components = hand_state().components().clone();
        components[LC_ZX] = RationalInterval::new(r(1, 2), r(1, 1)).unwrap();
        let crossing = PlanarLcIntervalState::from_interval_components(components).unwrap();
        assert!(matches!(
            replay_planar_lc_projection_default(&crossing_chart, &crossing),
            Err(PlanarLcProjectionError::RhoFloorNotStrict)
        ));
        let negative = unit_chart_with_floor("-1.0");
        assert_eq!(
            replay_planar_lc_projection_default(&negative, &hand_state()),
            Err(PlanarLcProjectionError::NegativeProjectionFloor)
        );
    }

    #[test]
    fn third_body_collision_and_precision_exhaustion_fail_closed() {
        let chart = unit_chart_with_floor("0.0");
        let mut values = hand_state().components().clone();
        values[LC_YX] = point(-1, 2);
        let collision = PlanarLcIntervalState::from_interval_components(values).unwrap();
        assert!(matches!(
            replay_planar_lc_projection_default(&chart, &collision),
            Err(PlanarLcProjectionError::Field(
                PlanarLcFieldError::ThirdBodyCollision { .. }
            ))
        ));
        assert_eq!(
            replay_planar_lc_projection(&chart, &hand_state(), HARD_MAX_SQRT_PRECISION_BITS + 1),
            Err(PlanarLcProjectionError::SqrtPrecisionExceeded {
                requested: HARD_MAX_SQRT_PRECISION_BITS + 1,
                limit: HARD_MAX_SQRT_PRECISION_BITS,
            })
        );
    }

    #[test]
    fn maximum_is_exactly_recomputed_from_all_six_residual_endpoints() {
        let replay =
            replay_planar_lc_projection_default(&unit_chart_with_floor("0.0"), &hand_state())
                .unwrap();
        let recomputed = replay
            .residuals()
            .iter()
            .flat_map(|residual| [residual.lower().abs(), residual.upper().abs()])
            .max()
            .unwrap();
        assert_eq!(replay.maximum_absolute_residual_endpoint(), &recomputed);
    }

    #[test]
    fn unrelated_chart_claims_and_time_metadata_do_not_enter_projection() {
        let mut wire = first_wire();
        wire.masses = vec![real("1.0"), real("1.0"), real("1.0")];
        wire.projection_rho_lower_bound = real("0.0");
        let baseline = replay_planar_lc_projection_default(
            &planar_lc_chart_input_from_wire(&wire).unwrap(),
            &hand_state(),
        )
        .unwrap();
        let mut claims = wire;
        claims.tail_bound = real("7.0");
        claims.coefficient_tolerance = real("8.0");
        claims.regularized_residual_tolerance = real("9.0");
        claims.projected_residual_tolerance = real("10.0");
        claims.parameter_interval = [real("2.0"), real("3.0")];
        claims.physical_time_interval = [real("4.0"), real("5.0")];
        claims.sample_count = crate::raw_schema::RawInteger::from_parsed_for_test(
            parse_json_number_lexeme("6", DEFAULT_JSON_NUMBER_LIMITS).unwrap(),
        );
        let changed = replay_planar_lc_projection_default(
            &planar_lc_chart_input_from_wire(&claims).unwrap(),
            &hand_state(),
        )
        .unwrap();
        assert_eq!(changed, baseline);
    }
}
