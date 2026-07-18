//! Exact-rational canonical planar Levi-Civita lift-cover arithmetic.
//!
//! This is not the Section 5.3 entry ledger.  A complete result invokes only
//! the pinned existential constrained-lift/deck/gauge implication; it does not
//! assert that every point of a rectangular lifted patch is constrained.

use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{
    outward_mass::{derive_planar_lc_mass_profile, MassProfileError},
    sqrt_enclosure_dyadic, ExactBinary64, NumericError, PlanarLcChartInput, RationalInterval,
    HARD_MAX_SQRT_PRECISION_BITS, PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION,
};

pub const EXACT_RATIONAL_PLANAR_LC_LIFT_COVER_V04_PROFILE_ID: &str =
    "exact_rational_planar_lc_lift_cover_v04";
pub const PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID: &str =
    "planar_lc_constrained_lift_deck_gauge_kernel_v1";
pub const PLANAR_LC_LIFT_DEFAULT_SQRT_PRECISION_BITS: usize = 256;
pub const PLANAR_CARTESIAN_STATE_DIMENSION: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcCanonicalLiftCase {
    ClosedUpperSingleton,
    ClosedLowerSingleton,
    RightHalfSingleton,
    StrictNegativeCutTwoPatch,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarLcLiftPatchLabel {
    Upper,
    Lower,
    Right,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcLiftParityEdge {
    source_patch: usize,
    target_patch: usize,
    parity: u8,
}

impl PlanarLcLiftParityEdge {
    pub const fn source_patch(&self) -> usize {
        self.source_patch
    }

    pub const fn target_patch(&self) -> usize {
        self.target_patch
    }

    pub const fn parity(&self) -> u8 {
        self.parity
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcLiftPatch {
    label: PlanarLcLiftPatchLabel,
    components: [RationalInterval; PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION],
    rho_lower_bound: BigRational,
}

impl PlanarLcLiftPatch {
    pub const fn label(&self) -> PlanarLcLiftPatchLabel {
        self.label
    }

    /// Component order is `(z,w,h,R,U,y,V)`.
    pub fn components(&self) -> &[RationalInterval; PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION] {
        &self.components
    }

    pub fn rho_lower_bound(&self) -> &BigRational {
        &self.rho_lower_bound
    }

    /// Exact LC deck involution, flipping only `z` and `w`.
    pub fn deck_transform(&self) -> Result<Self, PlanarLcLiftError> {
        let minus_one = -BigRational::one();
        let mut components = self.components.clone();
        for component in &mut components[..4] {
            *component = component.scale(&minus_one)?;
        }
        Ok(Self {
            label: self.label,
            components,
            rho_lower_bound: self.rho_lower_bound.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcLiftReplay {
    mass_kernel_id: &'static str,
    canonical_case: PlanarLcCanonicalLiftCase,
    complete_cover: bool,
    collision_free: bool,
    selected_pair_squared_distance_lower: BigRational,
    minimum_patch_rho_lower_bound: Option<BigRational>,
    relative_position: [RationalInterval; 2],
    relative_velocity: [RationalInterval; 2],
    patches: Vec<PlanarLcLiftPatch>,
    parity_edges: Vec<PlanarLcLiftParityEdge>,
}

impl PlanarLcLiftReplay {
    pub const fn profile_id(&self) -> &'static str {
        EXACT_RATIONAL_PLANAR_LC_LIFT_COVER_V04_PROFILE_ID
    }

    /// The analytic implication is exposed only after its arithmetic cover
    /// hypotheses have been established.  It is existential for the actual
    /// Cartesian state, not a whole-box constraint claim.
    pub fn analytic_kernel_id(&self) -> Option<&'static str> {
        self.complete_cover
            .then_some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID)
    }

    pub const fn mass_kernel_id(&self) -> &'static str {
        self.mass_kernel_id
    }

    pub const fn canonical_case(&self) -> PlanarLcCanonicalLiftCase {
        self.canonical_case
    }

    pub const fn complete_cover(&self) -> bool {
        self.complete_cover
    }

    pub const fn collision_free(&self) -> bool {
        self.collision_free
    }

    pub fn selected_pair_squared_distance_lower(&self) -> &BigRational {
        &self.selected_pair_squared_distance_lower
    }

    pub fn minimum_patch_rho_lower_bound(&self) -> Option<&BigRational> {
        self.minimum_patch_rho_lower_bound.as_ref()
    }

    pub fn relative_position(&self) -> &[RationalInterval; 2] {
        &self.relative_position
    }

    pub fn relative_velocity(&self) -> &[RationalInterval; 2] {
        &self.relative_velocity
    }

    pub fn patches(&self) -> &[PlanarLcLiftPatch] {
        &self.patches
    }

    pub fn parity_edges(&self) -> &[PlanarLcLiftParityEdge] {
        &self.parity_edges
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanarLcLiftError {
    CartesianStateDimension { expected: usize, actual: usize },
    SqrtPrecisionExceeded { requested: usize, limit: usize },
    MassDecode { index: usize, source: NumericError },
    MassProfile(MassProfileError),
    SquareRootDomain { stage: &'static str },
    InternalShapeInvariant,
    Numeric(NumericError),
}

impl fmt::Display for PlanarLcLiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "planar LC lift-cover replay failed: {self:?}")
    }
}

impl std::error::Error for PlanarLcLiftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MassDecode { source, .. } | Self::Numeric(source) => Some(source),
            Self::MassProfile(source) => Some(source),
            _ => None,
        }
    }
}

impl From<NumericError> for PlanarLcLiftError {
    fn from(source: NumericError) -> Self {
        Self::Numeric(source)
    }
}

impl From<MassProfileError> for PlanarLcLiftError {
    fn from(source: MassProfileError) -> Self {
        Self::MassProfile(source)
    }
}

pub fn replay_planar_lc_lift_cover_exact_rational_v04(
    chart: &PlanarLcChartInput,
    cartesian_state: &[RationalInterval],
) -> Result<PlanarLcLiftReplay, PlanarLcLiftError> {
    replay_planar_lc_lift_cover_exact_rational(
        chart,
        cartesian_state,
        PLANAR_LC_LIFT_DEFAULT_SQRT_PRECISION_BITS,
    )
}

pub fn replay_planar_lc_lift_cover_exact_rational(
    chart: &PlanarLcChartInput,
    cartesian_state: &[RationalInterval],
    sqrt_precision_bits: usize,
) -> Result<PlanarLcLiftReplay, PlanarLcLiftError> {
    if cartesian_state.len() != PLANAR_CARTESIAN_STATE_DIMENSION {
        return Err(PlanarLcLiftError::CartesianStateDimension {
            expected: PLANAR_CARTESIAN_STATE_DIMENSION,
            actual: cartesian_state.len(),
        });
    }
    if sqrt_precision_bits > HARD_MAX_SQRT_PRECISION_BITS {
        return Err(PlanarLcLiftError::SqrtPrecisionExceeded {
            requested: sqrt_precision_bits,
            limit: HARD_MAX_SQRT_PRECISION_BITS,
        });
    }
    let state = admit_state(cartesian_state)?;
    let masses = reconstruct_masses(chart)?;
    let mass_profile = derive_planar_lc_mass_profile(&masses, chart.pair())?;
    mass_profile.validate_against(&masses, chart.pair())?;

    let [first, second] = chart.pair();
    let third = mass_profile.third_index();
    let relative_position = relative_pair(&state, first, second, 0)?;
    let relative_velocity = relative_pair(&state, first, second, 6)?;
    let distance_squared =
        interval_square(&relative_position[0])?.add(&interval_square(&relative_position[1])?)?;
    let collision_free = distance_squared.lower().numer().sign() == Sign::Plus;
    let canonical_case = canonical_case(&relative_position);
    if !collision_free || canonical_case == PlanarLcCanonicalLiftCase::Unresolved {
        return Ok(PlanarLcLiftReplay {
            mass_kernel_id: mass_profile.kernel_id(),
            canonical_case,
            complete_cover: false,
            collision_free,
            selected_pair_squared_distance_lower: distance_squared.lower().clone(),
            minimum_patch_rho_lower_bound: None,
            relative_position,
            relative_velocity,
            patches: Vec::new(),
            parity_edges: Vec::new(),
        });
    }

    let common = CommonLiftData::new(
        &state,
        first,
        second,
        third,
        mass_profile.alpha().exact(),
        mass_profile.beta().exact(),
    )?;
    let mut patches = Vec::with_capacity(2);
    match canonical_case {
        PlanarLcCanonicalLiftCase::ClosedUpperSingleton => patches.push(build_patch(
            PlanarLcLiftPatchLabel::Upper,
            &relative_position,
            &relative_velocity,
            &common,
            mass_profile.pair_mass().exact(),
            sqrt_precision_bits,
        )?),
        PlanarLcCanonicalLiftCase::ClosedLowerSingleton => patches.push(build_patch(
            PlanarLcLiftPatchLabel::Lower,
            &relative_position,
            &relative_velocity,
            &common,
            mass_profile.pair_mass().exact(),
            sqrt_precision_bits,
        )?),
        PlanarLcCanonicalLiftCase::RightHalfSingleton => patches.push(build_patch(
            PlanarLcLiftPatchLabel::Right,
            &relative_position,
            &relative_velocity,
            &common,
            mass_profile.pair_mass().exact(),
            sqrt_precision_bits,
        )?),
        PlanarLcCanonicalLiftCase::StrictNegativeCutTwoPatch => {
            let zero = BigRational::zero();
            let upper_position = [
                relative_position[0].clone(),
                RationalInterval::new(zero.clone(), relative_position[1].upper().clone())?,
            ];
            let lower_position = [
                relative_position[0].clone(),
                RationalInterval::new(relative_position[1].lower().clone(), zero)?,
            ];
            patches.push(build_patch(
                PlanarLcLiftPatchLabel::Upper,
                &upper_position,
                &relative_velocity,
                &common,
                mass_profile.pair_mass().exact(),
                sqrt_precision_bits,
            )?);
            patches.push(build_patch(
                PlanarLcLiftPatchLabel::Lower,
                &lower_position,
                &relative_velocity,
                &common,
                mass_profile.pair_mass().exact(),
                sqrt_precision_bits,
            )?);
        }
        PlanarLcCanonicalLiftCase::Unresolved => {
            return Err(PlanarLcLiftError::InternalShapeInvariant);
        }
    }
    let complete_cover = patches
        .iter()
        .all(|patch| patch.rho_lower_bound.numer().sign() == Sign::Plus);
    let minimum_patch_rho_lower_bound = patches
        .iter()
        .map(|patch| patch.rho_lower_bound.clone())
        .min();
    let parity_edges = if canonical_case == PlanarLcCanonicalLiftCase::StrictNegativeCutTwoPatch {
        vec![PlanarLcLiftParityEdge {
            source_patch: 0,
            target_patch: 1,
            parity: 1,
        }]
    } else {
        Vec::new()
    };
    Ok(PlanarLcLiftReplay {
        mass_kernel_id: mass_profile.kernel_id(),
        canonical_case,
        complete_cover,
        collision_free,
        selected_pair_squared_distance_lower: distance_squared.lower().clone(),
        minimum_patch_rho_lower_bound,
        relative_position,
        relative_velocity,
        patches,
        parity_edges,
    })
}

struct CommonLiftData {
    center: [RationalInterval; 2],
    center_velocity: [RationalInterval; 2],
    third_offset: [RationalInterval; 2],
    third_offset_velocity: [RationalInterval; 2],
}

impl CommonLiftData {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: &[RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION],
        first: usize,
        second: usize,
        third: usize,
        alpha: &BigRational,
        beta: &BigRational,
    ) -> Result<Self, PlanarLcLiftError> {
        let center = weighted_pair(state, first, second, 0, beta, alpha)?;
        let center_velocity = weighted_pair(state, first, second, 6, beta, alpha)?;
        let third_position = body_vector(state, third, 0);
        let third_velocity = body_vector(state, third, 6);
        let third_offset = [
            third_position[0].subtract(&center[0])?,
            third_position[1].subtract(&center[1])?,
        ];
        let third_offset_velocity = [
            third_velocity[0].subtract(&center_velocity[0])?,
            third_velocity[1].subtract(&center_velocity[1])?,
        ];
        Ok(Self {
            center,
            center_velocity,
            third_offset,
            third_offset_velocity,
        })
    }
}

fn build_patch(
    label: PlanarLcLiftPatchLabel,
    relative_position: &[RationalInterval; 2],
    relative_velocity: &[RationalInterval; 2],
    common: &CommonLiftData,
    pair_mass: &BigRational,
    sqrt_precision_bits: usize,
) -> Result<PlanarLcLiftPatch, PlanarLcLiftError> {
    let distance_squared =
        interval_square(&relative_position[0])?.add(&interval_square(&relative_position[1])?)?;
    let radius = sqrt_interval(&distance_squared, sqrt_precision_bits, "radius")?;
    let half = BigRational::new(BigInt::one(), BigInt::from(2));
    let real_radicand = radius.add(&relative_position[0])?.scale(&half)?;
    let imaginary_radicand = radius.subtract(&relative_position[0])?.scale(&half)?;
    let zx = sqrt_nonnegative_forced(
        &real_radicand,
        sqrt_precision_bits,
        "real square-root radicand",
    )?;
    let zy_magnitude = sqrt_nonnegative_forced(
        &imaginary_radicand,
        sqrt_precision_bits,
        "imaginary square-root radicand",
    )?;
    let zy = match label {
        PlanarLcLiftPatchLabel::Upper => zy_magnitude,
        PlanarLcLiftPatchLabel::Lower => zy_magnitude.scale(&(-BigRational::one()))?,
        PlanarLcLiftPatchLabel::Right => {
            RationalInterval::new(-zy_magnitude.upper().clone(), zy_magnitude.upper().clone())?
        }
    };
    let wx = zx
        .multiply(&relative_velocity[0])?
        .add(&zy.multiply(&relative_velocity[1])?)?
        .scale(&half)?;
    let wy = zx
        .multiply(&relative_velocity[1])?
        .subtract(&zy.multiply(&relative_velocity[0])?)?
        .scale(&half)?;
    let velocity_squared =
        interval_square(&relative_velocity[0])?.add(&interval_square(&relative_velocity[1])?)?;
    let h = velocity_squared
        .scale(&half)?
        .subtract(&RationalInterval::try_point(pair_mass.clone())?.divide(&radius)?)?;
    let rho = interval_square(&zx)?.add(&interval_square(&zy)?)?;
    let components = [
        zx,
        zy,
        wx,
        wy,
        h,
        common.center[0].clone(),
        common.center[1].clone(),
        common.center_velocity[0].clone(),
        common.center_velocity[1].clone(),
        common.third_offset[0].clone(),
        common.third_offset[1].clone(),
        common.third_offset_velocity[0].clone(),
        common.third_offset_velocity[1].clone(),
    ];
    Ok(PlanarLcLiftPatch {
        label,
        rho_lower_bound: rho.lower().clone(),
        components,
    })
}

fn canonical_case(relative_position: &[RationalInterval; 2]) -> PlanarLcCanonicalLiftCase {
    let zero = BigRational::zero();
    let x = &relative_position[0];
    let y = &relative_position[1];
    if y.lower() >= &zero {
        PlanarLcCanonicalLiftCase::ClosedUpperSingleton
    } else if y.upper() <= &zero {
        PlanarLcCanonicalLiftCase::ClosedLowerSingleton
    } else if x.lower() > &zero {
        PlanarLcCanonicalLiftCase::RightHalfSingleton
    } else if y.lower() < &zero && y.upper() > &zero && x.upper() < &zero {
        PlanarLcCanonicalLiftCase::StrictNegativeCutTwoPatch
    } else {
        PlanarLcCanonicalLiftCase::Unresolved
    }
}

fn sqrt_interval(
    value: &RationalInterval,
    precision_bits: usize,
    stage: &'static str,
) -> Result<RationalInterval, PlanarLcLiftError> {
    if value.lower().numer().sign() == Sign::Minus {
        return Err(PlanarLcLiftError::SquareRootDomain { stage });
    }
    let lower = sqrt_enclosure_dyadic(value.lower(), precision_bits)?;
    let upper = sqrt_enclosure_dyadic(value.upper(), precision_bits)?;
    Ok(RationalInterval::new(
        lower.interval().lower().clone(),
        upper.interval().upper().clone(),
    )?)
}

fn sqrt_nonnegative_forced(
    value: &RationalInterval,
    precision_bits: usize,
    stage: &'static str,
) -> Result<RationalInterval, PlanarLcLiftError> {
    if value.upper().numer().sign() == Sign::Minus {
        return Err(PlanarLcLiftError::SquareRootDomain { stage });
    }
    let clamped = RationalInterval::new(
        BigRational::zero().max(value.lower().clone()),
        value.upper().clone(),
    )?;
    sqrt_interval(&clamped, precision_bits, stage)
}

fn admit_state(
    state: &[RationalInterval],
) -> Result<[RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION], PlanarLcLiftError> {
    let mut admitted = Vec::with_capacity(PLANAR_CARTESIAN_STATE_DIMENSION);
    for interval in state {
        admitted.push(RationalInterval::new(
            interval.lower().clone(),
            interval.upper().clone(),
        )?);
    }
    admitted
        .try_into()
        .map_err(|_| PlanarLcLiftError::InternalShapeInvariant)
}

fn reconstruct_masses(chart: &PlanarLcChartInput) -> Result<[ExactBinary64; 3], PlanarLcLiftError> {
    let mut masses = Vec::with_capacity(3);
    for (index, bits) in chart.mass_binary64_bits().iter().copied().enumerate() {
        masses.push(
            ExactBinary64::from_bits(bits)
                .map_err(|source| PlanarLcLiftError::MassDecode { index, source })?,
        );
    }
    masses
        .try_into()
        .map_err(|_| PlanarLcLiftError::InternalShapeInvariant)
}

fn relative_pair(
    state: &[RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION],
    first: usize,
    second: usize,
    offset: usize,
) -> Result<[RationalInterval; 2], PlanarLcLiftError> {
    let first = body_vector(state, first, offset);
    let second = body_vector(state, second, offset);
    Ok([second[0].subtract(first[0])?, second[1].subtract(first[1])?])
}

fn weighted_pair(
    state: &[RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION],
    first: usize,
    second: usize,
    offset: usize,
    first_weight: &BigRational,
    second_weight: &BigRational,
) -> Result<[RationalInterval; 2], PlanarLcLiftError> {
    let first = body_vector(state, first, offset);
    let second = body_vector(state, second, offset);
    Ok([
        first[0]
            .scale(first_weight)?
            .add(&second[0].scale(second_weight)?)?,
        first[1]
            .scale(first_weight)?
            .add(&second[1].scale(second_weight)?)?,
    ])
}

fn body_vector(
    state: &[RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION],
    body: usize,
    offset: usize,
) -> [&RationalInterval; 2] {
    [&state[offset + 2 * body], &state[offset + 2 * body + 1]]
}

fn interval_square(value: &RationalInterval) -> Result<RationalInterval, PlanarLcLiftError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ordinary_chart_input_from_wire, ordinary_tube_input_from_wire, parse_wire_json,
        planar_lc_chart_input_from_wire,
        raw_schema::{decode_raw_chain, SchemaProfile, SegmentWire},
        replay_carried_ordinary_bridge_exact_rational_v04, DEFAULT_WIRE_JSON_LIMITS,
    };

    const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
    ));
    const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../artifacts/v0.3.0-review/planar-chain/failed-revisit.raw.json"
    ));

    fn rational(value: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn point(value: i64) -> RationalInterval {
        RationalInterval::try_point(rational(value)).unwrap()
    }

    fn interval(lower: i64, upper: i64) -> RationalInterval {
        RationalInterval::new(rational(lower), rational(upper)).unwrap()
    }

    fn charts() -> Vec<PlanarLcChartInput> {
        let syntax = parse_wire_json(SUCCESS_RAW, DEFAULT_WIRE_JSON_LIMITS).unwrap();
        let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
        let mut charts = chain
            .segments
            .into_iter()
            .filter_map(|segment| match segment {
                SegmentWire::PlanarLcPassage(passage) => {
                    Some(planar_lc_chart_input_from_wire(&passage.lc_chart).unwrap())
                }
                SegmentWire::OrdinaryBridge(_) => None,
            })
            .collect::<Vec<_>>();
        charts.sort_by_key(PlanarLcChartInput::pair);
        charts.dedup_by_key(|chart| chart.pair());
        charts
    }

    fn state_for_pair(
        pair: [usize; 2],
        relative_position: [RationalInterval; 2],
        relative_velocity: [RationalInterval; 2],
    ) -> [RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION] {
        let zero = point(0);
        let mut state = std::array::from_fn(|_| zero.clone());
        let third = (0..3).find(|index| !pair.contains(index)).unwrap();
        state[2 * pair[0]] = point(1);
        state[2 * pair[0] + 1] = point(2);
        state[2 * pair[1]] = point(1).add(&relative_position[0]).unwrap();
        state[2 * pair[1] + 1] = point(2).add(&relative_position[1]).unwrap();
        state[2 * third] = point(10);
        state[2 * third + 1] = point(20);
        state[6 + 2 * pair[0]] = point(1);
        state[6 + 2 * pair[0] + 1] = point(0);
        state[6 + 2 * pair[1]] = point(1).add(&relative_velocity[0]).unwrap();
        state[6 + 2 * pair[1] + 1] = relative_velocity[1].clone();
        state[6 + 2 * third] = point(7);
        state[6 + 2 * third + 1] = point(8);
        state
    }

    fn branch_state(
        chart: &PlanarLcChartInput,
        qx: RationalInterval,
        qy: RationalInterval,
    ) -> [RationalInterval; PLANAR_CARTESIAN_STATE_DIMENSION] {
        state_for_pair(chart.pair(), [qx, qy], [point(0), point(0)])
    }

    fn lc_square(patch: &PlanarLcLiftPatch) -> [RationalInterval; 2] {
        let z = patch.components();
        let two = rational(2);
        [
            interval_square(&z[0])
                .unwrap()
                .subtract(&interval_square(&z[1]).unwrap())
                .unwrap(),
            z[0].multiply(&z[1]).unwrap().scale(&two).unwrap(),
        ]
    }

    #[test]
    fn perfect_square_points_pin_all_pairs_mass_roles_and_velocity_convention() {
        let charts = charts();
        assert_eq!(
            charts
                .iter()
                .map(PlanarLcChartInput::pair)
                .collect::<Vec<_>>(),
            vec![[0, 1], [0, 2], [1, 2]]
        );
        for chart in charts {
            let state = state_for_pair(chart.pair(), [point(3), point(4)], [point(4), point(2)]);
            let replay = replay_planar_lc_lift_cover_exact_rational_v04(&chart, &state).unwrap();
            assert!(replay.complete_cover());
            assert_eq!(
                replay.canonical_case(),
                PlanarLcCanonicalLiftCase::ClosedUpperSingleton
            );
            let patch = &replay.patches()[0];
            let lifted = patch.components();
            assert_eq!(lifted[0], point(2));
            assert_eq!(lifted[1], point(1));
            assert_eq!(lifted[2], point(5));
            assert_eq!(lifted[3], point(0));
            assert_eq!(lc_square(patch), replay.relative_position().clone());

            // Frozen §3.4 uses L(z)=2[[zx,-zy],[zy,zx]], so its
            // (1/4)L^T v is exactly the componentwise one-half formula.
            // Conversely v=2 L0(z) w/rho for L0 without that factor two.
            let rho = lifted[0]
                .multiply(&lifted[0])
                .unwrap()
                .add(&lifted[1].multiply(&lifted[1]).unwrap())
                .unwrap();
            let recovered_vx = lifted[0]
                .multiply(&lifted[2])
                .unwrap()
                .subtract(&lifted[1].multiply(&lifted[3]).unwrap())
                .unwrap()
                .scale(&rational(2))
                .unwrap()
                .divide(&rho)
                .unwrap();
            let recovered_vy = lifted[1]
                .multiply(&lifted[2])
                .unwrap()
                .add(&lifted[0].multiply(&lifted[3]).unwrap())
                .unwrap()
                .scale(&rational(2))
                .unwrap()
                .divide(&rho)
                .unwrap();
            assert_eq!(
                [recovered_vx, recovered_vy],
                replay.relative_velocity().clone()
            );

            let masses = chart.masses();
            let pair_mass = &masses[chart.pair()[0]] + &masses[chart.pair()[1]];
            let expected_h = BigRational::from_integer(BigInt::from(10))
                - &pair_mass / BigRational::from_integer(BigInt::from(5));
            assert_eq!(lifted[4], RationalInterval::try_point(expected_h).unwrap());
            let alpha = &masses[chart.pair()[1]] / &pair_mass;
            let beta = &masses[chart.pair()[0]] / &pair_mass;
            let expected_rx = &beta + &alpha * rational(4);
            let expected_ry = &beta * rational(2) + &alpha * rational(6);
            assert_eq!(
                lifted[5],
                RationalInterval::try_point(expected_rx.clone()).unwrap()
            );
            assert_eq!(
                lifted[6],
                RationalInterval::try_point(expected_ry.clone()).unwrap()
            );
            assert_eq!(
                lifted[9],
                RationalInterval::try_point(rational(10) - expected_rx).unwrap()
            );
            assert_eq!(
                lifted[10],
                RationalInterval::try_point(rational(20) - expected_ry).unwrap()
            );
            let expected_ux = &beta + &alpha * rational(5);
            let expected_uy = &alpha * rational(2);
            assert_eq!(
                lifted[7],
                RationalInterval::try_point(expected_ux.clone()).unwrap()
            );
            assert_eq!(
                lifted[8],
                RationalInterval::try_point(expected_uy.clone()).unwrap()
            );
            assert_eq!(
                lifted[11],
                RationalInterval::try_point(rational(7) - expected_ux).unwrap()
            );
            assert_eq!(
                lifted[12],
                RationalInterval::try_point(rational(8) - expected_uy).unwrap()
            );
        }
    }

    #[test]
    fn canonical_decision_order_and_equality_boundaries_are_exact() {
        let chart = &charts()[0];
        for (qx, qy, expected, patch_count) in [
            (
                interval(-3, -2),
                interval(0, 1),
                PlanarLcCanonicalLiftCase::ClosedUpperSingleton,
                1,
            ),
            (
                interval(3, 4),
                interval(-2, 0),
                PlanarLcCanonicalLiftCase::ClosedLowerSingleton,
                1,
            ),
            (
                interval(1, 2),
                interval(-1, 1),
                PlanarLcCanonicalLiftCase::RightHalfSingleton,
                1,
            ),
            (
                interval(-3, -2),
                interval(-1, 1),
                PlanarLcCanonicalLiftCase::StrictNegativeCutTwoPatch,
                2,
            ),
            (
                interval(-1, 1),
                interval(-2, 2),
                PlanarLcCanonicalLiftCase::Unresolved,
                0,
            ),
        ] {
            let replay =
                replay_planar_lc_lift_cover_exact_rational_v04(chart, &branch_state(chart, qx, qy))
                    .unwrap();
            assert_eq!(replay.canonical_case(), expected);
            assert_eq!(replay.patches().len(), patch_count);
        }
        let equality = replay_planar_lc_lift_cover_exact_rational_v04(
            chart,
            &branch_state(chart, point(-2), point(0)),
        )
        .unwrap();
        assert_eq!(
            equality.canonical_case(),
            PlanarLcCanonicalLiftCase::ClosedUpperSingleton
        );
    }

    #[test]
    fn collision_containing_box_is_a_sound_non_cover() {
        let chart = &charts()[0];
        let replay = replay_planar_lc_lift_cover_exact_rational_v04(
            chart,
            &branch_state(chart, interval(-1, 1), interval(0, 1)),
        )
        .unwrap();
        assert!(!replay.collision_free());
        assert!(!replay.complete_cover());
        assert!(replay.patches().is_empty());
        assert_eq!(replay.analytic_kernel_id(), None);
    }

    #[test]
    fn negative_cut_has_canonical_parity_one_and_deck_is_an_involution() {
        let chart = &charts()[0];
        let replay = replay_planar_lc_lift_cover_exact_rational_v04(
            chart,
            &branch_state(chart, interval(-3, -2), interval(-1, 1)),
        )
        .unwrap();
        assert!(replay.complete_cover());
        assert_eq!(replay.patches()[0].label(), PlanarLcLiftPatchLabel::Upper);
        assert_eq!(replay.patches()[1].label(), PlanarLcLiftPatchLabel::Lower);
        assert_eq!(replay.parity_edges().len(), 1);
        let edge = &replay.parity_edges()[0];
        assert_eq!(
            (edge.source_patch(), edge.target_patch(), edge.parity()),
            (0, 1, 1)
        );
        for patch in replay.patches() {
            let deck = patch.deck_transform().unwrap();
            assert_eq!(deck.deck_transform().unwrap(), *patch);
            assert_eq!(deck.rho_lower_bound(), patch.rho_lower_bound());
            for index in 4..PLANAR_LC_LIFTED_STATE_WITHOUT_TIME_DIMENSION {
                assert_eq!(deck.components()[index], patch.components()[index]);
            }
        }
    }

    #[test]
    fn representative_point_lifts_are_contained_by_each_applicable_patch() {
        let chart = &charts()[0];
        for (box_qx, box_qy, samples) in [
            (interval(2, 4), interval(1, 3), vec![(3, 2, 0_usize)]),
            (interval(2, 4), interval(-3, -1), vec![(3, -2, 0)]),
            (interval(2, 4), interval(-2, 2), vec![(3, 1, 0), (3, -1, 0)]),
            (
                interval(-4, -2),
                interval(-2, 2),
                vec![(-3, 1, 0), (-3, -1, 1)],
            ),
        ] {
            let cover = replay_planar_lc_lift_cover_exact_rational_v04(
                chart,
                &branch_state(chart, box_qx, box_qy),
            )
            .unwrap();
            assert!(cover.complete_cover());
            for (qx, qy, cover_patch_index) in samples {
                let point_lift = replay_planar_lc_lift_cover_exact_rational_v04(
                    chart,
                    &branch_state(chart, point(qx), point(qy)),
                )
                .unwrap();
                assert!(point_lift.complete_cover());
                for (cover_component, point_component) in cover.patches()[cover_patch_index]
                    .components()
                    .iter()
                    .zip(point_lift.patches()[0].components())
                {
                    assert!(cover_component.contains_interval(point_component));
                }
            }
        }
    }

    #[test]
    fn malformed_dimension_and_precision_fail_typed() {
        let chart = &charts()[0];
        assert_eq!(
            replay_planar_lc_lift_cover_exact_rational_v04(chart, &[]),
            Err(PlanarLcLiftError::CartesianStateDimension {
                expected: 12,
                actual: 0
            })
        );
        let state = branch_state(chart, point(3), point(4));
        assert_eq!(
            replay_planar_lc_lift_cover_exact_rational(
                chart,
                &state,
                HARD_MAX_SQRT_PRECISION_BITS + 1,
            ),
            Err(PlanarLcLiftError::SqrtPrecisionExceeded {
                requested: HARD_MAX_SQRT_PRECISION_BITS + 1,
                limit: HARD_MAX_SQRT_PRECISION_BITS,
            })
        );
    }

    #[test]
    fn both_first_carried_lc_entry_source_boxes_have_complete_positive_rho_covers() {
        for bytes in [SUCCESS_RAW, FAILED_REVISIT_RAW] {
            let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
            let chain = decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap();
            let SegmentWire::OrdinaryBridge(bridge) = &chain.segments[0] else {
                panic!("first canonical segment must be an ordinary bridge")
            };
            let SegmentWire::PlanarLcPassage(passage) = &chain.segments[1] else {
                panic!("second canonical segment must be an LC passage")
            };
            let source_chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
            let source_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
            let carried_chart = ordinary_chart_input_from_wire(&bridge.target_chart).unwrap();
            let carried_tube = ordinary_tube_input_from_wire(&bridge.target_tube);
            let bridge_replay = replay_carried_ordinary_bridge_exact_rational_v04(
                &bridge.transition,
                &source_chart,
                &source_tube,
                &carried_chart,
                &carried_tube,
                &RationalInterval::try_point(BigRational::zero()).unwrap(),
            )
            .unwrap();
            assert!(bridge_replay.local_bridge_satisfied());
            let radius = bridge_replay
                .target_tube_replay()
                .unwrap()
                .gronwall_upper()
                .unwrap();
            let parameter =
                RationalInterval::try_point(carried_chart.parameter_interval().upper().clone())
                    .unwrap();
            let mut centers = carried_chart
                .position_polynomial()
                .evaluate(&parameter)
                .unwrap();
            centers.extend(
                carried_chart
                    .velocity_polynomial()
                    .evaluate(&parameter)
                    .unwrap(),
            );
            assert_eq!(centers.len(), PLANAR_CARTESIAN_STATE_DIMENSION);
            let inflation = RationalInterval::new(-radius.clone(), radius.clone()).unwrap();
            let source_box = centers
                .iter()
                .map(|center| center.add(&inflation).unwrap())
                .collect::<Vec<_>>();
            let lc_chart = planar_lc_chart_input_from_wire(&passage.lc_chart).unwrap();
            let lift =
                replay_planar_lc_lift_cover_exact_rational_v04(&lc_chart, &source_box).unwrap();
            assert!(lift.collision_free());
            assert!(lift.complete_cover());
            assert!(lift.selected_pair_squared_distance_lower() > &BigRational::zero());
            assert!(lift.minimum_patch_rho_lower_bound().unwrap() > &BigRational::zero());
            assert_eq!(
                lift.analytic_kernel_id(),
                Some(PLANAR_LC_CONSTRAINED_LIFT_DECK_GAUGE_KERNEL_V1_ID)
            );
        }
    }
}
