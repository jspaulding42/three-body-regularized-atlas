//! Exact outward binary64 rounding and the raw-v1 planar LC mass profile.
//!
//! This module never asks the host floating-point implementation to round an
//! exact rational.  It locates the two neighboring finite binary64 values by
//! a monotone binary search over their bit encodings, compares exact rational
//! distances to select the nearest-even value, and then checks explicit
//! containment, adjacency, and nearest-even postconditions.
//!
//! The mass-profile formulas are the formulas in raw-v1 specification
//! Section 3.3.  Inputs are already-decoded finite binary64 masses, so their
//! [`ExactBinary64`] rational values are the serialized dyadics themselves.

use core::fmt;

use num_bigint::Sign;
use num_rational::BigRational;
use num_traits::Zero;

use crate::{ExactBinary64, NumericError};

const SIGN_MASK: u64 = 1_u64 << 63;
const POSITIVE_MAX_FINITE_BITS: u64 = 0x7fef_ffff_ffff_ffff;

/// Non-configurable component-size ceiling for public exact-rational inputs.
///
/// This is an **input** cap, not a claim that every arithmetic temporary fits
/// below 8,192 bits.  After an input is bounded and proved canonical, the
/// outward algorithm combines it only with finite binary64 endpoints, whose
/// numerator/denominator have at most 1,024/1,075 bits.  An exact endpoint
/// difference therefore has unreduced components of at most 9,268/9,267
/// bits.  Even a cross-product comparison of two such differences, including
/// one carry bit, is bounded by
/// [`HARD_MAX_OUTWARD_TRANSIENT_COMPONENT_BITS`].  Euclidean normalization
/// does not increase component sizes.
///
/// The ceiling is above the conservative 5,325-bit maximum needed by every
/// normalized Section 3.3 coefficient derived from three finite binary64
/// masses.  To see that bound, write a decoded mass as `a/b`: its numerator
/// and denominator have at most 1,024 and 1,075 bits.  Using the safe rules
/// `bits(x*y) <= bits(x)+bits(y)` and
/// `bits(x+y) <= max(bits(x),bits(y))+1`, unreduced pair-mass components are
/// at most 2,100/2,150 bits; a mass/pair-mass ratio is at most
/// 3,174/3,175; a center coefficient is at most 4,198/4,250; and an offset
/// coefficient is at most 5,275/5,325.  Rational normalization can only
/// reduce those component sizes.  The fixed 8,192-bit ceiling therefore has
/// substantial headroom while keeping every lattice comparison bounded.
pub const HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS: u64 = 8_192;

/// Fixed conservative component bound for transient outward-rounding
/// arithmetic after a valid capped input crosses the public boundary.
pub const HARD_MAX_OUTWARD_TRANSIENT_COMPONENT_BITS: u64 = 18_536;

/// Conservative algebraic upper bound justified in the hard-limit comment.
pub const DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS: u64 = 5_325;

/// Identifier pinned by raw-v1 for the exact-derived outward mass kernel.
pub const PLANAR_LC_MASS_KERNEL_ID: &str =
    "planar_lc_mass_coefficients_exact_binary64_fraction_outward_v1";

/// Which IEEE-754 zero encoding to retain when the exact rational is zero.
///
/// A rational has no signed-zero identity.  The ordinary entry point chooses
/// [`ZeroSign::Positive`]; callers reconstructing a signed serialized zero can
/// request [`ZeroSign::Negative`] explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZeroSign {
    Positive,
    Negative,
}

impl ZeroSign {
    const fn bits(self) -> u64 {
        match self {
            Self::Positive => 0,
            Self::Negative => SIGN_MASK,
        }
    }
}

/// A failed exact outward-rounding construction or postcondition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutwardBinary64Error {
    Numeric(NumericError),
    /// An arbitrary rational exceeded the non-configurable kernel ceiling.
    RationalComponentBitLimitExceeded {
        numerator_bits: u64,
        denominator_bits: u64,
        limit: u64,
    },
    /// `BigRational::new_raw` admitted a denominator of zero.
    RationalZeroDenominator,
    /// Canonical rationals require a strictly positive denominator.
    RationalNegativeDenominator,
    /// Raw components were not the unique reduced canonical form.
    RationalNotCanonical,
    /// The exact rational lies strictly outside the finite binary64 range.
    NoFiniteEnclosure,
    Postcondition(OutwardPostcondition),
}

/// The exact invariant that an alleged outward enclosure violated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutwardPostcondition {
    BoundsReversed,
    DoesNotContainExactValue,
    ExactPointNotRepresentedAsPoint,
    NonExactValueRepresentedAsPoint,
    BoundsNotAdjacent,
    BoundSignMismatch,
    NearestNotAnEndpoint,
    NearestNotNearestEven,
    SignedZeroMismatch,
}

impl fmt::Display for OutwardBinary64Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric(error) => write!(formatter, "binary64 decoding failed: {error}"),
            Self::RationalComponentBitLimitExceeded {
                numerator_bits,
                denominator_bits,
                limit,
            } => write!(
                formatter,
                "exact rational component limit exceeded: numerator={numerator_bits} bits, \
                 denominator={denominator_bits} bits, limit={limit} bits"
            ),
            Self::RationalZeroDenominator => {
                formatter.write_str("exact rational has a zero denominator")
            }
            Self::RationalNegativeDenominator => {
                formatter.write_str("exact rational has a negative denominator")
            }
            Self::RationalNotCanonical => formatter
                .write_str("exact rational is not in reduced canonical numerator/denominator form"),
            Self::NoFiniteEnclosure => formatter.write_str(
                "exact rational has no containing interval with finite binary64 endpoints",
            ),
            Self::Postcondition(condition) => {
                write!(
                    formatter,
                    "outward binary64 postcondition failed: {condition}"
                )
            }
        }
    }
}

impl fmt::Display for OutwardPostcondition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::BoundsReversed => "lower endpoint exceeds upper endpoint",
            Self::DoesNotContainExactValue => "endpoints do not contain the exact rational",
            Self::ExactPointNotRepresentedAsPoint => {
                "an exactly representable rational is not encoded as one point"
            }
            Self::NonExactValueRepresentedAsPoint => {
                "a non-representable rational is encoded as one point"
            }
            Self::BoundsNotAdjacent => "non-point endpoints are not adjacent binary64 values",
            Self::BoundSignMismatch => "an endpoint has the wrong signed branch",
            Self::NearestNotAnEndpoint => "nearest value is not one of the two endpoints",
            Self::NearestNotNearestEven => "nearest value fails exact nearest-even selection",
            Self::SignedZeroMismatch => "zero endpoint does not preserve the requested sign",
        };
        formatter.write_str(description)
    }
}

impl std::error::Error for OutwardBinary64Error {}

impl From<NumericError> for OutwardBinary64Error {
    fn from(error: NumericError) -> Self {
        Self::Numeric(error)
    }
}

/// The tight one- or two-point finite binary64 enclosure of an exact rational.
///
/// For a non-point interval, `lower` and `upper` are adjacent in the finite
/// binary64 lattice.  `nearest` is one of them, selected by exact distance and
/// the IEEE-754 ties-to-even rule.  Signed zero is retained in endpoint bits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutwardBinary64Enclosure {
    lower: ExactBinary64,
    upper: ExactBinary64,
    nearest: ExactBinary64,
}

impl OutwardBinary64Enclosure {
    /// Decode an externally supplied bit-level claim and accept it only after
    /// all exact tight-enclosure postconditions pass.
    pub fn from_bits_checked(
        exact: &BigRational,
        zero_sign: ZeroSign,
        lower_bits: u64,
        upper_bits: u64,
        nearest_bits: u64,
    ) -> Result<Self, OutwardBinary64Error> {
        validate_bounded_canonical_rational(exact)?;
        let enclosure = Self {
            lower: ExactBinary64::from_bits(lower_bits)?,
            upper: ExactBinary64::from_bits(upper_bits)?,
            nearest: ExactBinary64::from_bits(nearest_bits)?,
        };
        enclosure.validate(exact, zero_sign)?;
        Ok(enclosure)
    }

    pub fn lower(&self) -> &ExactBinary64 {
        &self.lower
    }

    pub fn upper(&self) -> &ExactBinary64 {
        &self.upper
    }

    pub fn nearest(&self) -> &ExactBinary64 {
        &self.nearest
    }

    pub fn is_point(&self) -> bool {
        self.lower.bits() == self.upper.bits()
    }

    /// Recheck containment, tight adjacency, signed-zero, and nearest-even
    /// selection using exact rational arithmetic.
    pub fn validate(
        &self,
        exact: &BigRational,
        zero_sign: ZeroSign,
    ) -> Result<(), OutwardBinary64Error> {
        validate_outward_binary64_enclosure(exact, zero_sign, self)
    }
}

/// Construct the tight finite binary64 enclosure, choosing positive zero for
/// an exact rational zero.
pub fn tight_outward_binary64(
    exact: &BigRational,
) -> Result<OutwardBinary64Enclosure, OutwardBinary64Error> {
    tight_outward_binary64_with_zero_sign(exact, ZeroSign::Positive)
}

/// Construct the tight finite binary64 enclosure without host float rounding.
pub fn tight_outward_binary64_with_zero_sign(
    exact: &BigRational,
    zero_sign: ZeroSign,
) -> Result<OutwardBinary64Enclosure, OutwardBinary64Error> {
    validate_bounded_canonical_rational(exact)?;
    if exact.is_zero() {
        let zero = ExactBinary64::from_bits(zero_sign.bits())?;
        let enclosure = OutwardBinary64Enclosure {
            lower: zero.clone(),
            upper: zero.clone(),
            nearest: zero,
        };
        enclosure.validate(exact, zero_sign)?;
        return Ok(enclosure);
    }

    let negative = exact < &BigRational::zero();
    let magnitude = if negative {
        -exact.clone()
    } else {
        exact.clone()
    };
    let positive = positive_magnitude_enclosure(&magnitude)?;

    let enclosure = if negative {
        // Negation reverses the rational endpoints.  Retain -0 as the upper
        // endpoint for exact values between -min_subnormal and zero.
        OutwardBinary64Enclosure {
            lower: ExactBinary64::from_bits(SIGN_MASK | positive.upper.bits())?,
            upper: ExactBinary64::from_bits(SIGN_MASK | positive.lower.bits())?,
            nearest: ExactBinary64::from_bits(SIGN_MASK | positive.nearest.bits())?,
        }
    } else {
        positive
    };

    enclosure.validate(exact, zero_sign)?;
    Ok(enclosure)
}

/// Independently validate an alleged tight outward enclosure.
pub fn validate_outward_binary64_enclosure(
    exact: &BigRational,
    zero_sign: ZeroSign,
    enclosure: &OutwardBinary64Enclosure,
) -> Result<(), OutwardBinary64Error> {
    validate_bounded_canonical_rational(exact)?;
    let max_finite = ExactBinary64::from_bits(POSITIVE_MAX_FINITE_BITS)?;
    if exact > max_finite.rational() || exact < &(-max_finite.rational().clone()) {
        return Err(OutwardBinary64Error::NoFiniteEnclosure);
    }

    let lower = enclosure.lower.rational();
    let upper = enclosure.upper.rational();
    if lower > upper {
        return postcondition_error(OutwardPostcondition::BoundsReversed);
    }
    if lower > exact || upper < exact {
        return postcondition_error(OutwardPostcondition::DoesNotContainExactValue);
    }

    if exact.is_zero() {
        let expected_bits = zero_sign.bits();
        if enclosure.lower.bits() != expected_bits
            || enclosure.upper.bits() != expected_bits
            || enclosure.nearest.bits() != expected_bits
        {
            return postcondition_error(OutwardPostcondition::SignedZeroMismatch);
        }
        return Ok(());
    }

    let exact_is_lower = lower == exact;
    let exact_is_upper = upper == exact;
    if exact_is_lower || exact_is_upper {
        if !exact_is_lower
            || !exact_is_upper
            || enclosure.lower.bits() != enclosure.upper.bits()
            || enclosure.nearest.bits() != enclosure.lower.bits()
        {
            return postcondition_error(OutwardPostcondition::ExactPointNotRepresentedAsPoint);
        }
        return Ok(());
    }

    if enclosure.lower.bits() == enclosure.upper.bits() {
        return postcondition_error(OutwardPostcondition::NonExactValueRepresentedAsPoint);
    }

    let positive = exact > &BigRational::zero();
    if positive {
        if enclosure.lower.bits() & SIGN_MASK != 0 || enclosure.upper.bits() & SIGN_MASK != 0 {
            return postcondition_error(OutwardPostcondition::BoundSignMismatch);
        }
        if enclosure.lower.bits().checked_add(1) != Some(enclosure.upper.bits()) {
            return postcondition_error(OutwardPostcondition::BoundsNotAdjacent);
        }
    } else {
        if enclosure.lower.bits() & SIGN_MASK == 0 || enclosure.upper.bits() & SIGN_MASK == 0 {
            return postcondition_error(OutwardPostcondition::BoundSignMismatch);
        }
        if enclosure.upper.bits().checked_add(1) != Some(enclosure.lower.bits()) {
            return postcondition_error(OutwardPostcondition::BoundsNotAdjacent);
        }
    }

    let nearest_bits = enclosure.nearest.bits();
    if nearest_bits != enclosure.lower.bits() && nearest_bits != enclosure.upper.bits() {
        return postcondition_error(OutwardPostcondition::NearestNotAnEndpoint);
    }

    let lower_distance = exact - lower;
    let upper_distance = upper - exact;
    let expected_nearest_bits = if lower_distance < upper_distance {
        enclosure.lower.bits()
    } else if upper_distance < lower_distance {
        enclosure.upper.bits()
    } else if enclosure.lower.bits() & 1 == 0 {
        enclosure.lower.bits()
    } else {
        enclosure.upper.bits()
    };
    if nearest_bits != expected_nearest_bits {
        return postcondition_error(OutwardPostcondition::NearestNotNearestEven);
    }

    Ok(())
}

fn postcondition_error<T>(condition: OutwardPostcondition) -> Result<T, OutwardBinary64Error> {
    Err(OutwardBinary64Error::Postcondition(condition))
}

fn validate_bounded_canonical_rational(exact: &BigRational) -> Result<(), OutwardBinary64Error> {
    // These are raw field accessors.  Do not invoke any invariant-dependent
    // BigRational operation until the fields pass size, zero, and sign checks:
    // `new_raw` can construct 1/0 and other malformed values.
    let numerator = exact.numer();
    let denominator = exact.denom();
    let numerator_bits = numerator.bits();
    let denominator_bits = denominator.bits();
    if numerator_bits > HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS
        || denominator_bits > HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS
    {
        return Err(OutwardBinary64Error::RationalComponentBitLimitExceeded {
            numerator_bits,
            denominator_bits,
            limit: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS,
        });
    }

    match denominator.sign() {
        Sign::NoSign => return Err(OutwardBinary64Error::RationalZeroDenominator),
        Sign::Minus => return Err(OutwardBinary64Error::RationalNegativeDenominator),
        Sign::Plus => {}
    }

    // Normalization is safe only after the bounded positive denominator
    // checks.  Compare raw BigInt fields, never the possibly malformed Ratio.
    let normalized = BigRational::new(numerator.clone(), denominator.clone());
    if normalized.numer() != numerator || normalized.denom() != denominator {
        return Err(OutwardBinary64Error::RationalNotCanonical);
    }
    Ok(())
}

fn positive_magnitude_enclosure(
    magnitude: &BigRational,
) -> Result<OutwardBinary64Enclosure, OutwardBinary64Error> {
    debug_assert!(magnitude > &BigRational::zero());

    let max_finite = ExactBinary64::from_bits(POSITIVE_MAX_FINITE_BITS)?;
    if magnitude > max_finite.rational() {
        return Err(OutwardBinary64Error::NoFiniteEnclosure);
    }

    // Positive finite binary64 values (including +0) are monotonically
    // ordered by their unsigned bit patterns.  Find the greatest value not
    // exceeding the exact magnitude.  This performs at most 63 exact dyadic
    // comparisons and never converts the input rational to f64.
    let mut below_bits = 0_u64;
    let mut search_upper = POSITIVE_MAX_FINITE_BITS;
    while below_bits < search_upper {
        let midpoint = below_bits + (search_upper - below_bits).div_ceil(2);
        let midpoint_value = ExactBinary64::from_bits(midpoint)?;
        if midpoint_value.rational() <= magnitude {
            below_bits = midpoint;
        } else {
            search_upper = midpoint - 1;
        }
    }

    let lower = ExactBinary64::from_bits(below_bits)?;
    if lower.rational() == magnitude {
        return Ok(OutwardBinary64Enclosure {
            lower: lower.clone(),
            upper: lower.clone(),
            nearest: lower,
        });
    }

    let above_bits = below_bits
        .checked_add(1)
        .filter(|bits| *bits <= POSITIVE_MAX_FINITE_BITS)
        .ok_or(OutwardBinary64Error::NoFiniteEnclosure)?;
    let upper = ExactBinary64::from_bits(above_bits)?;

    let lower_distance = magnitude - lower.rational();
    let upper_distance = upper.rational() - magnitude;
    let nearest = if lower_distance < upper_distance {
        lower.clone()
    } else if upper_distance < lower_distance {
        upper.clone()
    } else if below_bits & 1 == 0 {
        lower.clone()
    } else {
        upper.clone()
    };

    Ok(OutwardBinary64Enclosure {
        lower,
        upper,
        nearest,
    })
}

/// Names the eight exact-derived coefficients in raw-v1 Section 3.3.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MassCoefficient {
    PairMass,
    Alpha,
    Beta,
    ThirdOverPair,
    CenterFirst,
    CenterSecond,
    OffsetFirst,
    OffsetSecond,
}

impl MassCoefficient {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::PairMass => "M",
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::ThirdOverPair => "third_over_pair",
            Self::CenterFirst => "center_first",
            Self::CenterSecond => "center_second",
            Self::OffsetFirst => "offset_first",
            Self::OffsetSecond => "offset_second",
        }
    }
}

/// One exact coefficient and its tight finite binary64 enclosure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MassCoefficientWitness {
    exact: BigRational,
    outward: OutwardBinary64Enclosure,
}

impl MassCoefficientWitness {
    fn derive(exact: BigRational) -> Result<Self, OutwardBinary64Error> {
        let outward = tight_outward_binary64(&exact)?;
        Ok(Self { exact, outward })
    }

    pub fn exact(&self) -> &BigRational {
        &self.exact
    }

    pub fn outward(&self) -> &OutwardBinary64Enclosure {
        &self.outward
    }

    pub fn validate(&self) -> Result<(), OutwardBinary64Error> {
        self.outward.validate(&self.exact, ZeroSign::Positive)
    }
}

/// A complete exact-derived mass witness for one ordered planar LC pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarLcMassProfile {
    kernel_id: &'static str,
    mass_bits: [u64; 3],
    pair: [usize; 2],
    third_index: usize,
    pair_mass: MassCoefficientWitness,
    alpha: MassCoefficientWitness,
    beta: MassCoefficientWitness,
    third_over_pair: MassCoefficientWitness,
    center_first: MassCoefficientWitness,
    center_second: MassCoefficientWitness,
    offset_first: MassCoefficientWitness,
    offset_second: MassCoefficientWitness,
}

impl PlanarLcMassProfile {
    pub const fn kernel_id(&self) -> &'static str {
        self.kernel_id
    }

    pub const fn mass_bits(&self) -> &[u64; 3] {
        &self.mass_bits
    }

    pub const fn pair(&self) -> [usize; 2] {
        self.pair
    }

    pub const fn third_index(&self) -> usize {
        self.third_index
    }

    pub fn pair_mass(&self) -> &MassCoefficientWitness {
        &self.pair_mass
    }

    pub fn alpha(&self) -> &MassCoefficientWitness {
        &self.alpha
    }

    pub fn beta(&self) -> &MassCoefficientWitness {
        &self.beta
    }

    pub fn third_over_pair(&self) -> &MassCoefficientWitness {
        &self.third_over_pair
    }

    pub fn center_first(&self) -> &MassCoefficientWitness {
        &self.center_first
    }

    pub fn center_second(&self) -> &MassCoefficientWitness {
        &self.center_second
    }

    pub fn offset_first(&self) -> &MassCoefficientWitness {
        &self.offset_first
    }

    pub fn offset_second(&self) -> &MassCoefficientWitness {
        &self.offset_second
    }

    /// Recompute every exact formula and recheck every outward enclosure.
    pub fn validate_against(
        &self,
        masses: &[ExactBinary64; 3],
        pair: [usize; 2],
    ) -> Result<(), MassProfileError> {
        validate_planar_lc_mass_profile(self, masses, pair)
    }
}

/// A rejected mass-profile input, derivation, or exact postcondition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MassProfileError {
    MassNotPositive {
        index: usize,
    },
    PairIndexOutOfRange {
        position: usize,
        index: usize,
    },
    PairIndicesNotDistinct {
        index: usize,
    },
    MetadataMismatch,
    ExactFormulaMismatch {
        coefficient: MassCoefficient,
    },
    DerivedCoefficientComponentBitBoundExceeded {
        coefficient: MassCoefficient,
        numerator_bits: u64,
        denominator_bits: u64,
        limit: u64,
    },
    CoefficientEnclosure {
        coefficient: MassCoefficient,
        source: OutwardBinary64Error,
    },
}

impl fmt::Display for MassProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MassNotPositive { index } => {
                write!(formatter, "mass[{index}] is not a positive finite dyadic")
            }
            Self::PairIndexOutOfRange { position, index } => {
                write!(formatter, "pair[{position}] index {index} is outside 0..3")
            }
            Self::PairIndicesNotDistinct { index } => {
                write!(formatter, "ordered pair repeats mass index {index}")
            }
            Self::MetadataMismatch => {
                formatter.write_str("mass-profile kernel, input bits, pair, or third index changed")
            }
            Self::ExactFormulaMismatch { coefficient } => write!(
                formatter,
                "exact mass formula failed for {}",
                coefficient.wire_name()
            ),
            Self::DerivedCoefficientComponentBitBoundExceeded {
                coefficient,
                numerator_bits,
                denominator_bits,
                limit,
            } => write!(
                formatter,
                "derived {} component bound exceeded: numerator={numerator_bits} bits, \
                 denominator={denominator_bits} bits, limit={limit} bits",
                coefficient.wire_name()
            ),
            Self::CoefficientEnclosure {
                coefficient,
                source,
            } => write!(
                formatter,
                "outward enclosure failed for {}: {source}",
                coefficient.wire_name()
            ),
        }
    }
}

impl std::error::Error for MassProfileError {}

/// Derive the complete raw-v1 Section 3.3 mass profile.
pub fn derive_planar_lc_mass_profile(
    masses: &[ExactBinary64; 3],
    pair: [usize; 2],
) -> Result<PlanarLcMassProfile, MassProfileError> {
    let third_index = validate_mass_inputs(masses, pair)?;
    let exact = exact_mass_coefficients(masses, pair, third_index)?;

    let profile = PlanarLcMassProfile {
        kernel_id: PLANAR_LC_MASS_KERNEL_ID,
        mass_bits: [masses[0].bits(), masses[1].bits(), masses[2].bits()],
        pair,
        third_index,
        pair_mass: derive_coefficient(MassCoefficient::PairMass, exact.pair_mass)?,
        alpha: derive_coefficient(MassCoefficient::Alpha, exact.alpha)?,
        beta: derive_coefficient(MassCoefficient::Beta, exact.beta)?,
        third_over_pair: derive_coefficient(MassCoefficient::ThirdOverPair, exact.third_over_pair)?,
        center_first: derive_coefficient(MassCoefficient::CenterFirst, exact.center_first)?,
        center_second: derive_coefficient(MassCoefficient::CenterSecond, exact.center_second)?,
        offset_first: derive_coefficient(MassCoefficient::OffsetFirst, exact.offset_first)?,
        offset_second: derive_coefficient(MassCoefficient::OffsetSecond, exact.offset_second)?,
    };
    profile.validate_against(masses, pair)?;
    Ok(profile)
}

/// Independently recompute exact formulas and validate a complete profile.
pub fn validate_planar_lc_mass_profile(
    profile: &PlanarLcMassProfile,
    masses: &[ExactBinary64; 3],
    pair: [usize; 2],
) -> Result<(), MassProfileError> {
    let third_index = validate_mass_inputs(masses, pair)?;
    let mass_bits = [masses[0].bits(), masses[1].bits(), masses[2].bits()];
    if profile.kernel_id != PLANAR_LC_MASS_KERNEL_ID
        || profile.mass_bits != mass_bits
        || profile.pair != pair
        || profile.third_index != third_index
    {
        return Err(MassProfileError::MetadataMismatch);
    }

    let expected = exact_mass_coefficients(masses, pair, third_index)?;
    validate_coefficient(
        MassCoefficient::PairMass,
        &profile.pair_mass,
        &expected.pair_mass,
    )?;
    validate_coefficient(MassCoefficient::Alpha, &profile.alpha, &expected.alpha)?;
    validate_coefficient(MassCoefficient::Beta, &profile.beta, &expected.beta)?;
    validate_coefficient(
        MassCoefficient::ThirdOverPair,
        &profile.third_over_pair,
        &expected.third_over_pair,
    )?;
    validate_coefficient(
        MassCoefficient::CenterFirst,
        &profile.center_first,
        &expected.center_first,
    )?;
    validate_coefficient(
        MassCoefficient::CenterSecond,
        &profile.center_second,
        &expected.center_second,
    )?;
    validate_coefficient(
        MassCoefficient::OffsetFirst,
        &profile.offset_first,
        &expected.offset_first,
    )?;
    validate_coefficient(
        MassCoefficient::OffsetSecond,
        &profile.offset_second,
        &expected.offset_second,
    )?;
    Ok(())
}

fn validate_mass_inputs(
    masses: &[ExactBinary64; 3],
    pair: [usize; 2],
) -> Result<usize, MassProfileError> {
    for (index, mass) in masses.iter().enumerate() {
        if mass.rational() <= &BigRational::zero() {
            return Err(MassProfileError::MassNotPositive { index });
        }
    }
    for (position, index) in pair.into_iter().enumerate() {
        if index >= masses.len() {
            return Err(MassProfileError::PairIndexOutOfRange { position, index });
        }
    }
    if pair[0] == pair[1] {
        return Err(MassProfileError::PairIndicesNotDistinct { index: pair[0] });
    }

    // Exactly one element of {0,1,2} is absent from a validated distinct pair.
    // Spell out the finite complement instead of leaving a panic path.
    if pair[0] != 0 && pair[1] != 0 {
        Ok(0)
    } else if pair[0] != 1 && pair[1] != 1 {
        Ok(1)
    } else {
        Ok(2)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactMassCoefficients {
    pair_mass: BigRational,
    alpha: BigRational,
    beta: BigRational,
    third_over_pair: BigRational,
    center_first: BigRational,
    center_second: BigRational,
    offset_first: BigRational,
    offset_second: BigRational,
}

fn exact_mass_coefficients(
    masses: &[ExactBinary64; 3],
    pair: [usize; 2],
    third_index: usize,
) -> Result<ExactMassCoefficients, MassProfileError> {
    let first = masses[pair[0]].rational();
    let second = masses[pair[1]].rational();
    let third = masses[third_index].rational();
    let pair_mass = first + second;
    let alpha = second / &pair_mass;
    let beta = first / &pair_mass;
    let third_over_pair = third / &pair_mass;
    let center_first = first * &third_over_pair;
    let center_second = second * &third_over_pair;
    let offset_first = first + &center_first;
    let offset_second = second + &center_second;
    let coefficients = ExactMassCoefficients {
        pair_mass,
        alpha,
        beta,
        third_over_pair,
        center_first,
        center_second,
        offset_first,
        offset_second,
    };
    for (coefficient, exact) in [
        (MassCoefficient::PairMass, &coefficients.pair_mass),
        (MassCoefficient::Alpha, &coefficients.alpha),
        (MassCoefficient::Beta, &coefficients.beta),
        (
            MassCoefficient::ThirdOverPair,
            &coefficients.third_over_pair,
        ),
        (MassCoefficient::CenterFirst, &coefficients.center_first),
        (MassCoefficient::CenterSecond, &coefficients.center_second),
        (MassCoefficient::OffsetFirst, &coefficients.offset_first),
        (MassCoefficient::OffsetSecond, &coefficients.offset_second),
    ] {
        validate_derived_coefficient_bound(coefficient, exact)?;
    }
    Ok(coefficients)
}

fn derive_coefficient(
    coefficient: MassCoefficient,
    exact: BigRational,
) -> Result<MassCoefficientWitness, MassProfileError> {
    validate_derived_coefficient_bound(coefficient, &exact)?;
    MassCoefficientWitness::derive(exact).map_err(|source| MassProfileError::CoefficientEnclosure {
        coefficient,
        source,
    })
}

fn validate_coefficient(
    coefficient: MassCoefficient,
    witness: &MassCoefficientWitness,
    exact: &BigRational,
) -> Result<(), MassProfileError> {
    validate_derived_coefficient_bound(coefficient, exact)?;
    validate_bounded_canonical_rational(&witness.exact).map_err(|source| {
        MassProfileError::CoefficientEnclosure {
            coefficient,
            source,
        }
    })?;
    if witness.exact != *exact {
        return Err(MassProfileError::ExactFormulaMismatch { coefficient });
    }
    witness
        .validate()
        .map_err(|source| MassProfileError::CoefficientEnclosure {
            coefficient,
            source,
        })
}

fn validate_derived_coefficient_bound(
    coefficient: MassCoefficient,
    exact: &BigRational,
) -> Result<(), MassProfileError> {
    let numerator_bits = exact.numer().bits();
    let denominator_bits = exact.denom().bits();
    if numerator_bits > DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS
        || denominator_bits > DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS
    {
        return Err(
            MassProfileError::DerivedCoefficientComponentBitBoundExceeded {
                coefficient,
                numerator_bits,
                denominator_bits,
                limit: DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::One;

    use super::*;

    const ONE_BITS: u64 = 0x3ff0_0000_0000_0000;
    const TWO_BITS: u64 = 0x4000_0000_0000_0000;
    const THREE_BITS: u64 = 0x4008_0000_0000_0000;

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn power_of_two(exponent: usize) -> BigInt {
        BigInt::one() << exponent
    }

    fn exact(bits: u64) -> ExactBinary64 {
        ExactBinary64::from_bits(bits).unwrap()
    }

    fn simple_masses() -> [ExactBinary64; 3] {
        [exact(ONE_BITS), exact(TWO_BITS), exact(THREE_BITS)]
    }

    fn assert_adjacent(enclosure: &OutwardBinary64Enclosure) {
        assert!(!enclosure.is_point());
        if enclosure.lower.bits() & SIGN_MASK == 0 {
            assert_eq!(enclosure.lower.bits() + 1, enclosure.upper.bits());
        } else {
            assert_eq!(enclosure.upper.bits() + 1, enclosure.lower.bits());
        }
    }

    fn assert_witness(witness: &MassCoefficientWitness, numerator: i64, denominator: i64) {
        assert_eq!(witness.exact(), &rational(numerator, denominator));
        assert!(witness.outward().lower().rational() <= witness.exact());
        assert!(witness.outward().upper().rational() >= witness.exact());
        witness.validate().unwrap();
        if !witness.outward().is_point() {
            assert_adjacent(witness.outward());
        }
    }

    fn assert_invalid_rational_rejected_by_every_public_path(
        invalid: &BigRational,
        expected: OutwardBinary64Error,
    ) {
        let valid = tight_outward_binary64(&BigRational::one()).unwrap();
        assert_eq!(tight_outward_binary64(invalid), Err(expected.clone()));
        assert_eq!(
            tight_outward_binary64_with_zero_sign(invalid, ZeroSign::Negative),
            Err(expected.clone())
        );
        assert_eq!(
            valid.validate(invalid, ZeroSign::Positive),
            Err(expected.clone())
        );
        assert_eq!(
            validate_outward_binary64_enclosure(invalid, ZeroSign::Positive, &valid),
            Err(expected.clone())
        );
        assert_eq!(
            OutwardBinary64Enclosure::from_bits_checked(
                invalid,
                ZeroSign::Positive,
                valid.lower().bits(),
                valid.upper().bits(),
                valid.nearest().bits(),
            ),
            Err(expected.clone())
        );

        let witness = MassCoefficientWitness {
            exact: invalid.clone(),
            outward: valid,
        };
        assert_eq!(witness.validate(), Err(expected.clone()));

        let masses = simple_masses();
        let mut profile = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        profile.alpha.exact = invalid.clone();
        let profile_error = MassProfileError::CoefficientEnclosure {
            coefficient: MassCoefficient::Alpha,
            source: expected,
        };
        assert_eq!(
            profile.validate_against(&masses, [0, 1]),
            Err(profile_error.clone())
        );
        assert_eq!(
            validate_planar_lc_mass_profile(&profile, &masses, [0, 1]),
            Err(profile_error)
        );
    }

    #[test]
    fn signed_zero_is_explicit_and_exact() {
        let zero = BigRational::zero();
        let positive = tight_outward_binary64_with_zero_sign(&zero, ZeroSign::Positive).unwrap();
        let negative = tight_outward_binary64_with_zero_sign(&zero, ZeroSign::Negative).unwrap();
        assert_eq!(positive.lower().bits(), 0);
        assert_eq!(positive.upper().bits(), 0);
        assert_eq!(positive.nearest().bits(), 0);
        assert_eq!(negative.lower().bits(), SIGN_MASK);
        assert_eq!(negative.upper().bits(), SIGN_MASK);
        assert_eq!(negative.nearest().bits(), SIGN_MASK);
        assert_eq!(tight_outward_binary64(&zero).unwrap(), positive);
    }

    #[test]
    fn exactly_representable_normal_and_subnormal_values_are_points() {
        for bits in [
            1_u64,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            ONE_BITS,
            POSITIVE_MAX_FINITE_BITS,
        ] {
            let value = exact(bits);
            let enclosure = tight_outward_binary64(value.rational()).unwrap();
            assert!(enclosure.is_point());
            assert_eq!(enclosure.lower().bits(), bits);
            assert_eq!(enclosure.nearest().bits(), bits);

            let negative_value = -value.rational().clone();
            let negative = tight_outward_binary64(&negative_value).unwrap();
            assert!(negative.is_point());
            assert_eq!(negative.lower().bits(), SIGN_MASK | bits);
        }
    }

    #[test]
    fn subnormal_midpoints_obey_ties_to_even_and_keep_signed_zero() {
        let half_min_subnormal = BigRational::new(BigInt::one(), power_of_two(1075));
        let positive = tight_outward_binary64(&half_min_subnormal).unwrap();
        assert_eq!(positive.lower().bits(), 0);
        assert_eq!(positive.upper().bits(), 1);
        assert_eq!(positive.nearest().bits(), 0);
        assert_adjacent(&positive);

        let negative = tight_outward_binary64(&(-half_min_subnormal.clone())).unwrap();
        assert_eq!(negative.lower().bits(), SIGN_MASK | 1);
        assert_eq!(negative.upper().bits(), SIGN_MASK);
        assert_eq!(negative.nearest().bits(), SIGN_MASK);
        assert_adjacent(&negative);

        // Halfway between subnormals 1 and 2, the even low significand bit is
        // on bit pattern 2.
        let three_halves_min = BigRational::new(BigInt::from(3), power_of_two(1075));
        let second_tie = tight_outward_binary64(&three_halves_min).unwrap();
        assert_eq!(second_tie.lower().bits(), 1);
        assert_eq!(second_tie.upper().bits(), 2);
        assert_eq!(second_tie.nearest().bits(), 2);
    }

    #[test]
    fn normal_midpoint_ties_choose_the_even_endpoint_on_both_sides_of_one() {
        let above_one_midpoint =
            BigRational::new(power_of_two(53) + BigInt::one(), power_of_two(53));
        let above = tight_outward_binary64(&above_one_midpoint).unwrap();
        assert_eq!(above.lower().bits(), ONE_BITS);
        assert_eq!(above.upper().bits(), ONE_BITS + 1);
        assert_eq!(above.nearest().bits(), ONE_BITS);
        assert_adjacent(&above);

        let below_one_midpoint =
            BigRational::new(power_of_two(54) - BigInt::one(), power_of_two(54));
        let below = tight_outward_binary64(&below_one_midpoint).unwrap();
        assert_eq!(below.lower().bits(), ONE_BITS - 1);
        assert_eq!(below.upper().bits(), ONE_BITS);
        assert_eq!(below.nearest().bits(), ONE_BITS);
        assert_adjacent(&below);
    }

    #[test]
    fn representative_lattice_midpoints_stay_adjacent_and_even_at_boundaries() {
        for lower_bits in [
            0_u64,
            1,
            0x000f_ffff_ffff_ffff,
            ONE_BITS - 1,
            ONE_BITS,
            0x400f_ffff_ffff_ffff,
            POSITIVE_MAX_FINITE_BITS - 1,
        ] {
            let upper_bits = lower_bits + 1;
            let lower = exact(lower_bits);
            let upper = exact(upper_bits);
            let midpoint = (lower.rational() + upper.rational()) / BigInt::from(2);

            let positive = tight_outward_binary64(&midpoint).unwrap();
            assert_eq!(positive.lower().bits(), lower_bits);
            assert_eq!(positive.upper().bits(), upper_bits);
            assert_eq!(
                positive.nearest().bits(),
                if lower_bits & 1 == 0 {
                    lower_bits
                } else {
                    upper_bits
                }
            );
            assert_adjacent(&positive);

            let negative = tight_outward_binary64(&(-midpoint)).unwrap();
            assert_eq!(negative.lower().bits(), SIGN_MASK | upper_bits);
            assert_eq!(negative.upper().bits(), SIGN_MASK | lower_bits);
            assert_eq!(
                negative.nearest().bits(),
                SIGN_MASK
                    | if lower_bits & 1 == 0 {
                        lower_bits
                    } else {
                        upper_bits
                    }
            );
            assert_adjacent(&negative);
        }
    }

    #[test]
    fn exact_distance_not_host_rounding_selects_nearest_neighbor() {
        let lower = exact(ONE_BITS);
        let upper = exact(ONE_BITS + 1);
        let one_third_step =
            lower.rational() + (upper.rational() - lower.rational()) / BigInt::from(3);
        let down = tight_outward_binary64(&one_third_step).unwrap();
        assert_eq!(down.lower().bits(), ONE_BITS);
        assert_eq!(down.upper().bits(), ONE_BITS + 1);
        assert_eq!(down.nearest().bits(), ONE_BITS);

        let two_thirds_step = lower.rational()
            + (upper.rational() - lower.rational()) * BigInt::from(2) / BigInt::from(3);
        let up = tight_outward_binary64(&two_thirds_step).unwrap();
        assert_eq!(up.lower().bits(), ONE_BITS);
        assert_eq!(up.upper().bits(), ONE_BITS + 1);
        assert_eq!(up.nearest().bits(), ONE_BITS + 1);
    }

    #[test]
    fn values_outside_max_finite_fail_closed() {
        let max_finite = exact(POSITIVE_MAX_FINITE_BITS);
        let too_large = max_finite.rational() + BigRational::one();
        assert_eq!(
            tight_outward_binary64(&too_large),
            Err(OutwardBinary64Error::NoFiniteEnclosure)
        );
        assert_eq!(
            tight_outward_binary64(&(-too_large)),
            Err(OutwardBinary64Error::NoFiniteEnclosure)
        );
    }

    #[test]
    fn hard_component_limit_is_enforced_before_every_public_witness_path() {
        let at_limit_scale =
            BigInt::one() << (HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS as usize - 1);
        let at_limit = BigRational::new(&at_limit_scale + BigInt::one(), at_limit_scale.clone());
        assert_eq!(
            at_limit.numer().bits(),
            HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS
        );
        assert_eq!(
            at_limit.denom().bits(),
            HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS
        );
        tight_outward_binary64(&at_limit).unwrap();

        let oversized_scale = BigInt::one() << HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS as usize;
        let oversized = BigRational::new(&oversized_scale + BigInt::one(), oversized_scale.clone());
        let resource_error = OutwardBinary64Error::RationalComponentBitLimitExceeded {
            numerator_bits: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS + 1,
            denominator_bits: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS + 1,
            limit: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS,
        };

        assert_eq!(
            tight_outward_binary64(&oversized),
            Err(resource_error.clone())
        );

        let one = tight_outward_binary64(&BigRational::one()).unwrap();
        assert_eq!(
            validate_outward_binary64_enclosure(&oversized, ZeroSign::Positive, &one),
            Err(resource_error.clone())
        );

        // Even a non-finite alleged endpoint cannot change error priority:
        // resource rejection occurs before any endpoint decoding.
        assert_eq!(
            OutwardBinary64Enclosure::from_bits_checked(
                &oversized,
                ZeroSign::Positive,
                0x7ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
            ),
            Err(resource_error)
        );
    }

    #[test]
    fn malformed_new_raw_rationals_fail_closed_without_panicking() {
        for (invalid, expected) in [
            (
                BigRational::new_raw(BigInt::one(), BigInt::zero()),
                OutwardBinary64Error::RationalZeroDenominator,
            ),
            (
                BigRational::new_raw(BigInt::zero(), BigInt::zero()),
                OutwardBinary64Error::RationalZeroDenominator,
            ),
            (
                BigRational::new_raw(BigInt::one(), BigInt::from(-2)),
                OutwardBinary64Error::RationalNegativeDenominator,
            ),
            (
                BigRational::new_raw(BigInt::from(2), BigInt::from(4)),
                OutwardBinary64Error::RationalNotCanonical,
            ),
            (
                BigRational::new_raw(BigInt::zero(), BigInt::from(2)),
                OutwardBinary64Error::RationalNotCanonical,
            ),
        ] {
            assert_invalid_rational_rejected_by_every_public_path(&invalid, expected);
        }

        // The component cap has priority over malformed-denominator errors,
        // so normalization work can never be triggered by an oversized raw
        // numerator paired with zero.
        let oversized_numerator =
            BigInt::one() << HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS as usize;
        let oversized_zero_denominator = BigRational::new_raw(oversized_numerator, BigInt::zero());
        assert_eq!(
            tight_outward_binary64(&oversized_zero_denominator),
            Err(OutwardBinary64Error::RationalComponentBitLimitExceeded {
                numerator_bits: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS + 1,
                denominator_bits: 0,
                limit: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS,
            })
        );
    }

    #[test]
    fn deterministic_adjacent_interval_stress_checks_signs_fractions_and_ties() {
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let interior_fractions = [(1_i64, 3_i64), (1, 2), (2, 3)];

        for _ in 0..256 {
            // Fixed xorshift64 sequence: deterministic, dependency-free, and
            // broad across signless finite exponent/significand fields.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let lower_bits = state % POSITIVE_MAX_FINITE_BITS;
            let upper_bits = lower_bits + 1;
            let lower = exact(lower_bits);
            let upper = exact(upper_bits);
            let gap = upper.rational() - lower.rational();

            for (fraction_numerator, fraction_denominator) in interior_fractions {
                let target = lower.rational()
                    + &gap * BigInt::from(fraction_numerator) / BigInt::from(fraction_denominator);
                let expected_magnitude_nearest = if 2 * fraction_numerator < fraction_denominator {
                    lower_bits
                } else if 2 * fraction_numerator > fraction_denominator {
                    upper_bits
                } else if lower_bits & 1 == 0 {
                    lower_bits
                } else {
                    upper_bits
                };

                let positive = tight_outward_binary64(&target).unwrap();
                assert_eq!(positive.lower().bits(), lower_bits);
                assert_eq!(positive.upper().bits(), upper_bits);
                assert_eq!(positive.nearest().bits(), expected_magnitude_nearest);
                assert!(positive.lower().rational() < &target);
                assert!(positive.upper().rational() > &target);
                assert_adjacent(&positive);

                let negative_target = -target;
                let negative = tight_outward_binary64(&negative_target).unwrap();
                assert_eq!(negative.lower().bits(), SIGN_MASK | upper_bits);
                assert_eq!(negative.upper().bits(), SIGN_MASK | lower_bits);
                assert_eq!(
                    negative.nearest().bits(),
                    SIGN_MASK | expected_magnitude_nearest
                );
                assert!(negative.lower().rational() < &negative_target);
                assert!(negative.upper().rational() > &negative_target);
                assert_adjacent(&negative);
            }
        }
    }

    #[test]
    fn validator_rejects_nonadjacent_and_wrong_nearest_claims() {
        let target = rational(1, 3);
        let valid = tight_outward_binary64(&target).unwrap();
        assert_adjacent(&valid);

        let nonadjacent = OutwardBinary64Enclosure {
            lower: valid.lower.clone(),
            upper: exact(valid.upper.bits() + 1),
            nearest: valid.nearest.clone(),
        };
        assert_eq!(
            nonadjacent.validate(&target, ZeroSign::Positive),
            Err(OutwardBinary64Error::Postcondition(
                OutwardPostcondition::BoundsNotAdjacent
            ))
        );

        let wrong_nearest = OutwardBinary64Enclosure {
            lower: valid.lower.clone(),
            upper: valid.upper.clone(),
            nearest: if valid.nearest.bits() == valid.lower.bits() {
                valid.upper.clone()
            } else {
                valid.lower.clone()
            },
        };
        assert_eq!(
            wrong_nearest.validate(&target, ZeroSign::Positive),
            Err(OutwardBinary64Error::Postcondition(
                OutwardPostcondition::NearestNotNearestEven
            ))
        );
    }

    #[test]
    fn simple_masses_match_all_eight_hand_derived_formulas() {
        let masses = simple_masses();
        let profile = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        assert_eq!(profile.kernel_id(), PLANAR_LC_MASS_KERNEL_ID);
        assert_eq!(profile.pair(), [0, 1]);
        assert_eq!(profile.third_index(), 2);
        assert_witness(profile.pair_mass(), 3, 1);
        assert_witness(profile.alpha(), 2, 3);
        assert_witness(profile.beta(), 1, 3);
        assert_witness(profile.third_over_pair(), 1, 1);
        assert_witness(profile.center_first(), 1, 1);
        assert_witness(profile.center_second(), 2, 1);
        assert_witness(profile.offset_first(), 2, 1);
        assert_witness(profile.offset_second(), 4, 1);

        // These non-dyadic ratios have the known adjacent brackets around
        // their nearest binary64 approximations.
        assert_eq!(
            profile.beta().outward().lower().bits(),
            0x3fd5_5555_5555_5555
        );
        assert_eq!(
            profile.beta().outward().upper().bits(),
            0x3fd5_5555_5555_5556
        );
        assert_eq!(
            profile.alpha().outward().lower().bits(),
            0x3fe5_5555_5555_5555
        );
        assert_eq!(
            profile.alpha().outward().upper().bits(),
            0x3fe5_5555_5555_5556
        );
    }

    #[test]
    fn all_three_canonical_ordered_pairs_use_the_correct_third_mass() {
        let masses = simple_masses();

        let pair_01 = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        assert_eq!(pair_01.third_index(), 2);
        assert_witness(pair_01.pair_mass(), 3, 1);
        assert_witness(pair_01.third_over_pair(), 1, 1);

        let pair_02 = derive_planar_lc_mass_profile(&masses, [0, 2]).unwrap();
        assert_eq!(pair_02.third_index(), 1);
        assert_witness(pair_02.pair_mass(), 4, 1);
        assert_witness(pair_02.alpha(), 3, 4);
        assert_witness(pair_02.beta(), 1, 4);
        assert_witness(pair_02.third_over_pair(), 1, 2);
        assert_witness(pair_02.center_first(), 1, 2);
        assert_witness(pair_02.center_second(), 3, 2);
        assert_witness(pair_02.offset_first(), 3, 2);
        assert_witness(pair_02.offset_second(), 9, 2);

        let pair_12 = derive_planar_lc_mass_profile(&masses, [1, 2]).unwrap();
        assert_eq!(pair_12.third_index(), 0);
        assert_witness(pair_12.pair_mass(), 5, 1);
        assert_witness(pair_12.alpha(), 3, 5);
        assert_witness(pair_12.beta(), 2, 5);
        assert_witness(pair_12.third_over_pair(), 1, 5);
        assert_witness(pair_12.center_first(), 2, 5);
        assert_witness(pair_12.center_second(), 3, 5);
        assert_witness(pair_12.offset_first(), 12, 5);
        assert_witness(pair_12.offset_second(), 18, 5);

        for pair in [[0, 1], [0, 2], [1, 2]] {
            derive_planar_lc_mass_profile(&masses, pair)
                .unwrap()
                .validate_against(&masses, pair)
                .unwrap();
        }
    }

    #[test]
    fn reversing_an_ordered_pair_swaps_alpha_beta_and_center_roles() {
        let masses = simple_masses();
        let forward = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        let reverse = derive_planar_lc_mass_profile(&masses, [1, 0]).unwrap();
        assert_eq!(forward.alpha().exact(), reverse.beta().exact());
        assert_eq!(forward.beta().exact(), reverse.alpha().exact());
        assert_eq!(
            forward.center_first().exact(),
            reverse.center_second().exact()
        );
        assert_eq!(
            forward.center_second().exact(),
            reverse.center_first().exact()
        );
    }

    #[test]
    fn zero_negative_and_invalid_pairs_are_rejected() {
        let valid = simple_masses();
        let zero = [exact(0), exact(TWO_BITS), exact(THREE_BITS)];
        let negative = [
            exact(SIGN_MASK | ONE_BITS),
            exact(TWO_BITS),
            exact(THREE_BITS),
        ];
        assert_eq!(
            derive_planar_lc_mass_profile(&zero, [0, 1]),
            Err(MassProfileError::MassNotPositive { index: 0 })
        );
        assert_eq!(
            derive_planar_lc_mass_profile(&negative, [0, 1]),
            Err(MassProfileError::MassNotPositive { index: 0 })
        );
        assert_eq!(
            derive_planar_lc_mass_profile(&valid, [1, 1]),
            Err(MassProfileError::PairIndicesNotDistinct { index: 1 })
        );
        assert_eq!(
            derive_planar_lc_mass_profile(&valid, [3, 0]),
            Err(MassProfileError::PairIndexOutOfRange {
                position: 0,
                index: 3
            })
        );
        assert_eq!(
            derive_planar_lc_mass_profile(&valid, [0, 4]),
            Err(MassProfileError::PairIndexOutOfRange {
                position: 1,
                index: 4
            })
        );
    }

    #[test]
    fn mass_coefficient_overflow_identifies_the_failed_formula() {
        let masses = [
            exact(POSITIVE_MAX_FINITE_BITS),
            exact(ONE_BITS),
            exact(ONE_BITS),
        ];
        assert_eq!(
            derive_planar_lc_mass_profile(&masses, [0, 1]),
            Err(MassProfileError::CoefficientEnclosure {
                coefficient: MassCoefficient::PairMass,
                source: OutwardBinary64Error::NoFiniteEnclosure,
            })
        );
    }

    #[test]
    fn ratio_overflow_after_a_finite_pair_sum_is_attributed_precisely() {
        let masses = [exact(1), exact(1), exact(POSITIVE_MAX_FINITE_BITS)];
        assert_eq!(
            derive_planar_lc_mass_profile(&masses, [0, 1]),
            Err(MassProfileError::CoefficientEnclosure {
                coefficient: MassCoefficient::ThirdOverPair,
                source: OutwardBinary64Error::NoFiniteEnclosure,
            })
        );
    }

    #[test]
    fn conservative_mass_formula_bound_covers_extreme_binary64_components() {
        let representative_bits = [
            1_u64,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            ONE_BITS,
            THREE_BITS,
            POSITIVE_MAX_FINITE_BITS,
        ];
        let ordered_pairs = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]];

        for first_bits in representative_bits {
            for second_bits in representative_bits {
                for third_bits in representative_bits {
                    let masses = [exact(first_bits), exact(second_bits), exact(third_bits)];
                    for pair in ordered_pairs {
                        let third_index = validate_mass_inputs(&masses, pair).unwrap();
                        let coefficients =
                            exact_mass_coefficients(&masses, pair, third_index).unwrap();
                        for value in [
                            &coefficients.pair_mass,
                            &coefficients.alpha,
                            &coefficients.beta,
                            &coefficients.third_over_pair,
                            &coefficients.center_first,
                            &coefficients.center_second,
                            &coefficients.offset_first,
                            &coefficients.offset_second,
                        ] {
                            assert!(
                                value.numer().bits() <= DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS
                            );
                            assert!(
                                value.denom().bits() <= DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn internal_derived_coefficient_bound_fails_closed_with_attribution() {
        let scale = BigInt::one() << DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS as usize;
        let canonical_but_too_large = BigRational::new(&scale + BigInt::one(), scale);
        assert_eq!(
            derive_coefficient(MassCoefficient::OffsetSecond, canonical_but_too_large),
            Err(
                MassProfileError::DerivedCoefficientComponentBitBoundExceeded {
                    coefficient: MassCoefficient::OffsetSecond,
                    numerator_bits: DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS + 1,
                    denominator_bits: DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS + 1,
                    limit: DERIVED_MASS_COEFFICIENT_MAX_COMPONENT_BITS,
                }
            )
        );
    }

    #[test]
    fn profile_validator_detects_exact_formula_and_metadata_changes() {
        let masses = simple_masses();
        let mut profile = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        profile.alpha.exact = rational(1, 2);
        assert_eq!(
            profile.validate_against(&masses, [0, 1]),
            Err(MassProfileError::ExactFormulaMismatch {
                coefficient: MassCoefficient::Alpha
            })
        );

        let mut oversized_profile = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        let oversized_scale = BigInt::one() << HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS as usize;
        oversized_profile.alpha.exact =
            BigRational::new(&oversized_scale + BigInt::one(), oversized_scale);
        assert_eq!(
            oversized_profile.validate_against(&masses, [0, 1]),
            Err(MassProfileError::CoefficientEnclosure {
                coefficient: MassCoefficient::Alpha,
                source: OutwardBinary64Error::RationalComponentBitLimitExceeded {
                    numerator_bits: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS + 1,
                    denominator_bits: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS + 1,
                    limit: HARD_MAX_OUTWARD_RATIONAL_COMPONENT_BITS,
                },
            })
        );

        let profile = derive_planar_lc_mass_profile(&masses, [0, 1]).unwrap();
        assert_eq!(
            profile.validate_against(&masses, [0, 2]),
            Err(MassProfileError::MetadataMismatch)
        );
    }
}
