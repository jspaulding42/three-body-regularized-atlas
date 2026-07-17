use core::fmt;

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::One;

use crate::RationalInterval;

/// Absolute ceilings that callers cannot relax.
///
/// A supplied [`ExpEnclosureLimits`] is itself untrusted configuration.  These
/// ceilings keep it from turning a proof request into an unbounded allocation
/// or loop.  They are implementation limits, not mathematical restrictions on
/// the exponential lemma.
pub const HARD_MAX_EXP_INPUT_COMPONENT_BITS: u64 = 2_048;
pub const HARD_MAX_EXP_INTERMEDIATE_COMPONENT_BITS: u64 = 65_536;
pub const HARD_MAX_EXP_RANGE_REDUCTIONS: usize = 1_152;
pub const HARD_MAX_EXP_TAYLOR_CUTOFF: usize = 256;
pub const HARD_MAX_EXP_WORK_UNITS: u64 = 8_000_000;
pub const HARD_MAX_EXP_WITNESS_STORAGE_BITS: u64 = 4_194_304;

/// Resource bounds for exact rational exponential enclosure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpEnclosureLimits {
    /// Maximum bit length of the input numerator and denominator separately.
    pub max_input_component_bits: u64,
    /// Maximum bit length of every intermediate numerator and denominator.
    pub max_intermediate_component_bits: u64,
    /// Maximum number of divisions by two during range reduction.
    pub max_range_reductions: usize,
    /// Maximum accepted Taylor cutoff `N` (the partial sum includes term `N`).
    pub max_taylor_cutoff: usize,
    /// Aggregate limb-pair work units for construction plus mandatory replay.
    ///
    /// A rational scan costs its total 64-bit-limb count.  An exact binary
    /// operation or comparison costs the product of the two operand limb
    /// counts.  The accounting is deterministic and deliberately conservative
    /// with respect to the big-integer work performed underneath it.
    pub max_work_units: u64,
    /// Aggregate represented-bit budget for the returned proof witness.
    ///
    /// Each cached rational is charged its numerator and denominator bit
    /// lengths plus 128 metadata bits.  Every cached interval endpoint is
    /// charged separately, including every range-reduction squaring step.
    pub max_witness_storage_bits: u64,
}

pub const DEFAULT_EXP_ENCLOSURE_LIMITS: ExpEnclosureLimits = ExpEnclosureLimits {
    max_input_component_bits: 2_048,
    max_intermediate_component_bits: 32_768,
    max_range_reductions: 1_100,
    max_taylor_cutoff: 128,
    max_work_units: 2_000_000,
    max_witness_storage_bits: 1_048_576,
};

/// Fail-closed outcomes from the rational exponential proof kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpEnclosureError {
    NegativeExponent,
    NonpositiveMaxReducedTailUpper,
    InvalidResourceLimits,
    InputComponentBitLimitExceeded,
    IntermediateComponentBitLimitExceeded,
    MalformedRationalDenominator,
    NoncanonicalRational,
    RangeReductionLimitExceeded,
    TaylorCutoffLimitExceeded,
    TaylorCutoffArithmeticOverflow,
    InsufficientTaylorCutoff { cutoff: usize },
    ExactDivisionByZero,
    WorkBudgetExceeded,
    WitnessStorageBudgetExceeded,
    WitnessPostconditionFailure,
}

impl fmt::Display for ExpEnclosureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeExponent => {
                formatter.write_str("the rational exponential kernel requires x >= 0")
            }
            Self::NonpositiveMaxReducedTailUpper => {
                formatter.write_str("max_reduced_tail_upper must be positive")
            }
            Self::InvalidResourceLimits => formatter.write_str(
                "exponential resource limits are zero, inconsistent, or above a hard ceiling",
            ),
            Self::InputComponentBitLimitExceeded => {
                formatter.write_str("exponential input rational exceeds its component-bit limit")
            }
            Self::IntermediateComponentBitLimitExceeded => formatter.write_str(
                "exponential proof arithmetic exceeds its intermediate component-bit limit",
            ),
            Self::MalformedRationalDenominator => formatter
                .write_str("exact rational denominator must be strictly positive and nonzero"),
            Self::NoncanonicalRational => {
                formatter.write_str("exact rational is not in canonical reduced form")
            }
            Self::RangeReductionLimitExceeded => {
                formatter.write_str("exponential range-reduction limit exceeded")
            }
            Self::TaylorCutoffLimitExceeded => {
                formatter.write_str("exponential Taylor cutoff limit exceeded")
            }
            Self::TaylorCutoffArithmeticOverflow => {
                formatter.write_str("exponential Taylor cutoff index overflows usize")
            }
            Self::InsufficientTaylorCutoff { cutoff } => write!(
                formatter,
                "Taylor cutoff {cutoff} does not meet the declared reduced-tail tolerance"
            ),
            Self::ExactDivisionByZero => {
                formatter.write_str("exponential proof attempted exact division by zero")
            }
            Self::WorkBudgetExceeded => {
                formatter.write_str("exponential aggregate exact-work budget exceeded")
            }
            Self::WitnessStorageBudgetExceeded => {
                formatter.write_str("exponential witness storage budget exceeded")
            }
            Self::WitnessPostconditionFailure => formatter
                .write_str("exponential enclosure witness failed an exact algebraic postcondition"),
        }
    }
}

impl std::error::Error for ExpEnclosureError {}

/// Proof data for a rational enclosure of `exp(x)` with `x >= 0`.
///
/// If the cutoff is `N`, `partial_sum` is
///
/// `S_N(y) = sum_{j=0}^N y^j/j!`,
///
/// where `y = x / 2^range_reductions` and `0 <= y <= 1/2`.  The first
/// omitted term is `t_(N+1)`.  Every subsequent Taylor-term ratio is at most
/// `q = y/(N+2)`, so the nonnegative tail is no larger than
/// `t_(N+1)/(1-q)`.  Thus
///
/// `S_N(y) <= exp(y) <= S_N(y) + tail_upper`.
///
/// Squaring this positive interval `range_reductions` times encloses
/// `exp(x)`.  All stored quantities are exact rationals and [`Self::verify`]
/// recomputes the full algebraic witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalExpEnclosure {
    exponent: BigRational,
    range_reductions: usize,
    taylor_cutoff: usize,
    reduced_argument: BigRational,
    max_reduced_tail_upper: BigRational,
    partial_sum: BigRational,
    first_omitted_term: BigRational,
    geometric_ratio: BigRational,
    geometric_denominator_margin: BigRational,
    tail_upper: BigRational,
    reduced_enclosure: RationalInterval,
    squaring_enclosures: Vec<RationalInterval>,
    enclosure: RationalInterval,
}

impl RationalExpEnclosure {
    pub fn exponent(&self) -> &BigRational {
        &self.exponent
    }

    pub const fn range_reductions(&self) -> usize {
        self.range_reductions
    }

    pub const fn taylor_cutoff(&self) -> usize {
        self.taylor_cutoff
    }

    pub fn reduced_argument(&self) -> &BigRational {
        &self.reduced_argument
    }

    pub fn max_reduced_tail_upper(&self) -> &BigRational {
        &self.max_reduced_tail_upper
    }

    pub fn partial_sum(&self) -> &BigRational {
        &self.partial_sum
    }

    pub fn first_omitted_term(&self) -> &BigRational {
        &self.first_omitted_term
    }

    pub fn geometric_ratio(&self) -> &BigRational {
        &self.geometric_ratio
    }

    pub fn geometric_denominator_margin(&self) -> &BigRational {
        &self.geometric_denominator_margin
    }

    pub fn tail_upper(&self) -> &BigRational {
        &self.tail_upper
    }

    pub fn reduced_enclosure(&self) -> &RationalInterval {
        &self.reduced_enclosure
    }

    /// The enclosure after each successive squaring.  Its length is exactly
    /// [`Self::range_reductions`].
    pub fn squaring_enclosures(&self) -> &[RationalInterval] {
        &self.squaring_enclosures
    }

    pub fn enclosure(&self) -> &RationalInterval {
        &self.enclosure
    }

    /// Exact rational upper enclosure intended for direct use in a Gronwall
    /// multiplier.  This is the upper endpoint of [`Self::enclosure`].
    pub fn upper_bound(&self) -> &BigRational {
        self.enclosure.upper()
    }

    /// Exact width of the final, post-squaring enclosure.
    pub fn final_enclosure_width(&self) -> BigRational {
        self.enclosure.width()
    }

    /// Validate structural and resource properties of every cached field
    /// before any algebraic comparison or proof recomputation.
    fn preflight(
        &self,
        limits: ExpEnclosureLimits,
        work: &mut WorkBudget,
    ) -> Result<(), ExpEnclosureError> {
        if self.range_reductions > limits.max_range_reductions {
            return Err(ExpEnclosureError::RangeReductionLimitExceeded);
        }
        if self.taylor_cutoff > limits.max_taylor_cutoff {
            return Err(ExpEnclosureError::TaylorCutoffLimitExceeded);
        }
        if self.squaring_enclosures.len() != self.range_reductions {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }

        let mut storage = WitnessStorageBudget::new(limits);
        for value in [&self.exponent, &self.max_reduced_tail_upper] {
            preflight_input_rational(value, limits, work)?;
            storage.charge_rational(value)?;
        }
        for value in [
            &self.reduced_argument,
            &self.partial_sum,
            &self.first_omitted_term,
            &self.geometric_ratio,
            &self.geometric_denominator_margin,
            &self.tail_upper,
        ] {
            preflight_intermediate_rational(value, limits, work)?;
            storage.charge_rational(value)?;
        }
        for interval in core::iter::once(&self.reduced_enclosure)
            .chain(self.squaring_enclosures.iter())
            .chain(core::iter::once(&self.enclosure))
        {
            for endpoint in [interval.lower(), interval.upper()] {
                preflight_intermediate_rational(endpoint, limits, work)?;
                storage.charge_rational(endpoint)?;
            }
        }
        Ok(())
    }

    /// Recompute every theorem-facing postcondition using exact arithmetic.
    ///
    /// This checks the deterministic range reduction, Taylor recurrence,
    /// geometric-tail identity and inequality, declared tolerance, and every
    /// interval-squaring step.  It is deliberately independent of the cached
    /// fields except when comparing them with recomputed values.
    pub fn verify(&self, limits: ExpEnclosureLimits) -> Result<(), ExpEnclosureError> {
        validate_limits(limits)?;
        let mut work = WorkBudget::new(limits);
        self.verify_with_budget(limits, &mut work)
    }

    fn verify_with_budget(
        &self,
        limits: ExpEnclosureLimits,
        work: &mut WorkBudget,
    ) -> Result<(), ExpEnclosureError> {
        self.preflight(limits, work)?;
        if rational_is_negative(&self.exponent) {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }
        if !rational_is_positive(&self.max_reduced_tail_upper) {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }
        let (expected_reductions, expected_reduced) = range_reduce(&self.exponent, limits, work)?;
        if self.range_reductions != expected_reductions
            || !checked_equal(&self.reduced_argument, &expected_reduced, work)?
        {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }

        let taylor = taylor_tail_data(&expected_reduced, self.taylor_cutoff, limits, work)?;
        for (cached, expected) in [
            (&self.partial_sum, &taylor.partial_sum),
            (&self.first_omitted_term, &taylor.first_omitted_term),
            (&self.geometric_ratio, &taylor.geometric_ratio),
            (
                &self.geometric_denominator_margin,
                &taylor.geometric_denominator_margin,
            ),
            (&self.tail_upper, &taylor.tail_upper),
        ] {
            if !checked_equal(cached, expected, work)? {
                return Err(ExpEnclosureError::WitnessPostconditionFailure);
            }
        }
        if checked_compare(&self.tail_upper, &self.max_reduced_tail_upper, work)?.is_gt() {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }

        let expected_reduced_enclosure = interval_from_taylor(&taylor, limits, work)?;
        if !checked_interval_equal(&self.reduced_enclosure, &expected_reduced_enclosure, work)? {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }

        verify_squaring_transcript(
            expected_reduced_enclosure,
            &self.squaring_enclosures,
            &self.enclosure,
            limits,
            work,
        )
    }

    pub fn verifies(&self, limits: ExpEnclosureLimits) -> bool {
        self.verify(limits).is_ok()
    }
}

/// Enclose `exp(exponent)` by exact rational arithmetic.
///
/// `taylor_cutoff = N` means that the reduced Taylor partial sum contains
/// terms through degree `N`.  `max_reduced_tail_upper` is an exact upper limit
/// on the geometric majorant for the reduced (pre-squaring) tail.  It is not a
/// bound on the final enclosure width; use
/// [`RationalExpEnclosure::final_enclosure_width`] for that quantity.  A cutoff
/// that does not meet the declared reduced-tail accuracy is rejected rather
/// than silently returning a looser proof.
pub fn exp_enclosure_rational(
    exponent: &BigRational,
    taylor_cutoff: usize,
    max_reduced_tail_upper: &BigRational,
    limits: ExpEnclosureLimits,
) -> Result<RationalExpEnclosure, ExpEnclosureError> {
    validate_limits(limits)?;
    let mut work = WorkBudget::new(limits);
    preflight_input_rational(exponent, limits, &mut work)?;
    preflight_input_rational(max_reduced_tail_upper, limits, &mut work)?;
    if rational_is_negative(exponent) {
        return Err(ExpEnclosureError::NegativeExponent);
    }
    if !rational_is_positive(max_reduced_tail_upper) {
        return Err(ExpEnclosureError::NonpositiveMaxReducedTailUpper);
    }
    if taylor_cutoff > limits.max_taylor_cutoff {
        return Err(ExpEnclosureError::TaylorCutoffLimitExceeded);
    }

    let (range_reductions, reduced_argument) = range_reduce(exponent, limits, &mut work)?;
    let taylor = taylor_tail_data(&reduced_argument, taylor_cutoff, limits, &mut work)?;
    if checked_compare(&taylor.tail_upper, max_reduced_tail_upper, &mut work)?.is_gt() {
        return Err(ExpEnclosureError::InsufficientTaylorCutoff {
            cutoff: taylor_cutoff,
        });
    }
    let reduced_enclosure = interval_from_taylor(&taylor, limits, &mut work)?;

    let mut storage = WitnessStorageBudget::new(limits);
    for value in [
        exponent,
        &reduced_argument,
        max_reduced_tail_upper,
        &taylor.partial_sum,
        &taylor.first_omitted_term,
        &taylor.geometric_ratio,
        &taylor.geometric_denominator_margin,
        &taylor.tail_upper,
    ] {
        storage.charge_rational(value)?;
    }
    storage.charge_interval(&reduced_enclosure)?;
    let (squaring_enclosures, enclosure) = square_enclosure_steps(
        reduced_enclosure.clone(),
        range_reductions,
        limits,
        &mut work,
        &mut storage,
    )?;
    storage.charge_interval(&enclosure)?;

    let result = RationalExpEnclosure {
        exponent: exponent.clone(),
        range_reductions,
        taylor_cutoff,
        reduced_argument,
        max_reduced_tail_upper: max_reduced_tail_upper.clone(),
        partial_sum: taylor.partial_sum,
        first_omitted_term: taylor.first_omitted_term,
        geometric_ratio: taylor.geometric_ratio,
        geometric_denominator_margin: taylor.geometric_denominator_margin,
        tail_upper: taylor.tail_upper,
        reduced_enclosure,
        squaring_enclosures,
        enclosure,
    };
    result.verify_with_budget(limits, &mut work)?;
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TaylorTailData {
    partial_sum: BigRational,
    first_omitted_term: BigRational,
    geometric_ratio: BigRational,
    geometric_denominator_margin: BigRational,
    tail_upper: BigRational,
}

fn range_reduce(
    exponent: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<(usize, BigRational), ExpEnclosureError> {
    let half = BigRational::new(BigInt::one(), BigInt::from(2_u8));
    let two = BigRational::from_integer(BigInt::from(2_u8));
    let mut reduced = exponent.clone();
    ensure_intermediate_rational(&reduced, limits)?;
    let mut reductions = 0_usize;
    while checked_compare(&reduced, &half, work)?.is_gt() {
        if reductions == limits.max_range_reductions {
            return Err(ExpEnclosureError::RangeReductionLimitExceeded);
        }
        reduced = checked_divide(&reduced, &two, limits, work)?;
        reductions += 1;
    }

    // Retain a deterministic, minimal k.  This is not needed for soundness,
    // but makes the witness representation unique and easier to compare.
    if reductions > 0 {
        let previous = checked_multiply(&reduced, &two, limits, work)?;
        if !checked_compare(&previous, &half, work)?.is_gt() {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }
    }
    Ok((reductions, reduced))
}

fn taylor_tail_data(
    reduced_argument: &BigRational,
    cutoff: usize,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<TaylorTailData, ExpEnclosureError> {
    if cutoff > limits.max_taylor_cutoff {
        return Err(ExpEnclosureError::TaylorCutoffLimitExceeded);
    }
    let half = BigRational::new(BigInt::one(), BigInt::from(2_u8));
    if rational_is_negative(reduced_argument)
        || checked_compare(reduced_argument, &half, work)?.is_gt()
    {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }

    let one = BigRational::one();
    let mut term = one.clone();
    let mut partial_sum = one.clone();
    for index in 1..=cutoff {
        term = next_taylor_term(&term, reduced_argument, index, limits, work)?;
        partial_sum = checked_add(&partial_sum, &term, limits, work)?;
    }

    let first_omitted_index = cutoff
        .checked_add(1)
        .ok_or(ExpEnclosureError::TaylorCutoffArithmeticOverflow)?;
    let ratio_denominator = cutoff
        .checked_add(2)
        .ok_or(ExpEnclosureError::TaylorCutoffArithmeticOverflow)?;
    let first_omitted_term =
        next_taylor_term(&term, reduced_argument, first_omitted_index, limits, work)?;
    let geometric_ratio =
        checked_divide_by_positive_usize(reduced_argument, ratio_denominator, limits, work)?;
    let geometric_denominator_margin = checked_subtract(&one, &geometric_ratio, limits, work)?;
    if rational_is_negative(&geometric_ratio)
        || !checked_compare(&geometric_ratio, &one, work)?.is_lt()
        || !rational_is_positive(&geometric_denominator_margin)
    {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }
    let tail_upper = checked_divide(
        &first_omitted_term,
        &geometric_denominator_margin,
        limits,
        work,
    )?;
    let reconstructed_first =
        checked_multiply(&tail_upper, &geometric_denominator_margin, limits, work)?;
    if rational_is_negative(&tail_upper)
        || !checked_equal(&reconstructed_first, &first_omitted_term, work)?
    {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }

    Ok(TaylorTailData {
        partial_sum,
        first_omitted_term,
        geometric_ratio,
        geometric_denominator_margin,
        tail_upper,
    })
}

fn next_taylor_term(
    previous: &BigRational,
    argument: &BigRational,
    index: usize,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    let product = checked_multiply(previous, argument, limits, work)?;
    checked_divide_by_positive_usize(&product, index, limits, work)
}

fn interval_from_taylor(
    taylor: &TaylorTailData,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<RationalInterval, ExpEnclosureError> {
    let upper = checked_add(&taylor.partial_sum, &taylor.tail_upper, limits, work)?;
    charge_interval_construction(&taylor.partial_sum, &upper, work)?;
    RationalInterval::new(taylor.partial_sum.clone(), upper)
        .map_err(|_| ExpEnclosureError::WitnessPostconditionFailure)
}

fn square_enclosure_steps(
    reduced_enclosure: RationalInterval,
    count: usize,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
    storage: &mut WitnessStorageBudget,
) -> Result<(Vec<RationalInterval>, RationalInterval), ExpEnclosureError> {
    if count > limits.max_range_reductions {
        return Err(ExpEnclosureError::RangeReductionLimitExceeded);
    }
    if rational_is_negative(reduced_enclosure.lower()) {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }
    ensure_intermediate_rational(reduced_enclosure.lower(), limits)?;
    ensure_intermediate_rational(reduced_enclosure.upper(), limits)?;

    let mut current = reduced_enclosure;
    let mut steps = Vec::with_capacity(count);
    for _ in 0..count {
        let lower = checked_multiply(current.lower(), current.lower(), limits, work)?;
        let upper = checked_multiply(current.upper(), current.upper(), limits, work)?;
        charge_interval_construction(&lower, &upper, work)?;
        current = RationalInterval::new(lower, upper)
            .map_err(|_| ExpEnclosureError::WitnessPostconditionFailure)?;
        storage.charge_interval(&current)?;
        steps.push(current.clone());
    }
    Ok((steps, current))
}

fn verify_squaring_transcript(
    reduced_enclosure: RationalInterval,
    transcript: &[RationalInterval],
    final_enclosure: &RationalInterval,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<(), ExpEnclosureError> {
    if rational_is_negative(reduced_enclosure.lower()) {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }
    let mut current = reduced_enclosure;
    for cached in transcript {
        let lower = checked_multiply(current.lower(), current.lower(), limits, work)?;
        let upper = checked_multiply(current.upper(), current.upper(), limits, work)?;
        charge_interval_construction(&lower, &upper, work)?;
        let expected = RationalInterval::new(lower, upper)
            .map_err(|_| ExpEnclosureError::WitnessPostconditionFailure)?;
        if !checked_interval_equal(cached, &expected, work)? {
            return Err(ExpEnclosureError::WitnessPostconditionFailure);
        }
        current = expected;
    }
    if !checked_interval_equal(&current, final_enclosure, work)? {
        return Err(ExpEnclosureError::WitnessPostconditionFailure);
    }
    Ok(())
}

fn validate_limits(limits: ExpEnclosureLimits) -> Result<(), ExpEnclosureError> {
    let invalid = limits.max_input_component_bits == 0
        || limits.max_intermediate_component_bits == 0
        || limits.max_input_component_bits > limits.max_intermediate_component_bits
        || limits.max_input_component_bits > HARD_MAX_EXP_INPUT_COMPONENT_BITS
        || limits.max_intermediate_component_bits > HARD_MAX_EXP_INTERMEDIATE_COMPONENT_BITS
        || limits.max_range_reductions > HARD_MAX_EXP_RANGE_REDUCTIONS
        || limits.max_taylor_cutoff > HARD_MAX_EXP_TAYLOR_CUTOFF
        || limits.max_work_units == 0
        || limits.max_work_units > HARD_MAX_EXP_WORK_UNITS
        || limits.max_witness_storage_bits == 0
        || limits.max_witness_storage_bits > HARD_MAX_EXP_WITNESS_STORAGE_BITS;
    if invalid {
        Err(ExpEnclosureError::InvalidResourceLimits)
    } else {
        Ok(())
    }
}

/// Deterministic aggregate work accounting in 64-bit-limb-pair units.
///
/// Construction and the constructor's mandatory replay share one instance.
/// A standalone [`RationalExpEnclosure::verify`] starts a fresh instance with
/// the same non-relaxable ceiling.  Scans charge linear limb count; binary
/// rational operations and comparisons charge the product of operand counts.
struct WorkBudget {
    remaining: u64,
}

impl WorkBudget {
    const fn new(limits: ExpEnclosureLimits) -> Self {
        Self {
            remaining: limits.max_work_units,
        }
    }

    fn charge(&mut self, units: u64) -> Result<(), ExpEnclosureError> {
        self.remaining = self
            .remaining
            .checked_sub(units)
            .ok_or(ExpEnclosureError::WorkBudgetExceeded)?;
        Ok(())
    }

    fn charge_scan(&mut self, value: &BigRational) -> Result<(), ExpEnclosureError> {
        self.charge(rational_limb_units(value))
    }

    fn charge_binary(
        &mut self,
        left: &BigRational,
        right: &BigRational,
    ) -> Result<(), ExpEnclosureError> {
        let units = rational_limb_units(left)
            .checked_mul(rational_limb_units(right))
            .ok_or(ExpEnclosureError::WorkBudgetExceeded)?;
        self.charge(units)
    }
}

/// Deterministic storage accounting for every rational cached in a witness.
struct WitnessStorageBudget {
    used_bits: u64,
    limit_bits: u64,
}

impl WitnessStorageBudget {
    const fn new(limits: ExpEnclosureLimits) -> Self {
        Self {
            used_bits: 0,
            limit_bits: limits.max_witness_storage_bits,
        }
    }

    fn charge_rational(&mut self, value: &BigRational) -> Result<(), ExpEnclosureError> {
        // The 128-bit surcharge accounts deterministically for two signed
        // big-integer handles and rational metadata in addition to bit payload.
        let bits = value
            .numer()
            .bits()
            .checked_add(value.denom().bits())
            .and_then(|sum| sum.checked_add(128))
            .ok_or(ExpEnclosureError::WitnessStorageBudgetExceeded)?;
        self.used_bits = self
            .used_bits
            .checked_add(bits)
            .ok_or(ExpEnclosureError::WitnessStorageBudgetExceeded)?;
        if self.used_bits > self.limit_bits {
            return Err(ExpEnclosureError::WitnessStorageBudgetExceeded);
        }
        Ok(())
    }

    fn charge_interval(&mut self, interval: &RationalInterval) -> Result<(), ExpEnclosureError> {
        self.charge_rational(interval.lower())?;
        self.charge_rational(interval.upper())
    }
}

fn rational_limb_units(value: &BigRational) -> u64 {
    fn component_limb_units(bits: u64) -> u64 {
        bits.div_ceil(64).max(1)
    }
    component_limb_units(value.numer().bits()) + component_limb_units(value.denom().bits())
}

fn rational_is_negative(value: &BigRational) -> bool {
    value.numer().sign() == Sign::Minus
}

fn rational_is_positive(value: &BigRational) -> bool {
    value.numer().sign() == Sign::Plus
}

fn preflight_input_rational(
    value: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<(), ExpEnclosureError> {
    preflight_canonical_rational(
        value,
        limits.max_input_component_bits,
        ExpEnclosureError::InputComponentBitLimitExceeded,
        work,
    )
}

fn preflight_intermediate_rational(
    value: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<(), ExpEnclosureError> {
    preflight_canonical_rational(
        value,
        limits.max_intermediate_component_bits,
        ExpEnclosureError::IntermediateComponentBitLimitExceeded,
        work,
    )
}

/// Inspect raw numerator and denominator fields without invoking rational
/// arithmetic.  Only after the bit cap and positive-denominator checks pass do
/// we construct a bounded normalized clone and require exact canonical form.
fn preflight_canonical_rational(
    value: &BigRational,
    component_bit_limit: u64,
    bit_limit_error: ExpEnclosureError,
    work: &mut WorkBudget,
) -> Result<(), ExpEnclosureError> {
    if !rational_component_bits_within(value, component_bit_limit) {
        return Err(bit_limit_error);
    }
    if value.denom().sign() != Sign::Plus {
        return Err(ExpEnclosureError::MalformedRationalDenominator);
    }
    work.charge_scan(value)?;
    // Charge a conservative limb-pair unit for the bounded gcd/normalization.
    work.charge_binary(value, value)?;
    let normalized = BigRational::new(value.numer().clone(), value.denom().clone());
    if normalized.numer() != value.numer() || normalized.denom() != value.denom() {
        return Err(ExpEnclosureError::NoncanonicalRational);
    }
    Ok(())
}

fn ensure_intermediate_rational(
    value: &BigRational,
    limits: ExpEnclosureLimits,
) -> Result<(), ExpEnclosureError> {
    if rational_component_bits_within(value, limits.max_intermediate_component_bits) {
        Ok(())
    } else {
        Err(ExpEnclosureError::IntermediateComponentBitLimitExceeded)
    }
}

fn rational_component_bits_within(value: &BigRational, limit: u64) -> bool {
    value.numer().bits() <= limit && value.denom().bits() <= limit
}

fn checked_add(
    left: &BigRational,
    right: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    ensure_intermediate_rational(left, limits)?;
    ensure_intermediate_rational(right, limits)?;
    work.charge_binary(left, right)?;
    let first_cross_bits = checked_sum_bits(left.numer().bits(), right.denom().bits())?;
    let second_cross_bits = checked_sum_bits(right.numer().bits(), left.denom().bits())?;
    let numerator_bound = first_cross_bits
        .max(second_cross_bits)
        .checked_add(1)
        .ok_or(ExpEnclosureError::IntermediateComponentBitLimitExceeded)?;
    let denominator_bound = checked_sum_bits(left.denom().bits(), right.denom().bits())?;
    ensure_estimated_bits(numerator_bound, denominator_bound, limits)?;
    let result = left + right;
    ensure_intermediate_rational(&result, limits)?;
    Ok(result)
}

fn checked_subtract(
    left: &BigRational,
    right: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    ensure_intermediate_rational(left, limits)?;
    ensure_intermediate_rational(right, limits)?;
    work.charge_binary(left, right)?;
    let first_cross_bits = checked_sum_bits(left.numer().bits(), right.denom().bits())?;
    let second_cross_bits = checked_sum_bits(right.numer().bits(), left.denom().bits())?;
    let numerator_bound = first_cross_bits
        .max(second_cross_bits)
        .checked_add(1)
        .ok_or(ExpEnclosureError::IntermediateComponentBitLimitExceeded)?;
    let denominator_bound = checked_sum_bits(left.denom().bits(), right.denom().bits())?;
    ensure_estimated_bits(numerator_bound, denominator_bound, limits)?;
    let result = left - right;
    ensure_intermediate_rational(&result, limits)?;
    Ok(result)
}

fn checked_multiply(
    left: &BigRational,
    right: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    ensure_intermediate_rational(left, limits)?;
    ensure_intermediate_rational(right, limits)?;
    work.charge_binary(left, right)?;
    let numerator_bound = checked_sum_bits(left.numer().bits(), right.numer().bits())?;
    let denominator_bound = checked_sum_bits(left.denom().bits(), right.denom().bits())?;
    ensure_estimated_bits(numerator_bound, denominator_bound, limits)?;
    let result = left * right;
    ensure_intermediate_rational(&result, limits)?;
    Ok(result)
}

fn checked_divide(
    numerator: &BigRational,
    denominator: &BigRational,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    if denominator.numer().sign() == Sign::NoSign {
        return Err(ExpEnclosureError::ExactDivisionByZero);
    }
    ensure_intermediate_rational(numerator, limits)?;
    ensure_intermediate_rational(denominator, limits)?;
    work.charge_binary(numerator, denominator)?;
    let numerator_bound = checked_sum_bits(numerator.numer().bits(), denominator.denom().bits())?;
    let denominator_bound = checked_sum_bits(numerator.denom().bits(), denominator.numer().bits())?;
    ensure_estimated_bits(numerator_bound, denominator_bound, limits)?;
    let result = numerator / denominator;
    ensure_intermediate_rational(&result, limits)?;
    Ok(result)
}

fn checked_divide_by_positive_usize(
    value: &BigRational,
    denominator: usize,
    limits: ExpEnclosureLimits,
    work: &mut WorkBudget,
) -> Result<BigRational, ExpEnclosureError> {
    if denominator == 0 {
        return Err(ExpEnclosureError::ExactDivisionByZero);
    }
    let rational_denominator = BigRational::from_integer(BigInt::from(denominator));
    checked_divide(value, &rational_denominator, limits, work)
}

fn checked_compare(
    left: &BigRational,
    right: &BigRational,
    work: &mut WorkBudget,
) -> Result<core::cmp::Ordering, ExpEnclosureError> {
    work.charge_binary(left, right)?;
    Ok(left.cmp(right))
}

fn checked_equal(
    left: &BigRational,
    right: &BigRational,
    work: &mut WorkBudget,
) -> Result<bool, ExpEnclosureError> {
    work.charge_binary(left, right)?;
    Ok(left.numer() == right.numer() && left.denom() == right.denom())
}

fn checked_interval_equal(
    left: &RationalInterval,
    right: &RationalInterval,
    work: &mut WorkBudget,
) -> Result<bool, ExpEnclosureError> {
    Ok(checked_equal(left.lower(), right.lower(), work)?
        && checked_equal(left.upper(), right.upper(), work)?)
}

/// Account for the bounded canonical-normalization checks and endpoint
/// comparison performed by `RationalInterval::new` before invoking it.
fn charge_interval_construction(
    lower: &BigRational,
    upper: &BigRational,
    work: &mut WorkBudget,
) -> Result<(), ExpEnclosureError> {
    work.charge_binary(lower, lower)?;
    work.charge_binary(upper, upper)?;
    let _ = checked_compare(lower, upper, work)?;
    Ok(())
}

fn checked_sum_bits(left: u64, right: u64) -> Result<u64, ExpEnclosureError> {
    left.checked_add(right)
        .ok_or(ExpEnclosureError::IntermediateComponentBitLimitExceeded)
}

fn ensure_estimated_bits(
    numerator_bits: u64,
    denominator_bits: u64,
    limits: ExpEnclosureLimits,
) -> Result<(), ExpEnclosureError> {
    if numerator_bits > limits.max_intermediate_component_bits
        || denominator_bits > limits.max_intermediate_component_bits
    {
        Err(ExpEnclosureError::IntermediateComponentBitLimitExceeded)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rational(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn proof(
        exponent: BigRational,
        cutoff: usize,
        max_reduced_tail_upper: BigRational,
    ) -> RationalExpEnclosure {
        exp_enclosure_rational(
            &exponent,
            cutoff,
            &max_reduced_tail_upper,
            DEFAULT_EXP_ENCLOSURE_LIMITS,
        )
        .unwrap()
    }

    fn assert_witness_mismatch(witness: &RationalExpEnclosure) {
        assert_eq!(
            witness.verify(DEFAULT_EXP_ENCLOSURE_LIMITS),
            Err(ExpEnclosureError::WitnessPostconditionFailure)
        );
    }

    #[test]
    fn zero_exponent_is_the_exact_point_one() {
        let result = proof(rational(0, 1), 0, rational(1, 100));
        assert_eq!(result.range_reductions(), 0);
        assert_eq!(result.reduced_argument(), &rational(0, 1));
        assert_eq!(result.partial_sum(), &rational(1, 1));
        assert_eq!(result.first_omitted_term(), &rational(0, 1));
        assert_eq!(result.geometric_ratio(), &rational(0, 1));
        assert_eq!(result.tail_upper(), &rational(0, 1));
        assert_eq!(
            result.enclosure(),
            &RationalInterval::try_point(rational(1, 1)).unwrap()
        );
        assert!(result.squaring_enclosures().is_empty());
        assert!(result.verifies(DEFAULT_EXP_ENCLOSURE_LIMITS));
    }

    #[test]
    fn half_and_one_match_hand_derived_witnesses() {
        // For y=1/2 and N=2:
        // S_2=13/8, t_3=1/48, q=1/8, tail<=1/42.
        let half = proof(rational(1, 2), 2, rational(1, 40));
        assert_eq!(half.range_reductions(), 0);
        assert_eq!(half.partial_sum(), &rational(13, 8));
        assert_eq!(half.first_omitted_term(), &rational(1, 48));
        assert_eq!(half.geometric_ratio(), &rational(1, 8));
        assert_eq!(half.geometric_denominator_margin(), &rational(7, 8));
        assert_eq!(half.tail_upper(), &rational(1, 42));
        assert_eq!(half.reduced_enclosure().lower(), &rational(13, 8));
        assert_eq!(half.reduced_enclosure().upper(), &rational(277, 168));

        // x=1 reduces once to the same y and squares the positive bracket.
        let one = proof(rational(1, 1), 2, rational(1, 40));
        assert_eq!(one.range_reductions(), 1);
        assert_eq!(one.reduced_argument(), &rational(1, 2));
        assert_eq!(one.enclosure().lower(), &rational(169, 64));
        assert_eq!(one.enclosure().upper(), &rational(76_729, 28_224));
        assert_eq!(one.upper_bound(), &rational(76_729, 28_224));
        assert_eq!(
            one.final_enclosure_width(),
            one.enclosure().upper() - one.enclosure().lower()
        );
        assert_eq!(one.squaring_enclosures(), &[one.enclosure().clone()]);

        // Conservative textbook rational bounds, used only as a comparison:
        // the proof enclosure itself establishes the much sharper 5/2<e<3.
        assert!(one.enclosure().lower() > &rational(5, 2));
        assert!(one.enclosure().upper() < &rational(3, 1));
    }

    #[test]
    fn range_reduction_and_bounds_are_monotone_on_simple_rationals() {
        let quarter = proof(rational(1, 4), 4, rational(1, 10_000));
        let half = proof(rational(1, 2), 4, rational(1, 1_000));
        let one = proof(rational(1, 1), 4, rational(1, 1_000));

        for adjacent in [&quarter, &half, &one].windows(2) {
            assert!(adjacent[0].enclosure().lower() <= adjacent[1].enclosure().lower());
            assert!(adjacent[0].enclosure().upper() <= adjacent[1].enclosure().upper());
        }
        assert!(quarter.enclosure().lower() > &rational(1, 1));
        assert!(half.enclosure().upper() < &rational(2, 1));
        assert!(one.enclosure().upper() < &rational(3, 1));
    }

    #[test]
    fn increasing_cutoff_gives_nested_enclosures() {
        let coarse = proof(rational(1, 1), 2, rational(1, 40));
        let fine = proof(rational(1, 1), 6, rational(1, 500_000));
        assert!(coarse.enclosure().contains_interval(fine.enclosure()));
        assert!(fine.enclosure().lower() > coarse.enclosure().lower());
        assert!(fine.enclosure().upper() < coarse.enclosure().upper());
    }

    #[test]
    fn range_reduction_is_exact_at_half_plus_or_minus_epsilon() {
        let below = proof(rational(511, 1024), 4, rational(1, 1_000));
        let boundary = proof(rational(1, 2), 4, rational(1, 1_000));
        let above = proof(rational(513, 1024), 4, rational(1, 1_000));

        assert_eq!(below.range_reductions(), 0);
        assert_eq!(below.reduced_argument(), &rational(511, 1024));
        assert_eq!(boundary.range_reductions(), 0);
        assert_eq!(boundary.reduced_argument(), &rational(1, 2));
        assert_eq!(above.range_reductions(), 1);
        assert_eq!(above.reduced_argument(), &rational(513, 2048));
    }

    #[test]
    fn exact_powers_and_nondyadic_arguments_have_exact_reduction_transcripts() {
        let power = proof(rational(8, 1), 4, rational(1, 1_000));
        assert_eq!(power.range_reductions(), 4);
        assert_eq!(power.reduced_argument(), &rational(1, 2));
        assert_eq!(power.squaring_enclosures().len(), 4);

        let nondyadic = proof(rational(2, 3), 4, rational(1, 10_000));
        assert_eq!(nondyadic.range_reductions(), 1);
        assert_eq!(nondyadic.reduced_argument(), &rational(1, 3));
        assert!(nondyadic.verifies(DEFAULT_EXP_ENCLOSURE_LIMITS));
    }

    #[test]
    fn reduced_tail_threshold_accepts_equality_and_rejects_just_below() {
        let exact = exp_enclosure_rational(
            &rational(1, 2),
            2,
            &rational(1, 42),
            DEFAULT_EXP_ENCLOSURE_LIMITS,
        )
        .unwrap();
        assert_eq!(exact.tail_upper(), &rational(1, 42));
        assert_eq!(exact.max_reduced_tail_upper(), &rational(1, 42));

        assert_eq!(
            exp_enclosure_rational(
                &rational(1, 2),
                2,
                &rational(1, 43),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            ),
            Err(ExpEnclosureError::InsufficientTaylorCutoff { cutoff: 2 })
        );
    }

    #[test]
    fn multi_step_squaring_transcript_is_exact() {
        let result = proof(rational(2, 1), 2, rational(1, 40));
        assert_eq!(result.range_reductions(), 2);
        assert_eq!(result.squaring_enclosures().len(), 2);
        let first = &result.squaring_enclosures()[0];
        let second = &result.squaring_enclosures()[1];
        assert_eq!(first.lower(), &rational(169, 64));
        assert_eq!(first.upper(), &rational(76_729, 28_224));
        assert_eq!(second.lower(), &(first.lower() * first.lower()));
        assert_eq!(second.upper(), &(first.upper() * first.upper()));
        assert_eq!(result.enclosure(), second);
    }

    #[test]
    fn insufficient_cutoff_and_invalid_domain_fail_closed() {
        assert_eq!(
            exp_enclosure_rational(
                &rational(1, 1),
                0,
                &rational(1, 100),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            ),
            Err(ExpEnclosureError::InsufficientTaylorCutoff { cutoff: 0 })
        );
        assert_eq!(
            exp_enclosure_rational(
                &rational(-1, 10),
                3,
                &rational(1, 100),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            ),
            Err(ExpEnclosureError::NegativeExponent)
        );
        for tolerance in [rational(0, 1), rational(-1, 1)] {
            assert_eq!(
                exp_enclosure_rational(
                    &rational(1, 1),
                    3,
                    &tolerance,
                    DEFAULT_EXP_ENCLOSURE_LIMITS,
                ),
                Err(ExpEnclosureError::NonpositiveMaxReducedTailUpper)
            );
        }
    }

    #[test]
    fn malformed_and_noncanonical_raw_inputs_never_reach_rational_arithmetic() {
        let malformed = [
            BigRational::new_raw(BigInt::one(), BigInt::from(0_u8)),
            BigRational::new_raw(BigInt::from(0_u8), BigInt::from(0_u8)),
            BigRational::new_raw(BigInt::one(), BigInt::from(-2_i8)),
        ];
        for value in malformed {
            let outcome = std::panic::catch_unwind(|| {
                exp_enclosure_rational(&value, 2, &rational(1, 10), DEFAULT_EXP_ENCLOSURE_LIMITS)
            });
            assert_eq!(
                outcome.unwrap(),
                Err(ExpEnclosureError::MalformedRationalDenominator)
            );
        }

        let unreduced = BigRational::new_raw(BigInt::from(2_u8), BigInt::from(4_u8));
        let outcome = std::panic::catch_unwind(|| {
            exp_enclosure_rational(
                &unreduced,
                2,
                &rational(1, 10),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            )
        });
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::NoncanonicalRational)
        );

        let malformed_bound = BigRational::new_raw(BigInt::one(), BigInt::from(0_u8));
        let outcome = std::panic::catch_unwind(|| {
            exp_enclosure_rational(
                &rational(1, 2),
                2,
                &malformed_bound,
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            )
        });
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::MalformedRationalDenominator)
        );
    }

    #[test]
    fn bit_limits_precede_denominator_and_sign_checks() {
        let oversized_malformed =
            BigRational::new_raw(-(BigInt::one() << 3_000_usize), BigInt::from(0_u8));
        let outcome = std::panic::catch_unwind(|| {
            exp_enclosure_rational(
                &oversized_malformed,
                2,
                &rational(1, 10),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            )
        });
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::InputComponentBitLimitExceeded)
        );

        let oversized_negative = BigRational::from_integer(-(BigInt::one() << 3_000_usize));
        assert_eq!(
            exp_enclosure_rational(
                &oversized_negative,
                2,
                &rational(1, 10),
                DEFAULT_EXP_ENCLOSURE_LIMITS,
            ),
            Err(ExpEnclosureError::InputComponentBitLimitExceeded)
        );
    }

    #[test]
    fn every_resource_boundary_is_enforced() {
        assert_eq!(validate_limits(DEFAULT_EXP_ENCLOSURE_LIMITS), Ok(()));

        let cutoff_limits = ExpEnclosureLimits {
            max_taylor_cutoff: 2,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(1, 2), 3, &rational(1, 1), cutoff_limits,),
            Err(ExpEnclosureError::TaylorCutoffLimitExceeded)
        );

        let range_limits = ExpEnclosureLimits {
            max_range_reductions: 2,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(8, 1), 2, &rational(1, 1), range_limits,),
            Err(ExpEnclosureError::RangeReductionLimitExceeded)
        );

        let input_limits = ExpEnclosureLimits {
            max_input_component_bits: 8,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        let large_input = BigRational::from_integer(BigInt::one() << 20_usize);
        assert_eq!(
            exp_enclosure_rational(&large_input, 2, &rational(1, 1), input_limits,),
            Err(ExpEnclosureError::InputComponentBitLimitExceeded)
        );

        let intermediate_limits = ExpEnclosureLimits {
            max_input_component_bits: 4,
            max_intermediate_component_bits: 8,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(1, 2), 4, &rational(1, 1), intermediate_limits,),
            Err(ExpEnclosureError::IntermediateComponentBitLimitExceeded)
        );

        let invalid_limits = ExpEnclosureLimits {
            max_taylor_cutoff: HARD_MAX_EXP_TAYLOR_CUTOFF + 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), invalid_limits,),
            Err(ExpEnclosureError::InvalidResourceLimits)
        );

        let work_limits = ExpEnclosureLimits {
            max_work_units: 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), work_limits),
            Err(ExpEnclosureError::WorkBudgetExceeded)
        );

        let storage_limits = ExpEnclosureLimits {
            max_witness_storage_bits: 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), storage_limits),
            Err(ExpEnclosureError::WitnessStorageBudgetExceeded)
        );

        let above_hard_work = ExpEnclosureLimits {
            max_work_units: HARD_MAX_EXP_WORK_UNITS + 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), above_hard_work,),
            Err(ExpEnclosureError::InvalidResourceLimits)
        );
    }

    #[test]
    fn one_aggregate_work_budget_covers_construction_and_mandatory_replay() {
        let first_success = (1_u64..2_000)
            .find(|budget| {
                let limits = ExpEnclosureLimits {
                    max_work_units: *budget,
                    ..DEFAULT_EXP_ENCLOSURE_LIMITS
                };
                exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), limits).is_ok()
            })
            .expect("the zero witness fits a small deterministic work budget");
        assert!(first_success > 1);
        let just_below = ExpEnclosureLimits {
            max_work_units: first_success - 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), just_below),
            Err(ExpEnclosureError::WorkBudgetExceeded)
        );
        let exact = ExpEnclosureLimits {
            max_work_units: first_success,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert!(exp_enclosure_rational(&rational(0, 1), 0, &rational(1, 1), exact).is_ok());
    }

    #[test]
    fn exact_postcondition_verifier_rejects_tampering() {
        let valid = proof(rational(3, 2), 5, rational(1, 100_000));
        assert_eq!(valid.verify(DEFAULT_EXP_ENCLOSURE_LIMITS), Ok(()));

        let mut bad_exponent = valid.clone();
        bad_exponent.exponent = rational(7, 5);
        assert_witness_mismatch(&bad_exponent);

        let mut bad_reductions = valid.clone();
        bad_reductions.range_reductions += 1;
        assert_witness_mismatch(&bad_reductions);

        let mut bad_cutoff = valid.clone();
        bad_cutoff.taylor_cutoff += 1;
        assert_witness_mismatch(&bad_cutoff);

        let mut bad_reduced_argument = valid.clone();
        bad_reduced_argument.reduced_argument += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_reduced_argument);

        let mut bad_declared_tail = valid.clone();
        bad_declared_tail.max_reduced_tail_upper = &valid.tail_upper / rational(2, 1);
        assert_witness_mismatch(&bad_declared_tail);

        let mut bad_partial = valid.clone();
        bad_partial.partial_sum += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_partial);

        let mut bad_first_omitted = valid.clone();
        bad_first_omitted.first_omitted_term += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_first_omitted);

        let mut bad_ratio = valid.clone();
        bad_ratio.geometric_ratio += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_ratio);

        let mut bad_margin = valid.clone();
        bad_margin.geometric_denominator_margin += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_margin);

        let mut bad_tail = valid.clone();
        bad_tail.tail_upper += rational(1, 1_000_000);
        assert_witness_mismatch(&bad_tail);

        let mut bad_reduced_enclosure = valid.clone();
        bad_reduced_enclosure.reduced_enclosure =
            RationalInterval::try_point(rational(1, 1)).unwrap();
        assert_witness_mismatch(&bad_reduced_enclosure);

        let mut bad_step_value = valid.clone();
        bad_step_value.squaring_enclosures[0] =
            RationalInterval::try_point(rational(1, 1)).unwrap();
        assert_witness_mismatch(&bad_step_value);

        let mut bad_step = valid.clone();
        bad_step.squaring_enclosures.pop();
        assert_witness_mismatch(&bad_step);

        let mut bad_final = valid.clone();
        bad_final.enclosure = RationalInterval::try_point(rational(1, 1)).unwrap();
        assert_witness_mismatch(&bad_final);
    }

    #[test]
    fn witness_preflight_rejects_raw_malformed_fields_without_panicking() {
        let valid = proof(rational(3, 2), 5, rational(1, 100_000));
        let malformed = || BigRational::new_raw(BigInt::one(), BigInt::from(0_u8));

        let mut bad_input = valid.clone();
        bad_input.exponent = malformed();
        let outcome = std::panic::catch_unwind(|| bad_input.verify(DEFAULT_EXP_ENCLOSURE_LIMITS));
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::MalformedRationalDenominator)
        );

        let mut bad_bound = valid.clone();
        bad_bound.max_reduced_tail_upper = malformed();
        let outcome = std::panic::catch_unwind(|| bad_bound.verify(DEFAULT_EXP_ENCLOSURE_LIMITS));
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::MalformedRationalDenominator)
        );

        let mut bad_scalar = valid.clone();
        bad_scalar.partial_sum = malformed();
        let outcome = std::panic::catch_unwind(|| bad_scalar.verify(DEFAULT_EXP_ENCLOSURE_LIMITS));
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::MalformedRationalDenominator)
        );

        let mut bad_unreduced = valid.clone();
        bad_unreduced.geometric_ratio =
            BigRational::new_raw(BigInt::from(2_u8), BigInt::from(4_u8));
        let outcome =
            std::panic::catch_unwind(|| bad_unreduced.verify(DEFAULT_EXP_ENCLOSURE_LIMITS));
        assert_eq!(
            outcome.unwrap(),
            Err(ExpEnclosureError::NoncanonicalRational)
        );
    }

    #[test]
    fn verify_preserves_resource_errors_from_preflight_and_recomputation() {
        let valid = proof(rational(3, 2), 5, rational(1, 100_000));

        let too_tight_limits = ExpEnclosureLimits {
            max_input_component_bits: 4,
            max_intermediate_component_bits: 8,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            valid.verify(too_tight_limits),
            Err(ExpEnclosureError::InputComponentBitLimitExceeded)
        );

        let tiny_work = ExpEnclosureLimits {
            max_work_units: 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            valid.verify(tiny_work),
            Err(ExpEnclosureError::WorkBudgetExceeded)
        );

        let tiny_storage = ExpEnclosureLimits {
            max_witness_storage_bits: 1,
            ..DEFAULT_EXP_ENCLOSURE_LIMITS
        };
        assert_eq!(
            valid.verify(tiny_storage),
            Err(ExpEnclosureError::WitnessStorageBudgetExceeded)
        );

        let mut oversized_cached = valid.clone();
        oversized_cached.partial_sum = BigRational::from_integer(BigInt::one() << 40_000_usize);
        assert_eq!(
            oversized_cached.verify(DEFAULT_EXP_ENCLOSURE_LIMITS),
            Err(ExpEnclosureError::IntermediateComponentBitLimitExceeded)
        );

        let mut excessive_reductions = valid.clone();
        excessive_reductions.range_reductions = DEFAULT_EXP_ENCLOSURE_LIMITS
            .max_range_reductions
            .checked_add(1)
            .unwrap();
        assert_eq!(
            excessive_reductions.verify(DEFAULT_EXP_ENCLOSURE_LIMITS),
            Err(ExpEnclosureError::RangeReductionLimitExceeded)
        );

        let mut excessive_cutoff = valid.clone();
        excessive_cutoff.taylor_cutoff = DEFAULT_EXP_ENCLOSURE_LIMITS
            .max_taylor_cutoff
            .checked_add(1)
            .unwrap();
        assert_eq!(
            excessive_cutoff.verify(DEFAULT_EXP_ENCLOSURE_LIMITS),
            Err(ExpEnclosureError::TaylorCutoffLimitExceeded)
        );
    }
}
