# Rust mass identity and flow audit

## Status and scope

This note audits the mass data flow used by ordinary and planar
Levi--Civita (LC) source paths. It covers semantic admission, ordinary field,
defect and tube replay, LC field, interval series, lift, projection, tube,
entry and exit replay, and the mixed-chain handoffs that compose them.

Subject to the trust boundary in Section 6, the audited paths use one ordered
triple of positive serialized binary64 masses for each connected ordinary/LC
problem. Every LC ratio or mass combination is derived from that triple and
the selected pair by the validated mass-profile kernel. This is a source-level
audit, not a formal proof of Rust or its compiler.

## 1. Three distinct representations

The code deliberately distinguishes:

1. **binary64 identity:** the exact 64-bit encoding retained from an admitted
   JSON real;
2. **theorem-facing mass:** the exact dyadic rational value of those bits; and
3. **derived LC coefficient witness:** an exact rational coefficient together
   with a tight finite-binary64 outward enclosure.

`CheckedJsonBinary64` first parses the decimal exactly, verifies the proposed
finite binary64 nearest-even result, and stores both its bits and exact dyadic
value. `ExactBinary64::from_bits` independently decodes finite bits to a
`BigRational`.

The semantic mass checks require a strictly positive rational numerator.
Thus neither signed zero is an admitted mass. On positive finite binary64
values the map from bits to exact dyadic rational is injective: distinct
normal/subnormal encodings denote distinct positive real values. Consequently,
equality of admitted mass rationals is equivalent to equality of their
binary64 bits, even on ordinary paths whose semantic type retains only the
rational array. This is bit/value identity, not textual identity of the
original decimal lexemes; different lexemes may round to the same admitted
binary64 value.

## 2. Semantic admission and immutable chart identity

`ordinary_chart_input_from_wire` requires exactly three masses, copies each
`wire.masses[index].binary64_rational()` into the private ordered array
`OrdinaryChartInput::masses`, and rejects every nonpositive component. All
ordinary consumers receive this immutable array through `chart.masses()`.

`planar_lc_chart_input_from_wire` performs the same count and positivity
checks. From each same wire element it stores both:

\[
\texttt{masses}[b]=\operatorname{value}(\texttt{bits}[b])
\]

and `mass_binary64_bits[b]`. Both fields are private and have no mutation path.
It also admits only a canonical selected pair \([i,j]\) with
\(0\le i<j<3\).

LC-entry semantic construction compares every source ordinary mass rational
with the corresponding target LC mass rational. Ordinary bridges compare the
two ordinary chart arrays. Carried LC exit compares the source ordinary,
intermediate LC, and target ordinary arrays. By positivity and injectivity,
these value comparisons also establish componentwise binary64-bit identity.
No permutation is allowed: array index is body identity.

## 3. Ordinary mass flow

The ordinary proof path has no derived mass ratios.

- `evaluate_planar_three_body_ordinary_polynomial_defect` receives
  `chart.masses()` directly from ordinary tube replay. It preflights that same
  reference, then passes it unchanged to
  `evaluate_planar_three_body_ordinary_field`.
- The field revalidates all three canonical rationals as positive and scales
  each pair force by the corresponding exact rational `masses[other_body]`.
- `replay_ordinary_tube_exact_rational_v04` passes the same chart array both to
  the direct defect computation and to `analytic_lipschitz_upper`. Thus defect,
  field, collision-free geometry, and Lipschitz arithmetic refer to one mass
  problem.
- Ordinary bridge and mixed-chain progression accept a handoff only after the
  source and target chart arrays compare equal. They then freshly replay each
  target tube from its own equal chart array.

There is no conversion through host `f64` on these theorem-facing arithmetic
paths after semantic admission.

## 4. LC profile derivation and validation

Every audited LC arithmetic consumer starts from the chart's retained
`mass_binary64_bits`, reconstructs `[ExactBinary64;3]`, calls
`derive_planar_lc_mass_profile(masses,pair)`, and calls
`profile.validate_against(masses,pair)` before using the result.

For \([i,j]\), validation requires all three reconstructed masses positive,
records the exact ordered mass bits and pair, and computes the unique
complement \(k\). For the three canonical pairs:

| pair | third body |
|---|---:|
| `[0,1]` | 2 |
| `[0,2]` | 1 |
| `[1,2]` | 0 |

It then recomputes

\[
M=m_i+m_j,\quad \alpha={m_j\over M},\quad
\beta={m_i\over M},\quad \gamma={m_k\over M},
\]

\[
c_i={m_im_k\over M},\quad c_j={m_jm_k\over M},\quad
o_i=m_i+c_i,\quad o_j=m_j+c_j.
\]

For each coefficient, `MassCoefficientWitness` contains the exact rational and
the tight finite-binary64 outward enclosure. `validate_against` checks:

- the kernel identifier;
- componentwise original mass bits;
- ordered pair and complement index;
- equality of every stored exact coefficient with the independently
  recomputed formula; and
- the outward enclosure postconditions for that exact value.

Those postconditions include containment of the exact coefficient and the
appropriate exact-point or adjacent-endpoint representation.

An important distinction is that the current LC verifier is exact-rational:
field, series, lift and projection use `witness.exact()` as their scalar. They
do **not** substitute either binary64 outward endpoint into interval formulas.
The outward interval is validated witness metadata attached to the same
profile. This is stronger for arithmetic inclusion than using a rounded point
coefficient, but a claim that the outward endpoints themselves are propagated
through the LC equations would be false.

## 5. LC consumer-by-consumer flow

### Field, defect, tube, and series

`evaluate_planar_lc_field` reconstructs and validates the profile from its
single `PlanarLcChartInput`. It uses:

- exact \(\alpha,\beta\) in the two third-body displacement blocks;
- exact reconstructed `masses[k]` in \(m_k(f_j-f_i)\); and
- exact \(c_i,c_j,o_i,o_j\) in the center and third-offset accelerations.

`evaluate_planar_lc_polynomial_defect` constructs its state polynomial and
calls that field with the same chart, so coefficients, residual, masses, pair,
and field cannot come from different chart objects.

`replay_planar_lc_tube_exact_rational_v04` independently reconstructs and
validates the profile, evaluates the direct defect with the same chart, and
evaluates the interval-Jacobian field with the same chart. Its optional
pair-energy constraint uses that profile's exact \(M\).

`replay_planar_lc_interval_series` also reconstructs and validates once, then
passes the same masses and profile together into `field_series`. The series
uses exact \(\alpha,\beta,c_i,c_j,o_i,o_j\), exact `masses[k]`, and exact
\(M\) for the constraint series. This series is used by the separately labeled
claimed-tail chart diagnostic; its mass identity is sound even though the
claimed-tail result is not decisive in the proof-grade profile.

### Lift and projection

`replay_planar_lc_lift_cover_exact_rational_v04` reconstructs and validates the
target LC chart profile.
It uses the profile complement \(k\), exact \(\alpha,\beta\) for the binary
center, and exact \(M\) for pair energy. At entry, source ordinary and target
LC arrays must first compare equal, and `revalidate_mass_kernel` separately
reconstructs and validates the target profile. The direct lift evidence then
uses that target chart.

Both `project_planar_lc_full_state_exact_rational` and the residual projection
reconstruct and validate from the supplied LC chart on each call. They use the
same profile's \(i,j,k\), exact \(\alpha,\beta\), and exact reconstructed masses
in the Newton comparison. They therefore cannot reuse a stale profile from a
different mass triple or pair.

At carried exit, projection is called with the intermediate LC chart, while
the target tube is freshly replayed from the target ordinary chart. The
`common_problem` obligation requires source ordinary = intermediate LC =
target ordinary before target containment can succeed. The exit's inherited
mass-kernel identifier is an additional gate; projection still rederives and
revalidates its own profile rather than trusting that identifier alone.

### Mixed-chain composition

The mixed replay does not introduce a new mass representation or perform force
arithmetic. It delegates ordinary bridges and LC entry/exit to the audited
replays and advances the current chart only after their local obligations
succeed. Therefore a successful connected fold preserves the same ordered
positive mass triple across every segment.

## 6. Trust boundary and precise residual statement

The source audit assumes:

- Rust privacy and ownership execute as specified, and the reviewed immutable
  chart fields are not changed by compiler/runtime faults;
- `CheckedJsonBinary64`, `ExactBinary64`, `BigInt`, and `BigRational` correctly
  implement finite-binary64 decoding and exact integer/rational equality and
  arithmetic;
- the compiled verifier source is identical to the reviewed source; and
- every decoding, validation, resource, or arithmetic `Err` prevents
  certification.

No proof assistant, verified compiler, independently verified big-integer
kernel, or source-to-binary attestation proves those assumptions here.

Subject to that boundary, the mass-identity and exact-derived-profile gate is
closed: every decisive ordinary and LC consumer uses the same positive
serialized triple, and every LC ratio/composite coefficient comes from a
freshly validated profile for that triple and selected pair. The only wording
qualification is that the profile's outward binary64 enclosures are validated
but not numerically propagated; exact rational members are used instead. This
is not a soundness gap for the exact-rational verifier, but it must remain
explicit in any claim about "outward mass arithmetic."
