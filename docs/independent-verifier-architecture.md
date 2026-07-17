# Independent verifier architecture

## Decision

The second verifier will be a standalone Rust program under
`verifiers/rust-v1/`. It will consume the raw-v1 JSON certificate directly and
will not import, invoke, translate, or link any Python checker module. The
v0.3.0-review tag remains the immutable reference implementation and evidence
baseline.

The second implementation is intended to test the mathematical conclusion,
not reproduce Python implementation objects. It will emit its own obligation
ledger and a small versioned semantic result containing:

- parser outcome;
- raw byte SHA-256;
- `CERTIFIED_TO_T` or `UNRESOLVED`;
- accepted segment-prefix length and first failed normative obligation;
- current clock enclosure and typed retained frontier; and
- final fixed-time enclosure when certified.

The project may use the phrase **independently replayed** only after the Rust
verifier covers the complete raw-v1 grammar and agrees with the Python checker
on the release corpus and mutation campaign.

## Independence boundary

The Rust verifier must independently implement all of the following:

1. strict JSON parsing, duplicate-key rejection, field and tag checking, and
   canonical raw-byte validation;
2. decimal JSON number to finite IEEE-754 binary64 conversion and exact
   binary64-bit-pattern to dyadic-rational decoding;
3. exact rational interval addition, subtraction, multiplication, division,
   containment, intersection, and Horner evaluation;
4. polynomial differentiation and interval automatic differentiation;
5. outward square-root and negative half-integer power enclosures;
6. an explicit rational upper enclosure for the exponential used by the
   Gronwall bound;
7. ordinary Newtonian and fourteen-dimensional planar Levi--Civita vector
   fields and Jacobian bounds;
8. mass-coefficient derivation from the exact serialized mass dyadics;
9. ordinary tubes, LC tubes, complete entry lifts, gauge cover, complete exit
   projection, and endpoint containment;
10. the physical-clock ledger, pair-local gauge rules, fixed-time preimage,
    final inflation, status, prefix, and frontier semantics; and
11. its own deterministic obligation ledger and result serializer.

It may use general-purpose Rust crates for JSON tokenization, big integers,
hashing, and command-line handling. It must not use Python through FFI or a
subprocess, consume Python replay transcripts as proof inputs, or port cached
Python result objects into Rust.

## Numeric design

### Exact input layer

Every accepted real-valued wire field denotes the exact finite binary64 value
obtained by correctly rounded parsing of its JSON number. The verifier parses
the decimal lexeme into an exact rational and validates the candidate
binary64 against its adjacent exact dyadics, including nearest-even tie
handling. It then decodes the binary64 sign, exponent, and significand
directly into an exact rational. This prevents the host float parser from
silently defining the theorem input. Signed zero is preserved at the
lexical/bit layer where canonicalization needs it and is equal to zero in
theorem-facing rational arithmetic.

The exact layer also handles schema integers, dimensions, identifiers, pair
labels, clock translations, gauge parity, mass coefficients, target-time
preimages, polynomial coefficients, and endpoint evaluations.

### Algebraic interval layer

The principal interval type admits canonical rational endpoints under fixed
component and operation-size ceilings. Algebraic operations are exact within
those admitted bounds; widening occurs only for genuinely irrational
operations or an explicitly selected precision boundary. This is a different
arithmetic implementation from the Python binary64 interval backend, but the
numeric separation alone is not certificate replay.

For a nonnegative rational `x`, square root is enclosed on a rational dyadic
grid at a hard-capped declared precision using integer square root, with the
endpoint-square inequalities checked exactly. Reciprocal square-root powers
are derived from that enclosure using exact interval multiplication and
reciprocal operations.

For a nonnegative rational exponent, `exp(x)` is bounded using exact rational
range reduction and a Taylor partial sum with an explicit geometric tail
majorant. Range-reduction squaring is then exact. The release result records
the precision and Taylor cutoff; increasing precision must not invalidate a
previously accepted stable certificate.

### Automatic differentiation

The numeric foundation now includes a locally implemented, bounded first-order
rational interval-dual type. Each dual value consists of one checked rational
interval and a checked interval gradient with at most 64 coordinates. The
primitive implements checked value/gradient algebra and reciprocal and
square-root chain rules. It now independently evaluates the planar ordinary
12-state Newton RHS and a 12-by-12 rational-interval Jacobian, including an
exact infinity-row-sum Lipschitz upper bound, fixed-pair separation, exact
positive-mass preflight, and an unequal-mass conservation regression. This
ordinary field/Jacobian slice now supplies the field enclosure used by the
direct polynomial-defect kernel below, but neither API has raw-schema or
certificate semantics.

### Exact polynomials and direct ordinary defect

The arithmetic foundation includes a bounded nonempty rectangular
exact-rational vector polynomial in ascending degree-major order. It evaluates
each component by rational interval Horner arithmetic and forms derivative
coefficients by exact integer multiplication before either derivative
construction or direct derivative evaluation. Dimension, degree, rational
component size, and aggregate work are hard-capped.

For six-component position and velocity polynomials, the direct ordinary
defect kernel evaluates the 12 residual intervals in `q' - v`, `v' - a(q)`
order over one rational parameter interval. It returns the exact rational
maximum of the absolute values of all 24 interval endpoints. This is a
numeric enclosure of the nominal finite polynomial's direct ODE defect. It is
not the primitive chart coefficient-recurrence or Taylor-model residual/tail
ledger, and it does not decode an `OrdinaryChartWire`, compare a certificate
cap, or accept a tube. Regressions cover unequal masses, an acceleration-block
maximum, and mass/precision preflight before polynomial evaluation.

## Implementation status at the current checkpoint

The standalone Rust crate now implements a strict, resource-bounded JSON byte
layer, duplicate-key and canonical-byte rejection, exact decimal-to-binary64
nearest-even validation, exact binary64-to-dyadic decoding, bounded canonical
rational admission, checked rational intervals and Horner evaluation, a
bounded exact-rational degree-major vector-polynomial kernel with exact formal
differentiation and interval Horner evaluation, a direct 12-component ordinary
polynomial-defect enclosure with an exact maximum-absolute-endpoint bound, a
bounded first-order rational interval-dual primitive with a 64-coordinate cap,
checked value/gradient algebra, and reciprocal and square-root chain rules,
an independently evaluated planar 12-state Newton RHS, a 12-by-12
rational-interval Jacobian, an exact infinity-row-sum Lipschitz upper bound,
fixed-pair separation, exact positive-mass preflight, and an unequal-mass
conservation regression, plus exact-postcondition dyadic square-root
enclosures. It also implements an exact rational exponential enclosure with
range reduction, a Taylor partial sum and geometric tail, exact squaring, full
witness replay, and hard component/work/storage budgets. Its exact outward
binary64 mass kernel derives all eight raw-v1 mass formulas and validates
adjacent nearest-even endpoints;
hand-derived boundary cases and deterministic lattice stress tests exercise
that selection independently of host floating-point rounding. The crate also
has an explicit `V03Compatible` typed decoder
for the complete finite v0.3 raw-v1 record grammar. The record-by-record parser
boundary is documented in the
[`Rust schema map`](raw-v1-rust-schema-map.md).

The current validation checkpoint is 121 passing Rust unit tests and two
passing corpus tests, with `cargo fmt --check` and warning-denying Clippy clean.

The implementation-neutral
[`raw-v1 seed corpus`](../conformance/raw-v1/README.md) contains two accepted
v0.3 baseline payloads and sixteen single-mutation rejection payloads, with
neutral expectations and file hashes. The public Python admission API and an
explicit Rust integration table both enforce the same 2-accept/16-reject
classification. Its manifest deliberately says `seed_incomplete`: it does not
yet cover semantic failures, numeric boundaries, or the
root/fold/LC-entry/LC-tube/LC-exit/fixed-time result boundaries required for
release.

Delivery slices 1 and 2 are therefore only partial. `OPEN-V1-01` remains open
because the tested Rust float rendering path is not yet a portable normative
shortest-decimal algorithm and the Rust toolchain is not pinned. Raw SHA-256,
semantic ordinary-chart decoding, primitive chart recurrence and Taylor-model
residual/tail replay, complete tube collision and analytic Lipschitz and Grönwall
acceptance, root/bridge/fixed-time semantics, every LC field, all ordinary/LC
chart, tube, and transition checks, chain-fold semantics, clock and gauge logic,
the obligation ledger, and the semantic result serializer remain unimplemented.
The polynomial and direct-defect APIs have no certificate/schema coupling. No
certificate has been replayed by this crate. The corpus remains
`seed_incomplete`, no slice beyond the byte/schema and partial numeric and
ordinary-field foundations is complete, and neither the conformance gate nor
the independent-verifier gate passes at this checkpoint.

## Delivery slices

1. **Normative wire and arithmetic specification.** Freeze exact field sets,
   dimensions, numeric semantics, canonicalization, record grammar, and every
   acceptance obligation before relying on cross-implementation agreement.
2. **Parser and exact kernel.** Validate strict JSON, binary64 dyadics, hashes,
   rational intervals, polynomial evaluation, square root, reciprocal powers,
   exponential, and mass coefficients against hand-derived vectors.
3. **Ordinary-only replay.** Reproduce the exact root, ordinary tube, complete
   ordinary handoff, clock translation, fixed-time enclosure, and width gate.
4. **LC replay.** Add the LC tube, constrained entry lift cover, gauge graph,
   complete punctured projection, and exit clock enclosure for all three
   canonical pairs.
5. **Full finite-chain replay.** Fold arbitrary finite raw-v1 segment words,
   including pair revisits, and emit sound terminal or typed unresolved
   results.
6. **Differential audit.** Run golden cases, boundary cases, targeted invalid
   cases, and deterministic mutations through both implementations. Any
   unexplained disagreement blocks release.
7. **Reproducibility and publication.** Pin both environments, replay in two
   clean environments, publish all commands and hashes, and release the
   specification, corpus, verifier, and review addendum as v0.4.0-review.

## Agreement rules

Agreement does not require bit-identical intermediate bounds: the two
arithmetic backends are intentionally different. For each corpus member the
release harness compares:

- parser acceptance versus parser rejection;
- terminal status;
- accepted prefix length;
- failed normative obligation class;
- retained frontier representation;
- target physical time and chart/pair identity; and
- mutual consistency of final enclosures when both return
  `CERTIFIED_TO_T`.

If one sound enclosure is wider, certification is acceptable only when it
still satisfies the certificate's declared width gate. A case accepted by one
verifier and unresolved or parser-rejected by the other is a disagreement,
not a majority vote, and blocks the v0.4.0-review release until explained and
resolved.

## Explicit non-goals for v0.4

This milestone does not build an adaptive certificate producer, prove
producer termination, cover initial-condition families, certify collision
occurrence, add spatial or total-collision charts, establish all-time
continuation, or solve the general three-body problem. Its contribution is a
material reduction in correlated implementation risk for the already stated
finite planar supplied-chain theorem.
