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

The principal interval type has arbitrary-size rational endpoints. Algebraic
operations are exact; widening occurs only for genuinely irrational
operations or an explicitly selected precision boundary. This is a different
arithmetic implementation from the Python binary64 interval backend and
provides a useful higher-precision consistency check.

For a nonnegative rational `x`, square root is enclosed by rational dyadic
bisection at a declared precision, with the endpoint-square inequalities
checked exactly. Reciprocal square-root powers are derived from that enclosure
using exact interval multiplication and reciprocal operations.

For a nonnegative rational exponent, `exp(x)` is bounded using exact rational
range reduction and a Taylor partial sum with an explicit geometric tail
majorant. Range-reduction squaring is then exact. The release result records
the precision and Taylor cutoff; increasing precision must not invalidate a
previously accepted stable certificate.

### Automatic differentiation

The verifier uses a locally implemented first-order interval-dual type. Each
dual value consists of one rational interval and a fixed-size interval
gradient. Vector fields are expressed once over the dual operations so the
Jacobian bound is derived independently from the Python implementation.

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
