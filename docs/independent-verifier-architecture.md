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
ledger. By itself it does not decode an `OrdinaryChartWire`, compare a
certificate cap, or accept a tube; the conditional semantic layer below is
the only current theorem-facing consumer. Regressions cover unequal masses,
an acceleration-block maximum, and mass/precision preflight before polynomial
evaluation.

### Ordinary semantic inputs, conditional tube, and local root replay

Before semantic conversion, `CanonicalRawV1Admission` provides a fixed-limit
opaque boundary that owns both the exact canonical input bytes and their
strict `V03Compatible` typed decode. Its public constructor fixes
`DEFAULT_WIRE_JSON_LIMITS`; immutable accessors cannot replace either side of
the binding, and errors preserve the canonical-wire versus typed-schema
stage. This is not SHA-256, namespace validation, or replay of outer
obligations 1--5.

The `V03Compatible` wire decoder now feeds a non-certifying semantic adapter
for finite ordinary chart and tube records. Every real is taken from its
proved binary64 bits as an exact dyadic, coefficient tensors are checked and
flattened in body-major `q`/`v` order into bounded exact-rational polynomials,
and both chart intervals must be strictly increasing. This adapter deliberately
does not replay the primitive coefficient recurrence or Taylor-model
residual/tail obligations.

The adapter and direct-defect kernel feed a conditional replay of the six
ordered Section 4.4 tube obligations under the named arithmetic profile
`exact_rational_ordinary_tube_v04`. The profile preserves the exact analytic
v0.3 formula family for the nominal pair floor,
`d_tube = d_nominal - 2 sqrt(2) r`, the three body Lipschitz bounds, and
`L = max(1,L_0,L_1,L_2)`. It fixes square-root precision at 256 bits, rounds
the exact nonnegative exponential argument upward to the `2^-32` dyadic grid,
uses Taylor cutoff 32, and requires the reduced exponential tail to be at most
`2^-128`. Exact dyadic cap comparisons and the strict Gronwall inequality are
then evaluated without host floating-point arithmetic.

All six ordinary tubes embedded in each of the two baseline chains, twelve
replays total, satisfy all six obligations under this profile. This is only a
conditional a-posteriori tube result: by itself it does not prove that either
baseline's IVP lies in the initial ball, and it does not establish status
parity with the historical binary64 checker. That parity question remains
`OPEN-V1-08`.

The same planar semantic boundary now feeds
`exact_rational_initial_value_binding_v04`. This profile reproduces the eight
direct Python binding obligations with exact binary64 dyadics and exact
rational polynomial and affine-time evaluation. It deliberately does not make
`source` part of direct binding identity. Its scope is admitted planar inputs,
not parity with the Python direct-object API on three-dimensional inputs. Both
canonical roots satisfy all eight obligations.

`exact_rational_validated_ordinary_root_v04` then composes that exact binding
with the conditional ordinary-tube replay. Its eight proof-oriented
obligations strengthen the historical six-obligation wrapper with exact chart
unit speed and an exact physical-time anchor. It does not consume the
claimed-tail ordinary-chart ledger. The implementation computes the actual
initial error whenever the binding position and velocity gaps are available;
it retains the exact root clock origin only when both exact unit speed and the
exact physical-time anchor hold. Both canonical roots satisfy all eight
obligations.

The next local component is
`exact_rational_carried_ordinary_bridge_v04`. It replays the nine historical
ordinary-bridge IDs, including fresh exact-rational replay of both conditional
tubes. At the exact source right endpoint it evaluates all positions followed
by all velocities, inflates that 12-vector by the source tube replay's rational
Gronwall upper bound, and requires inclusive containment in the target initial
ball at the exact target left anchor. It derives the clock interval by the
exact cocycle `B'=B+e-a` and exposes the pinned analytic-kernel ID
`ordinary_autonomous_uniqueness_bridge_kernel_v1`. The first bridge in each
canonical chain passes all nine obligations and advances `[0,0]` to
`[2^-40,2^-40]`. Neither chart's claimed-tail ledger is consumed. Physical-time
metadata is used only for semantic/schema admission and never updates or
overrides the bridge clock.

These are component profiles, not raw-chain admission. Raw-chain replay
must additionally require a nonempty binding `source`, exact-zero time,
position, and velocity tolerances, and equality of the binding parameter,
chart left endpoint, and tube anchor. The bridge profile does not fold or
commit a raw-chain segment. It is a separate exact-rational arithmetic profile,
not frozen-v0.3 binary64 status parity, so `OPEN-V1-08` remains open. The
profiles do not implement fixed-time evaluation, any LC replay obligation,
full chain folding, or end-to-end independent certificate replay.

### Planar LC semantic admission

The exact-dyadic polynomial foundation now admits non-certifying planar LC
chart and tube inputs. It checks six planar vector series and two scalar
series against one bounded common coefficient count, positive masses,
ascending canonical pairs, strictly increasing parameter and physical-time
intervals, positive sample count, and strict tube identity/sign/anchor rules.
The five finite chart tolerance/bound claims retain their serialized signs;
this schema-level adapter does not silently add primitive-ledger sign
hypotheses. Public `PlanarLcEntryInput` construction requires an opaque
`CanonicalRawV1Admission` and segment index, then checks transition constants,
carried identifiers, mass equality, and exact ordinary-right/LC-left/anchor
bindings. Bare chart and tube conversion remains explicitly non-provenance and
non-certifying.

The next arithmetic layer concatenates those eight admitted polynomial
families degree-by-degree in the fixed lifted order
`(zx,zy,wx,wy,h,Rx,Ry,Ux,Uy,yx,yy,Vx,Vy,t)`. It provides exact interval and
derivative evaluation, a checked 13-plus-time anchor point, uniform
nonnegative inflation, dependency-aware interval squares for `rho`, the LC
square and pair-energy constraint, and the LC deck involution. Original
positive binary64 mass encodings are retained beside exact rational masses as
input identity for future mass replay, not as a mass-formula witness.

This layer proves no LC chart, tube, or entry obligation, supplies no outward
mass witness, and does not evaluate the LC field, construct a gauge or lift,
replay an exit, or commit a chain segment. It establishes no parity,
cross-profile agreement, implementation independence, or release gate.

The lifted-state operations are arithmetic only: evaluating a serialized
constraint expression does not prove constraint satisfaction or any LC
differential, tube, entry, gauge, projection, exit, or chain theorem.

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

The semantic adapter, conditional ordinary-tube replay, exact planar binding,
proof-oriented local-root composition, and local carried ordinary bridge
described above now couple decoded ordinary records to the exact polynomial,
square-root, exponential, defect, collision, analytic Lipschitz, cap, strict
Gronwall, binding-gap, endpoint-containment, and local clock-cocycle kernels.
The current validation checkpoint is 174 passing Rust unit tests and 15
passing integration cases across five integration test targets, with
`cargo fmt --check` and
warning-denying Clippy clean.

The newest bounded slice is
`exact_rational_raw_ordinary_only_chain_v04`. It accepts an already decoded
raw wire value and covers top-level obligations 6--13 only. It enforces the
strict raw root equalities, uses the claimed-tail chart ledger solely as a
fail-closed compatibility gate (the allowance is unproved and does not close
`OPEN-V1-06`). Root support comes from exact binding and the
direct-defect/a-posteriori tube argument; subsequent bridge handoff uses the
named autonomy/local-uniqueness kernel. Ordinary bridge state is
transactional: chart, tube, clock, count, and ledger advance only when the
entire nine-obligation bridge passes. A wholly ordinary word receives exact
clock/preimage arithmetic, absolute-parameter interval Horner evaluation,
fresh tube-radius inflation, and the inclusive width test. Width-only failure
retains the fixed-time enclosure; otherwise a freshly justified ordinary
right frontier is retained when possible. Malformed later ordinary records
are structured failures, segment count is capped at 256 before replay, and LC
is an explicit unsupported stop. The canonical inputs therefore demonstrate
root plus one committed bridge and a retained frontier, not full-chain replay.

The implementation-neutral
[`raw-v1 seed corpus`](../conformance/raw-v1/README.md) contains two accepted
v0.3 baseline payloads and sixteen single-mutation rejection payloads, with
neutral expectations and file hashes. The public Python admission API and an
explicit Rust integration table both enforce the same 2-accept/16-reject
classification. Its manifest deliberately says `seed_incomplete`: it does not
yet cover semantic failures, numeric boundaries, or the
root/fold/LC-entry/LC-tube/LC-exit/fixed-time result boundaries required for
release.

Delivery slices 1 and 2 remain partial. Slice 3 now includes the conditional
ordinary tube, exact admitted-planar binding, proof-oriented local root, local
ordinary bridge, strict root admission, transactional ordinary-prefix fold,
and fixed-time enclosure described above. It is still bounded to ordinary
segments and begins after outer obligations 1--5.
`OPEN-V1-01` remains open because the tested Rust
float rendering path is not yet a portable normative shortest-decimal
algorithm and the Rust toolchain is not pinned. `OPEN-V1-08` remains open
because the new exact-rational tube profile does not claim status parity with
the historical binary64 outward profile. Raw SHA-256, frozen-v0.3 primitive
ordinary-chart parity or a proof-grade verified-tail replay, every LC field,
full chain replay, and the semantic result serializer remain unimplemented.
The local profiles provide neither direct
three-dimensional API parity nor chain-level independence. No complete
certificate has been replayed by this crate. The corpus remains
`seed_incomplete`, and neither the conformance gate nor the
independent-verifier gate passes at this checkpoint.

This slice establishes neither outer obligations 1--5 nor
namespace/canonicalization/hash results, frozen parity, cross-profile
agreement, implementation independence, or any v0.4 release gate.

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
