# Raw-v1 Rust schema map

## Purpose and authority

This note is the implementation map for decoding the raw-v1 planar
continuation-chain certificate into the independent Rust verifier. It covers
only the outer and nested JSON records. It does not specify interval
arithmetic, the obligation ledger, or replay formulas.

The authoritative implementation anchors are:

- `three_body_symmetry/proof_carrying_planar_chain.py:114-347` for the outer
  record, segment union, exact field sets, tag dispatch, and canonical
  round-trip;
- `three_body_symmetry/certificate_language.py:40-205,317-347,563-678` for
  binding, chart, tube, and LC-exit records;
- `three_body_symmetry/proof_carrying_ordinary_bridge.py:64-145` and
  `three_body_symmetry/proof_carrying_carried_planar_lc_entry.py:112-206` for
  the two strict private transition records;
- `three_body_symmetry/certificate_language.py:2209-2248` for the coercing
  pair, interval, coefficient, and state-array helpers;
- `three_body_symmetry/proof_carrying_planar_chain.py:1311-1412` and
  `three_body_symmetry/proof_carrying_carried_planar_lc_entry.py:1089-1218`
  for semantic shape and value checks; and
- `docs/raw-v1-certificate-specification.md` Sections 1.1-1.3 and 2.1-2.10
  for the normative reconstruction. This map addresses the record-inventory
  part of OPEN-V1-12; portable corpus parity remains a separate release gate.

The archived v0.3 parser boundary and a possible stricter shape boundary
differ for some wrong-length arrays. They must be named as separate
conformance profiles:

- **`V03Compatible`** reproduces the Python parse/canonical-round-trip
  boundary. Some structurally wrong arrays parse and later yield
  `UNRESOLVED`.
- **`V04StrictExperimental`** rejects every violation of the declared raw-v1
  array shapes during typed schema decoding. It does not enlarge the
  certifiable language, but it intentionally changes malformed-input
  classification and therefore is not raw-v1 parser parity.

`V03Compatible` is the required explicit profile for the v0.4 raw-v1 release because
parser rejection versus `UNRESOLVED` is an explicit cross-verifier agreement
dimension. `V04StrictExperimental` may be an opt-in diagnostic or a raw-v2
candidate. It may become a release profile only if the Python verifier and
the frozen corpus are deliberately migrated at the same time; it must never
be used to explain away an otherwise unexplained verifier disagreement.

## Scalar and container types

Use these conceptual Rust aliases. The names describe the intended API; they
need not be public type aliases with exactly these spellings.

```rust
type S = String;
type F = CheckedJsonBinary64;

// Keep the arbitrary-precision integer value and original/canonical lexeme.
// Do not narrow to i64/usize until the field-specific semantic bound is proved.
struct I {
    parsed: ParsedJsonNumber,
    value: BigInt,
}

type F2 = [F; 2];
type I2 = [I; 2];

// Legacy v0.3 wire containers; validated shapes are described below.
type FSeries = Vec<F>;
type FVectorSeries = Vec<Vec<F>>;
type FMatrix = Vec<Vec<F>>;
type FTensor3 = Vec<Vec<Vec<F>>>;
```

`S`, `F`, `I`, and `B` mean exact JSON string, real-number token, integer
token, and Boolean. `F` is finite and carries both its verified binary64 bits
and theorem-facing exact dyadic rational. A token such as `1` is `I`, not `F`;
`1.0` can be `F`. Python `int` is unbounded, so narrowing an `I` is not a
parser operation.

The byte/AST layer rejects malformed UTF-8/JSON, duplicate keys, nonfinite or
overflowing reals, noncanonical CPython-v1 bytes, and resource-limit failures
according to the separately selected resource policy. The typed decoder then
rejects missing/unknown fields, `null`, wrong scalar/container classes, and an
unknown segment tag. It should extract fields manually from `WireJsonValue`;
ordinary Serde derivation must not silently accept unknown fields or erase the
integer/real distinction.

## Complete proposed Rust wire model

Every object below has **exactly** the listed fields. JSON object key order is
handled by canonical-byte validation, not by struct declaration order.

```rust
struct RawPlanarChainWire {
    certificate_id: S,
    root_binding: RootBindingWire,
    initial_chart: OrdinaryChartWire,
    initial_tube: OrdinaryTubeWire,
    segments: Vec<SegmentWire>,
    requested_target_time: F,
    requested_maximum_component_width: F,
    schema_version: I,
    certificate_type: S,
    source: S,
}

enum SegmentWire {
    OrdinaryBridge(OrdinaryBridgeSegmentWire),
    PlanarLcPassage(PlanarLcPassageSegmentWire),
}

struct OrdinaryBridgeSegmentWire {
    transition: OrdinaryBridgeTransitionWire,
    target_chart: OrdinaryChartWire,
    target_tube: OrdinaryTubeWire,
    segment_type: S,
}

struct PlanarLcPassageSegmentWire {
    entry_transition: LcEntryTransitionWire,
    lc_chart: PlanarLcChartWire,
    lc_tube: PlanarLcTubeWire,
    exit_transition: LcExitTransitionWire,
    target_chart: OrdinaryChartWire,
    target_tube: OrdinaryTubeWire,
    segment_type: S,
}

struct RootBindingWire {
    binding_id: S,
    chart_id: S,
    masses: Vec<F>,
    initial_time: F,
    chart_parameter: F,
    positions: FMatrix,
    velocities: FMatrix,
    time_tolerance: F,
    position_tolerance: F,
    velocity_tolerance: F,
    source: S,
}

struct OrdinaryChartWire {
    certificate_id: S,
    chart_id: S,
    chart_type: S,
    masses: Vec<F>,
    position_coefficients: FTensor3,
    velocity_coefficients: FTensor3,
    parameter_interval: F2,
    physical_time_interval: F2,
    coefficient_tolerance: F,
    residual_tolerance: F,
    tail_bound: F,
    sample_count: I,
    source: S,
}

struct OrdinaryTubeWire {
    tube_id: S,
    chart_id: S,
    anchor_parameter: F,
    initial_error_bound: F,
    tube_radius: F,
    max_defect_bound: F,
    max_lipschitz_bound: F,
    source: S,
}

struct OrdinaryBridgeTransitionWire {
    transition_id: S,
    source_chart_id: S,
    source_tube_id: S,
    target_chart_id: S,
    target_tube_id: S,
    source_parameter: F,
    target_parameter: F,
    schema_version: I,
    record_type: S,
    source: S,
}

struct LcEntryTransitionWire {
    transition_id: S,
    source_chart_id: S,
    source_tube_id: S,
    target_chart_id: S,
    target_tube_id: S,
    source_right_parameter: F,
    target_left_parameter: F,
    schema_version: I,
    record_type: S,
    source: S,
}

struct PlanarLcChartWire {
    certificate_id: S,
    chart_id: S,
    chart_type: S,
    masses: Vec<F>,
    pair: I2,
    z_coefficients: FVectorSeries,
    z_velocity_coefficients: FVectorSeries,
    pair_energy_coefficients: FSeries,
    binary_center_coefficients: FVectorSeries,
    binary_center_velocity_coefficients: FVectorSeries,
    third_offset_coefficients: FVectorSeries,
    third_offset_velocity_coefficients: FVectorSeries,
    physical_time_coefficients: FSeries,
    parameter_interval: F2,
    physical_time_interval: F2,
    coefficient_tolerance: F,
    regularized_residual_tolerance: F,
    projected_residual_tolerance: F,
    tail_bound: F,
    sample_count: I,
    projection_rho_lower_bound: F,
    source: S,
}

struct PlanarLcTubeWire {
    tube_id: S,
    chart_id: S,
    anchor_parameter: F,
    initial_error_bound: F,
    tube_radius: F,
    max_defect_bound: F,
    max_lipschitz_bound: F,
    require_pair_energy_constraint: bool,
    source: S,
}

struct LcExitTransitionWire {
    transition_id: S,
    source_chart_id: S,
    target_chart_id: S,
    source_parameter: F,
    target_parameter: F,
    source: S,
}
```

The segment discriminator is parser grammar, not a later theorem
obligation:

| `segment_type` value | Rust arm | Nested exact field set |
|---|---|---|
| `ordinary_bridge_v1` | `SegmentWire::OrdinaryBridge` | `transition`, `target_chart`, `target_tube`, `segment_type` |
| `planar_lc_passage_v1` | `SegmentWire::PlanarLcPassage` | `entry_transition`, `lc_chart`, `lc_tube`, `exit_transition`, `target_chart`, `target_tube`, `segment_type` |

Any other tag, a non-string tag, or fields belonging to the other arm are a
typed-parser rejection in both profiles. The `segments` array itself has
dynamic finite length; zero segments is syntactically valid.

## Shape map and the v0.3/v0.4 boundary

“Parse” below assumes the nesting containers and all elements still have the
declared JSON classes. A number or object where an array is expected is a
parser rejection even when the array's lengths are legacy-dynamic.

| Field family | Successful replay shape | `V03Compatible` wrong length | `V04StrictExperimental` wrong length | Validated Rust form |
|---|---|---|---|---|
| all `masses` | `F[3]` | parses, then `UNRESOLVED` | reject | `[F; 3]` |
| binding `positions`, `velocities` | `F[3][2]` | parses even when ragged, then `UNRESOLVED` | reject | `[[F; 2]; 3]` |
| ordinary position/velocity coefficients | equal `F[d][3][2]`, `d >= 2` | arbitrary/ragged lengths parse, then `UNRESOLVED` | reject | `Vec<[[F; 2]; 3]>`, with equal `d >= 2` |
| ordinary `parameter_interval`, `physical_time_interval` | `F[2]` | **parser rejection**: `_pair_of_float` substitutes a sentinel and canonical round-trip changes | reject explicitly | `[F; 2]` |
| LC `pair` | `I[2]` | **parser rejection**: `_pair_of_int` substitutes a sentinel and canonical round-trip changes | reject explicitly | `[I; 2]`, then validated pair enum |
| six LC vector coefficient blocks | equal `F[d][2]`, `d >= 2` | arbitrary/ragged lengths parse, then `UNRESOLVED` | reject | `Vec<[F; 2]>` with common `d` |
| LC pair-energy and physical-time coefficients | `F[d]`, same common `d` | arbitrary lengths parse, then `UNRESOLVED` | reject | `Vec<F>` with common `d` |
| LC `parameter_interval`, `physical_time_interval` | `F[2]` | **parser rejection** | reject explicitly | `[F; 2]` |
| outer `segments` | any finite length | accepted; semantic fold is ordered | accepted | `Vec<SegmentWire>` |

Thus it is incorrect to implement all bracketed shapes in Section 2 of the
specification as v0.3 parser constraints. It is also undesirable to preserve
that accidental inconsistency as the sole Rust interface. Implement the
profiles explicitly:

```rust
enum SchemaProfile { V03Compatible, V04StrictExperimental }

fn decode_raw_chain(
    value: WireJsonValue,
    profile: SchemaProfile,
) -> Result<RawPlanarChainWire, SchemaDecodeError>;
```

For `V03Compatible`, keep the legacy-dynamic `Vec` fields until replay-shape
validation. For `V04StrictExperimental`, validate them during decoding and
convert them at the raw-to-semantic boundary to the fixed validated forms
above. The release entry point must pin `V03Compatible` explicitly rather than
rely on a library default. A diagnostic corpus should include at least one
wrong-length case from every row and record rejection versus `UNRESOLVED` by
profile.

## Parser versus replay validation by record

The parser column is exhaustive for the typed schema layer. Everything in the
replay column produces `UNRESOLVED` after a canonical typed object exists,
except where the chosen v0.4 strict-shape profile moves a shape rule into
decoding.

| Record | Parser/typed-decoder checks | Replay/semantic checks |
|---|---|---|
| outer chain | exact ten fields; scalar classes; three nested objects; `segments` array; known segment tags | nonempty/globally unique certificate ID; `schema_version == 1`; `certificate_type == "raw_planar_continuation_chain"`; `source == "raw_planar_continuation_chain_v1"`; nonnegative requested width; target/width gates and all chain obligations |
| ordinary segment | exact four fields; known exact tag; nested record classes | nested schemas, identifier matches, endpoint handoff, tube replay, and clock advance |
| LC segment | exact seven fields; known exact tag; nested record classes | nested schemas, canonical pair, entry/tube/exit replay, gauge and clock advance |
| root binding | exact eleven fields and scalar/nesting classes; legacy dynamic arrays as in the shape table | nonempty `binding_id`, `chart_id`, `source`; three positive masses; planar state shapes; chart/mass identity; parameter/state binding; at chain root, all three tolerances are exact dyadic zero and parameter equals the chart/tube left anchor |
| ordinary chart | exact thirteen fields and scalar/nesting classes; fixed interval lengths; legacy-dynamic masses/coefficient blocks | raw-chain composition admission requires nonempty IDs/source, `chart_type == "ordinary_taylor"`, positive `F[3]` masses, equal planar coefficient shapes with `d >= 2`, strictly increasing intervals, and `sample_count >= 1`; replay the separate primitive recurrence/residual/tail ledger only where the composition theorem requires it |
| ordinary tube | exact eight fields and scalar classes | nonempty IDs/source; anchor in chart interval; `initial_error_bound >= 0`; `tube_radius > 0`; both caps nonnegative; identity, defect, separation, Lipschitz, and Gronwall checks |
| ordinary transition | exact ten fields and scalar classes | all IDs/source nonempty; `schema_version == 1`; `record_type == "ordinary_bridge_transition"`; `source == "private_carried_ordinary_bridge_v1"`; referenced identities and exact endpoint parameters |
| LC-entry transition | exact ten fields and scalar classes | all IDs/source nonempty; `schema_version == 1`; `record_type == "carried_planar_lc_entry_transition"`; `source == "private_carried_planar_lc_entry_v1"`; referenced identities and exact right/left endpoints |
| LC chart | exact twenty-two fields and scalar/nesting classes; pair and interval lengths fixed; other arrays legacy-dynamic | nonempty IDs/type/source; `chart_type == "planar_levi_civita_binary"`; positive `F[3]` masses; common `d >= 2` shapes; pair entries distinct in `{0,1,2}` and chain pair ascending; increasing intervals; five nonnegative tolerance/bound fields; `sample_count >= 1`; recurrence, constraint, time-containment, and residual checks |
| LC tube | exact nine fields, including an exact Boolean | nonempty IDs/source; ordinary-tube anchor/sign rules; LC defect/separation/Jacobian/Gronwall checks; pair-energy constraint when flag is true |
| LC-exit transition | exact six fields and scalar classes | all four strings nonempty; referenced identities and exact endpoint parameters; `source` is not pinned to a constant in raw-v1 |

The ordinary-chart row deliberately describes the raw planar-chain composition
surface.  It is narrower than the frozen 13-obligation direct-object primitive:
that primitive admits two- or three-dimensional coefficient blocks, tests no
`source` obligation, permits finite point intervals, and uses `sample_count`
only to enable a non-gating diagnostic when the value is at least `2`.  Passing
the Rust row's planar/source/strict-width/positive-sample checks is therefore
not evidence that the primitive ledger has been replayed.

The implemented adapter now feeds a distinct partial replay profile,
`exact_rational_ordinary_chart_claimed_tail_v04`.  It consumes each admitted
binary64 value as its exact dyadic, uses bounded adaptive square-root interval
precision from 256 through 2048 bits, and performs the planar formal-series
recurrence, 12-component residual construction, and interval Horner evaluation
with exact rational interval arithmetic.  All six ordinary charts in each of
the two canonical chains, twelve chart replays total, satisfy its 13 ordered
ledger entries.  This is not frozen-v0.3 primitive parity or complete chain
replay: the tail remains a claimed allowance, with no collision-free,
convergence, or Taylor-remainder witness.

Finite `F` values are already established by byte parsing, so “finite” is
not repeated as a semantic rule above. Empty strings, negative finite values,
wrong constant strings, reversed length-two intervals, out-of-range pair
integers, and arbitrary-size but value-invalid integers all parse and then
become `UNRESOLVED`.

After semantic shape validation, convert the raw LC pair to a closed enum so
later code cannot accidentally admit a reversed or out-of-range pair:

```rust
enum CanonicalPair { P01, P02, P12 }
```

Similarly, convert `schema_version` only after proving exact `1`.  On the
raw-chain/v0.4 semantic-admission surface, convert `sample_count` to `usize`
only after proving it is positive and fits the implementation/resource limit.
That conversion is a composition-profile choice, not parity with the frozen
primitive direct-object ledger; a compatibility implementation of that ledger
must preserve the diagnostic-only behavior separately.

## Specification audit findings

1. **OPEN-V1-12 is substantive, not editorial.** Sections 2.3, 2.4, and 2.8
   display replay shapes using bracket notation, while the v0.3 parser accepts
   wrong lengths in masses, state matrices, and coefficient blocks. Section
   2's introductory paragraph says this, but the per-record declarations do
   not identify each legacy-dynamic field. The shape table above is the
   missing exhaustive boundary.

2. **The ordinary-chart sign statement needs profile scoping.** Section 2.4
   unconditionally says ordinary chart tolerances and `tail_bound` must be
   nonnegative. That is true of the primitive ordinary-chart ledger. It is not
   uniformly enforced by the archived v0.3 chain aggregate: OPEN-V1-05 records
   the root chart-result omission, and the ordinary-bridge ledger certifies
   source/target tubes without carrying an ordinary-chart result. The private
   `_ordinary_chart_schema` functions require these fields to be finite but do
   not require nonnegativity. A sound v0.4 verifier should require the
   primitive chart ledger whenever the theorem claims it, but exact v0.3
   malformed-case parity must record the historical divergence.

3. **“Accepted mass” is boundary-ambiguous.** Section 1.1 says every accepted
   mass must be positive, although canonical negative `F` masses pass the
   strict parser and fail semantic replay. The text should say “every mass in
   a certified replay” (or explicitly put positivity in the typed v0.4
   schema) to preserve the rejection/`UNRESOLVED` distinction.

4. **Wrong nesting and wrong length are different in v0.3.** Section 2 names
   nonconforming lengths but does not state that the coercing helpers still
   require iterable arrays at each expected nesting level. Ragged arrays of
   `F` survive canonical round-trip; replacing an expected nested array with a
   scalar/object does not. The conformance corpus should exercise both.

5. **Canonical numeric spelling remains an independent blocker.** Correctly
   proving decimal-to-binary64 rounding is necessary but not sufficient for
   raw-v1 byte acceptance. The Rust emitter must reproduce CPython-v1 float
   spellings, including exponent signs/zero padding and `-0.0`, until
   OPEN-V1-01 is replaced by a language-neutral rule. Schema tests must not
   validate canonicality by re-emitting the original lexeme.

6. **Ordinary admission is not primitive-ledger parity.** The private planar
   composition schemas impose two spatial dimensions, nonempty `source`,
   positive interval width, and `sample_count >= 1`. The frozen primitive
   obligation ledger imposes none of those four restrictions: it accepts two
   or three dimensions, ignores `source`, permits point intervals, and treats
   sampling as diagnostic only. The semantic corpus must include one isolated
   case for each divergence and state whether its expectation belongs to the
   raw-chain admission profile or the direct primitive profile.

7. **The ordinary tail is an unverified allowance.** The primitive forms its
   recurrence and residual coefficients in binary64, embeds the rounded
   residuals as point dyadics, takes the largest exact-rational interval
   residual, adds `max(0, tail_bound)` once, converts to binary64, and applies
   one upward `nextafter`. Its final tail obligation checks only finite
   nonnegativity; it does not establish a Taylor-remainder theorem. A negative
   direct-object tail can therefore leave residual obligations 11 and 12 true
   after clamping while obligation 13 and overall certification are false.
   The implemented `exact_rational_ordinary_chart_claimed_tail_v04` profile
   replaces rounded coefficient formation with exact rational interval
   formal-series arithmetic, but deliberately inherits the same unproved
   claimed-tail premise. In that profile the inherited
   `ordinary_taylor_exact_rational_residual_polynomials` ID denotes exact
   rational interval arithmetic, not a convergence or remainder theorem.
   Accordingly OPEN-V1-06 remains open. The separate
   `exact_rational_ordinary_tube_v04` profile addresses only the conditional
   tube ledger. Historical parity and a stronger proof-grade primitive still
   require distinct, unimplemented profiles:
   `binary64_embedded_ordinary_chart_v03` and
   `exact_rational_ordinary_chart_with_verified_tail_v04`, respectively.

No outer or nested field is missing from Sections 2.1-2.10, and the two tag
field sets match the current source. The issues above concern validation
stage, profile scope, arithmetic semantics, and canonicalization rather than
record inventory.
