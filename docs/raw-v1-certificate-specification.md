# Raw-v1 planar continuation-chain certificate specification

## Status

This document is a normative, implementation-facing reconstruction of the
raw version-1 certificate language implemented by the v0.3 checker.  Its
authority is the tracked v0.3 source, tests, and review artifacts listed in
Appendix A.  It does not enlarge the theorem proved by that code.

Where this document records a v0.3 composition defect, the v0.4 development
checker may already contain a fail-closed repair.  Such repairs do not alter
the archived v0.3 baseline; they are called out explicitly below and remain
part of the cross-verifier conformance work.

The certificate proves a statement about one supplied, finite, planar,
binary64 point-IVP continuation chain.  It does **not** prove that a producer
can find such a chain for every IVP, that any LC segment contains a collision,
that the chain extends for all time, or that the three-body problem has a
general closed-form solution.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, and **MAY** are
to be interpreted as normative requirements.  “Reject” means that no replay
status exists.  `UNRESOLVED` is a replay result and is mathematically distinct
from rejection.

This specification has two conformance levels:

1. A **sound verifier** MUST issue `CERTIFIED_TO_T` only after proving every
   obligation below.  It MAY return `UNRESOLVED` more often than the reference
   checker.
2. A **v0.3-compatible verifier** must additionally reproduce the reference
   arithmetic and parsing profile.  Several parts of that profile are not yet
   stated in a language-neutral way; those gaps are explicitly marked
   **OPEN-V1**.  Until they are closed, bit-for-bit cross-language parity is
   not a justified claim.

## 1. Data and exact-number model

### 1.1 JSON scalar classes

The notation used below is:

- `S`: a JSON string.  Where an obligation says “present”, it MUST be
  nonempty.  No Unicode normalization is performed.
- `F`: a finite IEEE-754 binary64 JSON number whose parsed runtime class is a
  real, not an integer or Boolean.  Its theorem-facing exact value is the
  dyadic rational represented by that binary64 value.
- `I`: a JSON integer whose parsed runtime class is an integer, not a Boolean.
- `B`: a JSON Boolean.
- `F[n]`, `I[n]`, and nested brackets denote arrays of exact length.

All values used as masses, coefficients, state components, times,
parameters, tolerances, bounds, or radii are `F`.  `schema_version`,
`sample_count`, and LC pair entries are `I`.  The LC tube constraint flag is
`B`.  JSON `null` is not admitted in any raw-v1 field.

The strict byte loader MUST reject nonfinite numbers, including `NaN`,
`Infinity`, and values whose parsing overflows to infinity.  Mass positivity
is a replay obligation, not byte grammar: a canonical negative finite `F`
mass parses but MUST produce `UNRESOLVED`; every mass in a certified replay
MUST be strictly positive.  Further sign requirements are likewise stated
with their replay obligations unless a parser rule is named explicitly.

### 1.2 Canonical JSON evidence

A canonical raw-v1 evidence payload is UTF-8 JSON with:

- exactly one value, whose root is an object;
- no duplicate object keys at any depth;
- object keys sorted lexicographically by the reference serializer;
- separators `,` and `:` with no surrounding whitespace;
- non-ASCII characters emitted directly, not ASCII-escaped;
- no final newline, byte-order mark, or other leading/trailing bytes;
- no unknown or missing fields at any record level; and
- numeric tokens that deserialize and reserialize to identical bytes.

The canonical evidence digest is

```text
SHA256(canonical UTF-8 evidence bytes)
```

The v0.3 reference serialization operation is equivalent to CPython
`json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)`.  Therefore `1` and `1.0` are distinct
wire spellings and are admitted only in fields of the corresponding scalar
class.

**OPEN-V1-01 — numeric lexical grammar.**  Raw-v1 does not yet define a
language-neutral shortest-round-trip grammar for binary64 numbers.  Canonical
float spelling is presently tied to CPython's JSON emitter.  A portable
revision should pin an RFC 8785-style rule or encode binary64 bits directly.

**OPEN-V1-02 — Unicode ordering.**  The current profile relies on Python
string-key sorting and performs no normalization.  Identifiers can contain
arbitrary nonempty Unicode.  A portable revision should define comparison by
Unicode scalar value and decide whether normalization is forbidden or
required.

### 1.3 Parsing versus replay

The strict parser MUST reject:

- malformed UTF-8 or JSON, duplicate keys, nonfinite values, or noncanonical
  bytes;
- a wrong scalar class, missing field, unknown field, wrong array/container
  kind, or unknown segment tag; and
- any record whose parse/serialize round trip changes the canonical JSON.

Once a canonical raw object has been constructed, mathematical or structural
failures discovered by replay produce `UNRESOLVED`, not parser rejection.
Examples include a negative requested width, a reversed LC pair, an empty
semantic identifier, a mass mismatch, an invalid endpoint handoff, a failed
tube bound, or inadequate final accuracy.

This distinction is mandatory for interoperability: malformed evidence has
no certified prefix, while unresolved evidence can retain a certified prefix
or frontier.

**OPEN-V1-03 — dual API.**  The Python object API can be called directly with
values impossible in strict JSON.  For example, a direct-object nonfinite
request can yield `UNRESOLVED`, while strict JSON rejects it.  This document
specifies the strict byte-wire path.  A future API specification should either
remove the direct-object divergence or define it separately.

## 2. Raw grammar

The field sets and scalar/container classes below are parser grammar.  Array
lengths and matrix dimensions shown in brackets are the shapes required for
successful replay unless stated otherwise.  The v0.3-compatible boundary is:

- mass arrays, root position/velocity matrices, ordinary coefficient tensors,
  the six LC vector-coefficient blocks, and the two LC scalar-coefficient
  series may have arbitrary or ragged array lengths at parse time; a shape
  mismatch becomes `UNRESOLVED` during replay;
- every chart parameter/physical-time interval and every LC pair must have
  exactly two elements, because the v0.3 coercing helper otherwise substitutes
  a sentinel and the required canonical round trip fails; and
- the array nesting and the scalar class of every leaf remain parser grammar.
  Replacing an expected nested array with a scalar/object, or an `F` leaf with
  an `I`/`B`, is parser rejection rather than a wrong-length replay failure.

The exhaustive field-by-field boundary and proposed Rust types are recorded
in [`raw-v1-rust-schema-map.md`](raw-v1-rust-schema-map.md).  The v0.4 raw-v1
release profile must retain this v0.3-compatible classification so parser
rejection versus `UNRESOLVED` can be compared across implementations.  A
strict-shape profile is an experimental raw-v2 candidate unless both
verifiers and the frozen corpus are deliberately migrated together.  This
historical unevenness is one reason OPEN-V1-12 remains a release gate.

The complete chart grammar is

```text
Chain ::= N (OrdinaryBridge | PlanarLCPassage)*
OrdinaryBridge ::= N -> N
PlanarLCPassage ::= N -> LC_ij -> N
(i,j) ::= (0,1) | (0,2) | (1,2)
```

Pairs may be revisited and different pairs may occur in any finite order.
Every segment starts and ends at an ordinary chart.  An empty segment array is
syntactically valid.

### 2.1 Outer record

The outer object has exactly these fields:

| Field | Type | Replay requirement |
|---|---:|---|
| `certificate_id` | `S` | nonempty and globally unique |
| `root_binding` | root-binding record | exact root IVP |
| `initial_chart` | ordinary-chart record | freshly certified; see OPEN-V1-05 |
| `initial_tube` | ordinary-tube record | freshly certified |
| `segments` | array | strict tagged union, in order |
| `requested_target_time` | `F` | finite |
| `requested_maximum_component_width` | `F` | nonnegative |
| `schema_version` | `I` | exactly `1` |
| `certificate_type` | `S` | exactly `raw_planar_continuation_chain` |
| `source` | `S` | exactly `raw_planar_continuation_chain_v1` |

### 2.2 Segment records

An `ordinary_bridge_v1` object has exactly:

```text
transition, target_chart, target_tube, segment_type
```

with `segment_type = "ordinary_bridge_v1"`.

A `planar_lc_passage_v1` object has exactly:

```text
entry_transition, lc_chart, lc_tube, exit_transition,
target_chart, target_tube, segment_type
```

with `segment_type = "planar_lc_passage_v1"`.

### 2.3 Root-binding record

Exactly:

```text
binding_id:S, chart_id:S, masses:F[3], initial_time:F,
chart_parameter:F, positions:F[3][2], velocities:F[3][2],
time_tolerance:F, position_tolerance:F, velocity_tolerance:F, source:S
```

Replay requires the three identifiers/source strings to be nonempty, all
masses positive, and all arrays finite and planar.  At the chain root all
three tolerances MUST have exact dyadic value zero.

### 2.4 Ordinary-chart record

Exactly:

```text
certificate_id:S, chart_id:S, chart_type:S, masses:F[3],
position_coefficients:F[d][3][2],
velocity_coefficients:F[d][3][2],
parameter_interval:F[2], physical_time_interval:F[2],
coefficient_tolerance:F, residual_tolerance:F, tail_bound:F,
sample_count:I, source:S
```

Here `d >= 2`; both coefficient arrays MUST have the same shape.
For primitive ordinary-chart certification, `chart_type` MUST equal
`ordinary_taylor`; identifiers and source MUST be nonempty; intervals MUST be
strictly increasing; tolerances and tail bound MUST be nonnegative; and
`sample_count >= 1`.  These are replay obligations, not byte grammar.

The chain does not consume the primitive ordinary-chart ledger at every
vertex.  In particular, an ordinary bridge proves local existence with its
fresh a-posteriori tube and checks only the finite chart schema needed for
that theorem; its schema admits finite negative declared chart tolerances.
The v0.4 root consumes the primitive chart ledger after the OPEN-V1-05 repair,
and LC entry explicitly consumes its source ordinary-chart ledger.  A
cross-language verifier must reproduce the ledger actually required at each
composition surface rather than silently promote every ordinary vertex to a
primitive-chart claim.

The outer coefficient index is the raw monomial degree:

\[
q(s)=\sum_{n=0}^{d-1}q_n s^n,\qquad
v(s)=\sum_{n=0}^{d-1}v_n s^n.
\]

There is no implicit subtraction of the chart's left endpoint.

### 2.5 Ordinary-tube record

Exactly:

```text
tube_id:S, chart_id:S, anchor_parameter:F,
initial_error_bound:F, tube_radius:F, max_defect_bound:F,
max_lipschitz_bound:F, source:S
```

Identifiers/source MUST be nonempty.  The anchor MUST lie in the chart
parameter interval.  The initial error and both caps MUST be nonnegative; the
tube radius MUST be strictly positive.

### 2.6 Ordinary-bridge transition record

Exactly:

```text
transition_id:S, source_chart_id:S, source_tube_id:S,
target_chart_id:S, target_tube_id:S,
source_parameter:F, target_parameter:F,
schema_version:I, record_type:S, source:S
```

Replay requires nonempty identifiers,
`schema_version = 1`, `record_type = "ordinary_bridge_transition"`, and
`source = "private_carried_ordinary_bridge_v1"`.

### 2.7 LC-entry transition record

Exactly:

```text
transition_id:S, source_chart_id:S, source_tube_id:S,
target_chart_id:S, target_tube_id:S,
source_right_parameter:F, target_left_parameter:F,
schema_version:I, record_type:S, source:S
```

Replay requires nonempty identifiers,
`schema_version = 1`,
`record_type = "carried_planar_lc_entry_transition"`, and
`source = "private_carried_planar_lc_entry_v1"`.

### 2.8 Planar LC chart record

Exactly:

```text
certificate_id:S, chart_id:S, chart_type:S, masses:F[3], pair:I[2],
z_coefficients:F[d][2], z_velocity_coefficients:F[d][2],
pair_energy_coefficients:F[d],
binary_center_coefficients:F[d][2],
binary_center_velocity_coefficients:F[d][2],
third_offset_coefficients:F[d][2],
third_offset_velocity_coefficients:F[d][2],
physical_time_coefficients:F[d],
parameter_interval:F[2], physical_time_interval:F[2],
coefficient_tolerance:F, regularized_residual_tolerance:F,
projected_residual_tolerance:F, tail_bound:F, sample_count:I,
projection_rho_lower_bound:F, source:S
```

Here `d >= 2` and every coefficient block has the same degree count.
`chart_type` MUST equal `planar_levi_civita_binary`; identifiers/source MUST
be nonempty; masses MUST be positive; pair entries MUST be distinct members
of `{0,1,2}`; intervals MUST be strictly increasing; all four tolerance/bound
fields and `projection_rho_lower_bound` MUST be nonnegative; and
`sample_count >= 1`.  On the chain surface the pair MUST additionally be one
of the three canonical ascending pairs.

### 2.9 Planar LC tube record

Exactly:

```text
tube_id:S, chart_id:S, anchor_parameter:F,
initial_error_bound:F, tube_radius:F, max_defect_bound:F,
max_lipschitz_bound:F, require_pair_energy_constraint:B, source:S
```

The ordinary-tube sign and anchor rules apply.  The constraint flag is an
exact Boolean; truthy stand-ins are not wire-equivalent.

### 2.10 LC-exit transition record

Exactly:

```text
transition_id:S, source_chart_id:S, target_chart_id:S,
source_parameter:F, target_parameter:F, source:S
```

All strings MUST be nonempty and both parameters finite.  Unlike the private
ordinary and LC-entry transition records, v0.3 does not pin `source` to a
constant.

## 3. State and coordinate conventions

### 3.1 Ordinary state

The ordinary 12-vector order is exactly

```text
(q0x,q0y,q1x,q1y,q2x,q2y,
 v0x,v0y,v1x,v1y,v2x,v2y)
```

Matrix records use body-major, then axis-major order.  Every final ordinary
enclosure and ordinary retained region uses the label
`ordinary_cartesian_q_then_v_12` and this order.

The ordinary differential equation is the planar Newtonian three-body system

\[
q_i'=v_i,\qquad
v_i'=\sum_{j\ne i}m_j\frac{q_j-q_i}{|q_j-q_i|^3}.
\]

### 3.2 LC state

For canonical pair `(i,j)`, let `k` be the remaining body and

\[
M=m_i+m_j,\quad \alpha=m_j/M,\quad \beta=m_i/M,
\quad q=q_j-q_i=Q(z),
\]

\[
Q(z_x,z_y)=(z_x^2-z_y^2,2z_xz_y),\qquad \rho=|z|^2,
\]

\[
R=(m_iq_i+m_jq_j)/M,\quad y=q_k-R.
\]

The LC 14-vector order is exactly

```text
(zx,zy, wx,wy, h, Rx,Ry, Ux,Uy, yx,yy, Vx,Vy, t)
```

The 13-dimensional lifted anchor omits only the final `t`.  The LC polynomial
blocks in Section 2.8 appear in this same order.  The deck action flips only
the first four components:

\[
(z,w,h,R,U,y,V,t)\mapsto(-z,-w,h,R,U,y,V,t).
\]

For `rho > 0`, the Cartesian reconstruction is

\[
q_i=R-\alpha Q(z),\quad q_j=R+\beta Q(z),\quad q_k=R+y,
\]

\[
v_{rel}=v_j-v_i=L(z)w/\rho,\quad
L(z)=2\begin{pmatrix}z_x&-z_y\\z_y&z_x\end{pmatrix},
\]

\[
v_i=U-\alpha v_{rel},\quad v_j=U+\beta v_{rel},\quad v_k=U+V.
\]

For

\[
d_i=y+\alpha Q(z),\quad d_j=y-\beta Q(z),\quad
f_i=d_i/|d_i|^3,\quad f_j=d_j/|d_j|^3,
\]

define

\[
B={m_k\over M}(m_if_i+m_jf_j),\quad
A_k=-m_if_i-m_jf_j,\quad Y=A_k-B,\quad P=m_k(f_j-f_i).
\]

The regularized ODE is

\[
\begin{aligned}
z'&=w,\\
w'&=\tfrac12hz+\tfrac14\rho L(z)^TP,\\
h'&=(L(z)w)\cdot P,\\
R'&=\rho U,&U'&=\rho B,\\
y'&=\rho V,&V'&=\rho Y,\\
t'&=\rho.
\end{aligned}
\]

The constrained physical branch satisfies

\[
C=2|w|^2-M-\rho h=0.
\]

The chart recurrence permits a coefficient tolerance; carried-entry and
analytic invariance obligations establish the exact existential constraint
for the actual branch.  These are different claims.

### 3.3 Exact and outward mass profile

Each mass `m_l` is first converted to its exact binary64 dyadic rational.  For
the selected ordered pair, replay MUST derive exactly:

```text
M                 = m_i + m_j
alpha             = m_j / M
beta              = m_i / M
third_over_pair   = m_k / M
center_first      = m_i * third_over_pair
center_second     = m_j * third_over_pair
offset_first      = m_i + center_first
offset_second     = m_j + center_second
```

Each exact rational is then enclosed by the tightest finite binary64 interval
formed from the nearest binary64 value and, when necessary, its one adjacent
`nextafter` neighbor in the direction of the exact rational.  If no finite
binary64 enclosure exists, replay is `UNRESOLVED`.  Accepted LC entry, tube,
exit, and chain results pin the kernel identifier
`planar_lc_mass_coefficients_exact_binary64_fraction_outward_v1`.  This
profile is used by the decisive LC tube, carried entry, carried exit, and
top-level chain surfaces.  It is not used uniformly by the LC chart
recurrence/residual path; see OPEN-V1-07.

## 4. Primitive certificate obligations

An obligation ledger is ordered.  Every obligation named in a primitive's
own ledger MUST be true for that primitive result's `certified` property.
Whether a chain aggregate actually consults that property is a separate
composition question; the v0.3 root fails to consult the ordinary chart result
as explained in Section 4.6 and OPEN-V1-05.  Sampled residuals may be reported
as diagnostics, but primitive acceptance is controlled by the interval
obligations below.

### 4.1 Ordinary chart

In exact order:

1. `ordinary_chart_type`
2. `certificate_identity_present`
3. `coefficient_array_shape`
4. `finite_coefficients`
5. `positive_masses`
6. `finite_nonempty_time_intervals`
7. `ordinary_physical_parameter_unit_speed`
8. `finite_checker_tolerances`
9. `initial_noncollision`
10. `ordinary_taylor_coefficient_recurrence`
11. `ordinary_taylor_exact_rational_residual_polynomials`
12. `interval_taylor_model_newton_residual`
13. `tail_bound_admissible`

The ordinary-chart checker verifies shape and finiteness, positive masses,
positive pair distance at the zeroth coefficient, the Taylor coefficient
recurrence within `coefficient_tolerance`, formation and dyadic embedding of
Newton residual coefficient arrays, a rational-interval evaluation not
exceeding `residual_tolerance`, and a nonnegative tail bound.  Despite the
obligation's `exact_rational` name, acceleration and residual coefficient
arrays are first computed in binary64 and only then embedded exactly as
dyadics.  See OPEN-V1-06.

The physical and parameter interval widths are considered unit-speed when

```text
abs(parameter_width - physical_width)
    <= max(coefficient_tolerance, 1e-14)
```

under the v0.3 binary64 calculation.

**OPEN-V1-04 — ordinary clock interface.**  Later chain logic treats
`t = s + B` exactly although this primitive permits unequal interval widths
within tolerance.  The theorem interface should require exact dyadic width
equality, or explicitly propagate the discrepancy as clock uncertainty.

### 4.2 LC chart

In exact order:

1. `planar_levi_civita_binary_chart_type`
2. `certificate_identity_present`
3. `planar_lc_coefficient_array_shape`
4. `finite_coefficients`
5. `positive_masses`
6. `binary_pair_valid`
7. `finite_nonempty_time_intervals`
8. `finite_checker_tolerances`
9. `interval_physical_time_containment`
10. `planar_lc_regularized_coefficient_recurrence`
11. `planar_lc_pair_energy_constraint`
12. `planar_lc_exact_rational_regularized_residual_polynomials`
13. `interval_planar_lc_regularized_rhs_residual`
14. `interval_projected_newton_residual_away_from_binary_collision`
15. `tail_bound_admissible`

The LC-chart checker outwardly encloses the physical-time polynomial plus tail
inside the declared physical interval; checks the regularized recurrence and
pair-energy coefficient residual within `coefficient_tolerance`; forms
regularized RHS and residual coefficient arrays in binary64, embeds those
resulting values as exact dyadics for rational-interval evaluation; bounds the
full regularized residual by `regularized_residual_tolerance`; and, on the
region `rho >= projection_rho_lower_bound`, bounds the projected Newton
residual by `projected_residual_tolerance`.  The `exact_rational` obligation
name does not mean the RHS was derived exactly from the raw coefficients; see
OPEN-V1-06.  This path also uses point-valued binary64 mass ratios rather than
the outward mass witness of Section 3.3; see OPEN-V1-07.

### 4.3 Root IVP binding

In exact order:

1. `initial_value_binding_identity_present`
2. `initial_value_binding_chart_present`
3. `initial_value_problem_finite_positive_mass_state`
4. `initial_value_binding_tolerances_finite`
5. `initial_value_binding_chart_masses_match`
6. `initial_value_binding_parameter_inside_chart`
7. `initial_value_binding_state_shape_matches`
8. `initial_value_binding_polynomial_state_matches`

The chart polynomial is evaluated at the binding parameter using exact
dyadic-rational arithmetic.  Exact time, maximum position, and maximum
velocity gaps MUST not exceed the corresponding supplied tolerances.  At the
chain root those tolerances are zero.

### 4.4 Ordinary a-posteriori tube

In exact order:

1. `ordinary_tube_identity_matches_chart`
2. `ordinary_tube_inputs_finite`
3. `ordinary_tube_polynomial_defect_within_cap`
4. `ordinary_tube_collision_free`
5. `ordinary_tube_lipschitz_within_cap`
6. `ordinary_tube_gronwall_self_consistent`

Let `I` be the chart's parameter interval and let

\[
P(s)=(q(s),v(s))
\]

be the serialized polynomial in the 12-state `q`-then-`v` order of Section
3.1.  Every serialized coefficient is interpreted as the exact dyadic value
of its binary64 encoding.  If `p_n` is a component coefficient, the
corresponding derivative coefficient is the exact rational product
`(n + 1) * p[n + 1]`; replay MUST NOT first round that product through
binary64.  Writing `[P](I)` and `[P'](I)` for outward componentwise
interval-Horner enclosures and `[F]` for an outward interval extension of the
ordinary Newton field, replay forms

\[
[R](I)=[P'](I)-[F]([P](I))
\]

and takes `delta` to be an outward upper enclosure of

\[
\sup_{s\in I}\lVert P'(s)-F(P(s))\rVert_\infty,
\]

namely the maximum, over the 12 components of `[R](I)`, of the absolute
values of both interval endpoints.  This is the direct defect of the
serialized finite polynomial.  The chart's `tail_bound`, which belongs to the
primitive Taylor recurrence/residual ledger in Section 4.1, MUST NOT be added
to this direct tube defect.

The scalar initial error `eps` and tube radius `r` are radii in the state
ℓ∞ norm on all 12 coordinates.  The collision and Lipschitz domain is

\[
P(I)+[-r,r]^{12};
\]

replay may enclose it by the componentwise box `[P](I)+[-r,r]^12`.  In
particular, the same `r` is the position radius used in the pair-distance
reduction below.  Replay reconstructs the direct defect, a downward nominal
pair-distance floor, the radius-reduced tube pair-distance floor, and a
Newton-field Lipschitz bound on this complete domain.  Collision freedom
requires a strictly positive tube pair-distance floor.

The declared `max_defect_bound` and `max_lipschitz_bound` are untrusted
binary64 fields interpreted as exact dyadic caps, not witnesses.  The
recomputed outward `delta` and `L` MUST not exceed those caps.  With the exact
dyadic horizon `h = max(|left-anchor|,|right-anchor|)`, the outward Gronwall
error

\[
E=e^{Lh}\epsilon+\delta\,{e^{Lh}-1\over L}
\]

(with the continuous `L=0` interpretation) MUST satisfy the strict inequality
`E < r`.  Equality does not establish self-consistency.

In the v0.3 profile, `d_nominal` is a downward enclosure of the minimum
Euclidean distance between any pair of bodies over the nominal position box
`[q](I)`.  With `d=2`, the tube pair-distance floor is the downward enclosure
of

\[
d_{tube}=d_{nominal}-2\sqrt d\,r.
\]

For body `i`, the acceleration-Jacobian upper bound is the upward enclosure of

\[
2\Bigl(\sum_{j\ne i}m_j\Bigr)(1+3\sqrt d)/d_{tube}^3,
\]

and `L` is the maximum of `1` and the three body bounds.

A v0.3-conformance replay MUST use this published analytic pair-floor and
Lipschitz construction.  It MUST NOT silently substitute an automatic-
differentiation Jacobian row-sum, even if that row-sum is independently proved
to be a sound Lipschitz bound.  Such a bound may define a separately named,
versioned arithmetic profile.  This requirement fixes the v0.3 formula family;
the low-level status-parity questions about outward operations, square roots,
and exponentials remain those recorded in OPEN-V1-08.

Exhaustion of an implementation's declared arithmetic resources or of the
precision needed to prove separation, a cap comparison, or the strict
Gronwall inequality produces `UNRESOLVED` (or a separately reported
non-semantic resource status).  It is neither parser rejection nor
certification; see OPEN-V1-17.

An ordinary-tube result establishes this conditional a-posteriori estimate.
It does not by itself prove that a particular IVP lies in the initial ball.
The chain root must discharge that premise through the validated binding in
Section 4.6, and an ordinary bridge must discharge it through the complete
endpoint handoff in Section 5.2.

### 4.5 Planar LC a-posteriori tube

In exact order:

1. `planar_lc_tube_identity_matches_chart`
2. `planar_lc_tube_lifted_serialization_admissible`
3. `planar_lc_tube_outward_mass_arithmetic_certified`
4. `planar_lc_tube_inputs_finite`
5. `planar_lc_tube_direct_lifted_defect_within_cap`
6. `planar_lc_tube_separated_third_body`
7. `planar_lc_tube_interval_jacobian_within_cap`
8. `planar_lc_tube_pair_energy_constraint_when_required`
9. `planar_lc_tube_gronwall_self_consistent`

Replay evaluates the complete 14-dimensional lifted polynomial, direct
defect, interval Jacobian, and third-body denominators on the inflated tube.
The two third-body distances MUST stay positive.  Defect and infinity-norm
Jacobian bounds MUST not exceed their caps and the outward Gronwall error MUST
be strictly less than the radius.  If the Boolean constraint flag is true,
the exact anchor polynomial center MUST satisfy `C = 0`.

### 4.6 Validated ordinary root wrapper

In exact order:

1. `validated_ordinary_chart_serialization_admissible`
2. `validated_ordinary_ivp_binding_checked`
3. `validated_ordinary_tube_checked`
4. `validated_ordinary_component_chart_ids_match`
5. `validated_ordinary_anchor_parameter_matches_binding`
6. `validated_ordinary_actual_initial_error_covered`

It constructs results for the chart, binding, and tube and requires the
binding and tube results to certify, all chart identifiers and anchors to
agree, and the tube's initial ball to cover the actual exact
binding-to-polynomial gap.  In v0.3, `ValidatedOrdinaryIVPChartCheckResult`
checks only the exact class of `chart_result`, not `chart_result.certified`,
and its missing-obligation aggregation omits the chart ledger.  Therefore the
chain root can be accepted despite false ordinary-chart obligations if the
other aggregate conditions pass.  A sound independent verifier SHOULD return
`UNRESOLVED` in that case; exact v0.3 status parity is unsafe.  See
OPEN-V1-05.

## 5. Chain induction and segment replay

### 5.1 Induction invariant and clock

After the root and every accepted segment, replay retains one current
ordinary chart/tube and an exact rational interval `B = [B_lo,B_hi]` known to
contain the actual clock origin in `t = s + b`.

For root time `t0` and root parameter `s0`,

\[
B_0=[t_0-s_0,t_0-s_0]
\]

using exact dyadic arithmetic.  A segment advances the state only when its
entire nested obligation ledger is true.  Consequently the certified segment
count is always the longest consecutive accepted prefix.

The `physical_time_interval` field on later ordinary charts is checked for
finite positive width and approximate unit speed, but it does not update or
override this clock ledger.

### 5.2 Ordinary bridge obligations

In exact order:

1. `ordinary_bridge_exact_raw_schemas`
2. `ordinary_bridge_identifiers_match_and_are_unique`
3. `ordinary_bridge_parent_clock_origin_is_exact_interval`
4. `ordinary_bridge_common_planar_mass_problem`
5. `ordinary_bridge_source_and_target_tubes_freshly_certified`
6. `ordinary_bridge_exact_right_to_left_endpoint_handoff`
7. `ordinary_bridge_complete_source_endpoint_enclosure_reconstructed`
8. `ordinary_bridge_target_initial_ball_contains_complete_source_endpoint`
9. `ordinary_bridge_target_clock_origin_exactly_derived`

Let `e` be the source chart right endpoint and `a` the target chart left
endpoint.  Transition parameters MUST equal those endpoints exactly as
dyadics; both tube anchors MUST equal their chart left endpoints.  The two
mass tuples MUST agree exactly and both charts MUST be planar.

Replay evaluates all 12 source polynomial components at `e` by exact rational
Horner evaluation and symmetrically inflates them by the freshly recomputed
source Gronwall radius.  The resulting complete box MUST be contained in the
target polynomial center at `a`, symmetrically inflated by
`target_tube.initial_error_bound`.  Center agreement alone is insufficient.
The new clock is

\[
B'=B+e-a.
\]

### 5.3 LC-entry obligations

In exact order:

1. `carried_lc_entry_exact_raw_schemas`
2. `carried_lc_entry_transition_canonical_round_trip`
3. `carried_lc_entry_identifiers_match_and_are_unique`
4. `carried_lc_entry_parent_clock_origin_is_exact_interval`
5. `carried_lc_entry_source_ordinary_chart_freshly_certified`
6. `carried_lc_entry_source_ordinary_tube_freshly_certified`
7. `carried_lc_entry_target_lc_chart_freshly_certified`
8. `carried_lc_entry_target_lc_tube_freshly_certified`
9. `carried_lc_entry_common_planar_mass_problem`
10. `carried_lc_entry_pair_is_canonical_ascending`
11. `carried_lc_entry_outward_mass_arithmetic_certified`
12. `carried_lc_entry_exact_source_right_to_lc_left_anchor`
13. `carried_lc_entry_complete_source_endpoint_box_reconstructed`
14. `carried_lc_entry_selected_pair_collision_free`
15. `carried_lc_entry_canonical_square_root_atlas_reconstructed`
16. `carried_lc_entry_derived_parity_graph_certified`
17. `carried_lc_entry_physical_time_interval_exactly_derived`
18. `carried_lc_entry_all_lift_patches_have_positive_rho`
19. `carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed`
20. `carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel`
21. `carried_lc_entry_one_global_complement_contains_all_complete_patches`

The entry source parameter MUST be the ordinary right endpoint; the target
parameter and LC tube anchor MUST be the LC left endpoint.  Replay reconstructs
the complete ordinary endpoint box and requires the selected relative-position
box to exclude zero.

The current canonical square-root cover is selected by this ordered decision
tree for the relative box `x=[xlo,xhi]`, `y=[ylo,yhi]`:

1. if `ylo >= 0`: one closed-upper patch;
2. else if `yhi <= 0`: one closed-lower patch;
3. else if `xlo > 0`: one right-half patch;
4. else if `ylo < 0 < yhi` and `xhi < 0`: upper and lower patches meeting
   across the negative-axis cut, with deck parity `1`;
5. otherwise: `UNRESOLVED`.

For each patch, replay constructs outward principal square-root intervals
from

\[
z_x=\sqrt{(|q|+q_x)/2},\qquad
|z_y|=\sqrt{(|q|-q_x)/2},
\]

with the patch's prescribed sign, and derives
`w = (1/4)L(z)^T(v_j-v_i)`, `h`, `R`, `U`, `y`, and `V` from the complete
Cartesian box.  Every patch MUST have a positive lower bound for `rho`.

The exact entry time interval is

\[
D_{in}=B+e.
\]

It is appended as component 14.  The target LC 13-vector and time polynomial
are evaluated exactly at the left anchor.  The derived F2 gauge graph MUST be
connected and compatible, with edge equation
`g_source XOR g_target = parity`.  One of the two global complement choices
MUST place every component of every complete lifted patch inside the target
anchor center plus/minus `target_tube.initial_error_bound`.

The analytic kernel conclusion is existential: the one actual Cartesian
source state has a constrained lift in at least one complete patch.  It does
not claim that every point of each rectangular patch satisfies `C = 0`.

The entry transition reserves these identifiers, all of which participate in
the global namespace:

```text
<entry>:derived-gauge-cover
<entry>:patch:0-upper
<entry>:patch:0-lower
<entry>:patch:0-right
<entry>:patch:1-lower
<entry>:negative-axis-overlap
```

The derived gauge checker has obligations
`lc_gauge_certificate_identity`, `lc_gauge_chart_ids`,
`lc_gauge_overlap_schema`, `lc_gauge_graph_connected`, and
`lc_gauge_equations_compatible`.

### 5.4 LC-exit obligations

In exact order:

1. `carried_lc_exit_exact_raw_schemas`
2. `carried_lc_exit_identifiers_match_and_are_unique`
3. `carried_lc_exit_parent_source_invariant_is_explicit_condition`
4. `carried_lc_exit_entry_freshly_replayed_and_certified`
5. `carried_lc_exit_common_planar_mass_problem`
6. `carried_lc_exit_pair_is_canonical_ascending`
7. `carried_lc_exit_outward_mass_arithmetic_certified`
8. `carried_lc_exit_exact_right_to_left_endpoint_handoff`
9. `carried_lc_exit_constrained_entry_branch_carried`
10. `carried_lc_exit_constraint_invariance_kernel`
11. `carried_lc_exit_lc_tube_freshly_certified`
12. `carried_lc_exit_third_body_separated`
13. `carried_lc_exit_target_ordinary_tube_freshly_certified`
14. `carried_lc_exit_strict_physical_clock_kernel`
15. `carried_lc_exit_complete_inflated_slice_reconstructed`
16. `carried_lc_exit_complete_slice_rho_positive`
17. `carried_lc_exit_complete_cartesian_projection_reconstructed`
18. `carried_lc_exit_deck_equivariant_newton_projection_kernel`
19. `carried_lc_exit_target_initial_ball_contains_complete_projection`
20. `carried_lc_exit_time_interval_derived_from_component_fourteen`
21. `carried_lc_exit_target_clock_origin_exactly_derived`

Endpoint identities MUST form the exact chain

```text
ordinary source right = entry source
LC left = entry target = LC tube anchor
LC right = exit source
ordinary target left = exit target = target tube anchor
```

At the LC right endpoint replay outwardly evaluates and inflates all 14
components by the fresh LC Gronwall bound.  The complete slice MUST have
`rho > 0` and separated third-body denominators.  Replay projects the full
slice, not merely its center, to the 12 Cartesian intervals using the outward
mass profile.  The target ordinary initial ball MUST contain the complete
projection.

The fourteenth interval is the exit physical-time interval `D_out`; if `a` is
the target ordinary left anchor, the next clock is

\[
B'=D_{out}-a.
\]

Constraint invariance and strict physical time are pinned analytic lemmas.
An LC passage remains valid if no collision occurs; acceptance does not
assert that `z=0` occurs.

## 6. Global namespace

Every defining identifier below MUST be nonempty and no two defining
identifiers may have equal strings:

- outer certificate ID;
- root binding ID;
- each chart certificate ID and chart ID;
- each tube ID;
- every transition ID; and
- the six reserved derived IDs for every LC entry.

Reference fields such as a transition's `source_chart_id` are required to
equal the corresponding defining ID and are not additional definitions.

## 7. Top-level replay and target enclosure

The top-level obligation ledger has exactly this order:

1. `raw_planar_chain_outer_schema_exact`
2. `raw_planar_chain_global_identifier_namespace_unique`
3. `raw_planar_chain_canonical_evidence_serializable`
4. `raw_planar_chain_requested_target_finite`
5. `raw_planar_chain_requested_width_admissible`
6. `raw_planar_chain_root_exact_point_left_anchor`
7. `raw_planar_chain_root_freshly_certified`
8. `raw_planar_chain_all_segments_freshly_folded`
9. `raw_planar_chain_target_not_before_current_left_clock`
10. `raw_planar_chain_fixed_time_preimage_exactly_derived`
11. `raw_planar_chain_fixed_time_preimage_inside_forward_current_domain`
12. `raw_planar_chain_target_state_exactly_evaluated_and_inflated`
13. `raw_planar_chain_final_component_width_within_requested_bound`

The root binding parameter, initial chart left endpoint, and initial tube
anchor MUST be exactly equal as dyadics.  Binding and chart masses and all
three chart IDs MUST agree as required by their records.

For exact v0.3 reconstruction, top-level obligation 7 follows the aggregate
root result and can therefore be true while the nested ordinary-chart ledger
is false (OPEN-V1-05).  A sound verifier MUST NOT treat that implementation
omission as a theorem rule; it should return `UNRESOLVED` until the chart
ledger is also true.

After folding the longest certified segment prefix, let the current ordinary
domain be `[a,r]`, its clock enclosure be `[B_lo,B_hi]`, and the request be
`T`.  Replay MUST require

\[
T\ge a+B_{hi}.
\]

It derives, rather than accepts, the exact parameter preimage

\[
J=[T-B_{hi},T-B_{lo}]
\]

and requires `J` to lie in `[a,r]`.  It evaluates every current ordinary
position and velocity polynomial on `J` by exact rational interval Horner
evaluation and symmetrically inflates all components by the freshly checked
current ordinary Gronwall radius.  The maximum exact component width MUST not
exceed the requested bound.

If all 13 obligations are true, the result is `CERTIFIED_TO_T`, the covered
physical interval is `[t0,T]`, the final state order is the ordinary 12-vector
of Section 3.1, and no retained frontier is emitted.

## 8. `UNRESOLVED`, failure location, and retained frontiers

### 8.1 Meaning

`UNRESOLVED` means only that this evidence and checker did not establish the
requested theorem.  It is not evidence of nonexistence, collision, chaos,
singularity, or failure of another certificate.

`certified_segment_count` is the length of the longest consecutive segment
prefix accepted after a certified root.  A failed segment index, when
present, equals that count.

`first_failed_obligation` is the first false top-level obligation, except:

- a nested root failure is reported as `root:<first nested name>`; and
- a nested segment failure is reported as
  `segment[i]:<first nested name>`.

If the root is not freshly certified, no clock ledger, cocycle ledger,
current chart, or retained frontier is certified.

### 8.2 Retention priority

At most one retained region is emitted:

1. If the fixed-time mathematical enclosure succeeds but only the requested
   width fails, retain `certified_ordinary_fixed_time_enclosure`, with
   physical interval `[T,T]`, parameter interval `J`, and the ordinary
   12-vector order.
2. Otherwise, after a certified root/prefix, reconstruct the current ordinary
   right frontier when possible.  If the failed segment is LC and its entry
   and LC tube independently certify through the LC right endpoint, retain
   that later `certified_lifted_lc_right_frontier` in the LC 14-vector order;
   it has priority over the ordinary frontier.
3. If no LC-right frontier exists, retain the reconstructed ordinary frontier
   as `certified_current_ordinary_right_frontier`.
4. If neither can be freshly justified, retain nothing.

When at least one frontier is freshly reconstructed, the covered physical
interval on unresolved replay begins at `t0` and ends at the maximum of `t0`
and the **lower** physical-time endpoints of independently certified
ordinary-prefix and retained-LC frontiers.  If no frontier is reconstructed,
the covered interval is absent.  An uncertain frontier's upper time endpoint
MUST NOT be claimed as covered.

## 9. Arithmetic and transcript conformance gaps

The following are known limits of this reconstructed v1 specification.  A
verifier MUST NOT silently invent answers to them and claim exact v0.3 parity.

1. **OPEN-V1-05 — root chart certification gate.**  The archived v0.3
   aggregate root result does not test `chart_result.certified` and does not
   include the chart's missing obligations.  The v0.4 development branch now
   repairs both omissions and regression-tests fail-closed behavior.  A sound
   independent verifier must likewise require the chart ledger.  The gap
   remains open only as a versioned conformance decision: malformed cases can
   intentionally disagree with archived v0.3, and the v0.4 corpus must record
   the repaired `UNRESOLVED` outcome.
2. **OPEN-V1-06 — misleading exact-rational residual names.**  Both ordinary
   and LC `exact_rational_*` obligations first form RHS/residual coefficients
   through binary64 array arithmetic and only afterward embed the rounded
   coefficients as exact dyadics.  The names overstate what is exact.  A new
   profile must either specify and outwardly account for the binary64
   formation error or derive the coefficient algebra over exact rationals.
3. **OPEN-V1-07 — split LC mass semantics.**  The decisive LC tube,
   entry, and exit paths use exact-derived outward mass witnesses, but the LC
   chart recurrence and residual helpers use point-valued binary64 mass sums
   and ratios.  A chain therefore mixes two mass-arithmetic semantics.  The
   chart path must migrate to the outward witness, or the discrepancy must be
   bounded in its residual proof.
4. **OPEN-V1-08 — interval arithmetic profile.**  The source contains the
   decisive details for outward `nextafter`, interval products, powers,
   square roots, reciprocals, automatic differentiation, and a Decimal-based
   exponential enclosure, but raw-v1 has no standalone normative arithmetic
   standard.  A sound implementation may use wider proven enclosures and
   return more `UNRESOLVED`; status parity requires a separately pinned
   profile.
5. **OPEN-V1-09 — entry conversion path.**  Parts of the LC entry path convert
   exact rational endpoint boxes outward to binary64 and then embed those
   endpoints back into rationals.  This is sound but not equivalent to a
   purely rational lift.  The conversion points and rounding directions need
   a dedicated arithmetic profile.
6. **OPEN-V1-10 — source strings.**  Most chart/tube/binding `source` fields
   are required only to be nonempty; the exit source is also unpinned.  Their
   semantic purpose is therefore decorative in v1.  A revision should either
   enumerate meaningful constants or remove the fields.
7. **OPEN-V1-11 — ordinary physical-time metadata.**  Later ordinary
   `physical_time_interval` values do not drive the clock ledger.  This should
   be made explicit in a future schema or the redundant field should be
   removed from chain vertices.
8. **OPEN-V1-12 — parser schema.**  Nested Python `from_dict` coercions plus a
   canonical round trip indirectly enforce scalar types and impose the uneven
   shape boundary enumerated in Section 2.  The implementation map
   [`raw-v1-rust-schema-map.md`](raw-v1-rust-schema-map.md) now inventories
   every field and classifies every array family, but this gap remains open
   until the independent typed decoder and portable rejection/`UNRESOLVED`
   corpus pass.  A schema that makes every displayed bracket length a parser
   constraint would not be v0.3-compatible raw-v1.
9. **OPEN-V1-13 — gauge tie-break.**  Gauge compatibility and existence of a
   containing complement are semantically defined, but the deterministic
   assignment choice used in review transcripts is not specified
   independently of the implementation.
10. **OPEN-V1-14 — diagnostics.**  Some missing-obligation details include
   Python exception class names.  Those strings are not portable theorem
   data.
11. **OPEN-V1-15 — transcript layer.**  The v0.3 review transcript is a
   same-implementation audit artifact.  It is neither part of the raw wire nor
   currently a normative independent-verifier output schema.  A later release
   must decide whether independent verifiers reproduce it exactly or emit a
   smaller standardized result.
12. **OPEN-V1-16 — negative zero.**  Canonical JSON can preserve `-0.0`, while
   exact rational equality collapses its sign.  V1 has no explicit policy
   beyond the current field-specific comparisons.
13. **OPEN-V1-17 — resource limits.**  The semantic language has no maximum
    JSON size, degree, or segment count.  Independent verifiers need explicit
    implementation limits and must report limit exhaustion as `UNRESOLVED` or
    a distinct non-semantic resource status, never as certification.

## 10. Conformance checklist

A sound independent verifier of raw-v1 SHOULD, in order:

1. load exact bytes with duplicate-key and canonical-byte rejection;
2. validate the exact record grammar and scalar classes;
3. bind and log the SHA-256 of those bytes;
4. validate the global defining-ID namespace;
5. freshly replay the root chart, binding, and ordinary tube;
6. construct `B0` exactly and fold only a consecutive segment prefix;
7. freshly replay every nested chart, tube, transition, mass, lift, gauge, and
   projection obligation rather than trusting producer summaries;
8. derive every clock interval, state enclosure, and frontier internally;
9. issue `CERTIFIED_TO_T` only if the complete ordered top-level ledger is
   true and the final width bound holds; and
10. otherwise issue `UNRESOLVED` with only the prefix and frontier that were
    independently justified.

## Appendix A. Current authority map

The normative reconstruction above is derived from these tracked sources:

- `three_body_symmetry/proof_carrying_planar_chain.py`: raw records and tags
  (approximately lines 81–349), replay and top ledger (approximately
  712–1207), schema checks (approximately 1311–1563), exact evaluation and
  retained-frontier logic (approximately 1625 onward).
- `three_body_symmetry/proof_carrying_ordinary_bridge.py`: transition record,
  bridge replay, and nine obligations (approximately lines 48–466).
- `three_body_symmetry/proof_carrying_carried_planar_lc_entry.py`: strict entry
  transition, lift atlas, gauge replay, and 21 obligations (approximately
  lines 83–1059 and helper routines thereafter).
- `three_body_symmetry/proof_carrying_carried_planar_lc_exit.py`: exit replay,
  projection, and 21 obligations (approximately lines 57–839).
- `three_body_symmetry/certificate_language.py`: root, chart, tube, and exit
  wire dataclasses (notably classes beginning near lines 40, 130, 173, 317,
  563, and 596).
- `three_body_symmetry/certificate_checker.py`: ordinary and LC primitive
  ledgers, exact binding evaluation, tube checks, LC state packing, and root
  wrapper (notably functions beginning near lines 2931, 3142, 4842, 5008,
  5312, 7658, 8282, and 11148).  The archived v0.3 root aggregation gap was in
  `ValidatedOrdinaryIVPChartCheckResult` near lines 515–547; the v0.4
  development checker now closes it.  The binary64 residual-formation paths
  are visible near lines 11668–11945 and 12409–12455.
- `three_body_symmetry/binary_chart.py`: interval square-root branches and
  canonical finite atlas (approximately lines 234–674).
- `three_body_symmetry/binary_series.py`: regularized state pack/unpack order
  (approximately lines 625–659).
- `three_body_symmetry/planar_lc_mass_coefficients.py`: exact mass derivation
  and tight outward binary64 enclosure profile.
- `three_body_symmetry/planar_chain_review_artifact.py`: strict byte loader,
  canonical review bundle, exact rational transcript encoding, and manifest
  validation.
- `tests/test_proof_carrying_planar_chain.py` and
  `tests/test_planar_chain_review_artifact.py`: rejection/`UNRESOLVED`
  boundary, pair/revisit cases, retained-frontier behavior, namespace attacks,
  and canonical-byte tests.
- `artifacts/v0.3.0-review/planar-chain/success.raw.json` and
  `failed-revisit.raw.json`: concrete canonical examples.  Example values and
  hashes are test vectors, not additional language rules.

## Appendix B. Trust boundary

Raw-v1 is proof-carrying only relative to the following analytic and arithmetic
kernels, which a higher-assurance implementation should isolate, document,
and independently test or formalize:

- exact binary64-to-dyadic embedding and rational interval Horner evaluation;
- outward scalar/interval arithmetic and automatic differentiation;
- Newtonian ordinary recurrence, residual, collision, Lipschitz, and
  Gronwall lemmas;
- the LC regularized ODE, pair-energy invariance `C' = 0`, strict physical
  clock, and punctured deck-equivariant Newton projection;
- canonical square-root cover completeness and F2 gauge compatibility; and
- ordinary autonomous uniqueness and complete endpoint-containment handoff.

No producer-supplied checker result, clock value, gauge bit, endpoint box, or
final enclosure lies inside the trusted input.  All are replay-derived.
