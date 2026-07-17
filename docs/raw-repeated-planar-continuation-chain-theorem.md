# Raw repeated planar continuation-chain theorem

## Status and scope

This document states the implemented soundness contract of
`RawPlanarChainCertificate` and `check_raw_planar_chain(...)` in
`three_body_symmetry/proof_carrying_planar_chain.py`.  The raw wire format is
version 1; the current replay checker is
`raw_planar_continuation_chain_replay_checker_v2`.  This is a theorem about a
**supplied finite planar point-IVP certificate**, conditional on the trusted
kernel listed below.  It is not a completeness theorem for a certificate
producer and it is not a general solution of the three-body problem.

The checker accepts a finite chain made from ordinary Newtonian charts and
planar Levi--Civita charts for the three canonical ordered pair labels

\[
(0,1),\qquad(0,2),\qquad(1,2).
\]

The pair labels are canonical ascending wire labels.  Version 1 does not treat
the reversed labels as three additional chart kinds.  A pair may be revisited,
and different pairs may occur in any finite order, provided every LC passage
returns to an ordinary chart before the next segment begins.

## Exact input semantics

Every real-valued numeric field in an accepted version-1 wire record is a
finite built-in binary64 value; integer counts and schema versions remain
strict built-in integers. The checker interprets each real-valued field as the
exact dyadic rational represented by its binary64 bit pattern whenever it
enters a theorem-facing rational calculation. Thus a literal such as `0.1`
means the exact binary64 dyadic value stored by Python, not the mathematical
rational number \(1/10\). The checker does not silently replace the serialized
problem by a nearby decimal or rounded problem.

The root binding supplies positive masses, planar positions and velocities,
an initial physical time, and an ordinary-chart parameter.  Acceptance of the
root requires:

- exact zero time, position, and velocity binding tolerances;
- equality of the binding parameter, the initial chart's left endpoint, and
  the initial tube anchor by `Fraction.from_float` equality;
- exact agreement of the serialized mass tuple; and
- a fresh successful replay of the binding, ordinary chart, and ordinary
  a-posteriori tube.

Consequently the induction starts from one exact serialized point IVP, not an
initial-condition box and not a rounded representative of a box.

## Strict raw version-1 grammar

A `RawPlanarChainCertificate` contains exactly:

1. one root `InitialValueProblemBindingCertificate`;
2. one initial ordinary chart and ordinary tube;
3. a tuple of strict tagged-union segment records;
4. a finite requested physical target time \(T\); and
5. a finite nonnegative maximum final-component width.

The only segment tags are:

- `ordinary_bridge_v1`, carrying a raw ordinary transition, target ordinary
  chart, and target ordinary tube; and
- `planar_lc_passage_v1`, carrying a raw ordinary-to-LC entry transition, LC
  chart, LC tube, LC-to-ordinary exit transition, target ordinary chart, and
  target ordinary tube.

In chart notation the accepted language is

\[
N\bigl(\,N\to N\;\mid\;N\to LC_{ij}\to N\,\bigr)^*,
\qquad (i,j)\in\{(0,1),(0,2),(1,2)\}.
\]

Parsing is fail-closed: exact classes, exact field sets, strict tags, canonical
round trips, and a globally unique namespace for raw and reserved derived
identifiers are required.  The wire contains no later IVP binding, propagated
clock origin, selected gauge assignment, projected endpoint box, checker
result, or final enclosure.  Those values are derived during replay.

The checker serializes the complete raw certificate as canonical JSON and
records its aggregate SHA-256 digest.  This is an aggregate transport and
mutation binding for the version-1 evidence.  Version 1 does not yet carry a
separate digest and arithmetic manifest for every nested component.

## Conditional induction invariant

After the root and after every completely accepted segment, the checker is at
an ordinary-chart vertex.  The mathematical invariant is:

1. the unique solution of the exact root IVP has one actual carried branch in
   the current freshly checked ordinary tube;
2. the actual physical clock on that chart has the form \(t=s+b\) for one
   actual scalar \(b\);
3. the checker-derived rational interval \(B=[B_-,B_+]\) contains that actual
   clock origin, \(b\in B\); and
4. every previously accepted handoff identifies this branch with the same
   physical solution, modulo the local LC deck transformation used inside an
   LC passage.

The existential wording is essential.  The checker proves that the one actual
clock origin lies in \(B\).  It does **not** prove that every \(b\in B\) is
realizable, nor that arbitrary states in a rectangular tube belong to one
common physical branch.  Interval propagation may discard correlations while
remaining a sound enclosure of the actual carried branch.

At the root, if the initial physical time is \(t_0\) and the initial chart
parameter is \(s_0\), the clock interval is the exact point

\[
B_0=[t_0-s_0,t_0-s_0].
\]

The top-level checker, rather than a caller, discharges the conditional parent
invariant required by the ordinary and LC transition helpers.

## Ordinary bridge step

For an `ordinary_bridge_v1` segment, let \(e\) be the source right parameter
and \(a\) the target left parameter.  The helper freshly checks both ordinary
tubes, reconstructs the complete source endpoint enclosure by exact rational
polynomial evaluation plus the checked source tube radius, and requires that
complete enclosure to lie in the target initial ball.  Center-to-center or
sampled agreement is insufficient.

Autonomy and ordinary uniqueness then identify the target solution with the
same carried branch.  The checker derives the clock update exactly:

\[
B' = B + e-a.
\]

The associated cocycle record stores both clock intervals and the exact
parameter translation \(e-a\).  No producer-supplied clock value is accepted.

## Carried ordinary-to-LC entry

For a `planar_lc_passage_v1` segment, the entry helper starts from the complete
ordinary endpoint box.  It requires the selected pair to be separated on that
box and reconstructs a complete canonical finite square-root lift cover.  The
cover may contain one or two patches.  The checker derives the overlap parity
graph, checks the exact \(\mathbb F_2\) gauge relation, enumerates the two
global complements, and accepts only if one coherent complement places every
complete emitted patch box inside the LC target anchor.

The analytic entry kernel supplies the following conditional conclusion for
the one actual carried source state:

- at least one emitted patch contains an exact LC lift of that state;
- the selected lift obeys the pair-energy constraint;
- its physical-time coordinate lies in the exactly derived entry interval
  \(D_{\mathrm{in}}=B+e\); and
- the selected coherent gauge sends that lift into the checked LC anchor.

The conclusion is existential for the carried constrained lift.  It does not
assert that every point of every rectangular patch box satisfies the
constraint.

## LC tube and outward mass arithmetic

The LC tube checker freshly validates the fourteen-dimensional regularized
system

\[
X=(z,w,h,R,U,y,V,t),\qquad t'=\rho=|z|^2,
\]

on the complete parameter interval, including defect, derivative, Gronwall,
and separated-third-body obligations.  The carried entry constraint is
preserved by the pinned \(C'=0\) analytic lemma.  Away from \(\rho=0\), the
constrained LC projection is the Newtonian solution for the same exact mass
record.

For each LC tube, carried entry, and carried exit on this theorem surface, the
mass kernel first converts the three serialized binary64 masses to exact
`Fraction` values. For the selected ordered pair it then derives exactly the
pair mass, the two barycentric ratios, the third-over-pair ratio, and the center
and offset coefficients needed by the decisive tube and projection checks.
Each exact rational coefficient is converted only afterward to a tight outward
binary64 interval. The LC tube's interval-dual vector field and Jacobian and
the carried exit projection consume those enclosures. Accepted results pin

`planar_lc_mass_coefficients_exact_binary64_fraction_outward_v1`.

This removes the old requirement that every derived mass ratio itself be an
exact binary64 number on the new LC-tube/carried-entry/carried-exit/finite-chain
surface.  It does not claim that every legacy LC checker path has been
migrated.  Finite positive mass triples can also remain unresolved when an
exact derived coefficient has no finite binary64 enclosure or another
interval obligation overflows; that is checker incompleteness, not a physical
nonexistence result.

At entry, \(\rho>0\) proves that analytic \(z\) is not identically zero.  With
\(t'=|z|^2\geq0\) and isolated zeros of nonzero analytic \(z\), physical time
is strictly increasing along the carried LC segment.  This permits sound
continuation through an isolated zero if one occurs.  It does not prove that a
zero occurs.

## Complete LC-to-ordinary exit

At the exact LC right parameter, the exit helper evaluates all fourteen LC
components and inflates them by the freshly checked tube error.  It requires a
strictly positive lower bound for \(\rho\) on the complete exit slice.  It then
projects the full slice, using outward mass-coefficient arithmetic, to all
twelve Cartesian position and velocity intervals and requires their complete
containment in the target ordinary initial ball.

The fourteenth component gives an outward exit-time interval

\[
D_{\mathrm{out}}=[d_-,d_+]\ni t_{\mathrm{exit}}.
\]

If \(a\) is the target ordinary anchor parameter, the next clock interval is
derived, not supplied:

\[
B'=D_{\mathrm{out}}-a.
\]

Constraint invariance, strict LC time, deck-equivariant punctured projection,
complete endpoint containment, and ordinary uniqueness therefore reestablish
the induction invariant at the target ordinary vertex.

## Pair-local gauges and repetition

Each LC passage retains its derived patch graph and one coherent exact
\(\mathbb F_2\) assignment for the deck action

\[
(z,w,h,R,U,y,V,t)\longmapsto(-z,-w,h,R,U,y,V,t).
\]

Gauge namespaces are local to a passage and pair.  Gauges for different pairs
are never compared.  Two visits to the same pair separated by an ordinary
bridge are also independent lift problems; the checker does not impose an
unsupported equality between their gauge assignments.  The ordinary chart is
the shared physical representation through which pair changes and revisits
are composed.

The strict grammar accepts arbitrary finite repetitions.  The regression
suite includes one chain with pair word

\[
(0,1),(0,2),(1,2),(0,1),
\]

which exercises all three canonical pairs and a same-pair revisit.  This is
evidence for finite grammar composition, not evidence that those LC passages
contain physical collisions.

## Fixed physical target time

After all supplied segments fold successfully, the current vertex is an
ordinary chart with parameter domain \([a,r]\) and actual clock origin
\(b\in B=[B_-,B_+]\).  The checker first requires

\[
T\ge a+B_+.
\]

It then derives the complete parameter preimage interval exactly:

\[
J=T-B=[T-B_+,T-B_-].
\]

Acceptance requires \(J\subseteq[a,r]\).  Every final position and velocity
polynomial is evaluated over all of \(J\) by exact `RationalInterval` Horner
arithmetic.  The checker freshly replays the current ordinary tube and inflates
each component by its checked Gronwall error.  Exact rational endpoint
differences determine the maximum component width, which must not exceed the
serialized requested bound.

## Soundness theorem

Let \(C\) be a canonically parsed `RawPlanarChainCertificate` whose root is an
exact positive-mass planar point IVP with initially distinct positions, and
let \(T\) be its finite requested target.  Subject to the trusted kernel below,

\[
\operatorname{check\_raw\_planar\_chain}(C).\mathrm{status}
=\texttt{CERTIFIED\_TO\_T}
\]

implies:

1. the exact root IVP has one unique chain-compatible continuation from its
   initial time to \(T\);
2. on ordinary pieces it is the classical planar Newtonian solution, and on
   punctured constrained LC pieces its projection is that same Newtonian
   solution;
3. every accepted handoff carries that same branch at one globally coherent
   physical time, with no reversal across an LC passage;
4. different LC pair charts and repeated visits are composed through their
   intervening ordinary representations, with only pair-local deck ambiguity;
5. the returned final ordinary enclosure contains the state at exactly \(T\);
   and
6. every returned component width satisfies the requested bound.

The proof is finite induction on the segment tuple.  The exact root establishes
the invariant.  Each ordinary bridge or complete LC passage proves the
conditional induction step and derives the next clock enclosure.  The final
preimage and Horner enclosure then contain the state at \(T\).

An LC passage in an accepted chain is a regularized coordinate passage.  It
does **not** imply that the selected pair collided.  A collision claim requires
a separate checked event witness proving a zero of \(z\) (equivalently
\(\rho\)) at the asserted physical time.

## `UNRESOLVED` and retained-frontier semantics

Strict parsing and exact-class checks occur before replay: malformed canonical
JSON, unknown fields or tags, or wrong exact classes may raise a parse error or
`TypeError` and produce no replay result. For a successfully constructed
exact-class `RawPlanarChainCertificate`, a failed nested check, containment,
supported-arithmetic obligation, target-domain gate, or width gate produces
`UNRESOLVED`. The replay order and obligation manifest determine the first
failed obligation deterministically.

The result counts the longest consecutive segment prefix accepted before the
first failed segment and retains at most one typed region:

- the current ordinary right frontier after a certified root/prefix;
- a fourteen-dimensional lifted LC right frontier when the failing passage's
  entry and LC tube replay independently even though its exit fails; or
- the ordinary fixed-\(T\) enclosure when the mathematical enclosure succeeds
  but the requested width gate fails.

No root certification means no certified prefix or retained region.  An LC
frontier is never relabeled as a finite Cartesian velocity state.  When a
frontier time is an interval containing an unknown actual endpoint, the
guaranteed covered interval uses only its lower endpoint; if more than one
independently certified prefix frontier is available, it uses the furthest
such lower endpoint.  An upper endpoint is not reported as guaranteed reached.

`UNRESOLVED` proves only the accepted prefix and containment in the retained
region.  It says nothing about existence, uniqueness, collision behavior, or
certifiability beyond that frontier.

## Replay and review artifacts

The result stores the raw certificate, its canonical aggregate SHA-256, the
derived ledgers, the pinned checker/kernel identifiers, and the complete
obligation ledger.  Its `certified` property strictly re-runs
`check_raw_planar_chain(...)` on the retained raw object and requires exact
equality with the stored result.  Hostile subclasses, copied all-true ledgers,
mutated raw evidence, changed kernel identifiers, or altered derived snapshots
therefore do not inherit certification.

The successful result pins the top-level conditional-induction, clock-ledger,
and fixed-time kernels as
`exact_root_and_conditional_planar_segment_induction_kernel_v1`,
`forward_interval_clock_origin_ledger_v1`, and
`exact_rational_horner_fixed_time_kernel_v1`. Its accepted segment ledger
admits only `carried_ordinary_bridge_checker_v1` and
`carried_planar_lc_exit_checker_v2`; the LC path in turn exact-self binds the
v2 carried entry and v2 LC tube results.

This is exact-self replay by the same implementation, not verification by an
independent checker or proof assistant.  The tracked review bundle at
[`artifacts/v0.3.0-review/planar-chain/`](../artifacts/v0.3.0-review/planar-chain/)
contains canonical raw JSON, fresh replay transcripts, and a manifest of
transport hashes and pinned checker/kernel identifiers.

The successful raw fixture has theorem-evidence SHA-256
`ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11`.
Fresh replay returns `CERTIFIED_TO_T` after five accepted segments: one
ordinary bridge followed by LC pair word

\[
(0,1),(0,2),(1,2),(0,1).
\]

The failed-revisit raw fixture has theorem-evidence SHA-256
`c6830919726c22fbca6e45d5781b2465e54876bfa89c25bc26e97284ff918727`.
Fresh replay returns `UNRESOLVED`, with `certified_segment_count = 4`,
`failed_segment_index = 4`, and first failed obligation
`segment[4]:carried_lc_exit_target_initial_ball_contains_complete_projection`.
Thus the certified prefix comprises the ordinary bridge and the first three
LC passages; the failed final `(0,1)` revisit is not added to that prefix. Its
entry and LC tube do replay, so the result retains the typed fourteen-
dimensional lifted LC right frontier rather than claiming a Cartesian exit.

Run

```bash
python scripts/certify_repeated_planar_chain.py verify-bundle
```

to verify the recorded transport hashes and require freshly generated
same-implementation transcripts to match the tracked transcripts exactly.
Neither fixture is an independent-verifier result, and the occurrence of an
LC chart in either fixture is not a collision witness.

## Trusted kernel

The acceptance theorem depends on:

- Python integer and `Fraction` arithmetic and strict canonical JSON/SHA-256
  handling;
- `RationalInterval` polynomial arithmetic and the scalar outward binary64
  interval primitives used by local tube, lift, field, Jacobian, projection,
  square-root, exponential, and Gronwall bounds;
- exact-binary64-to-`Fraction` mass conversion and the reviewed outward LC
  mass-coefficient formulas;
- the explicit planar Newtonian and fourteen-dimensional LC vector fields,
  constraint, lift, projection, and deck action;
- the a-posteriori ODE enclosure lemma used by the ordinary and LC tube
  checkers;
- analytic local existence and uniqueness, LC constraint invariance,
  punctured projection equivalence, deck equivariance, and the nontrivial
  analytic-\(z\) strict-clock lemma;
- complete interval containment, canonical square-root cover, exact
  \(\mathbb F_2\) graph, conditional segment-induction, forward clock-ledger,
  and fixed-time evaluation kernels; and
- the correctness of the Python runtime and the implementation of these
  kernels.

Sampled residuals, nominal physical-time metadata, producer success flags,
reason strings, producer-selected gauge bits, object subclasses, and cached
checker results are not theorem hypotheses.  The current backend is a mixed
exact-rational/outward-binary64 implementation; an independently audited
arbitrary-precision backend remains a future hardening step.

## Precise nonclaims

The implemented theorem does not establish:

- that a producer exists which finds an accepted certificate for every exact
  point IVP, or that any current producer terminates;
- coverage of a box or family of initial conditions;
- that every value in a propagated clock or state interval is realizable;
- automatic collision detection, collision isolation, or the occurrence of a
  collision in any LC passage;
- a direct transition between different LC pair charts, or same-pair
  LC-to-LC recentering without an ordinary bridge;
- continuation through total collision or a complete simultaneous-close-pair
  classifier;
- spatial motion or Kustaanheimo--Stiefel chart gluing;
- escape completeness, all-time coverage, non-Zeno termination, or an
  infinite-atlas theorem;
- an elementary, finite-expression, or general closed-form solution;
- migration of every legacy checker to outward mass arithmetic;
- an independently implemented verifier, formal proof-assistant proof, or
  external mathematical review; or
- correctness without the trusted arithmetic and analytic kernels above.

The contribution is narrower: a finite, adversarially replayed induction
kernel whose acceptance carries one exact planar IVP through a supplied finite
ordinary/LC chain, including all canonical binary pairs and revisits, to a
rigorous fixed-time enclosure.
