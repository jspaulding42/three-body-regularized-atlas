# Raw mixed planar continuation theorem

## Status

This document specifies Milestone 3 of the general certified computational
solution goal and records its implemented private theorem surface.  The
version-1 checker in
`three_body_symmetry/proof_carrying_mixed_continuation.py` may return
`CERTIFIED_TO_T` only after every obligation below has been freshly replayed.
It is a supplied-certificate soundness theorem, not a claim of public API
stability or certificate-production completeness.

The implementation retains and freshly rechecks the Stage-3 entry evidence,
reconstructs the full fourteen-dimensional exit slice, proves positive exit
\(\rho\), contains its complete Cartesian projection in the ordinary anchor,
and derives the exact interval cocycles

\[
D=[d_-,d_+],\qquad B=D-a,\qquad
J=T-B=a+T-D=[a+T-d_+,a+T-d_-].
\]

It then evaluates the target chart over all of \(J\), adds the freshly replayed
ordinary-tube error, applies the exact rational component-width gate, and
returns either an evidence-bound `CERTIFIED_TO_T` enclosure or a structured
`UNRESOLVED` result.  Nonzero-anchor regression fixtures certify one passage
for each canonical pair `(0, 1)`, `(0, 2)`, and `(1, 2)`.  Repeated passages
and chains that change pairs remain the next implementation gap.

The replay result pins the checker and trusted-kernel identities
`raw_mixed_planar_continuation_replay_checker_v1`,
`planar_newton_lc_mixed_analytic_kernel_v1`,
`autonomous_target_clock_cocycle_kernel_v1`, and
`exact_rational_horner_fixed_time_kernel_v1`.  Its nested exit result pins
`raw_gauge_aware_planar_lc_exit_containment_checker_v1` and
`planar_lc_analytic_kernel_v1`.  These names identify the reviewed finite
arithmetic and analytic assumptions; they do not turn the analytic lemmas into
machine-checked proofs.

The first supported grammar is exactly

\[
N_0 \longrightarrow LC_{ij} \longrightarrow N_1,
\]

for one ordered pair \((i,j)\).  Direct changes between different LC pairs,
LC recentering, interval families of initial conditions, total collision,
escape, spatial motion, and producer completeness are later milestones.
Version 1 admits only the canonical ascending pairs
\((0,1),(0,2),(1,2)\); reversed labels are outside its wire grammar.

## Raw evidence

One certificate retains, in canonical order:

1. an exact point-IVP binding for `N_0`;
2. the `N_0` chart and ordinary a-posteriori tube;
3. the raw gauge-aware `N_0 -> LC_ij` transition;
4. the planar LC chart and its fourteen-dimensional a-posteriori tube;
5. the raw `LC_ij -> N_1` exit record;
6. the `N_1` chart and ordinary a-posteriori tube;
7. a finite requested physical time \(T\) and finite nonnegative maximum
   final-component width.

The checker accepts no supplied check result, branch label, gauge bit, clock
shift, projected endpoint box, or final enclosure.  All such objects are
derived from the retained primitives.  Identifiers are nonempty and globally
unique, and every chart has the exact same serialized positive mass triple and
planar dimension.  Source, LC, and target anchor parameters equal their
corresponding serialized left or right endpoints by exact `Fraction` equality,
not by a floating tolerance.

## Entry theorem

The exact point binding and `N_0` tube are freshly checked.  The entry
parameter is the right endpoint of the forward ordinary segment, the LC anchor
is the left endpoint of the LC segment, and the ordinary clock is derived as

\[
t=t_0+s-s_0.
\]

The gauge-aware entry checker reconstructs the complete ordinary handoff box.
It derives the canonical one- or two-patch square-root cover, the overlap
parity graph, and both global complementary gauge assignments.  For every
physical source state, at least one selected patch contains an exact
constrained lift.  Acceptance requires one coherent assignment for which the
entire emitted rectangular patch boxes, including the physical-time
coordinate, are contained in the LC anchor ball.  This does not assert that
every point of a rectangular patch box satisfies the constraint.  The result
must retain its raw evidence, replay it, and match the fresh immutable result
exactly.

## Regularized segment theorem

The LC checker proves a uniform fourteen-dimensional tube for the regularized
ODE on the entire parameter interval.  Exact rational endpoint checks require
\(s_{\mathrm{entry}}=s_-<s_+=s_{\mathrm{exit}}\), and the freshly evaluated
fourteenth ODE block verifies \(t'=\rho\) on that full tube.  The complete
inflated tube remains separated from the third body.  The selected lift
entering the tube satisfies the Newtonian constraint; invariance of that
constraint and the checked LC projection identities carry the same Newtonian
branch through any isolated zero of \(z\).

Entry with \(\rho=|z|^2>0\) proves that the analytic function \(z\) is not
identically zero.  Therefore \(t'(s)=\rho(s)\geq0\) and physical time is
strictly increasing on every nondegenerate parameter interval, even though
its derivative may vanish at isolated binary collisions.  A separate event
certificate is required to claim that a collision actually occurred.

The trusted analytic kernel for this conclusion consists of: local analytic
existence and uniqueness for the separated LC field; invariance of the
constraint \(C'=0\); equivariance under the antipodal deck action; equivalence
of the punctured LC projection with the Newtonian equations; and the
nonzero-at-entry analytic lemma that zeros of \(z\) are isolated and the
physical clock is strictly increasing.  Finite replay checks the hypotheses of
these lemmas; it does not re-prove the lemmas symbolically on each run.

## Complete exit containment

The exit parameter is the right endpoint of the LC segment and the `N_1`
anchor is the left endpoint of its ordinary parameter slab.  At the exit
parameter, interval evaluation of the LC polynomial plus the freshly proved
tube error gives the complete lifted slice

\[
E_{LC}\subset\mathbb R^{14}.
\]

The checker requires a strictly positive lower bound for \(\rho\) on this
slice.  It then projects the entire slice with outward interval arithmetic.
The complete Cartesian position/velocity box, not merely the polynomial
center, must lie in the `N_1` initial-error ball.  Gauge signs need not be
chosen at exit because the physical projection is deck invariant.

The fourteenth component supplies a rigorous physical exit-time interval

\[
D=[d_-,d_+]\ni t_{\mathrm{exit}}.
\]

No endpoint, midpoint, or declared handoff time is promoted to the exact
exit time.

The private exit checker consumes and freshly recomputes the six raw
gauge-aware entry primitives.  The existing legacy LC-to-ordinary checker,
which accepts a supplied legacy entry-result type, is not a theorem-facing
component of this construction.

## One global clock and the fixed-time enclosure

Let \(a\) be the `N_1` anchor parameter.  Autonomy of the Newtonian equations
means that the exact target-chart clock origin is

\[
b=t_{\mathrm{exit}}-a,
\]

so the checker carries the derived interval cocycle

\[
B=D-a\ni b.
\]

For a requested absolute time \(T\), every possible target parameter lies in
the outward interval

\[
J=T-B=a+T-D=[a+T-d_+,\ a+T-d_-].
\]

Acceptance requires \(T\geq d_+\) and complete containment of \(J\) in the
forward certified parameter slab of `N_1`.  The nominal
`physical_time_interval` metadata of `N_1` is not evidence for this step.
The final position and velocity enclosure is obtained by interval polynomial
evaluation over all of \(J\), followed by outward inflation with the freshly
recomputed ordinary Gronwall error.  Its exact rational endpoint differences
must not exceed the requested component-width bound.

This proves: for the one exact point IVP represented by the raw binding, the
regularized Newtonian continuation exists from its initial time through the
supplied `N -> LC_ij -> N` chain to physical time \(T\), and its state at
\(T\) lies in the returned enclosure.  It does not prove that a producer can
find such a chain for arbitrary input.

## Required fail-closed obligations

At minimum the replay ledger contains exact-boolean obligations for:

- canonical raw schemas, unique identities, exact point binding, and common
  problem identity;
- forward endpoint adjacency of all three segments;
- fresh ordinary source, gauge-aware entry, LC tube, and ordinary target-tube
  checks;
- directed mass arithmetic for every coefficient used by the LC field and
  physical projection;
- complete canonical lift coverage and one coherent derived gauge assignment;
- a constrained selected entry lift and a nontrivial LC branch;
- complete exit-slice reconstruction, positive exit \(\rho\), and complete
  Cartesian projection containment;
- the derived exit-time interval, clock-origin interval, and fixed-\(T\)
  preimage interval \(J\);
- target-domain containment, final enclosure construction, and the requested
  width bound; and
- a canonical evidence digest and exact equality with a fresh replay.

The first false obligation determines the structured `UNRESOLVED` result.
Only an independently replayed consecutive prefix may be retained as safe.
After a successful entry, that prefix may be a lifted LC tube/time enclosure,
not an `OrdinaryStateEnclosure`, because Cartesian velocity is undefined at an
interior binary collision.

The implemented result makes that frontier explicit with at most one typed
region: `certified_ordinary_entry_slice`,
`certified_lifted_lc_exit_slice`, or
`certified_ordinary_target_right_frontier`.  If the full fixed-time enclosure
is proved but only the requested width gate fails, it instead retains
`certified_ordinary_fixed_time_enclosure`.  Each region carries exact rational
physical-time and parameter intervals, complete component intervals, its
coordinate system, and a pinned provenance checker.  It never relabels an LC
frontier as an ordinary Cartesian state.

## Directed-arithmetic constraints

The implementation enforces the following soundness constraints; they are not
optional numerical improvements:

1. LC polynomial derivatives must form \(n a_n\) as exact rational products
   of the serialized binary64 coefficient before outward conversion.  A
   pre-rounded NumPy derivative cannot be treated as a point interval.
2. The LC horizon, exponential argument, and Gronwall expression must use
   exact rational combinations of serialized endpoints and previously proved
   upper bounds, with one final outward conversion for each result.
3. Every square root used in the canonical interval lift cover must have a
   proved directed endpoint.  `numpy.sqrt` followed by `nextafter` is not a
   documented proof of direction.
4. Rounded mass sums or ratios may be used as point coefficients only when an
   exact-ratio gate proves equality to the serialized rational problem;
   otherwise the coefficients themselves must be outward intervals.
5. Fixed-\(T\) parameter and state evaluation must be outward throughout;
   nominal chart times and sampled values are diagnostics only.

The focused regressions in
`tests/test_proof_carrying_mixed_continuation.py`, together with
`tests/test_proof_carrying_planar_lc_exit.py` and
`tests/test_lc_directed_arithmetic.py`, cover the three canonical pairs,
inclusive clock/domain/width boundaries, mutation of each retained raw input,
malformed and hostile wire values, typed prefix retention, copied all-true
ledgers, changed kernel identifiers, and mutated derived snapshots.  Such
mutations fail fresh replay rather than inheriting a prior success.

Version 1 may therefore return `UNRESOLVED` for a valid positive mass triple
whose required ratios are not exactly representable in the restricted
binary64 point-coefficient path.  This is checker incompleteness, not evidence
that the physical continuation does not exist.

The existing zero-error exact-collision-anchor and two-sided-passage checker
is separate from this theorem.  A mixed certificate without an additional
event witness makes no claim that its LC segment contains an actual binary
collision.

## Next generalization

With this exact three-segment theorem now exercised by nontrivial positive
fixtures, the next checker must compose a finite chain that alternates ordinary
bridges with any of `LC_01`, `LC_02`, and `LC_12`.  It should reuse this
transition theorem, carry a forward interval clock-origin ledger, derive
pair-indexed gauge graphs, and evaluate the final chart at fixed \(T\).  That
later supplied-chain theorem still makes no arbitrary-input termination or
completeness claim.
