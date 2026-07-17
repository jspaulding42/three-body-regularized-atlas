# Research Goal: Exact Levi-Civita Gauge Gluing

## Goal

Strengthen the validated planar Levi-Civita part of this project by adding a
small, exact, independently checkable local-to-global gluing layer.  The layer
will record the antipodal sign relating already-validated local LC sections,
solve the resulting finite system over \(\mathbb F_2\), and return either:

1. a primal assignment of one gauge bit to every chart; or
2. a dual obstruction consisting of a closed overlap cycle with odd total
   sign.

This is a deliberately bounded goal.  It is intended to make branch handling
more transparent and eventually less conservative.  It is not a general
solution of the three-body problem, a construction theorem for arbitrary
initial data, or a replacement for the analytic and interval arguments that
validate the local charts and their overlaps.

## Exact symmetry being used

For a fixed selected pair and fixed masses, write the regularized planar state
as

\[
X=(z,w,h,R,U,y,V,t).
\]

The nontrivial Levi-Civita deck transformation is

\[
\gamma X=(-z,-w,h,R,U,y,V,t).
\]

It is exact, including at binary collision while the third body is separated.
Indeed,

\[
Q(-z)=Q(z),\qquad |{-z}|^2=|z|^2,
\qquad L(-z)(-w)=L(z)w.
\]

Consequently the physical position and punctured physical velocity are
unchanged.  The pair-energy constraint

\[
C=2|w|^2-(m_i+m_j)-|z|^2h
\]

is unchanged as well.  The third-body force terms depend on \(z\) only through
\(Q(z)\), and the regularized vector field obeys

\[
F(\gamma X)=D\gamma\,F(X).
\]

Thus applying \(\gamma\) to a local LC solution gives another lifted
representative of the same physical solution.

## Exact same-IVP anchored-overlap theorem

An atlas vertex is not merely a chart name.  In the implemented Stage 2
language it is one fixed triple

\[
v=(\mathcal C_v,\mathcal T_v,a_v),
\]

consisting of a serialized LC chart, one zero-error a-posteriori tube, and that
tube's exact anchor parameter.  The same vertex identifier may not be reused
with a different chart, tube, or anchor.

Let two vertices \(u,v\) use the same positive masses and the same **ordered**
selected pair.  Write their chart intervals as \(I_u,I_v\), and suppose both
lifted tube checks accept with zero initial error.  Define the common shifted
parameter interval

\[
J=(I_u-a_u)\cap(I_v-a_v).
\]

Assume \(J\) is nondegenerate and exact rational evaluation of all fourteen
anchor coordinates gives, for one uniquely determined
\(\sigma_{uv}\in\mathbb F_2\),

\[
\bar X_v(a_v)=\gamma^{\sigma_{uv}}\bar X_u(a_u).
\]

Here equality includes \(h,R,U,y,V,t\), while both \(z\) and \(w\) must be
jointly equal or jointly antipodal.  Then the unique exact solutions selected
by the two tubes obey

\[
X_v(a_v+r)=\gamma^{\sigma_{uv}}X_u(a_u+r)
\qquad(r\in J).
\]

To prove this, set
\(Y(r)=\gamma^{\sigma_{uv}}X_u(a_u+r)\).  Equivariance gives
\(Y'=F(Y)\), and the exact anchor identity gives
\(Y(0)=X_v(a_v)\).  The a-posteriori tube theorem supplies existence and
uniqueness on both chart intervals, so ODE uniqueness gives \(Y=X_v\) on
their common shifted interval.  No condition \(\rho>0\) is used; the argument
therefore remains valid at binary collision.

The result deliberately reports three different strengths:

- `lifted_overlap_certified` proves the exact lifted same-IVP statement above;
- `constrained_newtonian_overlap_certified` additionally requires exact
  pair-energy constraints at both anchors and exact serialized mass-ratio
  arithmetic, so the punctured projections belong to one Newtonian IVP; and
- `two_sided_overlap` says \(\inf J<0<\sup J\).  A nondegenerate one-sided
  overlap is still a valid lifted edge, but it does not establish a two-sided
  local neighborhood of the anchor.

At collision, \(z=0\) cannot distinguish the two sheets.  Exact constraint
with positive pair mass forces \(w\ne0\), and the checker derives parity from
the joint \((z,w)\) relation.  It never accepts a supplied parity bit.

## Finite gluing theorem

Let \(G=(V,E)\) be a finite loopless multigraph.  A vertex is one connected LC
chart domain.  An edge is one connected, nonempty, independently validated
same-solution overlap component.  Suppose the checker for that overlap has
established a constant bit \(\sigma_e\in\mathbb F_2\) such that its two lifted
representatives differ by \(\gamma^{\sigma_e}\).

The charts can be re-gauged to agree on every overlap exactly when there are
bits \(t_v\) satisfying

\[
t_u+t_v=\sigma_e\pmod 2
\]

for each edge \(e=uv\).  Equivalently, the XOR of the edge labels is zero on
every graph cycle.  In incidence-matrix notation this is

\[
At=\sigma,
\]

and the dual condition is

\[
y\mathbin{\cdot}\sigma=0
\quad\text{for every }y\in\ker(A^T).
\]

A spanning-forest traversal proves the result constructively.  Set one root
bit to zero in each component and propagate the edge equations.  If every
non-tree edge agrees, the propagated bits are a primal witness.  If an edge
disagrees, that edge plus the unique tree path between its endpoints is a
closed cycle whose label XOR is one, hence a dual obstruction.  For a
connected compatible graph the assignment is unique up to one global
application of \(\gamma\).

For the generic graph kernel, an odd obstruction can diagnose inconsistent
supplied hypotheses or square-root monodromy in a proposed system of local
sections.  There is an important stronger conclusion for the implemented
raw-evidence aggregate.  Every aggregate vertex is bound to one fixed
chart/tube/anchor, every accepted edge is an exact relation between those
fixed anchor representatives, and the deck action is free on an accepted
relation.  Composing the exact relations around a cycle returns the initial
representative, so the XOR must be zero.  Consequently a fully certified
same-anchor aggregate cannot produce an odd cycle.  If its derived graph does,
at least one claimed edge, vertex binding, or checker computation is
inconsistent; monodromy requires a more general local-section or
recentered-handoff model than this same-anchor certificate.

## Exact checker boundary

Stage 1 remains an intentionally combinatorial kernel.  It checks IDs,
endpoints, supplied exact bits, connectivity, all \(\mathbb F_2\) equations,
and the returned primal or dual witness without floating-point arithmetic.
By itself it does **not** establish any edge label: intersection of two
interval boxes is not proof that they enclose the same exact solution.

Stage 2 supplies that missing evidence for the narrower same-anchor case.
`check_planar_lc_exact_overlap_anchor(...)` reruns both tube checkers, binds
their exact zero-error anchors, computes the exact common shifted interval,
compares the full lifted states, and derives the edge bit.  The raw-evidence
atlas aggregate binds positional manifests of unique chart, tube, and overlap
IDs; reruns each vertex and overlap from the supplied certificates rather than
accepting precomputed results; constructs every `PlanarLCGaugeOverlapEdge`
from the derived relation; and only then calls the Stage 1 \(\mathbb F_2\)
checker.  Its lifted, constrained-Newtonian, and all-overlaps-two-sided
statuses remain separate.

This boundary is intentionally narrow.  It does not yet certify a chain of
recentered continuation charts.  Such a chain would compare a target anchor
with a source solution at a different handoff parameter, not with the source
polynomial's own exact zero-error anchor.  A proof-bearing extension therefore
needs exact handoff witnesses and parameter-shift cocycles, including cycle
compatibility of those shifts; overlapping approximate tubes or equality of
their polynomial centers is insufficient.

Stage 3 is a separate opt-in ordinary-to-LC checker.  It retains a raw direct
IVP binding, raw ordinary tube, source ordinary chart, target LC chart and tube,
and transition certificate; it recomputes the direct ordinary validation and
freshly reproduces its full result snapshot from those raw inputs.  It accepts
no prebuilt source result and does not support a continuation-chain source.
Neither it nor Stages 1--2 is consumed by any pre-existing transition
`certified`, atlas `proof_certified`, continuation, or global route.

The legacy ordinary-to-LC checker remains separate.  Its time path is now
narrowed to a direct validated ordinary result with exact parameter/time width
agreement and zero binding time gap, but it still accepts a prebuilt result and
therefore retains that inherited-result trust boundary.  Raw Stage 3 is the
adversarially replayable route.

## Conservative integration roadmap

### Stage 1: exact cocycle kernel

- Implement deterministic spanning-forest solving.
- Return normalized chart bits for a compatible graph.
- Return a verifiable odd closed cycle for an incompatible graph, including
  contradictory parallel edges.
- Test malformed schemas, disconnected graphs, even and odd cycles, the exact
  LC symmetry, and the current two-patch branch-cut atlas.

This stage is implemented.

### Stage 2: independently checked LC-to-LC overlaps

The serialized overlap witness, exact parity derivation, and raw-evidence
aggregate are implemented for fixed zero-error anchors.  Only edges produced
by this checker enter the proof-bearing aggregate.  Projection-only overlaps
and recentered handoffs remain outside the certificate language.

### Stage 3: quantifier-correct branch containment

The raw opt-in checker implements the canonical theorem below.  It retains
universal source coverage and proves

\[
\text{there exists one global }t\text{ with }At=\sigma\text{ such that, for
every branch }v,\ \bigl(\gamma^{t_v}(B_v),t_S\bigr)
\subseteq\mathcal B_{14}(\bar X_*,\varepsilon).
\]

No patch is silently selected or discarded.  The bit for every patch comes
from the same exact graph assignment; unrelated per-patch choices are rejected.
Both global complements are enumerated, and acceptance requires one complement
to send every emitted 13-dimensional representative box, together with the
binding-derived physical-time coordinate, into one target 14-dimensional
initial ball.

### Stage 4: research evaluation

Measure whether the gauge-aware certificates materially reduce branch counts,
tube radii, or failed handoffs on nontrivial close-encounter examples.  If they
do, extract the exact LC gluing theorem, checker semantics, and experiments
into the next review paper.  If they do not, retain the kernel as a precise
negative result and diagnostic; the theorem itself is classical graph
cohomology, so the contribution must come from its validated-numerics use.

## Reusable local-to-global certificate pattern

The reusable pattern is:

1. construct rich local objects;
2. isolate their remaining mismatch in a finite algebraic system;
3. solve that system exactly; and
4. return either a primal solution or a dual parity obstruction.

This is a general certificate-design pattern, not a dependency on the
correctness or priority of any supplied manuscript: rich local analytic
objects are checked first, and only their remaining finite mismatch is passed
to the exact global kernel.

## Explicit nonclaims

This goal does not prove cover completeness, arbitrary-data chart
construction, global termination, spatial KS gauge gluing, total-collision
continuation, interval-rounding soundness, or a general closed solution of the
three-body problem.  Stage 3 proves only the direct-IVP, canonical local
containment theorem below.  It supplies no arbitrary-data construction,
continuation chain, global termination, or global/general solution claim.

## Current implementation status

Stage 1 is implemented in `three_body_symmetry/lc_gauge_gluing.py`.  The result
stores the checked graph, recomputes connectivity, derives rather than trusts
the chart assignment, rechecks every edge equation, and carries the complete
closed walk and edge records for an odd-cycle obstruction.  It rejects
malformed exact-type inputs without promoting truthy stand-ins.

The regression surface includes deterministic trees and cycles,
contradictory parallel edges, disconnected graphs, serialization, forged
result metadata, malformed schemas, fixed-seed multigraph comparison against
brute-force assignments, the production LC state projection/constraint/RHS,
and the existing two-patch branch-cut atlas as a non-proof diagnostic.  The
independent SymPy script now verifies twelve exact projection and gauge
identities, including the full displayed 14-dimensional RHS equivariance.
These checks are part of `scripts/fast_ci.py`.

Stage 2 is implemented in `three_body_symmetry/certificate_language.py` and
`three_body_symmetry/certificate_checker.py`.  The exact-overlap checker stores
the full rational anchors and common shifted interval, derives parity zero for
a renamed identical collision chart, and derives parity one for its antipodal
copy at \(z=0\) from the nonzero \(w\) coordinates.  Focused adversarial tests
cover JSON round trips, mixed \(z/w\) signs, fixed-coordinate and physical-time
mismatches, masses and ordered-pair mismatches, positive initial error,
degenerate common domains, malformed parameter types, duplicate identifiers,
and result/edge tampering.  The raw-evidence aggregate binds every vertex to
one fixed chart/tube/anchor, derives all edges, and invokes the exact graph
kernel.

Stage 3 is implemented in `three_body_symmetry/certificate_checker.py` as
`check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(...)`.  The result
retains all raw direct-IVP evidence, the freshly computed source validation,
the exact source and relative-position boxes, derived patch graph, both global
assignments and gaps, and the selected transformed boxes.  Its `certified`
property replays the raw inputs and requires exact snapshot equality.  Focused
adversarial tests cover source-evidence substitution, nested diagnostic
tampering, branch-cut and singleton complements, malformed schemas, ambiguous
and collision-containing source boxes, and physical time.  The next tasks are
raw continuation-chain evidence, recentered handoff/parameter-shift cocycles,
and nontrivial certified families.  Stage 3 remains outside all pre-existing
transition and `proof_certified` paths.

## Stage 3 theorem: raw canonical branch containment

The universal-quantifier step is implemented for a canonical one/two-patch
grammar reconstructed from raw direct-IVP evidence.  Its inputs are the raw
ordinary IVP binding, raw ordinary a-posteriori tube, source ordinary chart,
target LC chart and tube, and transition certificate.  The checker recomputes
the ordinary validation internally.  It does not accept a prebuilt source
result, and continuation-chain sources are outside this theorem.

Let \(S\) be the complete ordinary-source enclosure at the handoff, and let
\(q=(x,y)\) be the selected relative-position projection, with rectangle
\([x_-,x_+]\times[y_-,y_+]\).  Exact rational chart evaluation and the freshly
checked ordinary tube radius reconstruct the 12-dimensional source box.  The
relative rectangle is derived from those endpoints, and the checker applies
the following classification in the stated order.

1. If \(y_-\geq0\), use one upper-half-plane patch.  This includes
   \(y_-=y_+=0\), so the axis-only tie is resolved as upper.  Otherwise, if
   \(y_+\leq0\), use one lower-half-plane patch.  A rectangle certified in the
   right half-plane likewise uses one principal patch.  Each one-patch case
   emits one lift box \(B_0\), with the two complementary global choices
   \(t_0=0\) and \(t_0=1\).
2. Only a strict negative-cut crossing
   \(y_-<0<y_+\) with \(x_+<0\) uses two patches.  Split it into the convex
   closed sets

   \[
   P_+=S\cap\{y\geq0\},\qquad
   P_-=S\cap\{y\leq0\}.
   \]

   These patches cover \(S\).  Their intersection lies strictly on the
   negative axis.
   The principal LC representatives there are
   \(z_+=(0,+\sqrt{-x})\) and \(z_-=(0,-\sqrt{-x})\), with the corresponding
   velocity lifts also antipodal.  Hence the overlap parity is exactly one.
   The equation \(t_+\mathbin{\mathsf{XOR}}t_-=1\) has precisely the two global
   complementary assignments

   \[
   (t_+,t_-)=(0,1),\qquad (t_+,t_-)=(1,0).
   \]

Any rectangle not decided by this canonical grammar is rejected, as is any
source rectangle containing the selected-pair collision.  Constructor branch
booleans, names, reason strings, and parity metadata are not trusted.  The
checker derives either a singleton graph with no edge or the strict two-patch
graph with its parity-one edge, and requires the exact graph checker to return
a connected compatible primal assignment.

The canonical list is exhaustive on its accepted scope, so both global
complements are enumerated rather than guessed.  Let
\(B_p\subset\mathbb R^{13}\) be every emitted lifted patch box in coordinates
\((z,w,h,R,U,y,V)\).  Let the target LC tube's initial ball be
\(\mathcal B_{14}(\bar X_*,\varepsilon)\) in the corresponding thirteen
coordinates plus physical time.  Ordinary source time is not taken from the
chart's approximate display interval.  From the freshly validated raw binding
the checker derives it exactly as

\[
t_S=t_{\rm init}+s_S-s_{\rm bind}.
\]

Suppose one enumerated global assignment \(t\) satisfies

\[
\bigl(\gamma^{t_p}(B_p),t_S\bigr)
 \subseteq \mathcal B_{14}(\bar X_*,\varepsilon)
\qquad\text{for every emitted patch }p.
\]

The implementation checks every endpoint of all thirteen transformed box
coordinates and the exact source--target physical-time gap against the same
\(L^\infty\) radius \(\varepsilon\).  The separately declared handoff-time cap
remains an additional obligation and cannot replace the fourteenth-coordinate
test.

For every physical state \(X\in S\), patch coverage supplies at least one
\(p\) containing its relative position.  The interval lift construction then
supplies a principal constrained representative \(\ell_p(X)\in B_p\).  The
deck action changes only the signs of \((z,w)\), preserves its physical
projection and algebraic constraint, and the displayed containment gives

\[
\bigl(\gamma^{t_p}\ell_p(X),t_S\bigr)
 \in\mathcal B_{14}(\bar X_*,\varepsilon).
\]

Thus

\[
\exists\,t\;\forall X\in S\;\exists p:\quad
\bigl(\gamma^{t_p}\ell_p(X),t_S\bigr)
 \in\mathcal B_{14}(\bar X_*,\varepsilon),
\]

and every actual source state has a constrained selected lift in the accepted
target tube.  This is an existential statement about the selected physical
lifts, not a claim that every point of a rectangular interval box or every
point of a positive-error target tube is constrained.  No
physical state or emitted patch is silently discarded, the antipodal choice is
represented by the exhaustive global complements, and the bits cannot be
chosen independently patch by patch.  Checking the one or two exhaustive
global complements plus transformed interval-box containment is therefore a
constructive proof of the required universal statement for this restricted
branch grammar.

The three audited prerequisites are now closed without erasing why they were
necessary:

1. **Canonical boundary geometry and parity.**  Closed-upper, closed-lower,
   and right-half-plane cases are singletons; only
   \(y_-<0<y_+\), \(x_+<0\) creates the derived parity-one edge.  Ambiguous and
   collision-containing boxes reject, and metadata cannot extend the grammar.
2. **Exact or outward-enclosed mass arithmetic.**  Pair mass and weighted-center
   ratios are formed in exact rational arithmetic and converted to outward
   binary intervals.  Binary centers, center velocities, and pair energy use
   those outward exact-rational mass-coefficient enclosures, while the existing
   exact mass-ratio gate remains in force.
3. **One 14-dimensional clocked initial ball.**  All lifted endpoints and the
   binding-derived physical-time coordinate use the same target radius.  The
   raw binding is freshly validated, so an approximate chart time interval or
   a permissive binding tolerance cannot silently redefine the exact source
   clock.

The legacy checker is separately narrowed to direct validated inputs with
exact parameter/time width equality and zero binding time gap, but its prebuilt
source result remains an inherited trust boundary.  Raw Stage 3 has no such
prebuilt-result input and is the adversarially checkable route.  Neither route
is consumed by an existing continuation or global proof path, and this local
theorem is not a general or global solution of the three-body problem.
