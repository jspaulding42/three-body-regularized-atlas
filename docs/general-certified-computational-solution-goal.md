# Goal: A General Certified Computational Solution

## Active operational goal

Build a proof-carrying planar three-body integrator with the following contract.
For positive, exactly serialized masses, an exactly serialized planar initial
value problem at physical time \(t_0\), a finite requested time \(T\geq t_0\),
and a requested enclosure tolerance, the system returns exactly one of:

```text
CERTIFIED_TO_T
UNRESOLVED
```

`CERTIFIED_TO_T` carries a replay-checked trajectory atlas covering
\([t_0,T]\) and a rigorous final enclosure in a declared representation.
When a classical Cartesian phase-space enclosure is requested, certification
also requires the terminal chart to prove that \(T\) is collision-free and
that the complete terminal enclosure projects to finite positions and
velocities. A regularized terminal enclosure may instead be returned with an
explicit representation tag; it must not be presented as a finite Cartesian
velocity enclosure at a binary collision.

`UNRESOLVED` identifies the first failed checker obligation and retains the
last rigorously certified frontier region. It is an admissible, honest result.
It does not mean that the trajectory does not exist, that the requested
statement is false, or that no stronger certificate could succeed.

This is a computational notion of a general solution: produce a rigorous
finite-time trajectory enclosure when the supplied evidence is sufficient,
and otherwise expose exactly where certification stopped. It is not a request
for an elementary formula for arbitrary three-body motion.

## Producer and checker boundary

The numerical producer is untrusted. It may choose charts, step sizes,
polynomial orders, precision, recenterings, branch patches, and subdivisions.
It may use ordinary numerical integration, Taylor models, automatic
differentiation, interval Newton methods, or any other heuristic. None of its
success flags is evidence.

The replay checker is small, deterministic, and fail-closed. It accepts only
raw evidence and recomputes every theorem-facing quantity. In particular, a
proof-bearing chain must retain:

- the canonical problem record and its content digest;
- every serialized chart and polynomial coefficient;
- every local tube, residual, derivative, denominator, and tail input;
- every endpoint or time-slice enclosure used by a handoff;
- every coordinate transformation and pair index;
- every physical-clock and local-parameter relation;
- every Levi--Civita branch patch and derived gauge relation;
- the ordered chain and any redundant overlap edges; and
- the checker version, arithmetic backend, precision, and evaluation order.

The checker must not accept prebuilt checker-result objects, inherited
`certified` properties, `source` strings, reason prose, or producer-supplied
gauge bits as hypotheses. Identifiers are navigation aids, not content
bindings. All problem, chart, tube, transition, and chain identifiers must be
nonempty and unique in their namespace, and every reference must additionally
bind to a canonical digest of the exact serialized object. Reordering,
substitution, or mutation of any raw component must change the aggregate
digest or cause replay to reject.

## Exact problem model and arithmetic

The first theorem concerns one planar point initial value problem, not a box of
initial conditions. The canonical input is

\[
P=(m_0,m_1,m_2,q_0,q_1,q_2,v_0,v_1,v_2,t_0,T,\varepsilon),
\]

where the masses are positive and all scalar inputs have an exact serialized
meaning. A finite binary floating-point input is interpreted as the exact
dyadic rational represented by its bit pattern; a canonical rational format
may also be supported. The checker must never silently replace \(P\) by a
nearby problem.

For a selected pair \((i,j)\), quantities such as

\[
M=m_i+m_j,\qquad
\alpha=\frac{m_j}{M},\qquad
\beta=\frac{m_i}{M}
\]

are derived from the exact mass record. They are then evaluated exactly where
possible or enclosed by outward-rounded arbitrary-precision intervals. The
current restriction that a derived mass ratio must itself be exactly
representable in binary64 is not part of the target theorem. An accepted LC
chart must be proved to implement the vector field for the original exact
masses, not for rounded substitutes.

All interval operations are outward rounded. Bounds for elementary functions,
including the exponential in a Gronwall estimate, are part of the audited
arithmetic backend. Certificate-supplied bounds are admissible caps only; the
checker recomputes or independently bounds the underlying expressions.

## Planar chart grammar

The finite planar atlas has four segment kinds:

\[
N,\qquad LC_{01},\qquad LC_{02},\qquad LC_{12}.
\]

An ordinary segment \(N\) encloses the Cartesian Newtonian phase equation on a
tube whose complete position projection has a positive lower bound for all
three pair distances.

An \(LC_{ij}\) segment encloses the analytic fourteen-dimensional planar
Levi--Civita system

\[
X=(z,w,h,R,U,y,V,t),\qquad \frac{dt}{ds}=\rho=|z|^2,
\]

for the selected pair \((i,j)\). Its complete inflated tube must keep the third
body separated from both members of the pair. For the carried physical branch,
the entry evidence must establish the invariant pair-energy constraint

\[
C(X)=2|w|^2-(m_i+m_j)-|z|^2h=0.
\]

The local LC theorem is uniform over its admitted anchor set, while the
Newtonian projection conclusion is conditional on this exact constraint. The
checker must preserve that quantifier distinction: it need not claim that
every point in a rectangular LC tube is physical.

The first all-pair chain grammar uses ordinary bridges:

\[
N\,(N)^*\bigl(\,\to LC_{ij}\to N\,(N)^*\bigr)^*,
\qquad (i,j)\in\{(0,1),(0,2),(1,2)\}.
\]

Thus the same pair may be revisited and different pairs may occur in any
order, but a direct transition between different LC pair charts is not needed
for the first theorem. It must pass through an ordinary region on which every
pair is certified separated. This keeps pair changes explicit and avoids
conflating distinct regularized coordinate systems.

## Local segment obligations

For an ordinary polynomial model \(p\) with anchor set \(A\), parameter
interval \(I\), defect bound \(\delta\), Lipschitz bound \(L\), and bootstrap
radius \(R\), acceptance has the uniform meaning

\[
\forall x_a\in A\;\exists!\,x:I\to\mathbb R^{12}
\quad
x'=F_N(x),\quad x(a)=x_a,\quad
\sup_{s\in I}\|x(s)-p(s)\|_\infty\leq E<R,
\]

with the entire radius-\(R\) position tube collision-free. The checker derives
\(E\) from the checked defect, derivative, domain, and anchor mismatch.

For an LC model \(\bar X\) with anchor set \(A\), acceptance analogously means

\[
\forall X_a\in A\;\exists!\,X:I\to\mathbb R^{14}
\quad
X'=F^{LC}_{ij}(X),\quad X(a)=X_a,\quad
\sup_{s\in I}\|X(s)-\bar X(s)\|_\infty\leq E<R.
\]

The complete inflated tube must remain in the separated-third-body domain.
For every admitted anchor satisfying \(C(X_a)=0\), invariance gives
\(C(X(s))=0\); on every punctured subinterval with \(\rho>0\), the checked LC
projection is the Newtonian solution for the same exact masses.

Entry with \(\rho>0\), or an independently checked nontriviality witness at an
exact collision anchor, proves that the analytic function \(z\) is not
identically zero. Consequently \(t(s)\) is strictly increasing on every
nondegenerate interval even though \(t'(s)=0\) at isolated zeros of \(z\).
This permits a regularized segment to carry the branch through an isolated
selected-pair collision without pretending that Cartesian velocity is defined
there. A separate event record is required before the output may assert that a
collision actually occurred at a particular time.

## Transition obligations and complete containment

Every transition is a theorem about the complete source solution enclosure,
not about two polynomial centers. If \(E_u\) is the full checked source
handoff enclosure, \(A_v\) the target anchor set, and \(\Phi_e\) the checked
coordinate map, the basic obligation is

\[
\Phi_e(E_u)\subseteq A_v.
\]

The inclusion is componentwise, outward rounded, and includes the physical
clock. A tolerance supplied by the producer cannot make a failed inclusion
true.

The supported transitions are:

1. **\(N\to N\).** At a common physical time, the entire source phase box is
   contained in the target anchor set. Local uniqueness then makes the target
   solution the continuation of the carried source branch.
2. **\(N\to LC_{ij}\).** The selected-pair distance is positive on the full
   source box. The checker constructs a complete finite square-root lift
   cover. Formally, for every \(x\in E_u\), at least one emitted patch contains
   a lifted state \(X\) with \(\Pi_{ij}(X)=x\), \(C(X)=0\), and the same
   physical time. No midpoint choice and no silent branch deletion is
   permitted. After one coherent gauge choice, every emitted patch box must be
   contained in the target LC anchor set.
3. **\(LC_{ij}\to N\).** A checked physical-time slice of the source tube must
   have a positive lower bound on \(\rho\). The checker projects the complete
   lifted slice, including its tube error, and requires that entire Cartesian
   enclosure to lie in the ordinary target anchor set.
4. **\(LC_{ij}\to LC_{ij}\) recentering.** This is an optional strengthening,
   useful when one LC segment is too long. It requires a checked full-state
   handoff modulo a derived deck transformation and a compatible local-
   parameter translation. It is not needed for the first all-pair
   ordinary-bridge theorem.

Direct \(LC_{ij}\to LC_{k\ell}\) transitions with different pairs are outside
the first grammar. An interval box from which no single admissible transition
can be proved is returned as unresolved or subdivided by the producer; the
checker never chooses a pair from the midpoint.

## One global physical clock

Every accepted chart represents the same physical time. An ordinary chart has
an exactly checked affine clock

\[
t=s+c.
\]

An LC chart carries \(t(s)\) as a state component and checks
\(t'(s)=|z(s)|^2\), monotonicity, orientation, and outward endpoint image
enclosures. If the left and right endpoint values are enclosed by
\(L=[L_-,L_+]\) and \(R=[R_-,R_+]\), monotonicity proves at least the inner
physical-time image

\[
[L_+,R_-]\subseteq t(I).
\]

Only such proven inner images, checked exact-time slices, and replayed
handoffs contribute to coverage. Declared physical-time metadata does not.

A transition may carry an interval-valued clock shift \(D_e\), but it must be
derived from raw clock enclosures and must contain the one actual shift of the
carried branch. The global checker constructs clock-origin variables
\(b_v\) satisfying every edge relation

\[
b_v-b_u\in D_e.
\]

For a chain this can be propagated forward. If redundant overlaps create a
cycle, the checker must solve the complete finite rational interval constraint
system and exhibit a compatible assignment; merely observing that zero lies
in each interval cycle sum is not sufficient. The resulting clock must be
forward oriented, have no gap, and cover \([t_0,T]\). A physical-time slice
used to enter or leave an LC segment must be isolated from the checked
monotone clock, not inferred from a sampled center trajectory.

## Local-parameter and gauge cocycles

An additive local-parameter shift is meaningful only between charts using the
same parametrized vector field. For a recentering overlap it has the form

\[
s_v=s_u+c_e.
\]

The checker derives \(c_e\) from checked anchors or encloses it from a checked
handoff, constructs compatible parameter-origin variables, and verifies every
edge. Exact shifts must sum to zero around every cycle; interval shifts require
one simultaneous feasible assignment. There is no constant additive
parameter cocycle across an \(N\leftrightarrow LC\) switch, because Newtonian
physical time and LC regularized time have different rates. Such a switch is
bound through the global physical clock instead.

For a fixed pair, the planar LC deck transformation is

\[
\gamma(z,w,h,R,U,y,V,t)=(-z,-w,h,R,U,y,V,t).
\]

Gauge data therefore live in a separate \(\mathbb F_2\) system for each pair.
Every same-pair overlap bit \(\sigma_e\) is derived by comparing the complete
checked lifted relation: \(z\) and \(w\) must flip jointly, and all fixed
components, including physical time, must agree. The checker solves

\[
g_u\oplus g_v=\sigma_e
\]

and verifies that the XOR is zero on every accepted cycle. At an ordinary-to-
LC lift cover, all patches use one coherent graph assignment; unrelated
per-patch sign choices are rejected. Gauges belonging to different selected
pairs are not compared directly. The intervening ordinary chart is their
common physical representation.

## Result semantics

An accepted result has at least the following content:

```text
status: CERTIFIED_TO_T
problem_digest
chain_digest
certified_physical_time_interval
ordered_segment_and_transition_ids
global_clock_witness
pair-indexed_gauge_witnesses
terminal_representation
terminal_state_enclosure
requested_tolerance
checker_and_arithmetic_manifest
```

The terminal enclosure must contain the exact carried state at \(T\), and its
declared norm or componentwise widths must satisfy the requested tolerance.

An unresolved result has at least:

```text
status: UNRESOLVED
problem_digest
requested_target_time
certified_prefix, if any
last_certified_time_interval
retained_frontier_state_region
failed_segment_or_transition_id
first_failed_obligation_code
failure_class
checker_diagnostics
resource_limits_reached, if any
```

The first failure is chosen in a versioned deterministic replay order. The
retained region is checker-derived and contains the carried state at the last
certified frontier. If no initial binding was certified, no prefix theorem is
reported. Failure classes distinguish malformed evidence, failed numerical
bounds, ambiguous chart coverage, unsupported total-collision geometry, and
resource exhaustion, but none is a theorem about what happens after the
frontier.

## Supplied-chain soundness theorem

Let \(P\) be an admissible exact planar point IVP with positive masses and
initially distinct positions, let \(T\geq t_0\) be finite, and let \(C\) be any
finite raw certificate. Subject only to the trusted kernel listed below,

\[
\operatorname{Check}(P,T,\varepsilon,C)=\texttt{CERTIFIED\_TO\_T}
\]

implies that there exists a unique chain-compatible generalized planar
continuation of the bound IVP on \([t_0,T]\), unique in physical projection
and modulo the pair-indexed LC deck transformations. It is a classical
Newtonian solution on every ordinary chart and on every punctured LC chart;
every accepted LC segment is the constrained regularized continuation of that
same branch; all chart handoffs identify the same state at one globally
consistent physical time; the proven chart images cover \([t_0,T]\) without a
gap or reversal; and the returned terminal enclosure contains the state in
its declared representation and satisfies the requested tolerance.

The proof is finite induction. The IVP binding and first local tube select the
initial exact branch. Complete transition containment places that exact
handoff state in the next local theorem's anchor set. Local uniqueness,
LC equivariance, constraint invariance, and punctured projection equivalence
identify the next segment with the same physical branch. The clock and gauge
witnesses make these local identifications globally compatible. Induction over
the finite chain proves coverage and the terminal enclosure.

If the checker returns `UNRESOLVED` with a certified prefix, the same argument
proves only that prefix and containment in the retained frontier region. No
conclusion follows beyond it.

This is a soundness theorem for a supplied finite certificate. It does not say
that a producer can find such a certificate for every input.

## Trusted kernel

The intended trusted computing base consists of:

- strict parsing of a versioned canonical format and content digests;
- exact integer and rational arithmetic;
- audited arbitrary-precision outward interval arithmetic and elementary
  functions;
- interval polynomial evaluation, differentiation, and automatic
  differentiation;
- explicit Newtonian and planar LC vector fields, constraints, lifts,
  projections, and deck action;
- a small proved a-posteriori ODE enclosure lemma;
- checked denominator and collision-free-domain bounds;
- finite branch-cover, interval-containment, clock-feasibility, parameter-
  cocycle, and \(\mathbb F_2\) gauge kernels; and
- finite chain induction and terminal evaluation.

NumPy binary64 calculations, sampled residuals, heuristic chart selection,
producer metadata, mutable object identity, and cached checker-result objects
are not trusted theorem inputs.

## Milestones

### Milestone 1: exact problem and clock binding

Define the canonical point-IVP record, exact mass semantics, object digests,
unique identities, and one replayed physical-clock language. Ordinary chart
time must be derived as \(t=s+c\). Coverage must consume proven clock images,
not free interval metadata. Establish deterministic `CERTIFIED_TO_T` and
structured `UNRESOLVED` result schemas.

### Milestone 2: raw ordinary replay chain

Build a new top-level checker that accepts only raw IVP, ordinary charts,
tubes, and transitions. It reruns every local check, proves complete handoff
containment, propagates one global clock, computes the terminal enclosure at
the exact requested \(T\), and returns `CERTIFIED_TO_T` only when the requested
tolerance is met. This is the first vertical slice and the first publication-
quality theorem.

### Milestone 3: one raw ordinary--LC--ordinary passage

Consume raw ordinary source evidence rather than a prebuilt source result.
Add quantifier-correct complete LC lift covers, derived gauge assignments,
constraint-preserving entry, monotone LC physical-time images or isolated time
slices, punctured complete projection on exit, and raw content binding for the
entire aggregate. This milestone must close the current gap between a checked
LC endpoint projection and an exact continuation result.

### Milestone 4: repeated all-pair planar switching

Instantiate the LC construction for \((0,1)\), \((0,2)\), and \((1,2)\), and
accept arbitrary finite repetitions through certified ordinary bridges. Add
pair-indexed gauge graphs, optional same-pair LC recentering, clock and
parameter cocycle checks, adaptive precision, and subdivision diagnostics.
The theorem remains supplied-chain soundness; producer termination remains
open.

### Milestone 5: initial-condition families and wrapping control

Only after the point-IVP chain is sound, add parametric Taylor models,
Krawczyk or interval-Newton validation, Lohner/QR recentering, adaptive
subdivision, and a coverage ledger proving that accepted and unresolved child
regions cover the complete input box.

### Milestone 6: additional endpoint regimes

Treat total collision, escape, and long-time structural certificates as
separate theorem layers with their own chart and stop semantics. A certified
total-collision stop is preferable to an invented continuation. Spatial KS
charts and all-time statements require separate projection, gauge, and
asymptotic theorems.

## Explicit nonclaims

Completion of the first four milestones would not establish:

- an elementary or finite-expression closed form for the three-body problem;
- that the producer succeeds or terminates for every exact input;
- validated coverage of arbitrary boxes of initial conditions;
- a complete classifier for ambiguous simultaneous close encounters;
- continuation through total collision;
- escape or other infinite-time asymptotics;
- non-Zeno completeness of an automatically generated infinite chart atlas;
- spatial three-body motion or KS gauge gluing;
- an all-time trajectory theorem; or
- unconditional correctness without an audited outward-rounding and ODE-lemma
  kernel.

The first meaningful success is smaller and stronger: a replay checker whose
acceptance proves one exact finite planar IVP has been carried, without gaps or
branch loss, through a finite raw chain of ordinary and pair-indexed LC charts
to a rigorous enclosure at the requested time.
