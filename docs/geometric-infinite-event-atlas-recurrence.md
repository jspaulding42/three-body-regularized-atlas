# Geometric Infinite-Event Atlas Recurrence

The finite compact-atlas theorems cover any compact interval with only finitely
many separated binary collisions and identity-selector total collisions. This
note records the corresponding infinite-event recurrence when the event atlas is
already known to have a geometric compact-time tail.

This is not an arbitrary-data classification theorem. It is a conditional
all-future recurrence theorem: if a solution has an infinite sequence of
verified local event charts whose compact-time shells have summable geometry and
summable verification budgets, then the lift/construct/project/verify atlas has
a finite all-future budget.

The executable theorem assembler preserves that distinction.  A closed
`geometric_infinite_event_tail` recurrence now leaves
`geometric_event_regime_membership_from_initial_data` open until a separate
classifier derives the shell isolation and chart-family coverage assumptions
from the initial-value branch itself.

## Setup

Let `u in [u_0,1)` be a compact physical-time parameter, for example
`u=tanh(lambda t)` on a future tail. Let:

```text
delta_n = delta_0 theta^n,        0 < theta < 1,
S_n = [1-delta_n, 1-delta_(n+1)]
```

be endpoint shells exhausting `u=1`. Assume an infinite event atlas is already
constructed on `[u_0,1)` with the following properties.

1. Every compact subinterval `[u_0,U]` with `U<1` intersects only finitely many
   event neighborhoods.
2. In each shell `S_n`, the event neighborhoods are pairwise disjoint and are
   separated from the shell boundary after shrinking. Their total compact-time
   width satisfies:

```text
W_n <= W_0 theta^n.
```

3. Each event neighborhood is one of the verified local chart types:

```text
separated-binary Levi-Civita chart,
identity-selector Fuchsian-log total-collision chart.
```

The complement of the event neighborhoods in `S_n` is covered by finitely many
ordinary Taylor charts. Neighboring projected charts agree at every
noncollision handoff state. The separated-binary and identity-selector charts
provide the selected lifted continuation through their singular centers.

## Event Isolation and Shell Count

The first setup assumption above can be proved from uniform local isolation
data. This is the compact-prefix part of the infinite-event recurrence.

For a separated binary collision, the Levi-Civita chart has:

```text
z(s) = zeta(0)s + O(s^2),        |zeta(0)|^2 = M/2 > 0.
```

Thus `z=0` is a simple zero in regularized time. After shrinking the chart,
the selected pair collides only at `s=0`, and the third-body separation plus
the other two pair distances stay nonzero on the punctured chart. For an
identity-selector total collision:

```text
q_i(tau) = tau^2 S_i(tau),       S_i(0)-S_j(0) != 0,
```

so all three pair distances vanish simultaneously only at `tau=0` inside a
small enough Fuchsian-log chart. In both cases the singular event has a
punctured neighborhood free of any other collision event.

Now fix one compact-time shell `S_n` of length:

```text
Delta_n = delta_n - delta_(n+1) = delta_0(1-theta)theta^n.
```

Assume every event center `c` in `S_n` has an event-free compact-time isolation
interval:

```text
(c-h_{n,c}, c+h_{n,c}),          h_{n,c} >= alpha Delta_n,
```

and assume the event centers are at least `alpha Delta_n` away from the shell
boundary after assigning boundary events to the adjacent shell. Then the
intervals:

```text
[c-alpha Delta_n/2, c+alpha Delta_n/2]
```

for distinct event centers in `S_n` are disjoint subintervals of `S_n`. Hence
the number `E_n` of events in shell `S_n` satisfies:

```text
E_n alpha Delta_n <= Delta_n,
E_n <= floor(1/alpha).
```

If boundary assignment is not used and an event may lie within
`alpha Delta_n` of either shell endpoint, enlarge the bound by two:

```text
E_n <= floor(1/alpha)+2.
```

The same argument on any compact prefix `[u_0,U]` gives finiteness of the event
set there whenever that prefix has a positive minimum isolation radius
`h_* > 0`:

```text
E([u_0,U]) <= floor((U-u_0)/h_*) + 2.
```

Thus finite compact-prefix composition does not need to be assumed separately
once uniform local isolation is proved on that prefix. For the all-future
geometric recurrence, an isolation scale proportional to the shell width gives
a uniform per-shell event-count envelope:

```text
M_* = floor(1/alpha)+2.
```

This still leaves a dynamical task: prove such an `alpha` from the actual
future branch. The local binary and total-collision charts prove that every
individual verified event is isolated; the recurrence needs a uniform shell
isolation envelope.

## Uniform Event Isolation Supplies Chart-Family Counts

The primitive budget theorem needs separate count bounds for ordinary gap
atlases, separated-binary charts, and total-collision charts. These follow from
the same isolation packing estimate.

Let `E_n` be the total number of event centers in shell `S_n`, and let:

```text
L_n = number of separated-binary events in S_n,
T_n = number of total-collision events in S_n.
```

Because every event is one of those two types:

```text
L_n + T_n = E_n.
```

Therefore the shell isolation estimate gives:

```text
L_n <= M_*,
T_n <= M_*,
M_* = floor(1/alpha)+2
```

with the boundary-inclusive convention above. After the event neighborhoods in
`S_n` are chosen disjoint, their complement in the shell is a union of at most:

```text
E_n + 1
```

connected open intervals. Each connected complement component is a
collision-free compact-time gap after endpoint trimming, so the ordinary
compact Taylor-cover theorem supplies one finite ordinary gap atlas on that
component. Hence the ordinary gap-atlas count satisfies:

```text
G_n <= E_n + 1 <= M_* + 1.
```

Thus uniform event isolation supplies the primitive chart-family counts:

```text
M_L = M_*,
M_T = M_*,
M_O = M_* + 1.
```

If boundary events are assigned to adjacent shells before counting, replace
`M_*` by `floor(1/alpha)` throughout. If sharper type-specific information is
available, one can use smaller `M_L` or `M_T`, but the uniform bound above is
enough for the all-future recurrence.

## Geometric Shell Isolation Closes The Count Recurrence

The previous paragraph uses an abstract shell fraction `alpha`. In the endpoint
recurrence this fraction can be derived from geometric shell data.

Let:

```text
S_n = [1-delta_0 theta^n, 1-delta_0 theta^(n+1)],
Delta_n = |S_n| = delta_0(1-theta)theta^n.
```

Assume that every event center `c` in `S_n` has a compact-time event-free
radius and shell-boundary clearance with the same geometric scale:

```text
h_{n,c} >= H_0 theta^n,
dist(c, boundary(S_n)) >= B_0 theta^n.
```

Set:

```text
alpha = min(H_0,B_0) / (delta_0(1-theta)).
```

If `alpha>1`, replace it by `1` for the packing estimate. Then:

```text
h_{n,c} >= alpha Delta_n,
dist(c, boundary(S_n)) >= alpha Delta_n.
```

Distinct event centers in `S_n` are separated by at least `alpha Delta_n`,
because an event-free interval of radius `h_{n,c}` around one center cannot
contain another event center. The same packing argument gives:

```text
E_n <= floor(1/alpha)+2,
M_O=floor(1/alpha)+3,
M_L=M_T=floor(1/alpha)+2.
```

Combining these counts with the primitive Cauchy inputs from the later section
gives the all-future recurrence immediately:

```text
sum_(n>=N) B^X_n
 <= sum_K M_K A^X_K (r^X_K)^N/(1-r^X_K)
 <= B^X_0 r_X^N/(1-r_X).
```

This is now executable as
`derive_geometric_shell_event_isolation(...)`, followed by
`derive_all_future_event_budget_from_geometric_shell_isolation(...)` in
`three_body_symmetry/event_recurrence.py`. The constructor still assumes the
geometric local isolation scales `H_0 theta^n` and `B_0 theta^n`; it removes
the remaining free shell-count boolean once those scales are proved by the
ordinary, binary, or total-collision chart constructors.

## Compact Accumulation Dichotomy

The converse is also useful. Let `C` be a compact subinterval of the
compact-time domain away from the endpoint `u=1`, and suppose the event centers
`c_k in C` are all covered by the local chart types above. For each event, let
`h_k>0` be the largest certified compact-time radius for which the punctured
interval `(c_k-h_k,c_k+h_k)\{c_k}` is free of all other collision events and the
same local chart remains valid.

If infinitely many event centers lie in `C`, then:

```text
inf_k h_k = 0.
```

Indeed, if `h_k >= h_* > 0`, the disjoint intervals
`[c_k-h_*/2,c_k+h_*/2]` for a maximal separated subfamily have total length at
least `h_*` per event and all lie in a fixed bounded enlargement of `C`. The
packing bound above would allow only finitely many centers. Hence an infinite
compact-prefix event set forces the certified isolation radii to degenerate.

For the verified local charts, such degeneration has concrete meanings. In a
separated-binary chart, `|zeta(0)|^2=M/2` is fixed by the masses, so the simple
zero of `z` cannot degenerate for positive masses. Loss of binary isolation can
therefore only come from the chart domain shrinking: the third-body separation
or one of the noncolliding pair distances tends to zero, the branch atlas loses
a uniform split, or the analytic regularized Cauchy radius tends to zero. The
first possibility is exactly approach to a total-collision or binary-degenerate
cluster not covered by the separated-binary hypotheses.

In an identity-selector total-collision chart, the event is isolated while the
limiting shape remains collision-free:

```text
min_{i<j}|S_i(0)-S_j(0)| >= sigma_* > 0
```

and the Fuchsian-log chart has a positive uniform analytic radius. Therefore
compact-prefix accumulation through identity-selector total collisions also
forces degeneration of one of those quantities: the limiting shape approaches a
binary shape stratum, the selected Fuchsian-log radius tends to zero, or the
incoming branch leaves the finite-energy central-target class.

Thus a finite-time accumulation of verified events is not another ordinary
case of the recurrence. It is exactly the failure of the uniform isolation
hypotheses needed by the infinite-event theorem. The remaining global problem
is to prove that arbitrary-data dynamics either preserve these isolation
constants on every compact prefix or converge to one of the separately
classified degenerate regimes.

## Shell Budget Hypothesis

For each shell `S_n`, collect the complete verification budget of all ordinary
gaps and event neighborhoods in that shell. Write the value, first-jet,
lifted-residual, and projected physical-residual shell budgets as:

```text
B^V_n, B^J_n, B^E_n, B^R_n.
```

Assume there are constants:

```text
B^V_0, B^J_0, B^E_0, B^R_0 >= 0
0 <= r_V, r_J, r_E, r_R < 1
```

such that for every `n>=0`:

```text
B^V_n <= B^V_0 r_V^n,
B^J_n <= B^J_0 r_J^n,
B^E_n <= B^E_0 r_E^n,
B^R_n <= B^R_0 r_R^n.
```

The constants may already include a uniform bound on the number of event charts
per shell. Equivalently, if there are at most `M_*` event neighborhoods and
`G_*` ordinary gap atlases per shell, and each individual local chart budget is
bounded by a geometric envelope, then the shell-budget hypothesis holds after
multiplying the first-shell constants by `M_*+G_*`.

## Local Chart Envelopes Imply Shell Budgets

The shell-budget hypothesis follows from more primitive local estimates. For
each component

```text
X in {V,J,E,R}
```

suppose shell `S_n` contains at most:

```text
G_* ordinary gap Taylor atlases,
B_* separated-binary Levi-Civita charts,
T_* total-collision Fuchsian-log charts.
```

Assume each individual chart of each kind has a component budget bounded by a
geometric envelope:

```text
ordinary gap:          O^X_0 (r^X_O)^n,
separated binary:      L^X_0 (r^X_L)^n,
total collision:       T^X_0 (r^X_T)^n,
```

with all three ratios strictly smaller than one. Let

```text
r_X = max(r^X_O, r^X_L, r^X_T),
B^X_0 = G_* O^X_0 + B_* L^X_0 + T_* T^X_0.
```

Then the complete shell budget satisfies:

```text
B^X_n
 <= G_* O^X_0 (r^X_O)^n
  + B_* L^X_0 (r^X_L)^n
  + T_* T^X_0 (r^X_T)^n
 <= B^X_0 r_X^n.
```

Thus the global shell-budget hypothesis is reduced to three checkable local
inputs: a per-shell count envelope for each chart type, a first-shell budget
bound for each chart type, and a geometric decay ratio for each chart type.
The infinite-event theorem may then use:

```text
sum_(n>=N) B^X_n <= B^X_0 r_X^N/(1-r_X).
```

For the automatic positive-mass three-body zero-angular case, the
`total collision` row is the automatic identity-selector Fuchsian-log chart.
No extra selector-limit datum enters the arithmetic; its local residual and
projection budgets are simply part of `T^X_0`.

## Uniform Cauchy Tails Give The Local Geometric Envelopes

The previous reduction leaves one analytic task for each chart family: prove
the individual local envelopes. A standard Cauchy-tail argument gives exactly
such an envelope once the retained order grows and the chart evaluation stays a
fixed fraction inside its analytic disk.

Fix one chart family `K` and one budget component

```text
X in {V,J,E,R}.
```

Use the chart's natural lifted variable `zeta`: ordinary compact time for
collision-free Taylor charts, Levi-Civita regularized time for separated
binary charts, and the Fuchsian or Fuchsian-log collision variable for total
collision charts after the finite selector has been fixed. Suppose every
chart of this family in shell `S_n` has:

```text
|zeta-zeta_0| <= h_n,        h_n/R_n <= sigma_X < 1,
```

where `R_n` is a certified analytic radius for the lifted chart. Suppose the
component being estimated has a Cauchy majorant, including any derivative,
projection, or residual multiplier needed for that component:

```text
C^X_n <= C^X_0 Lambda_X^n,        Lambda_X >= 1,
```

and that the retained order satisfies:

```text
p_n >= p_0 + d n,        d >= 1.
```

The Cauchy estimate for the discarded tail gives:

```text
tail^X_n
  <= C^X_n sigma_X^(p_n+1)/(1-sigma_X)
  <= C^X_0 sigma_X^(p_0+1)/(1-sigma_X)
     (Lambda_X sigma_X^d)^n.
```

Thus this chart family has a local geometric envelope:

```text
K^X_0 = C^X_0 sigma_X^(p_0+1)/(1-sigma_X),
r^X_K = Lambda_X sigma_X^d,
```

provided:

```text
r^X_K < 1.
```

This is the missing bridge between local analytic chart construction and the
shell-budget hypothesis. The value and first-jet components use the usual
Cauchy estimates for a function and its derivative; lifted-equation residuals
use the coefficient recurrence, so all retained rows vanish and only the
discarded analytic tail remains; projected Newton residuals add the projection
and time-change multipliers, which are part of `C^X_n`.

Consequently, for ordinary gaps, separated binaries, and automatic
identity-selector total collisions, it is enough to prove:

```text
fixed step-to-radius ratios sigma_X < 1,
component majorant growth C^X_n <= C^X_0 Lambda_X^n,
retained-order growth p_n >= p_0 + d n,
Lambda_X sigma_X^d < 1.
```

The shell-budget theorem then applies with the local constants `K^X_0` and
ratios `r^X_K`. This still does not prove that arbitrary dynamics satisfy the
majorant-growth and step-to-radius hypotheses; it converts those dynamical
state estimates into the exact geometric local envelopes required by the
all-future recurrence.

## Primitive Cauchy Inputs Close The All-Future Budget

Combining the last two reductions gives a single all-future recurrence
theorem with no intermediate budget hypothesis.

Let the chart families be:

```text
K in {O,L,T}
```

for ordinary gap Taylor atlases, separated-binary Levi-Civita charts, and
automatic identity-selector total-collision Fuchsian-log charts. Assume shell
`S_n` contains at most `M_K` charts of family `K`. For each budget component

```text
X in {V,J,E,R}
```

assume every such chart has primitive Cauchy data:

```text
C^X_{K,n} <= C^X_{K,0} (Lambda^X_K)^n,
h^X_{K,n}/R^X_{K,n} <= sigma^X_K < 1,
p^X_{K,n} >= p^X_{K,0} + d^X_K n,
d^X_K >= 1.
```

Here `C` includes the component's Cauchy majorant and all projection or
time-change multipliers, `h/R` is the evaluation-to-radius ratio in the
family's lifted variable, and `p` is the retained order. Define:

```text
A^X_K = C^X_{K,0} (sigma^X_K)^(p^X_{K,0}+1)/(1-sigma^X_K),
r^X_K = Lambda^X_K (sigma^X_K)^(d^X_K).
```

If:

```text
r^X_K < 1       for every K and X,
```

then every chart of family `K` in shell `S_n` has component tail at most:

```text
A^X_K (r^X_K)^n.
```

Therefore the full shell budget satisfies:

```text
B^X_n <= sum_K M_K A^X_K (r^X_K)^n.
```

With:

```text
B^X_0 = sum_K M_K A^X_K,
r_X = max_K r^X_K,
```

the all-future tail from shell `N` has the explicit closed form:

```text
sum_(n>=N) B^X_n
 <= B^X_0 r_X^N/(1-r_X).
```

Equivalently, keeping the sharper family-wise decomposition gives:

```text
sum_(n>=N) B^X_n
 <= sum_K M_K A^X_K (r^X_K)^N/(1-r^X_K).
```

The second expression is often smaller; the first is the scalar shell
recurrence used by the global theorem. This proves the all-future verification
budget once the primitive dynamical estimates are known: per-shell chart
counts, Cauchy majorant growth, fixed step-to-radius ratios, and retained-order
growth. Those are now the remaining dynamical inputs, rather than an abstract
shell-budget assumption.

This reduction is now executable in `three_body_symmetry/event_recurrence.py`.
`derive_chart_family_counts_from_event_isolation(...)` implements the packing
bound

```text
M_*=floor(1/alpha)+2,
M_O=M_*+1,       M_L=M_T=M_*,
```

for boundary-inclusive shells. `construct_primitive_cauchy_tail_input(...)`
checks the local inequality

```text
r^X_K = Lambda^X_K (sigma^X_K)^(d^X_K) < 1,
```

and records

```text
A^X_K = C^X_{K,0}(sigma^X_K)^(p^X_{K,0}+1)/(1-sigma^X_K).
```

`derive_all_future_event_budget_from_primitive_cauchy_inputs(...)` then forms
the family-wise bound

```text
sum_K M_K A^X_K (r^X_K)^N/(1-r^X_K)
```

and the scalar bound

```text
B^X_0 r_X^N/(1-r_X),
```

for each component `X in {V,J,E,R}`. The certificate also checks observed
shell charts directly: if an observed chart in shell `n` has

```text
C_n <= C_0 Lambda^n,       h_n/R_n <= sigma,
p_n >= p_0+dn,
```

then its direct Cauchy tail is bounded by `A r^n`. Thus this constructor is not
a new boolean witness; it is the arithmetic proof that primitive local Cauchy
inputs imply a finite all-future event-atlas verification budget. It still
does not prove the primitive inputs for arbitrary dynamics.

## Uniform Noncollision State Envelopes Supply Ordinary Cauchy Inputs

One chart family can be closed directly. Suppose every ordinary gap chart in
all future shells lies in a uniform noncollision state envelope:

```text
min_{i<j}|q_i-q_j| >= d_* > 0,
max_{i<j}|q_i-q_j| <= D_*,
max_i |v_i| <= V_*.
```

For the Sundman-time Cauchy majorant with distance-power `p_s`, total mass
`M`, and the implementation's position radius `rho=d_*/10`, the state-envelope
lemma gives the uniform bounds:

```text
G_* = (D_*+d_*/5)^(3p_s),
A_* = M (D_*+d_*/5) / (((14/25)^(3/2)) d_*^3),
nu_* = sqrt(A_* rho),
R_* = min(
  rho/(G_*(V_*+nu_*)),
  sqrt(rho/A_*)/G_*
).
```

Because `d_*>0`, all these constants are finite and `R_*>0`. Thus every
ordinary gap chart has an analytic Sundman-time disk of radius at least `R_*`
and a component Cauchy majorant bounded by a constant depending only on:

```text
d_*, D_*, V_*, M, p_s
```

and on the fixed projection/time-change factors used for the component. Write
that constant as `C^X_{O,0}` for `X in {V,J,E,R}`.

Choose a fixed `sigma^X_O` with `0<sigma^X_O<1` and accept ordinary chart
steps only inside:

```text
h^X_{O,n} <= sigma^X_O R_*.
```

Then the primitive ordinary-gap data are:

```text
Lambda^X_O = 1,
C^X_{O,n} <= C^X_{O,0},
h^X_{O,n}/R^X_{O,n} <= sigma^X_O,
p^X_{O,n} >= p^X_{O,0}+d^X_O n.
```

The ordinary local ratio is therefore:

```text
r^X_O = (sigma^X_O)^(d^X_O) < 1.
```

Consequently every uniformly noncollision ordinary gap family has the local
geometric envelope required by the primitive all-future budget theorem:

```text
A^X_O =
  C^X_{O,0} (sigma^X_O)^(p^X_{O,0}+1)/(1-sigma^X_O),
tail^X_{O,n} <= A^X_O (r^X_O)^n.
```

This proves the ordinary-gap Cauchy input in the uniformly separated and
bounded-speed regime. It does not prove the corresponding primitive inputs for
separated-binary or total-collision charts, and it does not apply to unscaled
escape regimes where the Sundman Cauchy radius is known to decay.

## Uniform Separated-Binary Envelopes Supply Levi-Civita Cauchy Inputs

The separated-binary chart has the same primitive-input structure once its
regularized variables stay in a uniform separated-third-body envelope. For a
fixed colliding pair, write the Levi-Civita state as:

```text
z, zeta, h, R, Rdot, y, ydot,
```

where `z^2` is the pair separation, `h` is the pair-energy variable, `R` is the
binary center, and `y` is the third-body offset from that center. Let

```text
alpha = m_2/(m_1+m_2),       beta = m_1/(m_1+m_2).
```

Assume every separated-binary chart in all future shells satisfies uniform
state bounds:

```text
|z| <= Z_*,
|zeta| <= Xi_*,
|h| <= H_*,
|R| <= C_*,
|Rdot| <= W_*,
|y| <= Y_*,
|ydot| <= U_*,
```

and choose positive majorant radii:

```text
r_z, r_zeta, r_h, r_R, r_Rdot, r_y, r_ydot.
```

The only possible singularity in the regularized binary vector field is loss
of separation from the third body. Suppose the chosen radii preserve the
uniform lower bounds:

```text
|y + alpha z^2| - r_y - alpha(2Z_*r_z+r_z^2) >= gamma_* > 0,
|y - beta  z^2| - r_y - beta (2Z_*r_z+r_z^2) >= gamma_* > 0.
```

Then all third-body inverse-cube fields are analytic in the whole majorant
polydisk. With:

```text
U_3 = Y_* + r_y + max(alpha,beta)(Z_*+r_z)^2,
P_* = (m_3+m_1+m_2) U_3/gamma_*^3,
A^R_* = m_3 P_*,
A^y_* = (m_1+m_2+m_3) P_*,
```

the regularized binary RHS is bounded by finite constants. In particular, with

```text
Z = Z_*+r_z,
Xi = Xi_*+r_zeta,
H = H_*+r_h,
W = W_*+r_Rdot,
U = U_*+r_ydot,
rho = Z^2,
```

one may take:

```text
F_z      = Xi,
F_zeta   = (1/2)H Z + (1/2)rho Z P_*,
F_h      = 2Z Xi P_*,
F_R      = rho W,
F_Rdot   = rho A^R_*,
F_y      = rho U,
F_ydot   = rho A^y_*.
```

Therefore the lifted analytic self-map radius has the positive lower bound:

```text
R_L = min(
  r_z/F_z,
  r_zeta/F_zeta,
  r_h/F_h,
  r_R/F_R,
  r_Rdot/F_Rdot,
  r_y/F_y,
  r_ydot/F_ydot
),
```

omitting any quotient whose denominator is zero. Since every denominator is
finite and every radius is positive, `R_L>0`. The component Cauchy majorants
are bounded by constants depending only on the displayed uniform envelope,
the chosen radii, and the masses; call them `C^X_{L,0}`.

Choose fixed chart steps with:

```text
h^X_{L,n} <= sigma^X_L R_L,        0 < sigma^X_L < 1.
```

Then the separated-binary primitive Cauchy data are:

```text
Lambda^X_L = 1,
C^X_{L,n} <= C^X_{L,0},
h^X_{L,n}/R^X_{L,n} <= sigma^X_L,
p^X_{L,n} >= p^X_{L,0}+d^X_L n,
```

and hence:

```text
r^X_L = (sigma^X_L)^(d^X_L) < 1,
A^X_L =
  C^X_{L,0} (sigma^X_L)^(p^X_{L,0}+1)/(1-sigma^X_L).
```

This closes the separated-binary local Cauchy input whenever the global branch
supplies a uniform Levi-Civita envelope and a uniform third-body separation
margin in the lifted chart. It still leaves the dynamical task of proving such
uniform envelopes along arbitrary infinite binary-event tails, and it does not
address total-collision Fuchsian-log Cauchy inputs.

## Binary-Only Nonzero-Angular All-Future Constructor

The first global theorem target uses the same recurrence in a narrower regime:
positive masses, nonzero centered angular momentum, and no total-collision
chart family. The nonzero-angular invariant certificate supplies the analytic
reason for deleting the total-collision row. If a total collision occurred, the
centered angular momentum would vanish at the collision limit, and angular
momentum conservation would force the same zero value on the incoming branch.
Thus nonzero centered angular momentum excludes triple collision, leaving only
ordinary collision-free gaps and separated binary events.

The constructor `derive_nonzero_angular_event_recurrence_from_uniform_envelopes(...)`
implements the following scoped theorem. Assume:

```text
m_i > 0,
L_centered != 0,
delta_n = delta_0 theta^n,                 0 < theta < 1,
h_{n,c} >= H_0 theta^n,
dist(c,boundary(S_n)) >= B_0 theta^n,
```

for every binary event center `c` in shell `S_n`, and assume the ordinary gaps
and separated-binary charts satisfy the two uniform state envelopes proved in
the previous sections. There are two executable variants. The single-family
constructor assigns all binary events to one separated-binary Levi-Civita
family whose lifted envelope dominates the events being counted. The all-pair
constructor uses separate rows:

```text
L_01, L_02, L_12
```

with a separate uniform Levi-Civita envelope for each selected pair.

Set:

```text
Delta_n = delta_0(1-theta)theta^n,
alpha = min(H_0,B_0)/(delta_0(1-theta)),
M_* = floor(1/min(alpha,1))+2.
```

The shell-packing proof gives:

```text
number of binary events in S_n <= M_*,
number of ordinary gap atlases in S_n <= M_*+1.
```

Because triple collision is excluded, the chart-family counts are exactly:

```text
M_O = M_*+1,       ordinary_gap_taylor,
M_L = M_*,         each separated_binary_levi_civita row,
M_T = 0.           no total-collision row
```

Equivalently, the implementation calls
`derive_geometric_shell_event_isolation(..., total_collision_kind=None)`.
For the single-family constructor, `chart_family_counts` contains only
`ordinary_gap_taylor` and `separated_binary_levi_civita`. For the all-pair
constructor, it contains:

```text
ordinary_gap_taylor,
separated_binary_levi_civita_01,
separated_binary_levi_civita_02,
separated_binary_levi_civita_12.
```

The uniform ordinary-gap envelope derives primitive Cauchy inputs:

```text
C^X_{O,n} <= C^X_{O,0},
h^X_{O,n}/R^X_{O,n} <= sigma^X_O < 1,
p^X_{O,n} >= p^X_{O,0}+d^X_O n,
r^X_O = (sigma^X_O)^(d^X_O) < 1.
```

Each uniform separated-binary Levi-Civita envelope derives the analogous binary
inputs:

```text
C^X_{L,n} <= C^X_{L,0},
h^X_{L,n}/R^X_{L,n} <= sigma^X_L < 1,
p^X_{L,n} >= p^X_{L,0}+d^X_L n,
r^X_L = (sigma^X_L)^(d^X_L) < 1.
```

Here `X` ranges over the value, first-jet, lifted-residual, and projected
physical-residual budgets. The Cauchy tail theorem then gives the per-chart
estimates:

```text
tail^X_{O,n} <= A^X_O (r^X_O)^n,
A^X_O = C^X_{O,0}(sigma^X_O)^(p^X_{O,0}+1)/(1-sigma^X_O),

tail^X_{L,n} <= A^X_L (r^X_L)^n,
A^X_L = C^X_{L,0}(sigma^X_L)^(p^X_{L,0}+1)/(1-sigma^X_L).
```

Multiplying by the shell count envelopes gives the full binary-only shell
budget. In the single-family case:

```text
B^X_n
 <= (M_*+1) A^X_O (r^X_O)^n
  + M_*     A^X_L (r^X_L)^n.
```

In the all-pair case:

```text
B^X_n
 <= (M_*+1) A^X_O (r^X_O)^n
  + sum_(ij in {01,02,12}) M_* A^X_{L_ij} (r^X_{L_ij})^n.
```

Therefore every future tail after a checked prefix through shell `N-1` is
summable. The all-pair family-wise bound is:

```text
sum_(n>=N) B^X_n
 <= (M_*+1) A^X_O (r^X_O)^N/(1-r^X_O)
  + sum_(ij in {01,02,12})
      M_* A^X_{L_ij} (r^X_{L_ij})^N/(1-r^X_{L_ij}).
```

The scalar form used by the global theorem follows by taking:

```text
B^X_0 = (M_*+1)A^X_O + sum_ij M_*A^X_{L_ij},
r_X = max(r^X_O,r^X_{L_01},r^X_{L_02},r^X_{L_12}) < 1,
sum_(n>=N) B^X_n <= B^X_0 r_X^N/(1-r_X).
```

Thus the nonzero-angular binary-only constructors prove all-future recurrences
from explicit quantitative hypotheses. The all-time theorem uses these as
one-sided endpoint certificates: they must be supplied for the original future
endpoint and again for the time-reversed flow, which certifies the original
past endpoint. For arbitrary labeled data, each side must include all three
separated-binary primitive Cauchy rows unless a separate classifier proves that
only a subset of binary pairs can occur. A single-pair row remains a scoped
subregime and is reported as missing `nonzero_angular_all_pair_binary_coverage`
at theorem level. The all-pair executable path is
`derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes(...)`,
wrapped by
`construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(...)`.

This is a real theorem for the stated uniform-envelope regime. It is not yet
the arbitrary-data all-time theorem: the unresolved step is proving that an
arbitrary nonzero-angular solution either enters such binary-only uniform
envelopes, enters an already certified escape/scattering endpoint regime, or
belongs to another classified regime with its own constructor.

## Finite Fuchsian-Log Charts Supply Total-Collision Cauchy Inputs

The automatic total-collision row of the event recurrence needs the same
primitive input:

```text
C_n <= C_0 Lambda^n,
h_n/R_n <= sigma < 1,
p_n >= p_0+dn.
```

For a finite selected Fuchsian-log chart this can be proved directly on
punctured shells. Let the collision-variable shell centers be:

```text
tau_n = tau_0 theta^n,          0<theta<1,
```

and take local analytic disks:

```text
|tau-tau_n| <= a tau_n,         0<a<1.
```

The disk avoids `tau=0`, so one branch of `log(tau)` is analytic there. On the
disk,

```text
|tau| <= tau_0(1+a)theta^n,
|tau| >= tau_0(1-a)theta^n,
|log tau| <= |log tau_0| + n|log theta| - log(1-a).
```

For any chosen `gamma>1`, polynomial log growth is bounded geometrically. With
`L=|log theta|`, `B=|log tau_0|-log(1-a)`, and
`eta=log(gamma)/L`, the elementary inequality

```text
x^ell <= exp(eta x) (ell/(e eta))^ell
```

gives:

```text
(B+nL)^ell <= K_ell gamma^n.
```

Therefore every differentiated row

```text
d^m/dtau^m [tau^omega P(log tau)]
  = tau^(omega-m) Q_m(log tau)
```

has a geometric shell majorant. If `omega-m>=0`, the `tau` factor contributes
`theta^((omega-m)n)`; if `omega-m<0`, the lower radius gives the same formula
with a growth factor `theta^(omega-m)>1`. Summing the finitely many rows and
the scale term produces constants:

```text
C_0 = sum(row initial majorants),
Lambda = max(row geometric factors).
```

Thus a finite Fuchsian-log total-collision chart supplies the primitive
Cauchy input whenever the retained-order schedule satisfies:

```text
Lambda sigma^d < 1.
```

This reduction is executable as
`derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(...)` in
`three_body_symmetry/fuchsian.py`. It consumes an already constructed
`FiniteFuchsianLogBranch`, shell geometry `(tau_0,theta,a)`, a chosen
log-growth absorber `gamma`, component derivative orders, and retained-order
data. It returns `PrimitiveCauchyTailInput` objects that can be inserted
directly into `derive_all_future_event_budget_from_primitive_cauchy_inputs`.
This closes the total-collision primitive Cauchy input for supplied finite
Fuchsian-log branches. It still does not prove that arbitrary zero-angular
incoming collisions always supply such finite branch data.

The same finite branch data also supplies the local event-isolation proof.
Let:

```text
d_C = min_(i<j) |C_i-C_j| > 0,
S(tau)=C+Delta(tau).
```

For `0<|tau|<=rho<1`, each row has a finite real punctured-radius bound

```text
sup_(0<x<=rho) x^omega |log x|^ell
 = exp(-omega y_*) y_*^ell,
y_* = max(-log rho, ell/omega),
```

with the convention that the value is `rho^omega` when `ell=0`. Summing the
scale and finite Fuchsian-log row bounds gives:

```text
||Delta(tau)||_infty <= E_rho.
```

In dimension `d`, each body displacement is at most `sqrt(d) E_rho`, hence:

```text
|S_i(tau)-S_j(tau)|
 >= d_C - 2 sqrt(d) E_rho.
```

If the right-hand side is positive, then every shape pair distance is positive
for `0<|tau|<=rho`. Since projected physical positions satisfy
`q_i-q_j=tau^2(S_i-S_j)`, every physical pair distance is also positive on the
punctured chart and all three pair distances vanish simultaneously only at
`tau=0`.

This check is executable as
`certify_finite_fuchsian_log_total_collision_isolation(...)`. It provides the
local no-binary-collision event-isolation certificate for supplied finite
Fuchsian-log total-collision charts; a uniform shell-scale isolation fraction
for arbitrary future event tails remains a separate dynamical input.

To feed the compact-time event-count recurrence, the local `tau` isolation must
be projected to the compact physical-time parameter. A total-collision chart has
the cubic physical-time relation:

```text
t = t_* + tau^3.
```

If the punctured Fuchsian-log chart is isolated for `|tau|<=rho`, then its
physical-time window is:

```text
t in [t_*-rho^3, t_*+rho^3].
```

For the compactification `u=tanh(lambda t)`, monotonicity gives the compact
event-isolation interval:

```text
u_- = tanh(lambda(t_*-rho^3)),
u_* = tanh(lambda t_*),
u_+ = tanh(lambda(t_*+rho^3)).
```

Every compact parameter in `[u_-,u_+]` corresponds to a `tau` with
`|tau|<=rho`, so the punctured part of this compact interval is event-free by
the Fuchsian-log isolation certificate. Its centered compact-time isolation
radius is:

```text
h = min(u_*-u_-, u_+-u_*).
```

If the event lies in a shell interval `[a,b]`, the shell-scale isolation
fraction supplied to the packing theorem is:

```text
alpha = min(h, u_*-a, b-u_*) / (b-a).
```

When `alpha>0`, the event contributes to the same count bound:

```text
E_n <= floor(1/alpha)+2.
```

This compact-time projection is executable as
`certify_finite_fuchsian_log_compact_time_isolation(...)`; its
`shell_isolation_fraction` can be passed directly to
`derive_chart_family_counts_from_event_isolation(...)`.

## Automatic Three-Body Total-Collision Selectors In A Geometric Tail

In the positive-mass three-body zero-angular case, the setup can be weakened:
the total-collision selector data do not need to be supplied as independent
chart input. Assume every total collision in the event tail is a finite-energy
zero-centered-angular-momentum three-body total collision and that the local
entry hypotheses of the automatic selector theorem apply. Then each such event
automatically supplies a finite incoming Fuchsian-log selector list.

Indeed, the binary-degenerate Jacobi theorem excludes approach to a binary
shape stratum. McGehee monotonicity and the finite three-body central-target
classification give quotient-shape convergence. The reduced-hyperbolicity
theorem for all positive-mass three-body collision-free central targets gives
finite reduced length and an oriented normalized-shape limit. The
Poincare-Dulac stable normal form then produces the finite Fuchsian-log
selector expansion. Applying the canonical identity rule to those selector
constants gives the outgoing Fuchsian-log chart.

Thus in this subcase the event chart list can be written as:

```text
separated-binary Levi-Civita chart,
automatic identity-selector Fuchsian-log total-collision chart.
```

The geometric recurrence is unchanged. Each automatic total-collision chart
still has a value, first-jet, lifted-residual, and projected-residual budget,
and those budgets are included in the shell totals `B^V_n`, `B^J_n`,
`B^E_n`, and `B^R_n`. If the automatic total-collision chart count and
individual chart budgets satisfy the same geometric shell envelopes, the
infinite tail has the same closed-form remainders:

```text
B^V_0 r_V^N/(1-r_V),   B^J_0 r_J^N/(1-r_J),
B^E_0 r_E^N/(1-r_E),   B^R_0 r_R^N/(1-r_R).
```

This removes "preselect all total-collision Fuchsian-log data" from the
geometric infinite-event recurrence for positive-mass three-body
zero-angular total-collision events. It does not prove the shell geometry,
uniform per-shell event bounds, or geometric budget envelopes; those remain the
global recurrence hypotheses.

## Theorem

Under the setup and shell-budget hypotheses, the infinite mixed event atlas has
finite all-future verification budgets:

```text
sum_n B^V_n <= B^V_0/(1-r_V),
sum_n B^J_n <= B^J_0/(1-r_J),
sum_n B^E_n <= B^E_0/(1-r_E),
sum_n B^R_n <= B^R_0/(1-r_R).
```

More precisely, after a checked prefix through shell `N-1`, the remaining tail
is bounded by:

```text
sum_(n>=N) B^V_n <= B^V_0 r_V^N/(1-r_V),
sum_(n>=N) B^J_n <= B^J_0 r_J^N/(1-r_J),
sum_(n>=N) B^E_n <= B^E_0 r_E^N/(1-r_E),
sum_(n>=N) B^R_n <= B^R_0 r_R^N/(1-r_R).
```

Thus any finite prefix plus the displayed geometric remainders is a complete
all-future verification budget for the mixed ordinary/binary/total-collision
tail.

## Proof

Fix `U<1`. Since event accumulation is only at `u=1`, the interval `[u_0,U]`
meets finitely many event neighborhoods. On that compact interval, the finite
compact-atlas composition theorems apply: ordinary Taylor charts cover the
collision-free complement, separated-binary Levi-Civita charts cover isolated
binary events, and identity-selector Fuchsian-log charts cover the selected
total-collision events. The projected pieces glue at every noncollision
handoff by analytic Newtonian ODE uniqueness. At singular event centers, the
chosen lifted event chart is the continuation rule.

Let `A_N` be the finite atlas restricted to the prefix ending before shell
`S_N`. These finite atlases are compatible: `A_(N+1)` restricts to `A_N` on the
earlier compact subdomain because the chart data and handoff states are the
same there. The increasing union therefore defines one projected branch on
`[u_0,1)`, with the selected lifted continuation through each event.

The verification budgets of the prefix atlases are monotone partial sums of the
nonnegative shell budgets. The shell-budget hypothesis gives four scalar
geometric majorants. Hence the monotone partial sums have finite limits and the
tail after any prefix is bounded by the corresponding geometric remainder.
Those finite limits are the all-future value, first-jet, lifted-residual, and
physical-residual verification budgets.

If an endpoint chart at `u=1` is also present, for example a log-subtracted
scattering or homothetic escape endpoint chart, its endpoint-shell recurrence
is added as a separate geometric tail. The mixed event recurrence above
controls only the ordinary/binary/total-collision event shells before that
endpoint chart.

## What This Closes

This proves a genuine infinite-event recurrence subcase: infinitely many local
collision events no longer force the verification budget to be only finite
prefix data, provided the event distribution and chart residuals obey the
geometric shell hypotheses above.

It does not prove that arbitrary three-body data satisfy those hypotheses. In
particular, it does not rule out finite-time accumulation, prove a uniform
per-shell event-count bound, classify arbitrary escape/capture/bounded regimes,
or construct the missing global scattering map.
