# Compact Collision-Free Taylor Cover Lemma

## Claim

Every compact classical three-body trajectory segment that stays away from
collision is represented by a finite atlas of ordinary Newtonian Taylor charts.
This closes the ordinary finite-time part of the lift/construct/project/verify
pipeline; collision charts and all-time recurrence are separate obligations.

Let `q(t)` be a Newtonian solution for positive masses on a compact physical
time interval `[a,b]`, and assume:

```text
q_i(t) != q_j(t)       for all i<j and all t in [a,b].
```

By continuity on a compact interval:

```text
delta = min_{t in [a,b]} min_{i<j} |q_i(t)-q_j(t)| > 0,
Q = max_{t in [a,b]} max_i |q_i(t)| < infinity,
V = max_{t in [a,b]} max_i |q_i'(t)| < infinity.
```

The first-order Newtonian vector field:

```text
F(q,v) = (v, A(q))
```

is analytic on the open set where every pair distance is nonzero. In
particular it is analytic on the uniform tube:

```text
Omega =
  { (x,u) : |x_i-q_i(t0)| < delta/4,
            |u_i-q_i'(t0)| < 1 + V
            for some t0 in [a,b] }.
```

Every state in this tube has pair separations at least `delta/2`, so all
inverse-distance factors are analytic and uniformly bounded there. Hence the
analytic ODE theorem, or equivalently a Cauchy majorant/Picard proof on this
common tube, gives a radius `rho>0` depending only on `delta`, `Q`, `V`, and the
masses such that the Taylor series centered at any `t0 in [a,b]` converges and
solves Newton's equation for:

```text
|t-t0| < rho.
```

Choose a finite partition:

```text
a = t_0 < t_1 < ... < t_N = b,
0 < t_{k+1}-t_k < rho/2.
```

At each `t_k`, construct the ordinary Taylor chart by the coefficient
recurrence:

```text
q_{n+1} = v_n/(n+1),
v_{n+1} = A(q)_n/(n+1),
```

where `A(q)_n` is obtained by performing the pair-force inverse-distance
operations in the truncated power-series algebra. The recurrence is exactly
the Taylor coefficient form of:

```text
q' = v,
v' = A(q).
```

Projecting a chart back to physical coordinates is therefore just evaluating
its convergent Taylor series. Uniqueness for analytic ODEs implies that the
chart centered at `t_k` agrees with the original solution on the overlap with
the next chart, and the endpoint state of one chart is the initial state of the
next. Thus the finite chain:

```text
ordinary Taylor chart at t_0
  -> ordinary Taylor chart at t_1
  -> ...
  -> ordinary Taylor chart at t_{N-1}
```

represents the exact classical solution on the whole compact collision-free
interval `[a,b]`.

## Verification Role

This lemma supplies a theorem-level bridge between arbitrary noncollision
finite-time classical motion and the existing ordinary Taylor harness:

```text
lift:      convert q,v into power-series coefficients,
construct: solve the coefficient recurrence in the analytic force algebra,
project:   evaluate the convergent Taylor charts in physical time,
verify:    q'=v and v'=A(q) coefficient-by-coefficient.
```

Finite truncations inherit ordinary Cauchy tail bounds on each chart once a
chart radius and coefficient majorant are supplied. A finite compact segment
therefore needs only finitely many local tail budgets.

This is not the full general closed-form solution. It proves the ordinary
collision-free finite segment cannot be the missing obstruction. The remaining
global proof still has to handle binary collision charts, zero-angular total
collision reachability/conventions, escape endpoint classification, and the
all-future compact-time recurrence.

## All-Future Uniformly Collision-Free Recurrence

The compact-cover proof has an all-future form under stronger hypotheses. Let
`q(t)` be a classical Newtonian solution on `[0,infinity)`, and suppose the
internal motion has global bounds:

```text
delta = inf_{t>=0} min_{i<j} |q_i(t)-q_j(t)| > 0,
V = sup_{t>=0} max_i |q_i'(t)| < infinity.
```

No absolute position bound is needed for the force field, since the Newtonian
acceleration depends only on pair differences. Around every time `t0`, take the
same translated tube:

```text
|x_i - q_i(t0)| < delta/8,
|u_i - q_i'(t0)| < 1 + V.
```

Every point in this tube has pair separations at least `3delta/4`. Therefore
the inverse-distance force terms are analytic there and have uniform Cauchy
majorants depending only on `delta`, `V`, and the masses, not on `t0`. The
analytic ODE theorem with these common majorants gives constants:

```text
rho_* > 0,
M_* < infinity,
```

such that the ordinary Taylor chart centered at every `t0>=0` converges for
`|t-t0|<rho_*`, and the full state Taylor coefficients satisfy the uniform
Cauchy tail estimate:

```text
|X(t0+theta) - sum_{n=0}^N X_n(t0) theta^n|
  <= M_* (|theta|/rho_*)^(N+1) / (1 - |theta|/rho_*)
```

whenever `|theta|<rho_*`. Here `X=(q,v)`, and `X_n(t0)` is produced by the same
coefficient recurrence:

```text
q_{n+1}(t0) = v_n(t0)/(n+1),
v_{n+1}(t0) = A(q(t0))_n/(n+1).
```

Choose one fixed step `h` with `0<h<rho_*`. Define `t_k=kh`. The recurrence:

```text
construct Taylor coefficients at state X(t_k),
project the chart to theta=h,
use that endpoint as X(t_{k+1}),
```

is well defined for every `k>=0`, stays inside the same collision-free tube,
and covers all of `[0,infinity)`. The omitted tail after retaining `N`
coefficients has the same bound on every chart:

```text
R_N <= M_* (h/rho_*)^(N+1) / (1 - h/rho_*).
```

Thus uniformly collision-free bounded-speed solutions admit a genuine
all-future ordinary Taylor recurrence: the chart index may be infinite, but the
step size, convergence radius, and finite-truncation remainder formula are
time-independent.

The same proof applies on all of `R` if the two bounds hold for all real
times. Lagrange equilateral relative equilibria are an explicit family: their
pair distances and speeds are constant, so the ordinary Taylor recurrence can
be iterated with one fixed step and one uniform tail envelope for every future
period.

This is still a conditional subcase, not the unrestricted theorem. It closes
the all-future recurrence obligation for trajectories that are already known
to remain uniformly away from collision with bounded speed. The general
closed-form target still has to prove that arbitrary data enter one of the
global regimes: uniform collision-free recurrence, hyperbolic escape charts,
binary collision charts, or a certified total-collision stopping/continuation
convention.

## Periodic Collision-Free Orbits Give A Finite Cyclic Atlas

There is an even sharper all-future recurrence when the collision-free solution
is periodic. Suppose:

```text
q(t+P)=q(t),        q'(t+P)=q'(t),
P>0,
min_{t in [0,P]} min_{i<j}|q_i(t)-q_j(t)| > 0.
```

The compact-cover lemma applied to `[0,P]` gives a finite Taylor atlas:

```text
0=t_0<t_1<...<t_N=P.
```

The chart at `t_k` is constructed only from the state `X(t_k)=(q(t_k),q'(t_k))`
and the Newtonian coefficient recurrence. Since the state is periodic,
`X(t_k + mP)=X(t_k)` for every integer `m`, so the same chart coefficients
recur exactly at every translated center `t_k+mP`.

Thus the whole real line is represented by the finite cyclic chart list:

```text
chart 0 -> chart 1 -> ... -> chart N-1 -> chart 0 -> ...
```

with the same local convergence radii and Cauchy tail bounds on every cycle.
For any target time, write:

```text
t = mP + r,        0 <= r < P.
```

Choose the unique chart interval `t_k <= r <= t_{k+1}` and evaluate the stored
chart at `theta=r-t_k`. Periodicity then gives:

```text
X(t)=X(r)
```

and the finite chart's coefficient recurrence verifies Newton's equations on
the corresponding physical interval. The infinite all-time recurrence is
therefore reduced to a finite cyclic lift/construct/project/verify object.

This is exactly how a symmetry-rich special solution should enter the general
atlas program: the symmetry does not generalize to arbitrary data, but when it
is present it compresses an all-time solution into finitely many analytic
charts whose verification is local and reusable.
