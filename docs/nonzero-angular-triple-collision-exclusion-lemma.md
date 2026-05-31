# Nonzero Angular Triple-Collision Exclusion Lemma

## Claim

Inside the regularized total-collision chart used by the local triple-collision
normal form, nonzero centered angular momentum excludes total collision.

Work in center-of-mass coordinates and let the physical branch be represented
near total collision by the cubic-time regularization:

```text
t = tau^3,
q_i(tau) = tau^2 S_i(tau),
S(tau) analytic at tau=0,
S_i(0) != S_j(0).
```

For `tau != 0`, the physical velocity is:

```text
dq_i/dt = (1/(3 tau^2)) d/dtau (tau^2 S_i)
        = (2/(3 tau)) S_i + (1/3) S_i'.
```

Thus the centered angular momentum bivector is:

```text
L(tau)
 = sum_i m_i q_i wedge dq_i/dt
 = sum_i m_i tau^2 S_i wedge ((2/(3 tau))S_i + (1/3)S_i')
 = (tau^2/3) sum_i m_i S_i wedge S_i',
```

because `S_i wedge S_i = 0`. The last expression is analytic times `tau^2`,
so:

```text
L(tau) -> 0        as tau -> 0.
```

If the branch satisfies Newton's equation on a punctured side of the collision,
then angular momentum is conserved on that side because Newtonian pair forces
are central. A conserved quantity whose collision limit is zero is identically
zero on that punctured side. Hence every analytic cubic-time total-collision
branch in this chart has zero centered angular momentum.

Therefore an interval initial-data set whose centered angular momentum norm has
a strictly positive lower bound cannot enter such a total-collision chart. This
is the analytic content behind the one-sided `excluded` status used by the
harness:

```text
|L|^2 >= ell > 0
  and
analytic total-collision chart implies |L|^2 -> 0
```

are incompatible.

## Verification Role

This closes the nonzero-angular branch of the local triple-collision
continuation problem:

```text
lift:      use centered coordinates and cubic collision time tau,
construct: represent total-collision candidates as q=tau^2 S(tau),
project:   compute physical velocities through t=tau^3,
verify:    angular momentum is tau^2 times an analytic bounded factor.
```

The result is deliberately one-sided. It does not continue zero-angular total
collisions, and it does not prove that every arbitrary total collision has
already been captured by a convergent regularized germ with noncollision
limiting shape. Those remain separate global obligations. It does prove that
once the total-collision analysis is in this analytic chart, any nonzero
centered-angular-momentum branch is excluded rather than continued.

## Sundman Inequality For Nondegenerate Total Collapse

The same exclusion can be proved without first assuming an analytic
`q=tau^2S(tau)` germ if the collapsing shapes stay nondegenerate in the
standard Sundman sense.

Let:

```text
x_i = q_i - q_cm,
I = sum_i m_i |x_i|^2,
K = (1/2) sum_i m_i |x_i'|^2,
U = sum_{i<j} m_i m_j / |q_i-q_j|,
H = K - U.
```

Assume a finite-time total collapse along a classical branch:

```text
I(t) -> 0        as t -> T,
H(t) = H_0       finite and constant,
U(t) sqrt(I(t)) <= C
```

near `T`. The last condition is exactly a bounded normalized potential; it
holds when the normalized shape stays in a compact subset of the collision-free
shape sphere.

The mass Cauchy-Schwarz inequality gives Sundman's angular-momentum bound:

```text
|L|^2
 = |sum_i m_i x_i wedge x_i'|^2
 <= (sum_i m_i |x_i|^2)(sum_i m_i |x_i'|^2)
 = 2 I K.
```

Since `K=H_0+U`,

```text
2 I K = 2H_0 I + 2 I U
      <= 2H_0 I + 2C sqrt(I) -> 0.
```

Thus `|L|^2=0`. A branch with a strictly positive centered-angular-momentum
lower bound is incompatible with any such nondegenerate total collapse.

This argument is weaker than the full classical total-collision theorem because
it assumes the normalized-potential bound instead of proving it for every
possible total-collision approach. It is stronger than the analytic-germ
calculation above because it needs no convergent regularized expansion and no
preselected central configuration; bounded normalized shape control plus the
energy invariant is enough.

## Shape-Compact Collapse Supplies The Normalized-Potential Bound

The normalized-potential hypothesis is geometric. Let:

```text
R(t) = sqrt(I(t)),
y_i(t) = x_i(t)/R(t).
```

Then:

```text
sum_i m_i |y_i(t)|^2 = 1.
```

If the normalized shape stays a fixed distance from every binary-collision
stratum, there is `d_*>0` such that:

```text
|y_i(t)-y_j(t)| >= d_*       for all i<j
```

near the total-collapse time. Since:

```text
|q_i-q_j| = |x_i-x_j| = R |y_i-y_j|,
```

the Newtonian potential satisfies:

```text
U(t) sqrt(I(t))
 = sum_{i<j} m_i m_j / |y_i(t)-y_j(t)|
 <= (sum_{i<j} m_i m_j) / d_*.
```

Thus shape-compact total collapse is a concrete subcase of the
Sundman-inequality exclusion above. Nonzero centered angular momentum cannot
reach total collision while the normalized shape remains in any compact
collision-free subset of shape space.

After this estimate, the only remaining three-body nonzero-angular case is a
normalized shape tending toward a binary-collision stratum during total
collapse. The next two sections close that binary-degenerate alternative.

## Binary-Degenerate Collapse Kills The Tight-Pair Angular Part

The first estimate in the remaining binary-degenerate case is still elementary.
Suppose bodies `1` and `2` form the tight pair. In center-of-mass cluster
coordinates write:

```text
r = q_2-q_1,
rho = q_3 - (m_1q_1+m_2q_2)/(m_1+m_2),
mu = m_1m_2/(m_1+m_2).
```

Assume a finite-energy total collapse with:

```text
|r| -> 0,
|rho| -> 0,
|r|/|rho| -> 0,
H = K-U = H_0.
```

The tight-pair angular momentum is:

```text
L_12 = mu r wedge r'.
```

By Cauchy-Schwarz:

```text
|L_12|^2 <= mu^2 |r|^2 |r'|^2
          = 2 mu |r|^2 K_12
          <= 2 mu |r|^2 K.
```

Finite energy gives `K=H_0+U`. The potential has the form:

```text
U = m_1m_2/|r| + O(1/|rho|)
```

because the distances from the tight pair to body `3` are comparable to
`|rho|` when `|r|/|rho|` is small. Hence:

```text
|r|^2 K
 <= |H_0||r|^2 + C|r| + C |r|^2/|rho| -> 0.
```

Therefore:

```text
L_12 -> 0.
```

So a binary-degenerate total collapse cannot hide nonzero angular momentum in
the internal spin of the much tighter binary cluster.

## Outer Angular Barrier Closes The Binary-Degenerate Case

The cluster-versus-third angular component is also incompatible with total
collapse if it stays nonzero. Continue with the same Jacobi coordinates and
set:

```text
nu = (m_1+m_2)m_3/(m_1+m_2+m_3),
L_rho = nu rho wedge rho',
s = |rho|.
```

The Jacobi decomposition gives:

```text
L = L_12 + L_rho.
```

If the total angular momentum `L` is a nonzero conserved bivector, the estimate
above gives `L_12 -> 0`, hence `L_rho -> L`. Thus there are constants
`ell>0` and `t_0<T` such that:

```text
|L_rho(t)| >= ell       for t_0<t<T.
```

When `|r|/s` is small, the distances from body `3` to bodies `1` and `2` are
comparable to `s`. The outer Jacobi equation therefore has the force bound:

```text
|rho''| <= C/s^2.
```

The scalar radial acceleration satisfies the exact identity:

```text
s'' = |rho'_perp|^2/s + (rho/s) . rho''.
```

Since `|rho'_perp| = |rho wedge rho'|/s = |L_rho|/(nu s)`, this gives:

```text
s'' >= ell^2/(nu^2 s^3) - C/s^2.
```

For all sufficiently small `s`,

```text
s'' >= c/s^3             with c = ell^2/(2nu^2) > 0.
```

But no positive scalar function can satisfy `s(t)->0` at a finite terminal
time while obeying this inward angular-barrier inequality. Indeed, after
shrinking `t_0` if needed, `s''>0`. Since `s(t)->0`, one must have `s'<0`
near `T`; otherwise the increasing derivative would make `s` nondecreasing
near `T`. For:

```text
E_bar(t) = (s')^2 + c/s^2,
```

the inequality and `s'<0` imply:

```text
E_bar' = 2s'(s'' - c/s^3) <= 0.
```

So `E_bar` is nonincreasing as `t` approaches `T`. This is impossible because
`c/s(t)^2 -> infinity`.

Thus the binary-degenerate alternative also forces `L=0`.

## Full Three-Body Nonzero-Angular Total-Collapse Exclusion

The preceding pieces now cover all three-body total-collapse approaches.
Assume a finite-time total collapse in center-of-mass coordinates with finite
constant energy and nonzero conserved centered angular momentum.

If the normalized shape has a positive pair-distance floor along arbitrarily
late times, the shape-compact Sundman estimate gives a sequence on which
`|L|^2 <= 2IK -> 0`, contradicting conservation of nonzero `L`. Therefore, for
all sufficiently late times, the normalized shape must lie in a small
neighborhood of a binary-collision stratum. The three binary strata are
separated in normalized shape space, so after shrinking the neighborhood and
moving closer to the collision time, the same pair must be the tight pair.

That puts the branch in the binary-degenerate setting above. The tight-pair
angular momentum tends to zero by the finite-energy estimate, and any nonzero
remaining cluster angular momentum creates the radial angular barrier, which
prevents finite-time collapse. Hence the conserved centered angular momentum
must be zero.

Consequently, for the classical three-body problem, nonzero centered angular
momentum excludes total collision. This closes the nonzero-angular
triple-collision obstruction used by the harness. It still does not select or
continue zero-angular total-collision branches; those remain the separate
regularized-branch/convention problem.

## Compact-Interval Binary Finiteness For Nonzero Angular Momentum

The preceding exclusion combines with the finite-time binary-accumulation
theorem. On any compact physical-time interval, a nonzero-angular branch has
only finitely many binary collision events.

Assume otherwise. Let `[a,b]` be compact and suppose infinitely many binary
collision times occur in `[a,b]`, with the projected Levi-Civita continuation
used at separated binary events. By compactness, the event times have an
accumulation point `T in [a,b]`. The separated-binary accumulation theorem says
that a finite-time accumulation of binary events can only occur at total
collision: one pair repeats on an infinite subsequence, and if the third body
were separated at `T`, the local Levi-Civita simple-zero chart would give a
punctured neighborhood containing no other collision of that pair.

But total collision at `T` is impossible on a branch with conserved nonzero
centered angular momentum by the theorem above. This contradiction proves that
the binary event set in `[a,b]` is finite.

Thus the nonzero-angular compact-interval collision atlas needs no infinite
binary-event recurrence and no total-collision continuation. Its singular
content is a finite list of separated binary Levi-Civita charts plus ordinary
Taylor charts on the collision-free complement. Infinite binary recurrences
remain relevant only for zero-angular or endpoint-accumulating regimes where
the nonzero-angular total-collision obstruction is unavailable.

## Compact Nonzero-Angular Finite Atlas

This gives the full compact-interval atlas theorem for the nonzero-angular
branch. Let `[a,b]` be a compact physical-time interval of a classical
three-body branch with:

```text
|L| >= ell > 0,
H finite,
```

and suppose every binary collision on `[a,b]` is a separated-third-body binary
collision continued by the Levi-Civita chart. Then `[a,b]` has a finite
lift/construct/project/verify atlas made only of:

```text
ordinary Taylor charts,
separated-binary Levi-Civita charts.
```

Proof. Nonzero angular momentum excludes total collision, so the only possible
collision events are binary collisions. The compact-interval binary-finiteness
theorem above makes their set finite:

```text
a < c_1 < ... < c_N < b.
```

For each `c_k`, choose a small separated-binary Levi-Civita neighborhood. The
events are finite, so the neighborhoods can be made disjoint. Their complement
in `[a,b]` is a finite union of compact collision-free intervals. On each
complement component, the ordinary compact Taylor-cover lemma supplies finitely
many ordinary analytic Taylor charts. At noncollision handoffs, analytic ODE
uniqueness glues the ordinary and projected Levi-Civita pieces. At binary
centers, the Levi-Civita lifted chart is the selected continuation through
`z=0`.

The verification budget is finite. If the ordinary components contribute
budgets:

```text
O^V_j, O^J_j, O^E_j, O^R_j,
```

and the binary neighborhoods contribute:

```text
B^V_k, B^J_k, B^E_k, B^R_k,
```

then the compact nonzero-angular atlas budgets are:

```text
sum_j O^V_j + sum_k B^V_k,
sum_j O^J_j + sum_k B^J_k,
sum_j O^E_j + sum_k B^E_k,
sum_j O^R_j + sum_k B^R_k.
```

All sums are finite. Thus every compact nonzero-angular branch covered by the
separated-binary local hypotheses has a finite analytic atlas with direct
Newtonian verification after projection. The remaining compact collision
continuation difficulty is the zero-angular total-collision branch, where the
nonzero-angular exclusion cannot be used.
