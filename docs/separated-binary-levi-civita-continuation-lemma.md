# Separated Binary Levi-Civita Continuation Lemma

## Claim

In the planar three-body problem, an isolated binary collision with the third
body separated has a local analytic continuation in Levi-Civita regularized
time. This supplies the local binary-collision chart needed by the finite-time
atlas; it does not by itself solve chart selection, interval wrapping, or
all-time recurrence.

Select the colliding pair `(1,2)`, let:

```text
M = m_1 + m_2,
r = q_2-q_1,
R = (m_1 q_1 + m_2 q_2)/M,
y = q_3-R.
```

Use the planar Levi-Civita lift:

```text
r = z^2,
rho = |z|^2,
dt/ds = rho.
```

The third body is separated at collision exactly when:

```text
y(0) != 0.
```

In that case the two third-body force fields:

```text
y + (m_2/M) z^2,
y - (m_1/M) z^2
```

stay nonzero for all sufficiently small `|z|` and `|y-y(0)|`. Their inverse
cube fields are analytic functions of `(z,y)` near `z=0`. Therefore the
regularized chart equations:

```text
z'      = zeta,
zeta'  = (1/2) h z + (1/4) rho Dz^T P(z,y),
h'      = Dz zeta dot P(z,y),
R'      = rho Rdot,
Rdot'   = rho A_R(z,y),
y'      = rho ydot,
ydot'   = rho A_y(z,y),
t'      = rho,
```

have an analytic right hand side at `z=0`. Here `P(z,y)` is the analytic
third-body perturbation of the binary relative acceleration, `A_R` is the
binary-center acceleration, and `A_y` is the separated third-body offset
acceleration. The pair-energy constraint is:

```text
2|zeta|^2 - M - rho h = 0.
```

At an exact binary collision `z=0`, this gives:

```text
|zeta(0)|^2 = M/2.
```

Thus `zeta(0) != 0`, and the analytic solution has:

```text
z(s) = zeta(0)s + O(s^2),
rho(s) = |zeta(0)|^2 s^2 + O(s^3),
t(s) = (|zeta(0)|^2/3)s^3 + O(s^4).
```

After shrinking the interval, `t(s)` is strictly increasing through `s=0` in
the cubic sense: negative `s` gives incoming negative physical time, positive
`s` gives outgoing positive physical time, and the selected pair has
`r(s)=z(s)^2 != 0` for `s != 0` while `r(0)=0`.

For every `s != 0`, projection back to physical coordinates is:

```text
q_1 = R - (m_2/M) z^2,
q_2 = R + (m_1/M) z^2,
q_3 = R + y,
```

with physical velocities obtained by dividing `dq/ds` by `dt/ds=rho`. Direct
substitution into the projected acceleration formulas gives Newton's equations
for every punctured side of the collision. The positions extend continuously
through `s=0`; the pair's physical velocity is allowed to be singular at the
collision instant, which is exactly why the regularized chart is used.

Consequently, once the incoming branch has selected a Levi-Civita branch
`z -> -z` and the separated-third-body data, the analytic regularized solution
continues uniquely through `z=0` and projects to the incoming and outgoing
Newtonian binary-collision branches away from the collision instant.

## Verification Role

This is the binary analogue of the ordinary Taylor-cover and zero-angular
triple-collision local lemmas:

```text
lift:      replace r by z^2 and use dt/ds=|z|^2,
construct: solve the analytic regularized chart equations through z=0,
project:   evaluate q_1,q_2,q_3 for s != 0,
verify:    projected accelerations equal Newtonian accelerations.
```

The remaining global theorem still has to prove when a finite-time trajectory
enters such a chart, how interval branch atlases are propagated without losing
separation from the third body, and how these local binary continuations are
composed with ordinary charts and total-collision conventions.

## Finite Compact-Interval Binary Atlas

The local chart does compose into a finite analytic atlas on any compact
interval whose only singularities are finitely many separated binary
collisions. Let `[a,b]` be a physical-time interval and let:

```text
a < c_1 < ... < c_N < b
```

be the binary-collision times. Assume:

1. At each `c_k` exactly one pair collides.
2. The third body is separated at that event.
3. No total collision occurs on `[a,b]`.
4. The chosen Levi-Civita branch data at `c_k` match the incoming and outgoing
   one-sided Newtonian branch.

For each event, the local theorem gives an `epsilon_k>0` and a regularized
interval:

```text
s in [-epsilon_k, epsilon_k],
dt/ds = |z_k(s)|^2,
t_k(0)=c_k,
```

whose projection is Newtonian for `s != 0`. Shrink the `epsilon_k` so the
physical images:

```text
U_k = t_k([-epsilon_k,epsilon_k])
```

are pairwise disjoint, lie in `(a,b)`, and have the third body separated from
the binary pair throughout the chart. This is possible because the collision
times are finite in number and each third-body separation is positive at the
event.

The complement:

```text
[a,b] \ union_k U_k
```

is a finite union of compact intervals on which all pair distances are
positive. The ordinary compact Taylor-cover lemma applies to each such
component: every component is covered by finitely many ordinary Newtonian
Taylor charts, and each ordinary chart is verified coefficient by coefficient
from:

```text
q' = v,
v' = A(q).
```

At each boundary between an ordinary component and a binary chart, both
projected curves solve Newton's equations on a punctured one-sided
neighborhood and have the same finite state at the noncollision handoff time.
Uniqueness for the analytic Newtonian ODE on that handoff neighborhood makes
the ordinary chart and the projected Levi-Civita chart agree on their overlap.
Inside the regularized binary chart, uniqueness of the analytic lifted ODE
glues the incoming and outgoing sides through `z=0`. Therefore the finite list:

```text
ordinary Taylor charts on collision-free components
+ one Levi-Civita chart for each separated binary collision
```

is one projected Newtonian solution on `[a,b]` except at the binary collision
instants themselves, where the regularized coordinates give the continuation
and the physical positions remain continuous.

The verification budget is finite. If the ordinary components have tail and
coefficient-residual budgets `O_j`, and the binary charts have regularized
tail and projected-residual budgets `B_k`, then:

```text
sum_j O_j + sum_k B_k < infinity
```

because both sums are finite. This closes the finite compact-interval
composition problem for separated binary collisions. It does not prove a
long-time recurrence for infinitely many binary events, nor does it handle
finite accumulation into total collision; those remain global-regime
classification problems.

## Finite-Time Binary Accumulation Forces Total Collision

The finite-accumulation caveat can be sharpened. In the three-body problem,
infinitely many separated binary-collision events cannot accumulate at a finite
time unless the accumulation configuration is total collision.

Let `c_n -> T` be binary-collision times in a compact physical-time interval,
and assume the projected positions extend continuously through all separated
binary collisions by the Levi-Civita continuation above. There are only three
binary pairs. Passing to a subsequence, one selected pair, say `(1,2)`, collides
at every `c_n` in the subsequence:

```text
q_2(c_n)-q_1(c_n)=0.
```

Continuity gives:

```text
q_2(T)-q_1(T)=0.
```

If `T` is collision-free, this is impossible. If `T` is a separated binary
collision of the same pair, then the local Levi-Civita theorem applies at `T`:
the third-body offset satisfies `y(T) != 0`, the lifted equation is analytic,
and the pair coordinate has a simple zero

```text
z(s) = zeta(0)s + O(s^2),        zeta(0) != 0.
```

After shrinking the chart, the punctured neighborhood of `T` contains no other
collision of that pair, and the third body remains separated. That contradicts
the existence of the subsequence `c_n -> T`, `c_n != T`.

The only remaining possibility is that the limit is not a separated binary
collision. For three bodies this means total collision: at least one additional
pair distance vanishes at `T`, and then all three bodies coincide by the
triangle inequality.

Therefore any finite-time accumulation of binary collision events is a
total-collision approach. The separated-binary atlas needs no separate
finite-accumulation case away from total collision; the global theorem only has
to classify or continue the total-collision limit.
