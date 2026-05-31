# Hyperbolic Scattering Fixed-Point Convergence Lemma

## Claim

The log-subtracted escape endpoint is convergent for a genuine nonhomothetic
class of hyperbolic scattering data: prescribed distinct asymptotic velocities,
arbitrary offsets, and sufficiently large physical time. This is not yet an
asymptotic completeness theorem, but it proves that the nonhomothetic
log-subtracted construction is not merely formal.

Let `m_i > 0`, and let the asymptotic velocities `v_i` be distinct:

```text
d = min_{i<j} |v_i-v_j| > 0.
```

Set:

```text
B = A(v),
q_0(t) = v t - B log(t) + c,
```

where `c` is an arbitrary offset vector and `A` is the Newtonian acceleration
map. For all sufficiently large `T`, there is a unique correction `w(t)` on
`[T, infinity)` with:

```text
w(t) = O(log(t)^2/t),
```

such that:

```text
q(t) = q_0(t) + w(t)
```

solves Newton's equations for every `t >= T` and has:

```text
q_i(t)/t -> v_i.
```

## Fixed-Point Equation

Since:

```text
q_0''(t) = B/t^2,
```

the correction must solve:

```text
w''(t) = A(q_0(t)+w(t)) - B/t^2,
w(t), w'(t) -> 0    as t -> infinity.
```

Equivalently:

```text
(Phi w)(t)
  = int_t^infinity (s-t) [A(q_0(s)+w(s)) - B/s^2] ds.
```

Then `(Phi w)''` is the required right side, and the boundary conditions at
infinity are built in.

## Contraction Space

Work in the Banach space:

```text
X_T = { w continuous on [T,infinity) :
        ||w|| = sup_{t>=T} t |w(t)| / log(t)^2 < infinity }.
```

For large enough `T`, every pair separation in `q_0(t)` is bounded below by:

```text
|q_{0,j}(t)-q_{0,i}(t)| >= (d/2)t.
```

This follows from the elementary bound:

```text
|q_{0,j}(t)-q_{0,i}(t)|
 >= |v_j-v_i|t - |c_j-c_i-(B_j-B_i)log(t)|.
```

If:

```text
E_pair(T) = max_{i<j} ( |c_j-c_i|/log(T) + |B_j-B_i| )
```

then `|c_j-c_i-(B_j-B_i)log(t)| <= E_pair(T)log(t)` for `t>=T`, and
`log(t)/t <= log(T)/T` once `T>e`. Thus it is enough to choose `T` so that
`E_pair(T)log(T)/T <= d/2`.

Use the norm `|q|_* = max_i |q_i|`, and put `M=sum_i m_i`. On the tube where
all pair separations are at least `(d/4)t`, the acceleration map has the
explicit Lipschitz bound:

```text
|A(q)-A(p)| <= L |q-p| / t^3.
```

One can take:

```text
L = 256 M / d^3.
```

Indeed, the derivative of `r -> r/|r|^3` has operator norm at most `2/|r|^3`.
For each body this gives at most `4M/(rho^3 t^3)|q-p|_*` when all pair
separations are at least `rho t`; setting `rho=d/4` gives the displayed `L`.

Thus:

```text
|Phi w(t)-Phi z(t)|
 <= L ||w-z|| int_t^infinity (s-t) log(s)^2 / s^4 ds.
```

The integral is explicit:

```text
int_t^infinity (s-t) log(s)^2 / s^4 ds
 = t^-2 [
     (1/6)log(t)^2 + (5/18)log(t) + 19/108
   ].
```

Therefore:

```text
||Phi w-Phi z||
 <= (L/T) [
      1/6 + 5/(18 log T) + 19/(108 log(T)^2)
    ] ||w-z||.
```

Choosing `T` large makes this contraction factor less than `1`.

## Self-Map Bound

The uncorrected residual is:

```text
A(q_0(t)) - B/t^2
 = t^-2 [ A(v + (c-B log t)/t) - A(v) ].
```

Since `A` is analytic near the distinct velocity configuration `v`, this is:

```text
O(log(t)/t^3).
```

The same Lipschitz bound gives a usable quantitative version. If:

```text
E_body(T) = max_i |c_i|/log(T) + max_i |B_i|,
```

then `max_i |c_i-B_i log(t)| <= E_body(T)log(t)` for `t>=T`, and:

```text
|A(q_0(t)) - B/t^2|_*
 <= L E_body(T) log(t)/t^3.
```

Consequently:

```text
int_t^infinity (s-t) log(s)/s^3 ds
 = t^-1 [ (1/2)log(t) + 3/4 ],
```

and:

```text
||Phi 0|| = O(1/log T).
```

More explicitly, define:

```text
K_1(T) = 1/(2 log T) + 3/(4 log(T)^2),
K_2(T) = (1/T)[1/6 + 5/(18 log T) + 19/(108 log(T)^2)].
```

Then:

```text
||Phi 0|| <= L E_body(T) K_1(T),
||Phi w-Phi z|| <= L K_2(T) ||w-z||.
```

Choose `T` so that:

```text
E_pair(T) log(T)/T <= d/2,
L K_2(T) <= 1/2,
R >= 2 L E_body(T) K_1(T),
R log(T)^2/T^2 <= d/8.
```

The last condition keeps the whole `||w||<=R` ball inside the `(d/4)t`
separation tube. For this `T`, `Phi` maps the closed ball `||w||<=R` into
itself and is a contraction. Banach's fixed-point theorem gives a unique
correction `w`.

## Effective Tail-Start Existence

The tail-start inequalities above are not only asymptotic assumptions; they
give a terminating construction of a valid future chart start. Put:

```text
C_pair = max_{i<j}|c_j-c_i|,
B_pair = max_{i<j}|B_j-B_i|,
C_body = max_i |c_i|,
B_body = max_i |B_i|.
```

Then:

```text
E_pair(T) <= C_pair/log(T) + B_pair,
E_body(T) <= C_body/log(T) + B_body.
```

Thus:

```text
E_pair(T) log(T)/T
 <= [C_pair + B_pair log(T)]/T -> 0,
L K_2(T) -> 0,
L E_body(T) K_1(T) -> 0.
```

Choose the ball radius effectively as:

```text
R(T)=2 L E_body(T) K_1(T).
```

Then:

```text
R(T) log(T)^2/T^2 -> 0.
```

Consequently there exists a finite `T_*` such that every `T>=T_*`, with
`T>e^2`, satisfies all four start inequalities. A completely explicit
selection algorithm is: start at any `T>e^2` and repeatedly double `T` until:

```text
E_pair(T) log(T)/T <= d/2,
L K_2(T) <= 1/2,
2L E_body(T)K_1(T) log(T)^2/T^2 <= d/8.
```

The preceding limits prove that this loop terminates. With
`R=2L E_body(T)K_1(T)`, the fixed-point ball, all-future separation bound,
Picard tail bounds, compact endpoint recurrence, and projected residual
recurrence below become effective consequences of the prescribed scattering
data `(m,v,c)`.

## All-Future Separation And No-Collision Bound

The same constants give an explicit projected no-collision theorem on the
whole future interval, not only an abstract fixed point. For `t>=T`, the model
branch obeys:

```text
|q_{0,j}(t)-q_{0,i}(t)|
 >= d t - E_pair(T) log(t).
```

Every correction in the closed ball `||w||<=R` satisfies:

```text
|w_j(t)-w_i(t)| <= 2R log(t)^2/t.
```

Therefore every branch in the fixed-point ball has:

```text
|q_j(t)-q_i(t)|
 >= d t - E_pair(T) log(t) - 2R log(t)^2/t.
```

Since both `log(t)/t` and `log(t)^2/t^2` decrease for `t>=T>e^2`, the lower
bound can be made uniform in scale:

```text
|q_j(t)-q_i(t)|
 >= [d - E_pair(T)log(T)/T - 2R log(T)^2/T^2] t.
```

With the choices above, the bracket is at least `d/4`, so:

```text
|q_j(t)-q_i(t)| >= (d/4)t,       t>=T.
```

Thus the projected branch is collision-free for all future time. The same
estimate applies to every finite Picard approximant `q_N=q_0+w_N`, because the
self-map argument keeps all iterates inside the same ball. Consequently the
finite verify-by-residual approximants and the limiting Newtonian branch live
in one common collision-free tube on `[T,infinity)`.

## Center-Of-Mass And Momentum Matching

The scattering construction also preserves the inertial center-of-mass data
exactly. Newton's internal forces satisfy:

```text
sum_i m_i A_i(q) = 0
```

for every noncollision configuration `q`. Therefore:

```text
sum_i m_i B_i = sum_i m_i A_i(v) = 0.
```

The model branch has:

```text
sum_i m_i q_{0,i}(t)
  = t sum_i m_i v_i + sum_i m_i c_i,
```

because the logarithmic term has zero mass sum. Moreover, for every correction
`w` inside the collision-free tube:

```text
sum_i m_i [A_i(q_0(s)+w(s)) - B_i/s^2] = 0.
```

Thus:

```text
sum_i m_i (Phi w)_i(t) = 0.
```

Starting from `w_0=0`, every Picard approximant has zero mass-weighted
correction, and so does the limit. Consequently the constructed branch and
every finite Picard approximant satisfy the exact center-of-mass identity:

```text
sum_i m_i q_i(t)
  = t sum_i m_i v_i + sum_i m_i c_i,
```

and the total linear momentum is:

```text
sum_i m_i q_i'(t) = sum_i m_i v_i.
```

Thus the prescribed asymptotic velocity and offset data are not merely endpoint
parameters; their mass-weighted parts are exactly the inertial center-of-mass
motion of the projected Newtonian branch.

## Angular-Momentum And Energy Matching

The same scattering branch also has explicit classical invariants determined
by the asymptotic data. Write `a wedge b` for the angular-momentum bivector
with components `a_alpha b_beta - a_beta b_alpha`; in the planar case this is
the usual signed scalar angular momentum.

First, the logarithmic term carries no net angular-momentum drift. Pairwise
centrality gives:

```text
sum_i m_i v_i wedge A_i(v)
 = sum_{i<j} m_i m_j (v_i-v_j) wedge (v_j-v_i)/|v_j-v_i|^3
 = 0.
```

Thus:

```text
sum_i m_i v_i wedge B_i = 0.
```

For the exact branch:

```text
q_i(t)  = t v_i - B_i log(t) + c_i + w_i(t),
q_i'(t) = v_i - B_i/t + w_i'(t),
```

where the fixed-point estimates give:

```text
w_i(t) = O(log(t)^2/t),
w_i'(t) = O(log(t)^2/t^2).
```

The total angular momentum is conserved on `[T,infinity)` because:

```text
d/dt sum_i m_i q_i wedge q_i'
  = sum_i m_i q_i wedge A_i(q)
  = 0,
```

again by pairwise centrality. Taking the limit of the displayed asymptotic
expansion as `t -> infinity`, the possible `log(t)` term is
`log(t) sum_i m_i v_i wedge B_i`, hence vanishes, and the constant
`-sum_i m_i v_i wedge B_i` vanishes for the same reason. All terms containing
`w` or `w'` go to zero by the position and velocity tail bounds. Therefore the
constructed branch has the exact angular-momentum bivector:

```text
L = sum_i m_i c_i wedge v_i.
```

The total energy is matched just as directly. The all-future separation bound
gives:

```text
|q_j(t)-q_i(t)| >= (d/4)t,
```

so the potential energy is `O(1/t)`. The velocity estimate gives
`q_i'(t) = v_i + O(1/t)`, hence:

```text
1/2 sum_i m_i |q_i'(t)|^2 -> 1/2 sum_i m_i |v_i|^2.
```

Since Newtonian energy is conserved along the collision-free branch, the
constant energy of the constructed solution is:

```text
H = 1/2 sum_i m_i |v_i|^2.
```

Thus the prescribed distinct asymptotic velocities and offsets determine not
only the branch endpoint behavior and inertial center-of-mass motion, but also
the exact angular momentum and energy of this nonhomothetic scattering family.

## Analytic Dependence On Scattering Data

The fixed-point construction is not only pointwise in one chosen asymptotic
state. It gives a local analytic chart of scattering branches.

Fix one data pair `(v,c)` with distinct asymptotic velocities, and choose a
small parameter neighborhood `K` of nearby data `(tilde v, tilde c)` such that:

```text
d_0 = inf_K min_{i<j} |tilde v_j-tilde v_i| > 0,
|tilde c_i| <= C_0,
|A(tilde v)_i| <= B_0.
```

The acceleration map is analytic on the open set of distinct velocity
configurations, so:

```text
tilde B = A(tilde v)
```

depends analytically on `tilde v`. Using `d_0`, `C_0`, and `B_0` in the
estimates above, choose one sufficiently large `T` and one radius `R` so that
every parameter in `K` satisfies the same model separation estimate, the same
self-map bound, and a uniform contraction factor:

```text
||Phi_(tilde v,tilde c) w - Phi_(tilde v,tilde c) z||
  <= kappa ||w-z||,       kappa < 1.
```

For `(tilde v,tilde c) in K` and `||w||<=R`, the projected argument:

```text
s tilde v - A(tilde v) log(s) + tilde c + w(s)
```

stays in one common collision-free tube. On that tube the Newtonian
acceleration is analytic in the positions. The integrand:

```text
A(s tilde v - A(tilde v)log(s) + tilde c + w(s))
  - A(tilde v)/s^2
```

is therefore analytic in the finite-dimensional parameters and in the Banach
variable `w`. The same integrable majorants used in the contraction proof allow
termwise differentiation under the integral defining `Phi`, so:

```text
(tilde v,tilde c,w) -> Phi_(tilde v,tilde c)(w)
```

is analytic as a map from `K x X_T` to `X_T`.

The fixed point is the zero of:

```text
G(tilde v,tilde c,w) = w - Phi_(tilde v,tilde c)(w).
```

At every fixed point:

```text
D_w G = I - D_w Phi
```

is invertible by the Neumann series because `||D_w Phi|| <= kappa < 1`. The
analytic implicit-function theorem in Banach spaces then gives an analytic map:

```text
(tilde v,tilde c) -> w_(tilde v,tilde c) in X_T.
```

Consequently the projected Newtonian branch:

```text
q(t;tilde v,tilde c)
 = t tilde v - A(tilde v)log(t) + tilde c + w_(tilde v,tilde c)(t)
```

depends analytically on the asymptotic velocities and offsets throughout this
scattering chart. Finite Picard approximants are analytic by induction and
converge uniformly in the same all-future weighted norm to this analytic
branch. This makes the hyperbolic scattering construction a genuine local
lift/construct/project chart, not a separately tuned solution for one endpoint
datum.

## Picard Series And All-Future Tail Bound

The same contraction gives a constructive series, not only an existence
statement. Set:

```text
w_0(t) = 0,
w_{n+1}(t) = Phi(w_n)(t),
Delta_n(t) = w_{n+1}(t)-w_n(t).
```

Let:

```text
kappa = L K_2(T),
eta = L E_body(T) K_1(T).
```

The previous estimates give:

```text
||Delta_0|| = ||Phi(0)|| <= eta,
||Delta_n|| <= kappa^n eta.
```

Therefore:

```text
w(t) = sum_{n=0}^infinity Delta_n(t)
```

converges in `X_T`, and the truncation after `N` Picard corrections has the
all-future weighted remainder:

```text
||w - w_N|| <= kappa^N eta / (1-kappa).
```

Equivalently, for every `t >= T`:

```text
|w(t)-w_N(t)|_* <= [log(t)^2/t] kappa^N eta/(1-kappa).
```

This is the escape-tail analogue of the desired global recurrence estimate:
one set of constants controls the error for the entire future interval
`[T,infinity)`, and each retained Picard correction improves the bound by the
same factor `kappa < 1`.

## Velocity Tail And Asymptotic Velocity Bound

The same recurrence also controls the projected velocities. Differentiating the
fixed-point operator gives:

```text
(Phi w)'(t)
  = - int_t^infinity [A(q_0(s)+w(s)) - B/s^2] ds.
```

For differences:

```text
|(Phi w)'(t)-(Phi z)'(t)|
 <= L ||w-z|| int_t^infinity log(s)^2/s^4 ds.
```

The remaining integral is explicit:

```text
int_t^infinity log(s)^2/s^4 ds
 = t^-3 [
     (1/3)log(t)^2 + (2/9)log(t) + 2/27
   ].
```

Use the weighted velocity seminorm:

```text
||f'||_v = sup_{t>=T} t^2 |f'(t)| / log(t)^2.
```

Also:

```text
int_t^infinity log(s)/s^3 ds
 = t^-2 [ (1/2)log(t) + 1/4 ].
```

Define:

```text
K_1^v(T) = 1/(2 log T) + 1/(4 log(T)^2),
K_2^v(T) = (1/T)[1/3 + 2/(9 log T) + 2/(27 log(T)^2)].
```

Then the first Picard increment satisfies:

```text
||Delta_0'||_v <= L E_body(T) K_1^v(T),
```

and for every `n>=1`:

```text
||Delta_n'||_v
 <= L K_2^v(T) ||Delta_{n-1}||
 <= L K_2^v(T) kappa^(n-1) eta.
```

Thus, after `N>=1` Picard corrections:

```text
||w' - w_N'||_v
 <= L K_2^v(T) eta kappa^(N-1)/(1-kappa).
```

Equivalently, for every `t>=T`:

```text
|w'(t)-w_N'(t)|_*
 <= [log(t)^2/t^2] L K_2^v(T) eta kappa^(N-1)/(1-kappa).
```

The full correction has:

```text
||w'||_v
 <= L E_body(T)K_1^v(T) + L K_2^v(T) eta/(1-kappa).
```

Since:

```text
q'(t) = v - B/t + w'(t),
```

the asymptotic velocity is certified by the all-future bound:

```text
|q'(t)-v|_*
 <= |B|_*/t
    + [log(t)^2/t^2]
      [L E_body(T)K_1^v(T) + L K_2^v(T) eta/(1-kappa)].
```

This proves not only `q_i(t)/t -> v_i`, but also `q_i'(t) -> v_i`, with a
uniform future velocity tail for finite Picard truncations.

## Truncated Newton Residual Bound

The Picard series also gives a direct verification estimate for the finite
approximants. Define:

```text
q_N(t) = q_0(t) + w_N(t),       N >= 1.
```

Because `w_N = Phi(w_{N-1})`,

```text
w_N''(t) = A(q_0(t)+w_{N-1}(t)) - B/t^2.
```

Since `q_0''(t)=B/t^2`, this gives:

```text
q_N''(t) = A(q_{N-1}(t)).
```

Thus the Newton residual of the `N`th finite Picard approximant is exactly:

```text
Res_N(t) = q_N''(t) - A(q_N(t))
         = A(q_{N-1}(t)) - A(q_N(t)).
```

The same Lipschitz tube and Picard increment bound imply, for every `t >= T`,

```text
|Res_N(t)|_*
 <= (L/t^3) |w_N(t)-w_{N-1}(t)|_*
 <= L eta kappa^(N-1) log(t)^2 / t^4.
```

So every retained correction reduces the certified Newton residual envelope by
the same factor `kappa`, uniformly for all future time. In the limit
`N -> infinity`, the residual tends to zero on `[T,infinity)`, and the
projected curve solves Newton's equations there.

## Finite Taylor Handoff To The All-Future Scattering Chart

The fixed-point chart is an all-future chart starting at a large physical time
`T`. It can be glued to the ordinary finite Taylor atlas without introducing a
new proof layer.

Assume an ordinary collision-free Taylor atlas covers a compact interval
`[a,T]` and its final exact state is the scattering branch state
`(q(T),q'(T))` constructed above. Equivalently, the final chart endpoint is:

```text
q(T)  = v T - B log(T) + c + w(T),
q'(T) = v - B/T + w'(T).
```

The scattering proof supplies the handoff enclosure explicitly. For any
retained Picard truncation `w_N`:

```text
|w(T)-w_N(T)|_*
 <= [log(T)^2/T] kappa^N eta/(1-kappa),
```

and, for `N>=1`:

```text
|w'(T)-w_N'(T)|_*
 <= [log(T)^2/T^2] L K_2^v(T) eta kappa^(N-1)/(1-kappa).
```

The full branch lies in the endpoint tube:

```text
|q(T)-[v T - B log(T) + c]|_* <= R log(T)^2/T,
|q'(T)-[v-B/T]|_*
 <= log(T)^2/T^2 [
      L E_body(T)K_1^v(T) + L K_2^v(T) eta/(1-kappa)
    ].
```

At the same endpoint every pair separation has the positive lower bound:

```text
sigma_T =
  d T - E_pair(T)log(T) - 2R log(T)^2/T
 >= (d/4)T.
```

Therefore the endpoint is an ordinary analytic initial condition with a
nonzero pair-distance margin. The standard local Taylor existence theorem gives
ordinary charts immediately to the left and right of `T`, and the finite atlas
on `[a,T]` glues to the scattering chart by ODE uniqueness. The resulting
piecewise representation is one Newtonian solution on `[a,infinity)`.

The verification envelope is also piecewise explicit. On each finite Taylor
chart it is the chart's usual Cauchy tail and coefficient-residual bound. On
the all-future scattering chart it is:

```text
|q(t)-q_N(t)|_*
 <= [log(t)^2/t] kappa^N eta/(1-kappa),

|q'(t)-q_N'(t)|_*
 <= [log(t)^2/t^2] L K_2^v(T) eta kappa^(N-1)/(1-kappa),

|q_N''(t)-A(q_N(t))|_*
 <= L eta kappa^(N-1) log(t)^2/t^4,
        t >= T.
```

Thus a finite early-time atlas plus one log-subtracted scattering chart gives a
genuine all-future recurrence bound for this hyperbolic-scattering class. This
is still not an asymptotic-completeness theorem: it applies to branches already
known to enter the distinct-asymptotic-velocity scattering tube and whose
pre-`T` segment has a collision-free finite Taylor cover.

## Compact Physical-Time Endpoint Tail

The same fixed-point estimates also make the scattering chart compatible with
the global physical-time compactification recommended by the unrestricted
atlas route. Let:

```text
u = tanh(lambda t),
tau(u) = 1/t = lambda / atanh(u),
u_T = tanh(lambda T).
```

For `u_T <= u < 1`, define the scaled endpoint coordinate:

```text
X(u) = tau(u) q(1/tau(u)).
```

The unscaled coordinate `q(t)` diverges at `u=1`, so `q` is the wrong endpoint
unknown. The forced log-subtracted scaled coordinate is:

```text
Y(u) = X(u) - B tau(u) log(tau(u)),       B=A(v).
```

Using the scattering representation:

```text
q(t) = v t - B log(t) + c + w(t),
```

and `log(tau)=-log(t)`, this becomes:

```text
Y(u) = v + c tau(u) + tau(u) w(1/tau(u)).
```

Therefore the fixed-point ball `||w||<=R` gives the compact-endpoint bound:

```text
|Y(u) - v - c tau(u)|_*
  <= R tau(u)^2 log(1/tau(u))^2.
```

For the finite Picard approximation `w_N`, define:

```text
Y_N(u) = v + c tau(u) + tau(u) w_N(1/tau(u)).
```

The all-future Picard tail transfers directly to compact time:

```text
|Y(u)-Y_N(u)|_*
 <= tau(u)^2 log(1/tau(u))^2
    kappa^N eta/(1-kappa).
```

Once `T>e`, the function `tau^2 log(1/tau)^2` decreases as `t=1/tau`
increases. Hence each compact endpoint tail `[u_*,1)` has a single scalar
remainder bound obtained at its left endpoint. This is the compact-time form
of the all-future recurrence estimate: the infinite physical future is a
bounded endpoint interval, the lifted coordinate has a finite limit

```text
lim_{u -> 1^-} Y(u) = v,
```

and every retained Picard correction improves the whole endpoint tail by the
same factor `kappa`.

Projection back to Newtonian motion is exact away from the endpoint:

```text
q(t(u)) = [Y(u) + B tau(u) log(tau(u))] / tau(u).
```

Thus the compact physical-time representation does not pretend the unscaled
position is bounded. It lifts the escape branch into the bounded
log-subtracted scaled coordinate, constructs the endpoint there, and projects
back only for finite physical target times.

## Compact Endpoint First Jet And Offset Recovery

The log-subtracted compact endpoint also has the correct first jet in the
endpoint variable `tau`. Write:

```text
Y(tau)=v+c tau+tau w(1/tau).
```

The fixed-point correction is differentiable on `0<tau<=1/T`, and:

```text
dY/dtau = c + w(1/tau) - tau^-1 w'(1/tau).
```

From the fixed-point ball and the velocity estimate:

```text
|w(t)|_* <= R log(t)^2/t,
|w'(t)|_* <= W_v log(t)^2/t^2,
```

where:

```text
W_v = L E_body(T)K_1^v(T) + L K_2^v(T) eta/(1-kappa).
```

Thus:

```text
|dY/dtau - c|_*
 <= (R+W_v) tau log(1/tau)^2.
```

In particular:

```text
lim_{tau -> 0^+} dY/dtau = c.
```

So the asymptotic offset is recovered as the first log-subtracted scaled
endpoint jet. The same calculation gives a constructive derivative tail for
finite Picard approximants. For `N>=1`:

```text
d/dtau [Y(tau)-Y_N(tau)]
 = [w-w_N](1/tau) - tau^-1 [w'-w_N'](1/tau),
```

and hence:

```text
|dY/dtau-dY_N/dtau|_*
 <= tau log(1/tau)^2 [
      kappa^N eta/(1-kappa)
      + L K_2^v(T) eta kappa^(N-1)/(1-kappa)
    ].
```

This is the first-jet companion to the compact endpoint value bound. It shows
that the lifted scattering chart carries both endpoint velocity data `v` and
offset data `c` with uniform all-future Picard remainders. The derivative is
claimed in the endpoint coordinate `tau=lambda/atanh(u)`, not directly in `u`;
`u` deliberately compactifies infinite physical time and has a singular
Jacobian at the endpoint.

## Compact Endpoint Equation Residual

The finite Picard approximants also verify the lifted endpoint equation
directly. Put:

```text
X_N(tau)=Y_N(tau)+B tau log(tau),
q_N(t)=X_N(1/t)/tau,       tau=1/t.
```

The exact projection identity gives:

```text
q_N''(t)-A(q_N(t))
 = tau^2 [
     tau Y_N''(tau)
     - (A(Y_N(tau)+B tau log(tau))-B)
   ].
```

Thus the compact endpoint residual:

```text
E_N(tau)
 = tau Y_N''(tau)
   - (A(Y_N(tau)+B tau log(tau))-B)
```

is exactly the physical Newton residual divided by `tau^2`. From the truncated
Newton residual estimate:

```text
|q_N''(t)-A(q_N(t))|_*
 <= L eta kappa^(N-1) log(t)^2/t^4,
```

we obtain the endpoint-equation verification bound:

```text
|E_N(tau)|_*
 <= L eta kappa^(N-1) tau^2 log(1/tau)^2,
        0<tau<=1/T.
```

As with the endpoint value tail, the scalar factor is decreasing toward
`tau=0` once `T>e`, and each additional Picard correction improves the lifted
equation residual by the same factor `kappa`. In the limit, `E_N -> 0`
uniformly on every compact endpoint tail, so the limiting `Y` solves:

```text
tau Y'' = A(Y+B tau log(tau))-B
```

throughout the punctured compact endpoint interval. This is the
construct-in-the-lifted-space verification for the scattering chart; projection
then gives Newton's equation at every finite physical target time.

## Compact Endpoint Dyadic Shell Recurrence

The compact endpoint estimates above also give an honest all-future recurrence
over infinitely many endpoint shells. Let:

```text
tau_0 = 1/T,              tau_n = 2^(-n) tau_0,
I_n = [tau_{n+1}, tau_n],
L_n = log(1/tau_n) = log(T) + n log(2).
```

Assume `log(T) > log(2)/(sqrt(2)-1)`. For any exponent `a>=1`, the scalar
factor

```text
F_a(tau)=tau^a log(1/tau)^2
```

is increasing as a function of `tau` on every shell `I_n`, because
`log(1/tau_n) >= log(T) > 2/a`. Hence the maximum of `F_a` on `I_n` is
`F_a(tau_n)`. Consecutive shell maxima obey:

```text
F_a(tau_{n+1})/F_a(tau_n)
 = 2^(-a) ((L_n+log(2))/L_n)^2
 <= 2^(-a) (1+log(2)/log(T))^2.
```

The first-jet envelope has `a=1`, while the compact endpoint value tail and
the lifted endpoint residual have `a=2`. Therefore all three are controlled by
the common ratio:

```text
r_T = (1/2)(1+log(2)/log(T))^2 < 1.
```

If:

```text
V_N(tau)=C_V tau^2 log(1/tau)^2,
J_N(tau)=C_J tau  log(1/tau)^2,
E_N(tau)=C_E tau^2 log(1/tau)^2,
```

denote respectively the finite Picard value tail, first-jet tail, and compact
endpoint residual envelope, then their shell maxima:

```text
V_{N,n}=sup_{tau in I_n} V_N(tau),
J_{N,n}=sup_{tau in I_n} J_N(tau),
E_{N,n}=sup_{tau in I_n} E_N(tau),
```

satisfy the recurrence:

```text
V_{N,n+1} <= r_T V_{N,n},
J_{N,n+1} <= r_T J_{N,n},
E_{N,n+1} <= r_T E_{N,n}.
```

Consequently the infinite endpoint-shell tails are summable:

```text
sum_{n>=0} V_{N,n} <= V_{N,0}/(1-r_T),
sum_{n>=0} J_{N,n} <= J_{N,0}/(1-r_T),
sum_{n>=0} E_{N,n} <= E_{N,0}/(1-r_T).
```

This closes the all-future recurrence and summability algebra for the compact
scattering endpoint. It is stronger than monotone decay on sampled future
times: every shell in the infinite compact endpoint tail is covered by one
recursive bound. The statement remains scoped to the prescribed
distinct-asymptotic-velocity scattering class; it is not an asymptotic
classification of arbitrary solutions.

## Executable Constructor

The constants and shell recurrences above are implemented in
`three_body_symmetry/escape_endpoint.py`. Given masses, distinct asymptotic
velocities `v`, offsets `c`, and a proposed start time `T`, the constructor
computes:

```text
B=A(v),
d=min_{i<j}|v_i-v_j|,
L=256 sum(m_i)/d^3,
kappa=L K_2(T),
eta=L E_body(T)K_1(T),
R=2 eta,
```

plus the pair-separation loss, tube loss, velocity-transfer constants, and
dyadic shell ratio. It certifies the start only if:

```text
kappa<1,
pair separation loss <= d/2,
tube loss <= d/8,
r_T=(1/2)(1+log(2)/log(T))^2<1.
```

`find_scattering_endpoint_tail_start(...)` performs the terminating doubling
search for such a `T`. Once certified,
`construct_scattering_endpoint_dyadic_recurrence(...)` returns the all-future
shell bounds:

```text
V_{N,n}, J_{N,n}, E_{N,n},
sum_n V_{N,n}, sum_n J_{N,n}, sum_n E_{N,n},
```

for value, first-jet, and lifted endpoint residual envelopes. This makes the
prescribed-scattering endpoint recurrence a constructor-derived proof object
rather than a manual witness.

## Past-Infinity Scattering By Time Reversal

The same construction covers a prescribed past hyperbolic end. Let `t<=-T` and
write `s=-t>=T`. If the past asymptotic velocity in physical time is `v`, then
the time-reversed curve:

```text
Q(s)=q(-s)
```

has future asymptotic velocity:

```text
V=-v.
```

The Newtonian acceleration is odd under global inversion, so:

```text
A(V)=A(-v)=-A(v).
```

Applying the future theorem to `Q` gives:

```text
Q(s)=V s - A(V) log(s) + c + w_-(s)
     = -v s + A(v) log(s) + c + w_-(s).
```

Returning to physical past time `t=-s`:

```text
q(t)=v t + A(v) log(-t) + c + w_-(-t),        t<=-T.
```

Thus the logarithmic sign is the opposite of the future-end formula with the
same physical asymptotic velocity `v`. This sign is forced by Newton's
equation: for `t<0`, the leading acceleration of `q=t v` is `-A(v)/t^2`, and:

```text
d^2[A(v)log(-t)]/dt^2 = -A(v)/t^2.
```

All constants in the future theorem are unchanged after replacing `v` by
`V=-v`: the velocity gaps are the same, `|A(V)|=|A(v)|`, the pair-separation
tube is identical after time reversal, and the Picard contraction takes place
on the same positive half-line `s>=T`. Therefore every future estimate above
has a past counterpart with `s=-t`.

In compact physical time, the past endpoint is `u->-1^+`. Use:

```text
tau_-(u)= -lambda / atanh(u) = 1/(-t) > 0.
```

The scaled past coordinate:

```text
X_-(tau)=tau q(-1/tau)
```

has:

```text
X_-(tau) = -v - A(v) tau log(tau) + c tau + tau w_-(1/tau).
```

Equivalently, with `B_-=A(-v)=-A(v)`, the same endpoint subtraction as the
future chart gives:

```text
Y_-(tau)=X_-(tau)-B_- tau log(tau)
        = -v + c tau + tau w_-(1/tau).
```

Consequently:

```text
Y_-(tau)->-v,
dY_-/dtau -> c,
```

and the value, first-jet, and lifted-equation residual tails are exactly the
future bounds with `v` replaced by `-v`. Projection back to physical past time
is:

```text
q(t)= [Y_-(tau)+B_- tau log(tau)]/tau,
tau=1/(-t),        t<0.
```

Together with the future chart, this gives the local-at-infinity scattering
atlas at both ends of the real line for prescribed distinct asymptotic
velocities. It is still not asymptotic completeness: it constructs and
verifies branches with chosen scattering data at the ends; it does not prove
that every three-body solution has such data.

## Two-Ended Scattering Atlas With A Finite Middle

The past and future endpoint charts give an all-real-time representation for a
scattering branch once a collision-free finite middle segment is known. This is
the scattering analogue of the compact finite-atlas plus endpoint-chart route.

Assume:

1. A past scattering chart is constructed on `(-infinity,-T_-]` with physical
   asymptotic data `(v_-,c_-)` and endpoint state
   `(q_-(-T_-),q_-'(-T_-))`.
2. A future scattering chart is constructed on `[T_+,infinity)` with physical
   asymptotic data `(v_+,c_+)` and endpoint state
   `(q_+(T_+),q_+'(T_+))`.
3. A finite ordinary Taylor atlas covers `[-T_-,T_+]`, is collision-free there,
   and has endpoint states equal to the two scattering endpoint states.

Then the three pieces glue by uniqueness for Newton's ODE. The past scattering
chart and the first ordinary chart agree at `-T_-`; the finite ordinary charts
agree on overlaps by the usual analytic Taylor uniqueness; and the last
ordinary chart agrees with the future scattering chart at `T_+`. Therefore the
piecewise curve is one Newtonian solution on all of `R`.

The verification envelope is also a single piecewise formula. On the compact
middle interval use the finite Taylor atlas' Cauchy tails and coefficient
residuals. On the future endpoint use the future bounds:

```text
|q-q_N^+|_* <= log(t)^2 kappa_+^N eta_+/(t(1-kappa_+)),
|q'-q_N^{+prime}|_* <= log(t)^2 L_+K_{2,+}^v eta_+ kappa_+^(N-1)
                       /(t^2(1-kappa_+)),
|Res_N^+(t)|_* <= L_+ eta_+ kappa_+^(N-1)log(t)^2/t^4.
```

On the past endpoint, with `s=-t`, use the same formulas with the past
constants:

```text
|q-q_N^-|_* <= log(s)^2 kappa_-^N eta_-/(s(1-kappa_-)),
|q'-q_N^{-prime}|_* <= log(s)^2 L_-K_{2,-}^v eta_- kappa_-^(N-1)
                       /(s^2(1-kappa_-)),
|Res_N^-(t)|_* <= L_- eta_- kappa_-^(N-1)log(s)^2/s^4.
```

In compact physical time `u=tanh(lambda t)`, the same all-real-time solution is
covered by three bounded regions:

```text
[-1,u_-]        past log-subtracted scaled endpoint chart,
[u_-,u_+]       finite ordinary Taylor atlas in physical or compact time,
[u_+,1]         future log-subtracted scaled endpoint chart,
```

where `u_- = tanh(-lambda T_-)` and `u_+ = tanh(lambda T_+)`. The endpoint
charts remain bounded after scaling and log subtraction, while unscaled
positions are recovered only for finite `t`. Thus, under explicit finite-middle
handoff hypotheses, the lift/construct/project/verify pattern gives a genuine
all-real-time scattering atlas. The theorem still does not assert that
arbitrary data enter this class, or that arbitrary incoming and outgoing
scattering data can be connected by a collision-free middle atlas.

## Finite Collision Middle Atlas

The collision-free ordinary middle can be replaced by any finite verified
compact middle atlas. This is the all-real scattering composition obtained by
combining the endpoint charts with the finite compact-interval collision
theorems.

Assume the past and future scattering charts above are constructed with
handoff states at `-T_-` and `T_+`. Instead of requiring a collision-free
ordinary Taylor atlas on `[-T_-,T_+]`, assume a finite middle atlas whose chart
list is made of:

```text
ordinary Taylor charts on collision-free pieces,
separated-binary Levi-Civita charts,
identity-selector Fuchsian-log total-collision charts.
```

The middle atlas must have endpoint states equal to the two scattering handoff
states. At each noncollision overlap, neighboring projected charts agree by
analytic Newtonian ODE uniqueness. At each separated binary collision, the
Levi-Civita chart supplies the lifted continuation through `z=0`. At each
zero-angular total collision in the collision-free central-target class, the
identity-selector Fuchsian-log chart supplies the lifted continuation through
`tau=0`. Therefore the past endpoint, finite mixed middle, and future endpoint
again form one projected Newtonian branch on all finite physical target times.

The compact-time cover is still:

```text
[-1,u_-]        past log-subtracted scaled endpoint chart,
[u_-,u_+]       finite mixed middle atlas,
[u_+,1]         future log-subtracted scaled endpoint chart.
```

Only the finite middle budget changes. If the middle chart list contributes
finite value, first-jet, lifted-residual, and physical-residual budgets:

```text
M_V, M_J, M_E, M_R,
```

then the endpoint geometric sums from the previous sections give the same
all-real budgets:

```text
M_V + V^-_{N,0}/(1-r_-) + V^+_{N,0}/(1-r_+),
M_J + J^-_{N,0}/(1-r_-) + J^+_{N,0}/(1-r_+),
M_E + E^-_{N,0}/(1-r_-) + E^+_{N,0}/(1-r_+),
M_R + R^-_{N,0}/(1-r^-_R) + R^+_{N,0}/(1-r^+_R).
```

Thus prescribed two-ended scattering data plus a finite verified collision
middle atlas give an all-real lift/construct/project/verify representation.
The remaining theorem gap is not endpoint recurrence or finite collision
composition; it is the classification/existence problem that would show
arbitrary initial data enter one of the verified global regimes and, in the
scattering case, that the needed finite mixed middle atlas exists for the
chosen endpoint data.

## Two-Ended Scattering With A Nonzero-Angular Compact Middle

There is an important subcase where the finite middle hypothesis above is not
an independent atlas assumption. Suppose the two prescribed scattering endpoint
charts have handoff states at `-T_-` and `T_+`, and suppose the connecting
compact branch on `[-T_-,T_+]` has conserved nonzero centered angular momentum.
Assume also that every binary collision on the compact branch is a
separated-third-body binary event covered by the Levi-Civita local theorem.

Then the compact nonzero-angular finite-atlas theorem applies to the middle
branch. Nonzero angular momentum excludes total collision on the compact
interval. If the binary-event set were infinite, compactness would give a
finite accumulation point; the separated-binary accumulation theorem would
force that accumulation to be total collision, contradicting the nonzero
angular exclusion. Hence the binary events are finite. The complement of the
finite binary-event list is a finite union of collision-free compact intervals,
each covered by finitely many ordinary Taylor charts, and each binary event is
covered by one separated-binary Levi-Civita chart.

Thus the middle chart list has only the two kinds

```text
ordinary Taylor charts on collision-free pieces,
separated-binary Levi-Civita charts.
```

No `identity_selector_total_collision` chart is required or allowed in this
subcase, because total collision has already been excluded by the angular
momentum invariant. Let the finite ordinary and binary middle budgets be

```text
M_V, M_J, M_E, M_R.
```

The endpoint recurrence is unchanged, so the all-real value, first-jet,
lifted-residual, and projected physical-residual budgets are:

```text
M_V + V^-_{N,0}/(1-r_-) + V^+_{N,0}/(1-r_+),
M_J + J^-_{N,0}/(1-r_-) + J^+_{N,0}/(1-r_+),
M_E + E^-_{N,0}/(1-r_-) + E^+_{N,0}/(1-r_+),
M_R + R^-_{N,0}/(1-r^-_R) + R^+_{N,0}/(1-r^+_R).
```

Therefore prescribed two-ended scattering data plus a compact connecting branch
with conserved nonzero centered angular momentum and only separated binary
events give a finite all-real lift/construct/project/verify representation.
This closes the nonzero-angular compact-middle subcase of the prescribed
two-ended scattering theorem. It still does not prove that arbitrary initial
data are scattering, nor that arbitrary prescribed scattering endpoints admit
such a connecting nonzero-angular compact branch.

## Two-Ended Scattering With An Automatic Zero-Angular Finite-Event Middle

There is a parallel zero-angular subcase where the finite mixed middle atlas is
also a consequence rather than an independent input. Suppose the two prescribed
scattering endpoint charts have matching handoff states at `-T_-` and `T_+`.
Assume the compact connecting branch on `[-T_-,T_+]` has finite energy, zero
centered angular momentum, only finitely many collision events, every binary
event away from total collision is a separated-third-body Levi-Civita event,
and every total collision is a positive-mass three-body total-collision event.

At each total collision, the automatic three-body selector compact-atlas
theorem supplies the identity-selector Fuchsian-log chart. The local selector
data are not an extra hypothesis: binary-degenerate approach is excluded,
quotient shape converges to a Lagrange or ordered-Euler target, every
collision-free three-body central target is reduced-hyperbolic, and the
Poincare-Dulac stable normal form gives the finite incoming Fuchsian-log
selector list. The identity rule copies those selector constants to the
outgoing side and the triangular recurrence constructs the outgoing lifted
branch.

Since the collision-event set is finite, choose disjoint neighborhoods of all
automatic total-collision charts and all separated-binary Levi-Civita charts.
The remaining complement is a finite union of collision-free compact
intervals, covered by ordinary Taylor charts. Thus the middle chart list is
finite and consists of:

```text
ordinary Taylor charts on collision-free pieces,
separated-binary Levi-Civita charts,
automatic identity-selector Fuchsian-log total-collision charts.
```

At noncollision overlaps, analytic Newtonian ODE uniqueness glues neighboring
projected charts. At binary collisions, the Levi-Civita chart supplies the
lifted continuation. At zero-angular total collisions, the automatic
identity-selector chart supplies the lifted continuation through `tau=0`, with
zero angular momentum and matching finite energy.

Let the finite middle budgets obtained by summing those ordinary, binary, and
automatic total-collision chart bounds be:

```text
M_V, M_J, M_E, M_R.
```

The endpoint dyadic recurrences are unchanged. Therefore the whole all-real
scattering branch has finite verification budgets:

```text
M_V + V^-_{N,0}/(1-r_-) + V^+_{N,0}/(1-r_+),
M_J + J^-_{N,0}/(1-r_-) + J^+_{N,0}/(1-r_+),
M_E + E^-_{N,0}/(1-r_-) + E^+_{N,0}/(1-r_+),
M_R + R^-_{N,0}/(1-r^-_R) + R^+_{N,0}/(1-r^+_R).
```

This closes the all-real prescribed-scattering subcase with a finite
zero-angular positive-mass three-body event middle. The remaining assumptions
are global: the endpoints must actually be connected by such a compact branch,
the compact branch must have only finitely many events, and arbitrary-data
scattering classification/scattering-map existence are still not proved.

## Effective Two-Ended Tail Starts

The two endpoint start times can be chosen effectively from the prescribed
scattering data. For the future end, apply the effective tail-start search to
`(v_+,c_+)` and obtain `T_+`. For the past end, apply the same search to the
time-reversed future data:

```text
V_-=-v_-,        C_-=c_-,
```

and call the resulting positive half-line start `T_-`. Since velocity gaps,
the Lipschitz constant, and the acceleration norms are unchanged by
`v -> -v`, this is exactly the past endpoint chart after the time reversal
`s=-t`.

For both endpoints, require the same finite list of inequalities:

```text
log(T_\pm) > log(2)/(sqrt(2)-1),
E_pair^\pm(T_\pm) log(T_\pm)/T_\pm <= d_\pm/2,
L_\pm K_2^\pm(T_\pm) <= 1/2,
2L_\pm E_body^\pm(T_\pm)K_1^\pm(T_\pm) log(T_\pm)^2/T_\pm^2 <= d_\pm/8.
```

The first inequality is the dyadic shell-ratio condition; the last three are
the separation, contraction, and fixed-point tube conditions. The same limits
used in the one-ended proof show that doubling `T_+` and `T_-` terminates.

Therefore the prescribed two-ended scattering data give computable endpoint
charts before any finite-middle handoff is considered. If a collision-free
finite ordinary Taylor atlas then matches the two endpoint states on
`[-T_-,T_+]`, the all-real verification budget is fully effective:

```text
middle finite budget
 + past geometric endpoint sum with r_-
 + future geometric endpoint sum with r_+.
```

This closes the recurrence start for the prescribed two-ended scattering
subcase. The remaining hard problem is not the endpoint recurrence; it is the
existence/classification question of whether arbitrary data enter such
two-ended scattering charts and which endpoint data are connected by a
collision-free middle branch.

## Two-Ended Compact Shell Recurrence With A Finite Middle

The dyadic endpoint recurrence composes with the two-ended atlas. Let the past
and future handoff times be `T_-` and `T_+`, and define endpoint variables:

```text
tau^- = 1/(-t),        t <= -T_-,
tau^+ = 1/t,           t >=  T_+.
```

On the two endpoint regions use dyadic shells:

```text
tau^-_n = 2^(-n)/T_-,
tau^+_n = 2^(-n)/T_+.
```

Assume the endpoint starts are large enough that:

```text
log(T_-) > log(2)/(sqrt(2)-1),
log(T_+) > log(2)/(sqrt(2)-1).
```

Then the past and future compact endpoint value, first-jet, and lifted-residual
shell maxima satisfy geometric recurrences with:

```text
r_- = (1/2)(1+log(2)/log(T_-))^2 < 1,
r_+ = (1/2)(1+log(2)/log(T_+))^2 < 1.
```

For example, if `V^-_{N,n}`, `J^-_{N,n}`, `E^-_{N,n}` are the shell maxima for
the past log-subtracted scaled endpoint, and `V^+_{N,n}`, `J^+_{N,n}`,
`E^+_{N,n}` are the corresponding future maxima, then:

```text
V^-_{N,n+1} <= r_- V^-_{N,n},    J^-_{N,n+1} <= r_- J^-_{N,n},
E^-_{N,n+1} <= r_- E^-_{N,n},

V^+_{N,n+1} <= r_+ V^+_{N,n},    J^+_{N,n+1} <= r_+ J^+_{N,n},
E^+_{N,n+1} <= r_+ E^+_{N,n}.
```

The finite middle atlas contributes only finitely many Cauchy and coefficient
residual bounds. If `M_V`, `M_J`, and `M_E` denote any valid finite middle
budgets for the value, first-jet, and equation-residual envelopes, the whole
compact real line has finite verification budgets:

```text
M_V + V^-_{N,0}/(1-r_-) + V^+_{N,0}/(1-r_+),
M_J + J^-_{N,0}/(1-r_-) + J^+_{N,0}/(1-r_+),
M_E + E^-_{N,0}/(1-r_-) + E^+_{N,0}/(1-r_+).
```

This recurrence is now executable. `construct_two_ended_scattering_atlas_recurrence(...)`
applies the certified endpoint-start search to the time-reversed past data
`(-v_-,c_-)` and to the future data `(v_+,c_+)`, builds both dyadic endpoint
recurrences, coerces the finite middle list into value, first-jet, lifted
residual, and physical-residual budgets, and returns the two endpoint geometric
sums plus the finite middle budget. The constructor does not infer a scattering
map or a middle connector; those remain explicit hypotheses. It does remove the
test-local arithmetic for this subcase: the all-real endpoint sums now come from
the same fixed-point constants that prove the all-future endpoint recurrence.

Thus, once the finite middle handoff hypotheses hold, the all-real-time
scattering atlas has a finite verification envelope made of one finite middle
list plus two infinite but geometrically summable endpoint recurrences. This is
the endpoint-shell version of the lift/construct/project/verify method on the
whole compact physical-time interval `[-1,1]`. It still remains a theorem for
branches with prescribed scattering ends and a known collision-free middle; it
does not prove arbitrary-data classification or existence of a scattering map.

## Whole-Interval Newton Residual After Projection

The compact endpoint residuals are lifted-equation residuals. Projection back
to physical Newtonian coordinates improves their decay. On the future endpoint:

```text
q_N''(t)-A(q_N(t)) = (tau^+)^2 E_N^+(tau^+),
tau^+ = 1/t.
```

On the past endpoint the time-reversed identity is identical:

```text
q_N''(t)-A(q_N(t)) = (tau^-)^2 E_N^-(tau^-),
tau^- = 1/(-t).
```

Therefore if the lifted residual envelopes are:

```text
|E_N^\pm(tau)|_* <= C_E^\pm tau^2 log(1/tau)^2,
```

then the physical Newton residual envelopes are:

```text
|q_N''-A(q_N)|_* <= C_E^\pm tau^4 log(1/tau)^2.
```

These physical residual shell maxima satisfy the stronger dyadic recurrence:

```text
R^\pm_{N,n+1}
 <= 2^(-4)(1+log(2)/log(T_\pm))^2 R^\pm_{N,n}.
```

In particular they are dominated by the lifted-residual recurrence used above.
If the finite middle atlas has a finite coefficient-residual budget `M_R`, the
whole all-real scattering atlas has the physical Newton residual budget:

```text
M_R + R^-_{N,0}/(1-r^-_R) + R^+_{N,0}/(1-r^+_R),
```

where:

```text
r^\pm_R = 2^(-4)(1+log(2)/log(T_\pm))^2 < 1.
```

Thus the final projection step verifies Newton's equation on every finite
physical target time in the two endpoint regions, while the finite middle
atlas verifies it on the compact middle interval. In the limiting chart, the
endpoint residuals tend to zero by the Picard contraction and the middle
coefficient residuals vanish by the Taylor recurrence, so the glued curve
satisfies Newton's equations on the whole punctured real line.

## Two-Ended Scattering Invariant Matching

The two endpoint charts cannot be chosen independently. If one Newtonian
solution is represented by a past scattering chart with physical asymptotic
data `(v_-,c_-)` and by a future scattering chart with physical asymptotic data
`(v_+,c_+)`, then its classical invariants force:

```text
sum_i m_i v_{-,i} = sum_i m_i v_{+,i},
sum_i m_i c_{-,i} = sum_i m_i c_{+,i},
sum_i m_i c_{-,i} wedge v_{-,i}
    = sum_i m_i c_{+,i} wedge v_{+,i},
1/2 sum_i m_i |v_{-,i}|^2 = 1/2 sum_i m_i |v_{+,i}|^2.
```

The first two identities come from center-of-mass motion. At the future end,

```text
q_i(t)=v_{+,i}t-A_i(v_+)log(t)+c_{+,i}+o(1),
```

while at the past end,

```text
q_i(t)=v_{-,i}t+A_i(v_-)log(-t)+c_{-,i}+o(1).
```

For either sign, the logarithmic center-of-mass term vanishes because the
pairwise central force cancellation gives `sum_i m_i A_i(v)=0`. Hence the same
affine center-of-mass line has the two endpoint slopes and intercepts above.

Angular momentum gives the third identity. In the future expansion the possible
logarithmic drift is proportional to

```text
sum_i m_i v_i wedge A_i(v),
```

and the constant cross term has the same coefficient with the opposite sign.
At the past end the signs reverse, but the same coefficient appears. Pairwise
centrality cancels it:

```text
sum_i m_i v_i wedge A_i(v)
  = sum_{i<j} m_i m_j
      (v_i-v_j) wedge (v_j-v_i)/|v_i-v_j|^3
  = 0.
```

The remaining endpoint limit of angular momentum is therefore
`sum_i m_i c_i wedge v_i` at both ends. Finally, distinct asymptotic velocities
give linear pair separation at each end, so the potential energy tends to zero
and the conserved energy is the asymptotic kinetic energy. Any mismatch in one
of these four quantities obstructs an all-real-time two-ended scattering atlas
with a collision-free finite middle.

This is only a necessary matching theorem. It does not construct the scattering
map, prove that matched endpoint data are connected by a middle atlas, or prove
asymptotic completeness.

## Projection Back To Newtonian Motion

The constructed curve:

```text
q(t) = v t - A(v) log(t) + c + w(t)
```

satisfies Newton's equations by construction. In inverse time `tau=1/t`, the
scaled coordinate:

```text
X(tau) = tau q(1/tau)
```

has:

```text
X(tau) = v + A(v) tau log(tau) + c tau + O(tau^2 log(tau)^2).
```

Thus the log-subtracted endpoint variable:

```text
Y(tau) = X(tau) - A(v) tau log(tau)
```

converges to `v` and has a controlled nonhomothetic asymptotic expansion on
this scattering branch.

## Scope

This is a local-at-infinity convergence theorem for prescribed distinct
asymptotic velocities. It does not prove that every escape solution admits such
asymptotic data, nor does it handle capture, collision, or bounded recurrent
motions. It closes a real convergence subcase needed by the all-future route:
nonhomothetic hyperbolic escape endpoints can be constructed by the
lift/construct/project/verify pattern after subtracting the forced logarithmic
term.
