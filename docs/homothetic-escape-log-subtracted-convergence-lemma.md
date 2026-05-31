# Homothetic Escape Log-Subtracted Convergence Lemma

## Claim

For a positive-energy homothetic escape branch, the inverse-time
log-subtracted endpoint variable is not merely formal. It is represented by a
convergent analytic power series in the two small variables:

```text
tau = 1/t,
rho = tau log(tau).
```

This proves convergence of the escape transseries in the homothetic subcase and
gives a concrete model for the general log-subtracted escape endpoint route.

## Setup

Let `Q` be a central configuration with:

```text
A(Q) = -mu Q,       mu > 0.
```

On a homothetic escape branch:

```text
q_i(t) = R(t) Q_i,
R'' = -mu / R^2,
(1/2) R'^2 - mu / R = E > 0.
```

Set:

```text
c = sqrt(2E),
beta = 2mu / c^2.
```

On the expanding branch:

```text
R' = sqrt(c^2 + 2mu/R).
```

An antiderivative of `dR / sqrt(c^2 + 2mu/R)` is:

```text
F(R)
  = (1/c) [sqrt(R(R+beta)) - beta asinh(sqrt(R/beta))].
```

Thus every time translation of the branch has:

```text
t = t_* + F(R).
```

## Analytic Endpoint Equation

Set:

```text
tau = 1/t,
x(tau) = tau R(t),
rho = tau log(tau).
```

Substituting `R=x/tau` into `1=tau(t_*+F(R))` gives:

```text
H(x,tau,rho) = 0,
```

where:

```text
H(x,tau,rho)
  = tau t_*
    + (1/c) sqrt(x(x+beta tau))
    - (beta tau/c)
        [log(sqrt(x)+sqrt(x+beta tau)) - (1/2)log(beta)]
    + (beta/(2c)) rho
    - 1.
```

This identity uses:

```text
asinh(sqrt(x/(beta tau)))
  = log(sqrt(x)+sqrt(x+beta tau))
    - (1/2)log(beta)
    - (1/2)log(tau).
```

Near the escape endpoint `(x,tau,rho)=(c,0,0)`, the square-root and logarithm
terms in `H` are analytic because `c>0`. Also:

```text
H(c,0,0) = 0,
dH/dx(c,0,0) = 1/c != 0.
```

The analytic implicit function theorem therefore gives a convergent analytic
function:

```text
x = Phi(tau,rho)
```

near `(tau,rho)=(0,0)`.

## Forced Log Coefficient

Differentiate the implicit equation at the endpoint with respect to `rho`:

```text
Phi_rho(0,0)
  = - H_rho / H_x
  = - (beta/(2c)) / (1/c)
  = - beta/2
  = - mu/c^2.
```

Therefore:

```text
x(tau) = c - (mu/c^2) tau log(tau) + analytic terms in tau and rho.
```

Since `B=A(cQ)=-(mu/c^2)Q`, this is exactly the vector log coefficient from the
general hyperbolic scattering lemma:

```text
X_i(tau) = x(tau) Q_i
         = c Q_i + B_i tau log(tau) + convergent analytic remainder.
```

Equivalently, the log-subtracted variable:

```text
Y_i(tau) = X_i(tau) - B_i tau log(tau)
```

has a convergent analytic expansion in `(tau, rho)` at the escape endpoint.

## Consequence

This does not prove convergence for arbitrary hyperbolic scattering data.
It closes the homothetic escape subcase: after the forced logarithmic term is
lifted out, the remaining endpoint construction is convergent by the analytic
implicit function theorem. Any all-data global proof still has to prove the
same kind of convergence or a stronger replacement for nonhomothetic scattering,
and it still has to connect that endpoint analysis to the binary and
triple-collision continuation branches.

## Dyadic All-Future Shell Recurrence

The implicit-function proof also gives an actual all-future recurrence, not
only pointwise convergence at infinity. Since `Phi(tau,rho)` is analytic near
`(0,0)`, choose a closed polydisc:

```text
|tau| <= R_tau,       |rho| <= R_rho
```

inside its convergence domain, and let `M` be a Cauchy majorant for the
log-subtracted scalar endpoint function:

```text
Y(tau,rho)=Phi(tau,rho)+(mu/c^2)rho.
```

The theorem path now derives such an `M` for the scalar homothetic endpoint
instead of accepting it as an opaque global witness.  Write `x=c+y`.  On the
circle `|y|=R_x`, decompose the implicit equation as:

```text
H(c+y,tau,rho) = y/c + Rem(y,tau,rho).
```

If:

```text
R_x + beta R_tau < c,
```

then the square-root and logarithm arguments stay in the right half-plane.  The
constructor bounds the remainder by:

```text
|Rem|
 <= R_tau |t_*|
    + beta R_tau (c+R_x)/(c(c-R_x))
    + beta R_tau L/c
    + beta R_rho/(2c),
```

where `L` is a principal-log bound for
`log(sqrt(x)+sqrt(x+beta tau))-(1/2)log(beta)` on the same polydisc.  When:

```text
|Rem| < R_x/c,
```

Rouche's theorem gives a unique analytic root `x=Phi(tau,rho)` inside
`|x-c|<R_x`.  Therefore:

```text
|Y(tau,rho)| <= c + R_x + (beta/2)R_rho,
```

which supplies the Cauchy majorant `M` used by the dyadic recurrence.

For a start time `T` large enough that:

```text
theta_0 =
  max(1/(T R_tau), log(T)/(T R_rho)) < 1,
```

define dyadic inverse-time shells:

```text
tau_n = 2^(-n)/T,
I_n = [tau_(n+1), tau_n],
L_n = log(1/tau_n)=log(T)+n log(2).
```

On shell `I_n` the two small variables obey:

```text
|tau| <= tau_n,
|rho| = |tau log(tau)| <= tau_n L_n.
```

Moreover:

```text
tau_(n+1)/tau_n = 1/2,

(tau_(n+1)L_(n+1))/(tau_n L_n)
  = (1/2)(1 + log(2)/L_n)
  <= r_T,
```

where:

```text
r_T = (1/2)(1 + log(2)/log(T)) < 1
```

once `T>2`. Therefore:

```text
theta_(n+1) <= r_T theta_n,
theta_n = max(tau_n/R_tau, tau_n L_n/R_rho).
```

If `Y_N` is the truncation of the convergent two-variable power series to total
degree `N`, the Cauchy tail bound has the form:

```text
sup_(I_n) |Y-Y_N|
 <= M theta_n^(N+1)/(1-theta_n).
```

Since `theta_n` is decreasing, consecutive shell tails satisfy:

```text
sup_(I_(n+1)) |Y-Y_N|
 <= r_T^(N+1) sup_(I_n) |Y-Y_N|.
```

Thus the infinite future tail is geometrically summable:

```text
sum_(n>=0) sup_(I_n) |Y-Y_N|
 <= [M theta_0^(N+1)/(1-theta_0)] / [1-r_T^(N+1)].
```

The same argument applies to endpoint derivative tails after lowering the
retained degree by the number of derivatives and using the corresponding
Cauchy derivative majorant. Projection back to physical coordinates is exact
for every finite `t`:

```text
q_i(t)= [Y(tau,rho) - (mu/c^2)Q_i rho] Q_i / tau,
tau=1/t,
rho=tau log(tau).
```

The theorem bridge also records the projected Newton verification. In centered
coordinates the homothetic solution has `q_i(t)=R(t)Q_i` and `v_i(t)=R'(t)Q_i`.
Since the Newtonian acceleration is homogeneous of degree `-2`,

```text
A(RQ)=R^(-2)A(Q).
```

The scalar radial equation gives `R''=-mu/R^2`, so:

```text
q''-A(q)=R^(-2)(-mu Q-A(Q)).
```

Thus the all-future physical residual is bounded by the central-configuration
residual at `R=1` on the expanding branch. The same constructor checks the
centered linear momentum, angular-momentum bivector, and normalized energy
against the homothetic invariant formulas.

So the homothetic escape branch now has a complete endpoint recurrence in the
lifted log-subtracted variables: one finite start shell plus a geometric rule
covers all later physical times. This is still a scoped subcase, but it is an
all-future recurrence bound of the kind required by the global route.

The incoming endpoint is obtained by exact time reversal rather than by a new
asymptotic assumption. If the future endpoint recurrence covers `t >= S` and
the collision time in the normalization above is `T=t_*`, the identity-selector
homothetic continuation `q(T+tau^3)=tau^2u(tau^2)Q` is invariant under
`tau -> -tau` after reversing velocity. Therefore the same dyadic constants
and shell-tail bounds cover the incoming branch for `t <= 2T-S`; compact time
sends that handoff to `tanh(rate (2T-S))` in `(-1,0)`.

The scoped positive-energy homothetic branch now has an all-real atlas
composition. For positive energy on the expanding branch, the backward
continuation reaches a finite total-collision endpoint. The constructor builds
the nonzero-energy homothetic total-collision atlas there, uses the explicit
identity selector, and derives uniform collision-free ordinary Taylor
recurrence bounds for the two finite middle intervals. The collision-free
middle lower bound is not assumed: at the total-collision chart boundary
`tau=a`, the scalar majorant gives `u(a^2)>0` and
`u+a^2u'>0`, so the homothetic radius is positive and increasing away from the
collision. Since the scaled central shape has positive pair distances, both
middle intervals have a positive pair-distance floor; compactness gives finite
position and speed bounds, which feed the ordinary Taylor recurrence
constructor.

This remains a scoped homothetic theorem. It does not classify arbitrary
positive-energy escape data as homothetic, prove nonhomothetic scattering
completeness, or close full global regime exhaustion.

The executable gluing certificate records the finite verification budget
explicitly. If `h` is the uniform ordinary step, `B_mid` is the per-chart
Cauchy tail bound, and the two middle lengths are `L_-` and `L_+`, then the
middle chart counts are bounded by:

```text
N_- = ceil(L_-/h),        N_+ = ceil(L_+/h).
```

The finite middle tail budget is `(N_-+N_+)B_mid`. Adding the future endpoint
tail, the time-reversed past endpoint tail, and the local total-collision
chart tail gives the all-real tail budget recorded by the theorem object. The
same chart counts give an explicit handoff/internal transition-count bound for
the compact atlas composition.

## Executable Constructor

The recurrence above is implemented in
`three_body_symmetry/escape_endpoint.py`. The constructor first derives:

```text
E = (1/2)R_0'^2 - mu/R_0,
c = sqrt(2E),
beta = 2mu/c^2,
t_* = -F(R_0),
B = -mu/c^2.
```

It rejects nonpositive `E`, so the endpoint chart cannot be instantiated for a
non-escape homothetic branch. Given Cauchy polydisc radii `R_tau`, `R_rho`, an
implicit-function radius `R_x`, start time `T`, and retained degree `N`, the
theorem path first derives `M` by the Rouche check above. The lower-level
recurrence constructor can still accept an externally supplied `M`, but a
theorem-level positive-energy homothetic escape certificate requires the
derived `positive_energy_homothetic_implicit_cauchy_majorant` obligation, plus
`positive_energy_homothetic_projection_newton_invariants` for the projection
and invariant ledger. It also derives
`positive_energy_homothetic_past_endpoint_time_reversal` from the certified
future recurrence and `positive_energy_homothetic_all_real_gluing` from the
local total-collision atlas plus the finite middle recurrence. With the
derived or supplied majorant, the recurrence computes:

```text
theta_0=max(1/(TR_tau), log(T)/(TR_rho)),
r_T=(1/2)(1+log(2)/log(T)),
p=N+1-m.
```

The constructor certifies only when `theta_0<1`, `r_T<1`, and `p>0`, then
returns the all-future bound:

```text
M theta_0^p / ((1-theta_0)(1-r_T^p)).
```

Thus this is not a manual witness flag in the theorem pipeline: the future
recurrence bound is computed from the scalar implicit equation, and weaker
polydisc data are rejected.
