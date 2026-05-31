# Total-Collision Generalized Fuchsian Stop Proof

This note records the proof chain used by the pointwise finite-target
atlas-or-stop theorem when a finite target interval reaches an unselected total
collision.  Its role is deliberately narrower than a continuation theorem: under
the maximal-classical policy the atlas stops at the first total collision, and
does not select an outgoing branch.

The proof target is:

For positive masses in dimension two or three, every finite-time total-collision
germ of a noncollision Newtonian three-body solution admits finite generalized
Fuchsian/Puiseux-log entry data, a Cauchy-majorized analytic remainder, and a
verifier-checkable total-stop chart whose projection satisfies Newton's
equations on punctured slabs and collapses all mutual distances at the endpoint.

The checker-facing certificates are:

- `SuppliedGeneralizedFuchsianEntryCertificate`
- `SuppliedGeneralizedFuchsianAnalyticRemainderMajorantCertificate`
- `SuppliedGeneralizedFuchsianFiniteRowTailBudgetCertificate`
- `SuppliedGeneralizedFuchsianStopChartCertificate`
- `TotalCollisionGeneralizedFuchsianStopChartCertificate`
- `GeneralizedFuchsianRemainderMajorantCertificate`

The mathematical chain is split into TC1-TC7.  TC5 and TC6 are the proof-critical
normal-form and majorant steps.

## TC1. Total Collision Forces Zero Centered Angular Momentum

Work in center-of-mass coordinates.  Let

```text
I = sum_i m_i |q_i|^2,        L = sum_i m_i q_i wedge v_i,
U = sum_{i<j} m_i m_j / |q_i-q_j|.
```

At finite total collision, `I(t) -> 0`.  The Sundman inequality has the form

```text
|L|^2 <= C I (K_shape),
```

where `K_shape` is the mass-metric kinetic energy of the normalized shape plus
radial contributions controlled along a finite-energy collision germ.  Since
`I -> 0` and the normalized collision blow-up has bounded shape energy on the
compact shape quotient away from binary degeneration, the conserved centered
angular momentum must be zero:

```text
L = 0.
```

Binary-degenerate normalized approaches are excluded in TC2, so this argument
covers all total-collision approaches after quotienting translations.

## TC2. Binary-Degenerate Normalized Shape Approach Is Impossible

Assume two bodies form a tight pair while the third remains at normalized
distance comparable to the total scale.  In Jacobi coordinates `(x,y)`, total
collision means both `x` and `y` shrink in physical coordinates.  A
binary-degenerate normalized limit would have `|x|/|y| -> 0` or `infinity`.

The tight coordinate satisfies a perturbed Kepler equation:

```text
x'' = -mu_x x/|x|^3 + R_x,
```

where the perturbation `R_x` is lower order at the Jacobi scale forced by total
collapse.  The same blow-up applies to `y`.  Finite energy and zero total
angular momentum imply both Jacobi coordinates have parabolic Kepler collision
scaling:

```text
|x(t)| ~ c_x (t_c-t)^(2/3),
|y(t)| ~ c_y (t_c-t)^(2/3),
```

with positive constants determined by the effective Jacobi masses.  Hence their
ratio has a positive finite limit, contradicting normalized binary degeneration.

Therefore the normalized shape limit is collision-free.

## TC3. Total-Collision Branches Have Central-Configuration Shape Limits

Write the total scale as `rho = sqrt(I)` and normalized shape as
`s = q/rho`, with the mass metric normalization `||s||_m = 1` and center of mass
zero.  By TC2, every normalized limiting shape stays in the compact
collision-free shape quotient.

The Newton equation decomposes into scale and shape equations.  The leading
acceleration scale is

```text
q'' = A(q) = rho^(-2) A(s).
```

The Lagrange-Jacobi identity gives the parabolic scale law

```text
rho(t) = a (t_c-t)^(2/3) + lower-order terms.
```

Substituting `q = rho s` and comparing the leading `rho^(-2)` acceleration terms
shows that any collision-free normalized shape limit `s_*` must solve

```text
A(s_*) = -lambda s_*
```

for the appropriate mass-metric multiplier `lambda > 0`.  Thus `s_*` is a
central configuration.

The quotient by rotations/reflections leaves finitely many central targets for
fixed masses: two equilateral orientations in the planar quotient and the three
Euler collinear orderings, with their spatial embeddings obtained by a fixed
orthonormal two-frame.

## TC4. Reduced Central Targets Are Hyperbolic on the Quotient

After quotienting translation, scale, rotation, and reflection, the total
collision equations are a regular singular system near each collision-free
central target.  For the positive-mass three-body problem the reduced central
targets have no center directions other than the removed group and scale
directions.

Equivalently, the reduced McGehee/Fuchsian linearization has:

- one scale/radial mode fixed by the parabolic collision law;
- neutral modes only from the quotient symmetries already removed;
- stable/unstable shape exponents with nonzero real parts on the reduced shape
  space.

This hyperbolicity supplies the finite list of indicial exponents used in TC5.
Integer, rational, and irrational positive stable exponents are all allowed:
the certificate language records the exponents as algebraic/interval data rather
than assuming an ordinary integer-power Taylor series.

## TC5. Stable Branches Admit Finite Generalized Fuchsian/Puiseux-Log Data

Use the regularized collision time

```text
tau = (t_c-t)^(1/3)
```

and write

```text
q(tau) = tau^2 S(tau),
```

with `S(0)=C`, where `C` is a collision-free central configuration in centered
mass coordinates.  Split the variables into scale, removed group directions,
and reduced shape coordinates.  On the reduced stable manifold, the local
system has Fuchsian form

```text
tau dX/dtau = B X + F(tau, X),
```

where `F` is analytic in `tau` and polynomial/analytic in the lifted variables
after introducing monomials for the stable exponents.

Let the positive stable exponents be `alpha_1,...,alpha_k`.  The lifted
generalized monomial basis is generated by

```text
tau^n prod_j z_j^(beta_j) (log tau)^ell,
z_j = tau^(alpha_j),
```

where `n` and `ell` are nonnegative integers and `beta_j` ranges over finite
multi-indices needed below the chosen residual order.  Resonance occurs when
two generated weights differ by an integer eigenvalue of the linear row
operator.  In that case the row solve is performed in descending log degree;
the highest log-degree coefficient kills the resonant obstruction, and lower
log degrees are then solved by the triangular row equations.

More explicitly, write the stable-manifold time as `s=-log lambda`, where
`lambda` is any positive local scale comparable to `tau^2`.  After the quotient
normalizations from TC4, the stable variables have the analytic form

```text
dy/ds = -A y + P_res(y) + H(y),
```

where the eigenvalues of `A` have positive real part, `P_res` is the finite
Poincare-Dulac resonant polynomial, and `H` contains only nonresonant terms
above the retained weight.  Order the coordinates so that a resonant monomial
in row `j` has weight `<m,alpha>=alpha_j` only when it uses already solved
lower/equal weight coordinates.  The scalar row equation for a candidate
coefficient `c_{j,m,ell}` has the form

```text
(<m,alpha>-alpha_j)c_{j,m,ell}
    + (ell+1)c_{j,m,ell+1}
    = known lower-weight forcing.
```

If `<m,alpha> != alpha_j`, the coefficient is determined by division by the
nonzero spectral denominator.  If `<m,alpha> = alpha_j`, the equation is solved
triangularly in the log degree: choose the top log coefficient to cancel the
resonant obstruction, then solve downward until the constant log row remains.
That constant row is the selector parameter for the branch.  Since every
retained row has bounded total generalized weight, only finitely many
multi-indices and log degrees occur.

The finite row construction is made checkable by choosing an explicit retained
weight cutoff `W_*`.  Let

```text
W(n,beta) = n + sum_j beta_j alpha_j.
```

Only rows with `W(n,beta) <= W_*` are retained.  For each retained row `j`,
split the target coefficient space into the range and kernel of the row
operator:

```text
E_j = Range(M_j) direct_sum Ker(M_j),        Pi_j:E_j -> Ker(M_j).
```

The range part is solved by the bounded inverse of `M_j` on `Range(M_j)`.  The
kernel part is the only place where selector data may enter.  A nonresonant row
has `Pi_j forcing = 0` automatically because the spectral denominator is
nonzero.  A resonant row has `Pi_j forcing` killed by raising the logarithmic
degree once:

```text
M_j H_ell + (ell+1) H_{ell+1} = F_ell,
Pi_j F_ell = (ell+1) Pi_j H_{ell+1}.
```

After the top log coefficient is fixed this triangular system descends in
`ell`.  The final constant kernel component `Pi_j H_0` is not solved by the
equation; it is exactly the selector coordinate recorded in the certificate.
The constructor rejects nontriangular dependencies: every resonant forcing term
must be built from rows of strictly lower weight or from earlier rows in a
fixed total order at the same weight.  Therefore the finite table is computed
by induction over `(W,row,log-degree)`.

This also proves there is no hidden finite row datum.  Suppose two incoming
germs have the same central target, scale row, exponent table, and selector
coordinates through weight `W_*`.  Let `(W,row,ell)` be the first coefficient
where they differ.  The difference satisfies the homogeneous triangular row
equation.  Its range component vanishes by invertibility on `Range(M_j)`, and
its positive log-degree kernel components vanish by descending induction.  The
remaining constant kernel component is the selector coordinate, already equal
by hypothesis.  Hence the difference is zero, a contradiction.  All freedom
below `W_*` is therefore exhausted by the finite selector list.

Changing variables back from `s` to cubic collision time gives

```text
exp(-alpha_j s) = tau^(gamma_j) u_j(tau),
gamma_j > 0,
```

with `u_j` analytic and nonzero near `tau=0`.  Thus the finite stable rows are
Puiseux/log rows in `tau`, and irrational `gamma_j` are allowed because the
certificate records the exponent table rather than requiring an integer Taylor
series.

The finite generalized entry certificate records:

- the central target `C`;
- selected stable amplitudes or selector rows;
- the exponent table and generated weights;
- finite Puiseux/log coefficients through the retained order;
- per-row residual identities after substitution into the regularized Newton
  equation;
- energy and zero-angular consistency at the collision endpoint.

For a fixed finite target certificate, only finitely many weights can affect the
requested residual/tail tolerance.  All higher weights are relegated to the
analytic remainder handled in TC6.

The exact/computable pointwise theorem uses this finiteness in only one
direction: for every requested tolerance choose `W_*` high enough that the
first omitted weight dominates the desired residual and projection order.  It
does not require one uniform `W_*` for a whole interval box of incoming germs.
Uniform interval-box constants are part of the separate set-valued constructor
problem, not the pointwise closed-form theorem.

## TC6. Cauchy Estimates Produce a Finite Analytic Remainder Majorant

After TC5 solves all retained generalized rows, the unsolved remainder `R`
satisfies a Fuchsian fixed-point equation on a small polydisc in the lifted
variables:

```text
R = L^{-1}(N_retained + N_remainder),
```

where:

- `L` is the diagonal/triangular row operator with all retained resonances
  removed by the finite log rows;
- `N_retained` is the computable residual produced by truncating the finite
  generalized expansion;
- `N_remainder` is analytic and at least quadratic or higher weighted order in
  the remainder variables.

Choose a polyradius vector `r` for the lifted variables and a Banach norm
weighted by the first omitted generalized weight.  Cauchy estimates on the
analytic Newton vector field give computable bounds:

```text
||L^{-1} N_retained|| <= eta,
Lip(L^{-1} N_remainder) <= kappa,
```

with `kappa < 1` after shrinking the polyradius.  The contraction gives a unique
analytic remainder with

```text
||R|| <= eta/(1-kappa),
||R - R_N|| <= eta kappa^N/(1-kappa).
```

The checker-facing majorant records exactly the finite data needed to recheck
this:

- lifted polyradius and domain;
- first omitted weight;
- residual norm bound `eta`;
- nonlinear Lipschitz bound `kappa`;
- row inverse bound;
- value and derivative tail bounds for projection to physical variables;
- angular-momentum, center-of-mass, linear-momentum, energy, and endpoint
  collapse envelopes.

The constants are extracted from a finite Cauchy audit, not from an informal
small-o estimate.  Fix a closed lifted polydisc

```text
P(r) = {|tau| <= r_tau, |z_j| <= r_j, |R_a| <= R_0 w_a}
```

inside the collision-free central-target chart.  Let `M_a(r)` be a rational or
interval upper bound for each analytic numerator in the transformed Newton
vector field on a slightly larger polydisc `P(rho r)` with `rho > 1`.  Cauchy's
inequality gives coefficient and derivative bounds

```text
|partial^nu N_a(0)|/nu! <= M_a(rho r) prod_i (rho r_i)^(-nu_i),
||DN_a||_{P(r)} <= M_a(rho r) sum_i 1/((rho-1) r_i).
```

These finite bounds produce:

```text
D      = retained residual defect after all rows W <= W_* are substituted,
B      = sup omitted-row right-inverse norm on the weighted Banach space,
L_N    = Cauchy derivative/Lipschitz bound for the nonlinear omitted part,
R_0    = candidate weighted ball radius.
```

The first omitted weight `W_+` fixes the Banach weights: a component whose
lowest monomial has weight `omega_a >= W_+` is normed as
`||R_a|| / tau^{omega_a}` on the punctured sector.  Multiplication by `tau^2`
for positions and by the cubic-time chain rule for velocities is included in
the projection constants, so the projected Newton residual tail is bounded in
physical coordinates rather than only in lifted coordinates.

The finite data in the checker use the following Banach inequalities.  Let
`B` bound the inverse of the triangular row operator on the omitted-weight
subspace, let `D` bound the retained truncation defect, let `L_N` bound the
nonlinear remainder Lipschitz constant on the chosen ball, and let `R_0` be the
candidate ball radius.  The constructor only accepts the majorant when

```text
q = B L_N < 1,
B D + q R_0 <= R_0.
```

These are the concrete forms of `kappa<1` and the self-map inequality above.
They produce a unique analytic fixed point and the a priori norm bound
`||R|| <= R_0`.  The same polydisc Cauchy estimates give componentwise
primitive tail inputs

```text
(C_0, Lambda, sigma, p_0, d)
```

for value, first jet, lifted residual, projected physical residual, and
regularized-position value.  On shell `n` these inputs give a geometric bound
whose ratio is `Lambda sigma^d`; the independent checker rejects any component
whose ratio is not below one or whose first-shell tail does not cover the
declared remainder.  This is why the proof needs both the analytic existence
of a polydisc and the finite numeric majorant constants extracted from it.

The self-map and contraction proof is the standard Banach argument in explicit
certificate form.  For every `R` in the ball `||R|| <= R_0`,

```text
||L^{-1}(N_retained + N_remainder(R))||
    <= B D + B L_N R_0 <= R_0,
```

so the map preserves the ball.  For two remainders `R_1,R_2` in that ball,

```text
||Phi(R_1)-Phi(R_2)|| <= B L_N ||R_1-R_2|| = q ||R_1-R_2||,
q < 1.
```

The unique fixed point is analytic because the fixed-point operator is analytic
on the polydisc and is obtained as the uniform limit of analytic Picard
iterates.  The Cauchy tail certificate records the first-shell radius,
component majorant, derivative order, and geometric ratio for each component,
so the checker can verify the tail without trusting the constructor's floating
samples.

This is the step that prevents the generalized total-stop chart from being a
finite formal series only.  It turns the finite Puiseux-log rows into an
analytic chart with a computable tail.

## TC7. Generalized Entry Data Produce a Maximal-Classical Total-Stop Chart

Given TC5 entry rows and the TC6 remainder majorant, define the chart on a
punctured regularized slab `0 < tau <= tau_0` by

```text
q(tau) = tau^2 (S_retained(tau) + R(tau)),
t_c - t = tau^3.
```

The certificate checker verifies:

- finite grammar of the generalized exponent/log table;
- endpoint collapse `q_i-q_j -> 0` for every pair;
- center-of-mass and total linear momentum ledgers;
- zero angular momentum at the endpoint and on punctured slabs;
- projected Newton residual bounds away from `tau=0`;
- lifted residual and tail bounds in the generalized Fuchsian variables;
- physical-time containment and monotone approach to the stop time;
- absence of any selected outgoing continuation under the maximal-classical
  policy.

Thus a finite target before the stop is resolved by an ordinary/LC/KS chain; a
finite target at or beyond the first unselected total collision is resolved by a
finite ordinary/LC/KS/generalized-total-stop chain.  This is exactly the
atlas-or-stop alternative used by the pointwise closed-form route.

## Remaining Public-Proof Risks

The code now records TC1-TC7 as internal theorem evidence.  The public proof
still needs external audit of:

- the reduced hyperbolicity statement in TC4 for every positive-mass central
  target;
- the resonance and log-degree triangular solve in TC5;
- the extraction of finite selector/exponent data for arbitrary incoming
  total-collision germs;
- the Cauchy-majorant constants in TC6 using a proof-grade arithmetic backend.

These risks do not reintroduce the interval-box branch/event-order constructor
blocker.  They are analytic proof-audit risks for the pointwise theorem itself.
