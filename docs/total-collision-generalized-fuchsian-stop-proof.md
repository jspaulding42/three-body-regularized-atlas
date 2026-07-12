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
I = sum_i m_i |q_i|^2,        K = (1/2) sum_i m_i |v_i|^2,
L = sum_i m_i q_i wedge v_i,  U = sum_{i<j} m_i m_j / |q_i-q_j|,
H = K - U.
```

At finite total collision, `I(t) -> 0`.  The estimate used here is finite and
mass-metric.  In centered coordinates the exterior-product Cauchy-Schwarz
inequality gives the centered Sundman inequality

```text
|L|^2
 = |sum_i m_i q_i wedge v_i|^2
 <= (sum_i m_i |q_i|^2)(sum_i m_i |v_i|^2)
 = 2 I K.
```

The inequality itself does not use TC2; it is an algebraic inequality for each
centered state.  TC2 is used only in the next line, where it rules out
binary-degenerate normalized approach.  After that exclusion the normalized
shape remains in a compact collision-free subset of the mass-metric shape
sphere, so for some `d_*>0` every pair has the lower bound

```text
|q_i-q_j| >= d_* sqrt(I)
```

near the collision, up to the fixed mass-metric normalization convention.
Therefore, for a finite constant `C_U`,

```text
U <= C_U / sqrt(I),
```

and the conserved finite energy gives `K = H + U`.  Hence

```text
I K = I(H+U) <= |H| I + C_U sqrt(I) -> 0.
```

Therefore `|L|^2 <= 2 I K -> 0`.  Since `L` is conserved on the incoming
punctured branch, its constant value is zero:

```text
L = 0.
```

## TC2. Binary-Degenerate Normalized Shape Approach Is Impossible

Fix the candidate tight binary `(1,2)` and use Jacobi coordinates

```text
x = q_1-q_2,
y = q_3 - (m_1 q_1+m_2 q_2)/(m_1+m_2),
I = alpha |x|^2 + beta |y|^2,
alpha = m_1m_2/(m_1+m_2),
beta  = m_3(m_1+m_2)/(m_1+m_2+m_3).
```

Total collision means `x(t)->0` and `y(t)->0`.  A binary-degenerate normalized
limit for this pair would have `|x|/|y| -> 0`; the alternative
`|y|/|x| -> 0` is the same argument after relabeling the tight pair.

When `|x|/|y| -> 0`, the third-body denominators satisfy
`|y+a x| = |y|(1+o(1))` for the fixed mass fractions `a`.  The Jacobi equations
therefore split into two perturbed Kepler equations:

```text
x'' = -(m_1+m_2) x/|x|^3 + R_x,
|R_x| <= C |x|/|y|^3 = o(|x|^-2),

y'' = -M y/|y|^3 + R_y,
|R_y| <= C |x|/|y|^3 = o(|y|^-2),
```

where `M=m_1+m_2+m_3`.  The first remainder is lower order than the binary
Kepler force because `|R_x|/(|x|^-2) = O((|x|/|y|)^3) -> 0`; the second is
lower order than the cluster-third Kepler force because
`|R_y|/(|y|^-2) = O(|x|/|y|) -> 0`.

Use the standard collision asymptotic for a finite-energy perturbed Kepler
equation

```text
z'' = -mu z/|z|^3 + o(|z|^-2),        z(t)->0.
```

The angular part is negligible by the two-body Sundman estimate, and the
radial energy identity gives

```text
dr/dt = -sqrt(2 mu/r) (1+o(1)).
```

Integration to the collision time yields

```text
|z(t)| = (9 mu/2)^(1/3) (t_c-t)^(2/3) (1+o(1)).
```

Applying this to the two Jacobi equations gives

```text
|x(t)| = (9(m_1+m_2)/2)^(1/3) (t_c-t)^(2/3) (1+o(1)),
|y(t)| = (9M/2)^(1/3)             (t_c-t)^(2/3) (1+o(1)).
```

Hence

```text
|x(t)|/|y(t)| -> ((m_1+m_2)/M)^(1/3) > 0,
```

contradicting `|x|/|y| -> 0`.  Thus no normalized binary degeneration is
possible at total collision.  All pair distances are comparable to `sqrt(I)`
near the endpoint, which is the pair-distance floor used in TC1.

## TC3. Total-Collision Branches Have Central-Configuration Shape Limits

Write the total scale as `rho = sqrt(I)` and normalized shape as
`s = q/rho`, with the mass metric normalization `||s||_m = 1` and center of mass
zero.  By TC2, every normalized limiting shape stays in the compact
collision-free shape quotient.

Homogeneity gives the two basic identities

```text
U(rho s) = rho^(-1) U(s),
A(rho s) = rho^(-2) A(s).
```

The mass-metric kinetic energy splits as

```text
K = (1/2) rho_dot^2 + (1/2) rho^2 ||s_dot||_m^2.
```

Introduce the collision-scaled variables and McGehee time

```text
d sigma/dt = rho^(-3/2),
v = rho^(1/2) rho_dot,
w = rho^(3/2) s_dot.
```

Then the finite-energy identity becomes

```text
(1/2)(v^2 + ||w||_m^2) - U(s) = rho H -> 0.
```

The Lagrange-Jacobi identity

```text
I'' = 4H + 2U = 4H + 2 rho^(-1) U(s)
```

and the compact collision-free shape range imply the parabolic radial law.  At
any limiting shape `s_*`,

```text
rho(t) = (9 U(s_*)/2)^(1/3) (t_c-t)^(2/3) (1+o(1)).
```

The shape variable is controlled on the collision manifold.  In McGehee time
the shape equation has the form

```text
s_sigma = w,
w_sigma = Pi_s(grad_m U(s)) - (1/2) v w + quadratic terms in w,
```

where `Pi_s` is the mass-metric projection onto the tangent space of
`||s||_m=1` and the displayed tangential force is the only term that survives
when `w=0`.  The same Lagrange-Jacobi/McGehee Lyapunov estimate gives

```text
int_{sigma_0}^infinity ||w(sigma)||_m^2 d sigma < infinity,
```

so every compact omega-limit point of the incoming collision branch has
`w=0`.  Invariance of the omega-limit set then forces

```text
Pi_{s_*}(grad_m U(s_*)) = 0.
```

Equivalently, the Newton acceleration at the normalized limiting shape is
purely radial:

```text
A(s_*) = -lambda s_*,        lambda = U(s_*) > 0
```

with the sign convention of this note.  Thus every normalized total-collision
shape limit is a central configuration.

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

The quotient modes are explicit:

```text
translation: removed by center-of-mass reduction,
scale:       separated as the parabolic radial law,
rotation:    removed by the oriented-shape quotient,
reflection:  discrete, so it contributes no infinitesimal center mode.
```

The spectrum calculation is made in the mass metric.  Scale each central
configuration `C` so that:

```text
A(C)=-(2/9)C,
```

The indicial equation is not a separate assumption.  Use the cubic collision
time `t_c-t=tau^3`, so

```text
d/dt = -(1/(3 tau^2)) d/dtau.
```

For `q=tau^2 S(tau)`, Newton's equation `q_tt=A(q)` is equivalent on
punctured slabs to the lifted residual identity

```text
tau^2 S'' + 2 tau S' - 2S = 9 A(S).
```

Now write a regular-singular perturbation

```text
S(tau)=C+tau^k V+...
```

and let `DA(C)V=mu V`.  The constant terms cancel because
`-2C=9A(C)`.  The coefficient of `tau^k` gives

```text
(k^2+k-2)V = 9 DA(C)V,
```

or, for an eigenvector,

```text
(k+2)(k-1)=9mu.
```

A reduced McGehee center direction would require an indicial root with zero
real part.  The three-body collision-free central-configuration spectra are
real, so the only such root is `k=0`, equivalently:

```text
mu=-2/9.
```

That value is exactly the infinitesimal rotation mode, and it is removed in the
oriented-shape quotient.

For an arbitrary positive-mass Lagrange equilateral target, write:

```text
M=m_1+m_2+m_3,
beta=(m_1m_2+m_1m_3+m_2m_3)/M^2,        0 < beta <= 1/3.
```

The mass-metric scaled spectrum is:

```text
{0,0,4/9,-2/9,
 1/9+(1/3)sqrt(1-3beta),
 1/9-(1/3)sqrt(1-3beta)}.
```

Here is the finite mass-metric calculation behind the two shape eigenvalues.
The two translation modes are the kernel of center-of-mass reduction.  The
scale vector `C` has eigenvalue `4/9` by the normalization
`A(C)=-(2/9)C` and homogeneity of the Newtonian force, while the infinitesimal
rotation `J C` has eigenvalue `-2/9` by rotational covariance:

```text
DA(C) C = 4/9 C,
DA(C) J C = -2/9 J C.
```

After these four directions are removed, the remaining real two-dimensional
mass-orthogonal shape block has characteristic polynomial

```text
chi_Lag(mu)
  = mu^2 - (2/9) mu + (27 beta - 8)/81.
```

Equivalently,

```text
tr_shape = 2/9,
det_shape = (27 beta - 8)/81.
```

This block is obtained by evaluating the mass-weighted second variation of the
equilateral central configuration on any orthonormal pair of shape
perturbations.  The trace and determinant are symmetric in the masses, scale
invariant after `A(C)=-(2/9)C`, and reduce to the displayed functions of the
single shape parameter `beta`; substituting the equilateral pair-distance
normalization gives the coefficients above.  Solving `chi_Lag(mu)=0` gives the
two displayed shape eigenvalues.

The range of `beta` is also explicit.  With positive masses,

```text
0 < beta,
3(m_1m_2+m_1m_3+m_2m_3) <= (m_1+m_2+m_3)^2,
```

because the difference is

```text
m_1^2+m_2^2+m_3^2-m_1m_2-m_1m_3-m_2m_3
  = ((m_1-m_2)^2+(m_1-m_3)^2+(m_2-m_3)^2)/2.
```

Thus `0<beta<=1/3`, and the square root in `chi_Lag` is real.

The two zeros are translations, `4/9` is the scale mode, and `-2/9` is
rotation.  The remaining two shape eigenvalues obey:

```text
1/9+(1/3)sqrt(1-3beta) > 0,
1/9-(1/3)sqrt(1-3beta) > -2/9,
```

because `beta>0`.  Thus no reduced equilateral shape mode has `mu=-2/9`.

For an ordered Euler collinear target the mass-centered horizontal and
transverse force blocks satisfy:

```text
L_parallel = -2 L_perp.
```

The spectrum therefore has the form:

```text
{0,0,4/9,-2/9,sigma,-sigma/2}.
```

The horizontal mass-metric quadratic form is:

```text
sum_{i<j} 2 m_i m_j (xi_i-xi_j)^2 / |X_i-X_j|^3,
```

so it is positive semidefinite and vanishes only on translations.  After
center-of-mass reduction the remaining horizontal shape eigenvalue satisfies
`sigma>0`.  The Euler ratio calculation gives the sharper positive-mass bound:

```text
4/9 < sigma < 32/9.
```

The bound is a finite positive-mass calculation, not a numerical observation.
Normalize one ordered Euler line by

```text
x_1=0, x_2=1, x_3=1+r,        r>0,
```

and use Euler's quintic to eliminate the third mass.  If `lambda_c` is the
unscaled central multiplier, the horizontal trace is

```text
tr(L_parallel)
 = 2[(m_1+m_2) + (m_2+m_3)/r^3 + (m_1+m_3)/(1+r)^3],
```

and after scaling to `A(C)=-(2/9)C`,

```text
sigma = (2/(9 lambda_c)) [tr(L_parallel) - 2 lambda_c].
```

Substituting the eliminated mass gives the explicit positive-ratio formula

```text
sigma =
  8 r^2(1+r)(2r^2+3r+2)(m_1(1+r)+m_2 r)
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1).
```

The upper gap is

```text
32/9 - sigma =
  8[
    m_1 r^2(r-1)^2(r+2)(2r+1)
    + m_2(2r^6+11r^5+19r^4+22r^3+24r^2+16r+4)
  ]
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1),
```

so it is positive for `r>0` and positive masses.  The lower gap is

```text
sigma - 4/9 =
  4(3r^2+3r+1)
  [m_1 r^2(r^2+3r+3)+m_2(r-1)(r+1)(r^2+r+1)]
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1).
```

For `r>=1` the bracket is positive.  For `0<r<1`, positivity of the eliminated mass gives the inequality below:

```text
m_3 =
  [m_1 r^3(r^2+3r+3)
   + m_2(r-1)(r+1)^2(r^2+r+1)]
  /(3r^2+3r+1)
```

implies

```text
m_1 r^2(r^2+3r+3)
  > m_2(1-r)(r+1)(r^2+r+1),
```

which is exactly positivity of the lower-gap bracket.  This proves
`4/9 < sigma < 32/9` uniformly on the positive-mass ordered Euler family.

Hence the paired transverse shape eigenvalue satisfies `-sigma/2 < -2/9`; it
is not the rotation eigenvalue.  Again no reduced Euler shape mode has
`mu=-2/9`.

Equivalently, the reduced McGehee/Fuchsian linearization has:

- one scale/radial mode fixed by the parabolic collision law;
- neutral modes only from the quotient symmetries already removed;
- stable/unstable shape exponents with nonzero real parts on the reduced shape
  space.

This hyperbolicity supplies the finite list of indicial exponents used in TC5.
Integer, rational, and irrational positive stable exponents are all allowed:
the certificate language records the exponents as algebraic/interval data rather
than assuming an ordinary integer-power Taylor series.

This closes the TC4 reduced-hyperbolicity audit for the positive-mass
three-body collision-free central targets.  It does not prove the arbitrary
incoming-germ finite selector expansion; that is exactly the TC5 normal-form
and Fuchsian/Puiseux-log entry step.

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

The denominator/projector audit is finite.  For the retained cutoff define the
nonresonant denominator set

```text
D_*(W_*) = {|<m,alpha>-alpha_j|:
            W(m)<=W_*, <m,alpha> != alpha_j}.
```

Since the retained multi-index set is finite, public proof data must exhibit a
rational interval lower bound

```text
delta_*(W_*) <= min D_*(W_*),        delta_*(W_*) > 0.
```

Rows whose interval denominator contains zero are not treated as nonresonant;
they are moved to the resonant projector system.  For each resonant row the
certificate records finite matrices `Pi_j` and `Q_j` satisfying the algebraic
identities

```text
Pi_j^2 = Pi_j,
M_j Pi_j = 0,
Pi_j M_j = 0,
M_j Q_j = I-Pi_j,
Pi_j Q_j = Q_j Pi_j = 0.
```

Thus `Pi_j` is the kernel projector and Q_j is a right inverse on the range.
All identities are finite matrix equalities in the mass-metric coordinates,
checked with the same rational interval backend used in TC6.  The row solve is
therefore not hiding a choice of complement: the range coefficient is `Q_j`
applied to the range forcing, and the only unsolved part is the displayed
kernel selector `Pi_j H_0`.

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

The finite log-degree bound is explicit.  For the retained cutoff `W_*`, define
the finite index set

```text
I(W_*) = {(omega,j): omega <= W_*, row j retained at weight omega}.
```

Put a directed edge `(omega',i) -> (omega,j)` when the forcing in row `j` at
weight `omega` uses a coefficient from row `i` at weight `omega'`.  The
triangular dependency rule says every edge either has `omega'<omega` or has
`omega'=omega` with row `i` earlier than row `j`; hence this dependency graph
is acyclic.  Let

```text
rho(omega,j) = length of the longest directed path ending at (omega,j).
```

A nonresonant solve preserves the forcing log degree, while a resonant kernel
projection raises it by at most one:

```text
deg_log H_j(omega) <= 1 + max deg_log forcing predecessors.
```

Therefore induction on the acyclic graph gives

```text
deg_log H_j(omega) <= rho(omega,j) <= |I(W_*)|-1.
```

This is the proof-grade finite log-degree rule: the maximum log degree

```text
L_*(W_*) = max_{(omega,j) in I(W_*)} rho(omega,j)
```

is computed from the retained dependency graph itself.  No infinite logarithmic
tower can appear below `W_*`, because each new log power consumes one step in a
finite acyclic resonance chain.

The selector extraction algorithm is therefore finite.  Choose any topological
ordering of the acyclic retained graph `I(W_*)`.  At a node `(omega,j)`:

```text
1. subtract all previously constructed lower-node particular rows;
2. project the remaining forcing with I-Pi_j and solve the range part by Q_j;
3. if Pi_j forcing is nonzero, raise log degree once and solve downward;
4. record only the final constant kernel component as a selector.
```

For an incoming germ this final selector is computed by the finite limit

```text
a_(omega,j) =
  lim_{s->infinity} exp(omega s)
    Pi_j( y_j(s) - retained_particular_rows_(omega,j)(s) ).
```

The Cauchy property is a scalar row calculation, not a new hypothesis.  After
all earlier rows are subtracted, a retained row has the model form

```text
dY/ds + alpha_j Y = exp(-omega s) P_omega(s)
                    + O(exp(-(omega+epsilon)s) s^m),
```

where `P_omega` is a finite polynomial in `s`.  Put

```text
Y_part = exp(-omega s) B(s).
```

Then the polynomial part must solve

```text
B'(s) + (alpha_j-omega) B(s) = P_omega(s).
```

If `omega != alpha_j`, this equation is solved downward in degree by division
by the nonzero denominator `alpha_j-omega`; for the top polynomial coefficient,

```text
b_L = f_L/(alpha_j-omega),
```

and lower `b_l` are then fixed by the already known `b_{l+1}` term.  If
`omega = alpha_j`, the equation becomes

```text
B'(s)=P_alpha_j(s),
```

so a forcing term `f_L s^L` contributes

```text
f_L s^(L+1)/(L+1)
```

and raises the log degree exactly once.  After this forced resonant polynomial
is subtracted, the remainder `R` satisfies

```text
dR/ds + alpha_j R = O(exp(-(alpha_j+epsilon)s) s^m).
```

Thus

```text
|exp(alpha_j s)R(s)-exp(alpha_j S)R(S)|
 <= C int_S^s exp(-epsilon u) u^m du
 <= C' exp(-epsilon S) S^m,
```

so `exp(alpha_j s)R(s)` is Cauchy as `s->infinity`.  This limit is exactly the
constant kernel selector recorded above.

The public evidence field `selector_data_fields` is exactly this finite list:

```text
central target, exponent table, resonant row ids, selector constants.
```

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

The time-change formula is finite in the lifted selector class.  Let `lambda`
be the positive McGehee scale used in `s=-log lambda`.  Along the total
collision branch the cubic-time normalization gives

```text
lambda = c tau^2 (1+eta(tau,z)),
c>0,        eta(0,0)=0,
```

with `eta` analytic in the lifted variables `z_j=tau^(alpha_j)`.  Hence

```text
s = -2 log tau - log c - log(1+eta(tau,z)).
```

The last term is analytic and bounded on the punctured chart.  Consequently a
retained stable-time row

```text
exp(-omega s) s^ell
```

becomes

```text
c^omega tau^(2 omega) (1+eta(tau,z))^omega
  (-2 log tau - log c - log(1+eta(tau,z)))^ell.
```

For each fixed retained `ell`, expanding the last factor gives only finitely
many powers of `log tau`, with analytic lifted coefficients.  The analytic
unit `(1+eta)^omega` is handled by the same TC6 Cauchy majorant as the other
omitted analytic factors.  Therefore the projection from McGehee stable rows
to cubic-time Puiseux/log rows preserves the finite exponent table and the
finite log-degree bound already derived from the retained dependency graph.

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

The extraction of finite selector data from an incoming germ is also finite.
Use McGehee time `s -> +infinity` along the incoming branch.  TC4 gives
convergence to a reduced-hyperbolic central target, so after the quotient
normalizations the branch lies in the local stable manifold and the
Poincare-Dulac coordinates satisfy a triangular stable system:

```text
dy_j/ds = -alpha_j y_j + P_j^{res}(y_1,...,y_{j-1},y_j)
          + H_j(y),
alpha_j > 0.
```

Here "lies in the local stable manifold" is a finite analytic assertion, not a
new oracle.  In the quotient McGehee chart let `X=(lambda,y)` denote the scale
coordinate and reduced shape/velocity coordinates, with the central target at
`X=0`.  TC4 gives a hyperbolic splitting

```text
T_0 X = E^s direct_sum E^u,
spec(Df|E^s) subset {Re z < -a},
spec(Df|E^u) subset {Re z > a}
```

for some `a>0` after quotienting the zero group modes.  By the stable manifold theorem in this reduced analytic chart, there is an analytic graph

```text
W^s_loc = {x_s + h(x_s): x_s in E^s, h(0)=0, Dh(0)=0}.
```

An incoming total-collision branch that converges to the target has an
eventual entry time `S_0`: for `s>=S_0` it stays in the hyperbolic chart.  Its
unstable graph coordinate must vanish.  Otherwise the unstable component in
the variation-of-constants formula

```text
X_u(s) = e^{A_u(s-S)} X_u(S)
       + integral_S^s e^{A_u(s-u)} N_u(X(u)) du
```

has a nonzero leading term growing like `exp(a(s-S))`, contradicting
`X(s)->0`.  Equivalently, the Lyapunov-Perron fixed point defining
`W^s_loc` is the unique orbit in the local chart with bounded forward tail.
Thus every TC1-TC4 incoming branch is eventually represented by coordinates on
`W^s_loc`; no extra selector hypothesis is being inserted here.

On `W^s_loc`, put the stable vector field into finite-order
Poincare-Dulac form.  For the retained cutoff `W_*`, solve the homological
equations only for weights `W<=W_*`.  A nonresonant monomial is removed by
division by its homological denominator

```text
<m,alpha> - alpha_j,
```

while resonant monomials are kept in `P_j^{res}`.  Since all stable rates
`alpha_j` are positive, the finite retained set is in the Poincare domain:
there are only finitely many multiindices with `W(n,beta)<=W_*`, and every
denominator used by the normalizing map is bounded away from zero on that
finite nonresonant list.  The finite-order normalizing map is analytic,
near-identity, and invertible on a smaller stable chart.  Therefore applying
the selector extraction below in Poincare-Dulac coordinates is equivalent to
extracting finite data from the original incoming germ.

Order generated monomials by the same weight `W(n,beta)` used above.  Suppose
all rows of smaller weight, and earlier rows at the same weight, have already
been extracted.  Their finite selector data determine a known forcing
polynomial on row `j`:

```text
F_j^{<W}(s) = sum_{omega < W} sum_ell f_{j,omega,ell}
              exp(-omega s) s^ell.
```

Subtract the corresponding particular solution `P_j^{<W}(s)`.  The residual
component

```text
Y_j^{W}(s)=y_j(s)-P_j^{<W}(s)
```

satisfies

```text
dY_j^{W}/ds + alpha_j Y_j^{W}
    = O(exp(-(W+epsilon)s) s^m)
```

unless `W=alpha_j` is resonant.  In the nonresonant case the selector
coefficient is the convergent limit

```text
a_j = lim_{s->infinity} exp(alpha_j s) Y_j^{W}(s).
```

In the resonant case the known forcing contains terms

```text
f_ell exp(-alpha_j s) s^ell.
```

Solving

```text
dY/ds + alpha_j Y = f_ell exp(-alpha_j s) s^ell
```

gives the particular row

```text
exp(-alpha_j s) sum_{r=0}^{ell+1} b_r s^r.
```

Thus resonance creates exactly one higher log-degree triangular row, not an
infinite family of new data.  After subtracting all forced resonant log rows,
the remaining selector is again the finite limit:

```text
a_j = lim_{s->infinity} exp(alpha_j s)
      (Y_j^{W}(s) - forced_resonant_log_polynomial_j(s)).
```

The convergence follows by the variation-of-constants estimate

```text
|exp(alpha_j s)Y_j^{W}(s)
 - exp(alpha_j S)Y_j^{W}(S)|
 <= integral_S^s C exp(-(W+epsilon-alpha_j)u) u^m du,
```

after the resonant polynomial has been removed.  Since
`W+epsilon > alpha_j`, the integral is Cauchy as `S,s -> infinity`.
Induction over `(W,row,log-degree)` therefore extracts every selector
coefficient with `W <= W_*` from the incoming germ by finitely many limits and
finite polynomial subtractions.  This is the analytic content behind the
certificate fields `exponents`, `resonant_rows`, and `selector_coordinates`.

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

The projection constants are finite chain-rule consequences of

```text
q(tau)=tau^2 S(tau),        t_c-t=tau^3,
d/dt=-(1/(3 tau^2)) d/dtau.
```

For any omitted lifted component satisfying

```text
|R_a(tau)| <= C tau^omega,        omega >= W_+,
```

Cauchy bounds on a shell give derivative tails

```text
|R_a'(tau)| <= C_1 tau^(omega-1),
|R_a''(tau)| <= C_2 tau^(omega-2).
```

Therefore the physical position and velocity tails obey

```text
|delta q| <= C tau^(omega+2),
|delta v| = |delta q_t|
          <= (1/3) C_v tau^(omega-1),
```

and the acceleration tail is bounded by the explicit cubic-time formula

```text
delta q_tt =
  (1/(9 tau^4)) delta q_tautau
  - (2/(9 tau^5)) delta q_tau.
```

The lifted residual equation is exactly the numerator obtained by multiplying
the projected Newton residual by `9 tau^4`; equivalently

```text
projected_residual
  = [tau^2 S'' + 2 tau S' - 2 S - 9 A(S)]/(9 tau^4).
```

Thus a lifted residual tail of weight `omega` transfers to a physical residual
tail of weight `omega-4`.  The checker records this weight shift explicitly in
the primitive Cauchy inputs for `projected_physical_residual`, and it requires
the first omitted weight to exceed the requested physical residual order before
the total-stop chart is accepted.

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

The extraction of `D`, `B`, `L_N`, and the primitive constants is finite and
checker-facing.  Work with rational interval enclosures for all masses,
central-target coordinates, retained coefficients, exponent intervals, and
polyradii.  The analytic force denominators are bounded away from zero on
`P(rho r)` by the collision-free central-target margin, so every numerator and
denominator in the transformed Newton field has a rational interval enclosure.

The proof-grade arithmetic backend is an inclusion proof over a finite
expression DAG.  Every leaf is a rational interval, and every arithmetic node
has an exact rational outward enclosure:

```text
X op Y subset I_op(X,Y),        op in {+,-,*,/},
0 notin Y for division nodes.
```

Division is admitted only with a denominator exclusion certificate: if
`Y=[y_-,y_+]` and `0 notin Y`, then the reciprocal interval is the convex hull
of `{1/y_-,1/y_+}` and multiplication gives the enclosing quotient.  Algebraic
matrix inverses are checked by a rational residual or Neumann certificate; for
example `B A = I+E` with `||E||<1` proves

```text
A^(-1) = (I+E)^(-1) B,
||A^(-1)|| <= ||B||/(1-||E||).
```

The only non-polynomial shell quantities are also finite interval inputs, not
floating samples.  On each punctured shell `tau in [tau_-,tau_+]` with
`0<tau_-<tau_+<1`, the backend records rational enclosures for

```text
log_tau_shell,
tau_power_alpha_shell_j = tau^(alpha_j),
tau_power_weight_shell_a = tau^(omega_a),
```

validated by monotonicity in `tau` and in the exponent intervals, together with
Taylor or continued-fraction remainder bounds for `log`.  If any shell power or
log enclosure is absent, the public TC6 audit cannot certify the majorant.

Thus every retained residual, derivative bound, inverse bound, and primitive
tail constant is the value of a finite rational interval expression.  The
backend maintains the interval inclusion invariant:

```text
true analytic quantity on P(rho r) is contained in its computed interval.
```

The Banach checks are then exact rational comparisons of interval endpoints:

```text
upper(B) upper(L_N) < 1,
upper(B) upper(D) + upper(B) upper(L_N) R_0 <= R_0,
upper(Lambda) upper(sigma)^d < 1.
```

These comparisons are the reason the public proof requires a named
proof-grade backend id.  Without the expression-DAG inclusion invariant, the
numbers `D`, `B`, `L_N`, `C_0`, `Lambda`, `sigma`, `p_0`, and `d` would be
diagnostic estimates rather than certificate-grade constants.

The retained defect `D` is obtained by substituting the finite Puiseux/log table
from TC5 into the lifted regularized equation, deleting all rows of weight
`> W_*`, and interval-evaluating the remaining defect polynomial/rational
expression on `P(r)`.  Since each retained row is finite and the checker has
explicit interval enclosures for `tau`, the lifted exponent variables, and
`log tau` on each punctured shell, this produces a finite interval bound:

```text
D = sup_{P(r)} || retained_defect_{>W_*} ||_w.
```

The right-inverse bound `B` is split into a finite low-omitted part and a
uniform high-weight tail.  For every omitted row with weight
`W_+ <= omega <= W_inv`, the checker forms the interval matrix

```text
M_j(omega) = omega I - A_j
```

after applying the range/kernel projector already used in TC5.  A rational
interval inverse or Neumann certificate gives:

```text
||M_j(omega)^(-1)|| <= B_{j,omega}.
```

For all higher weights, the diagonal weight dominates the fixed finite
linearization matrix.  If `A_bound` is a rational interval operator-norm bound
for the row linearization and `omega >= W_inv > A_bound`, then:

```text
M_j(omega)^(-1)
  = omega^(-1) (I - A_j/omega)^(-1),
||M_j(omega)^(-1)|| <= 1/(omega-A_bound).
```

Thus the omitted inverse bound is the finite maximum:

```text
B = max( max_{W_+ <= omega <= W_inv} B_{j,omega},
         1/(W_inv-A_bound) ).
```

The Lipschitz constant `L_N` is extracted from the same Cauchy enclosure after
removing the linear retained rows.  Let `DN_rem` be the derivative of the
nonlinear omitted part with respect to the weighted remainder variables.  On
the ball `||R||_w <= R_0`, rational interval Cauchy estimates give:

```text
L_N = sup_{P(r), ||R||_w <= R_0} ||DN_rem||_w.
```

Because `N_remainder` is at least quadratic in the omitted variables, shrinking
the lifted polydisc and the ball radius decreases this interval bound.  The
constructor therefore searches over rational candidate radii until either

```text
B L_N < 1,        B D + B L_N R_0 <= R_0
```

is proved, or the certificate is left open.

Finally, the primitive Cauchy tuple for each checker component is derived from
the same rational interval majorants.  On shell radii

```text
rho_n = rho_0 sigma^n,
```

component `a` with first retained derivative order `p_0` and derivative step
`d` receives:

```text
C_0      = first-shell component majorant,
Lambda   = shell-to-shell growth multiplier from Cauchy scaling,
sigma    = shell contraction,
p_0      = first retained derivative/order,
d        = derivative/order increment.
```

The checker recomputes the geometric tail

```text
C_0 Lambda^n sigma^(p_0+n d) / (1 - Lambda sigma^d)
```

and requires `Lambda sigma^d < 1`.  These rational interval recomputations are
why a public TC6 audit must name both the primitive constants and the
proof-grade arithmetic backend; otherwise the Banach proof would only be an
informal small-radius assertion.

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

The constructive series is explicit.  Set `R^(0)=0` and

```text
R^(n+1) = Phi(R^(n)).
```

The first step satisfies

```text
||R^(1)-R^(0)|| <= B D.
```

The contraction inequality gives, by induction,

```text
||R^(n+1)-R^(n)|| <= q^n B D.
```

Consequently the analytic remainder is the normally convergent series

```text
R = sum_{n>=0} (R^(n+1)-R^(n)),
```

and every retained Picard truncation has the certified tail

```text
||R-R^(N)|| <= q^N B D/(1-q).
```

This is the majorant series used by the checker: its ratio is the same
certificate quantity `q=B L_N`, and its first term is the retained defect
transported through the omitted-row inverse.

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

The endpoint and invariant ledgers are finite interval consequences of the
same chart.  Endpoint collapse follows from bounded lifted shape variables:

```text
q_i-q_j = tau^2(S_i-S_j),
|S_i-S_j| <= C_shape
        => |q_i-q_j| <= C_shape tau^2 -> 0.
```

The center-of-mass and linear-momentum rows are checked on the retained series
and on the TC6 tails:

```text
CM(tau) = sum_i m_i q_i(tau),
P(tau)  = sum_i m_i q_{i,t}(tau).
```

Since the projection-tail step gives interval bounds for `delta q` and
`delta v`, the omitted parts contribute finite envelopes

```text
|delta CM| <= sum_i m_i |delta q_i|,
|delta P|  <= sum_i m_i |delta v_i|.
```

Angular momentum has an especially simple lifted form.  From
`q=tau^2S` and `d/dt=-(1/(3tau^2))d/dtau`,

```text
q_t = -(2/(3tau))S - (1/3)S',
L = sum_i m_i q_i wedge q_{i,t}
  = -(tau^2/3) sum_i m_i S_i wedge S_i'.
```

Thus bounded lifted shape and first-jet intervals force `L -> 0`; the checker
also interval-evaluates the displayed expression on punctured shells to verify
zero angular momentum within the declared tail budget.

Energy is verified by the same expression-DAG backend after the finite
central-scale rows cancel the singular terms:

```text
H = (1/2) sum_i m_i |q_{i,t}|^2
    - sum_{i<j} m_i m_j/|q_i-q_j|.
```

With `q=tau^2S` this becomes the explicit lifted expression

```text
H =
  tau^(-2) [ (2/9)||S||_m^2 - U(S) ]
  + tau^(-1) (2/9)<S,S'>_m
  + (1/18)||S'||_m^2.
```

The central row has `A(C)=-(2/9)C`, so Euler homogeneity gives

```text
U(C) = (2/9)||C||_m^2,
```

and the leading `tau^-2` term cancels at the endpoint.  The retained
Puiseux/log rows are required to cancel every remaining negative-power row in
the displayed energy expression.  The first nonnegative constant row is the
finite-energy matching constant `H_ret`, and the checker compares it with the
incoming conserved energy interval.

The omitted-row contribution is bounded on denominator-excluded shape boxes.
Since `|q_i-q_j|=tau^2|S_i-S_j|` and TC2/TC7 provide

```text
|S_i-S_j| >= d_shape > 0,
```

the shape potential `U(S)` has a finite Lipschitz bound `L_U` on the lifted
box.  The kinetic polynomial in `(S,S')` has a finite interval derivative bound
`L_K`.  Thus the TC6 value and first-jet tails give an energy envelope of the
form

```text
|delta H| <= L_U ||delta S|| + L_K (||delta S|| + ||delta S'||)
             + retained_energy_defect_tail.
```

Therefore the checker's invariant rows are not sampled diagnostics; they are
interval consequences of the lifted chart, projection chain rule, denominator
exclusion, and Cauchy tails.

Physical-time containment is equally explicit.  On the punctured slab
`0 < tau <= tau_0`,

```text
t(tau) = t_c - tau^3,
dt/dtau = -3 tau^2 < 0.
```

Hence the chart maps the slab monotonically onto

```text
t in [t_c-tau_0^3, t_c),
```

and the checker verifies any requested target interval by rational endpoint
comparisons against this interval.  A finite target strictly before `t_c` is
not evaluated by the stop chart; it is covered by the preceding ordinary/LC/KS
chain.  A target at or beyond `t_c` is resolved by the stop certificate because
the maximal classical branch has reached total collision at `t_c`.

The no-continuation row is a policy statement, not a hidden analytic branch
choice.  The chart certifies only the incoming punctured side:

```text
0 < tau <= tau_0,        t<t_c.
```

It records the collision endpoint and the maximal-classical policy id
`maximal_classical_stop`.  Under that policy, no outgoing selector constants
are supplied and no positive-`tau` continuation chart is part of the finite
target certificate.  Any selected continuation theorem must instead provide a
separate selector policy and a separate outgoing chart; it cannot be smuggled
into this stop certificate.

Thus a finite target before the stop is resolved by an ordinary/LC/KS chain; a
finite target at or beyond the first unselected total collision is resolved by a
finite ordinary/LC/KS/generalized-total-stop chain.  This is exactly the
atlas-or-stop alternative used by the pointwise closed-form route.

## Public TC4-TC6 Audit Line-Item Map

This section is a public-review checklist, not another certificate layer.  The
local proof package may resolve these document anchors and test/checker
artifacts, but public proof closure still requires independent line-by-line
review artifacts:

```text
external-public-review:tc4-reduced-hyperbolicity-line-audit
external-public-review:tc5-generalized-fuchsian-entry-line-audit
external-public-review:tc6-cauchy-majorant-backend-line-audit
```

The local audit map is intentionally finite.  Each row has a mathematical
claim, the exact formula or construction to be checked, and the current local
artifact that prevents the claim from being a prose-only assertion.

TC4-Audit-01, quotient and target coverage:

- Claim: after center-of-mass reduction and quotienting scale, rotation, and
  reflection, every positive-mass total-collision target is one of the
  `Euler_collinear_positive_mass` or `Lagrange_equilateral_positive_mass`
  families.
- Formula to audit: the only zero/neutral modes before quotienting are
  translation, scale, rotation, and the discrete reflection label.
- Local artifact ids:
  `tests/test_obstructions.py::test_arbitrary_mass_equilateral_linearized_spectrum_matches_beta_formula`,
  `tests/test_obstructions.py::test_ordered_euler_linearized_spectrum_has_single_horizontal_shape_parameter`,
  `tests/test_obstructions.py::test_ordered_euler_shape_eigenvalue_bounds_limit_higher_resonance_orders`,
  `tests/test_obstructions.py::test_ordered_euler_shape_gap_public_audit_polynomial_identities_are_exact`.
- Public reviewer conclusion required: the mass-metric spectra in TC4 cover all
  positive masses and leave no reduced indicial root with zero real part.

TC4-Audit-02, indicial equation:

- Claim: reduced hyperbolicity is equivalent to absence of the rotation
  eigenvalue `mu=-2/9` after quotienting.
- Formula to audit:

```text
q=tau^2 S,        t_c-t=tau^3,
tau^2 S'' + 2 tau S' - 2S = 9 A(S),
(k+2)(k-1)=9mu.
```

- Public reviewer conclusion required: the Lagrange beta spectrum and the Euler
  `4/9 < sigma < 32/9` calculation exclude `mu=-2/9` on the reduced shape
  block, not merely for sampled masses.

TC5-Audit-01, generalized entry language:

- Claim: incoming germs use finite generalized Fuchsian/Puiseux-log data, not
  an ordinary integer-power Taylor ansatz.
- Coordinates to audit: `mcgehee_time`, `scale_energy`,
  `stable_shape_modes`.
- Selector fields to audit: `exponents`, `resonant_rows`,
  `cauchy_tail_majorants`.
- Local artifact ids:
  `tests/test_obstructions.py::test_stable_log_selector_chain_constructor_recovers_coupled_log_selectors`,
  `tests/test_obstructions.py::test_stable_log_selector_chain_log_degree_bound_is_finite_and_triangular`,
  `tests/test_obstructions.py::test_fuchsian_log_resonant_projector_right_inverse_identities_are_exact`,
  `tests/test_obstructions.py::test_stable_log_selector_chain_projects_to_finite_fuchsian_log_branch`,
  `tests/test_obstructions.py::test_fuchsian_log_row_constructor_builds_resonant_selector_branch`,
  `tests/test_obstructions.py::test_finite_fuchsian_log_branch_composes_selector_rows_and_projects`.
- Public reviewer conclusion required: arbitrary incoming total-collision germs
  enter this finite generalized-entry language on the punctured incoming side.

TC5-Audit-02, resonance/projector solve:

- Claim: resonant rows are solved by finite triangular log-degree descent and
  nonresonant rows have a certified denominator floor.
- Formula to audit:

```text
(<m,alpha>-alpha_j)c_{j,m,ell}
    + (ell+1)c_{j,m,ell+1}
    = known lower-weight forcing,

Pi_j^2 = Pi_j,
M_j Pi_j = 0,
Pi_j M_j = 0,
M_j Q_j = I-Pi_j,
Pi_j Q_j = Q_j Pi_j = 0.
```

- Rule ids to audit: `poincare_dulac_denominator_projector`,
  `finite_log_degree_bound`.
- Order keys to audit: `weight`, `row`, `log_degree`.
- Local artifact id:
  `tests/test_obstructions.py::test_fuchsian_log_resonant_projector_right_inverse_identities_are_exact`.
- Public reviewer conclusion required: the row solve extracts exactly the
  finite selector constants for the incoming branch and does not choose an
  outgoing continuation.

TC6-Audit-01, Banach majorant:

- Claim: retained generalized rows plus a finite Cauchy audit determine a
  unique analytic remainder in the omitted-weight Banach ball.
- Primitive constants to audit: `(C0, Lambda, sigma, p0, d)` for value, first
  jet, lifted residual, projected physical residual, and regularized-position
  value.
- Polydisc and norm ids to audit: `tc6-polydisc:P(r)`,
  `tc6-norm:sup-polydisc`.
- Formula to audit:

```text
D      = retained truncation defect,
B      = omitted-row right-inverse bound,
L_N    = nonlinear omitted-part Lipschitz bound,
R_0    = candidate weighted ball radius,
q = B L_N < 1,
B D + q R_0 <= R_0.
```

- Public reviewer conclusion required: every constant is extracted by rational
  interval, ball, or otherwise directed-rounded proof arithmetic, not by
  sampled floating-point evidence.
- Local checker artifact ids:
  `tests/test_certificate_checker.py::test_generalized_fuchsian_remainder_majorant_picard_tail_formula_is_explicit`,
  `tests/test_certificate_checker.py::test_independent_checker_recomputes_generalized_fuchsian_primitive_tail_bounds`.

TC6-Audit-02, projected residual transfer:

- Claim: the checker certifies the direct interval physical Newton residual
  plus the physical-residual Cauchy tail, not only lifted residual tails.
- Formula to audit:

```text
projected_residual
  = [tau^2 S'' + 2 tau S' - 2 S - 9 A(S)]/(9 tau^4).
```

- Required backend id: `certify_rational_interval_arithmetic_backend_soundness`.
- Local checker artifact ids:
  `tests/test_certificate_checker.py::test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture`,
  `tests/test_certificate_checker.py::test_generalized_fuchsian_remainder_majorant_picard_tail_formula_is_explicit`,
  `tests/test_certificate_checker.py::test_independent_checker_recomputes_generalized_fuchsian_primitive_tail_bounds`,
  `tests/test_certificate_checker.py::test_generalized_fuchsian_projected_residual_direct_interval_is_certification_gated`,
  `tests/test_certificate_checker.py::test_generalized_fuchsian_projected_budget_does_not_weaken_lifted_gate`,
  `tests/test_certificate_checker.py::test_generalized_fuchsian_projected_residual_weight_shift_is_cubic_time_exact`,
  `tests/test_certificate_checker.py::test_generalized_fuchsian_stop_checker_certification_is_interval_not_sample_gated`,
  `tests/test_certificate_checker.py::test_independent_checker_rejects_corrupted_generalized_fuchsian_majorant`,
  `tests/test_certificate_checker.py::test_independent_checker_rejects_generalized_fuchsian_without_projected_tail`.
- Checker module artifacts:
  `TotalCollisionGeneralizedFuchsianStopChartCertificate`,
  `check_total_collision_generalized_fuchsian_stop_chart`.
- Public reviewer conclusion required: the weight shift from lifted residual
  order `omega` to physical residual order `omega-4` is included in the
  primitive Cauchy inputs, the direct projected residual interval is actually
  gated, and the optional projected-residual tolerance cannot weaken the
  lifted-residual tolerance.

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
