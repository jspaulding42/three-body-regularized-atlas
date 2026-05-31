# Triple-Collision Regularized Jet Lemma

## Claim

A zero-angular total-collision continuation cannot be selected from the
collapsed Newtonian state, nor from the first regularized collision-time jet.
For any analytic cubic-time total-collision branch whose leading collision
shape is nondegenerate, Newton's equation forces the second regularized-time
jet to be a scaled central configuration. For the parabolic homothetic branches,
that second jet is exactly the datum that selects the outgoing branch.

The homothetic model is:

```text
t = tau^3,
q_i(tau) = C_i tau^2.
```

Then `q_i(0) = 0` and `dq_i/dtau(0) = 0` for every branch. The branch is carried
by:

```text
J_i = d^2 q_i / dtau^2 at tau = 0 = 2 C_i.
```

The Newtonian equation away from `tau = 0` forces the algebraic condition:

```text
A(C) = -(2/9) C,
```

or equivalently:

```text
A(J) = -(1/36) J.
```

Thus a collision-continuation convention that passes through zero-angular total
collision must include a central-configuration second-jet datum, or an explicit
rule that chooses one.

## General Analytic Cubic-Time Branch

The same second-jet condition is necessary without assuming a homothetic branch.
Work after translating the collision point to the origin, and suppose:

```text
t = tau^3,
q_i(tau) = C_i tau^2 + O(tau^3),
```

where the shape `C` is noncollision, so all `C_i-C_j` are nonzero. Then:

```text
dq_i/dt
  = (dq_i/dtau)/(dt/dtau)
  = (2/3) C_i tau^-1 + O(1),
```

and:

```text
d^2q_i/dt^2
  = -(2/9) C_i tau^-4 + O(tau^-2).
```

On the other hand, Newtonian acceleration is homogeneous of degree `-2`, and
smooth near the noncollision shape `C`, so:

```text
A_i(q(tau))
  = A_i(tau^2(C + O(tau)))
  = tau^-4 A_i(C) + O(tau^-3).
```

If the projected curve satisfies Newton's equation for `tau != 0`, multiplying
`q''-A(q)` by `tau^4` and taking `tau -> 0` gives:

```text
-(2/9) C_i - A_i(C) = 0.
```

Thus every analytic cubic-time total-collision branch with a noncollision
second shape must satisfy:

```text
A(C) = -(2/9) C.
```

Equivalently, for the second regularized jet `J=2C`:

```text
A(J) = -(1/36) J.
```

Higher regularized-time jets may impose additional equations, but they cannot
change this leading central-configuration condition.

## Three-Body Second-Jet Dichotomy

For three bodies, the central-configuration condition already leaves only the
classical geometric types. A noncollinear second jet is necessarily a Lagrange
equilateral shape; otherwise it is a collinear Euler shape.

Let `C` be mass-centered:

```text
sum_i m_i C_i = 0,
```

and suppose:

```text
A(C) = -lambda C,       lambda > 0.
```

If `C_1`, `C_2`, and `C_3` are not collinear, then at vertex `1` the two edge
vectors:

```text
e_12 = C_2 - C_1,
e_13 = C_3 - C_1
```

are a basis of the collision plane. The acceleration equation at body `1` is:

```text
m_2 e_12/r_12^3 + m_3 e_13/r_13^3 = -lambda C_1.
```

The mass-centering identity gives:

```text
m_2 e_12 + m_3 e_13 = -M C_1,      M = m_1+m_2+m_3.
```

Because `e_12,e_13` are linearly independent, comparing the two coordinate
expansions of `-C_1` forces:

```text
M/r_12^3 = lambda = M/r_13^3.
```

Thus `r_12=r_13`. Repeating the same argument at vertex `2` gives
`r_12=r_23`. Hence:

```text
r_12 = r_13 = r_23.
```

So every noncollinear three-body total-collision second jet is equilateral,
after translation, rotation, scale, and reflection. If the three points are
collinear, the same central equation is exactly the Euler collinear central
configuration equation for the chosen ordering.

This is not yet a continuation theorem: resonances and branch-selection
parameters can still occur inside the equilateral or Euler families. It does
rule out any scalene noncollinear second-jet branch in a zero-angular total
collision normal form.

## Ordered Collinear Euler Ratio

The collinear side of the dichotomy is also one-dimensional before scale. Fix
an ordering on the line and write:

```text
x_1 = 0,
x_2 = 1,
x_3 = 1 + r,       r > 0.
```

The masses attached to those ordered bodies are `m_1,m_2,m_3`. Translation to
the mass center is irrelevant for pair differences. Let `A_i` be the scalar
accelerations along the line. The central-configuration equation is equivalent
to:

```text
A_2 - A_1 = -lambda (x_2-x_1),
A_3 - A_2 = -lambda (x_3-x_2).
```

Eliminating `lambda` gives:

```text
r(A_2-A_1) = A_3-A_2.
```

Using:

```text
A_1 = m_2 + m_3/(1+r)^2,
A_2 = -m_1 + m_3/r^2,
A_3 = -m_1/(1+r)^2 - m_2/r^2,
```

and multiplying by `r^2(1+r)^2` gives Euler's quintic:

```text
(m_1+m_2)r^5
+(3m_1+2m_2)r^4
+(3m_1+m_2)r^3
-(m_2+3m_3)r^2
-(2m_2+3m_3)r
-(m_2+m_3) = 0.
```

For positive masses, the coefficient signs are:

```text
++,---.
```

There is exactly one sign change, so Descartes' rule gives at most one positive
root. The polynomial is negative at `r=0` and positive as `r -> infinity`, so
there is at least one positive root. Hence each ordered collinear mass triple
has exactly one Euler ratio `r>0`. After translating to the mass center and
scaling by `s=(9 lambda/2)^(1/3)`, this gives the unique normalized collinear
second jet for that ordering:

```text
A(sX) = -(2/9)sX.
```

Thus the zero-angular second-jet selector has only the two Lagrange orientations
and three Euler orderings, up to rotation, scale, and reflection. The later
regularized coefficients can still resonate, so this is a classification of
the second jet, not a full branch-continuation theorem.

## Ordered Euler Linearized Spectrum Reduction

For any ordered Euler second jet, the linearized acceleration has only one
shape parameter left after the universal translation, scale, and rotation
modes are removed.

Put the normalized collinear central configuration on the `x`-axis:

```text
C_i = (X_i, 0),       A(C) = -(2/9)C.
```

The derivative of one pair force

```text
r -> r/|r|^3
```

in a perturbation `u` is:

```text
u/|r|^3 - 3r(r.u)/|r|^5.
```

When `r` is horizontal, this equals `-2u_x/|r|^3` in the horizontal direction
and `u_y/|r|^3` in the transverse direction. Thus the horizontal and transverse
blocks satisfy:

```text
L_parallel = -2 L_perp.
```

The universal modes give:

```text
translations: 0, 0,
scale:        DA(C)[C] = (4/9)C,
rotation:     DA(C)[iC] = -(2/9)iC.
```

There is one remaining mass-centered horizontal shape eigenvalue; call it
`sigma`. The paired transverse shape eigenvalue is then `-sigma/2`. Hence:

```text
spec(DA(C)) = {0, 0, 4/9, -2/9, sigma, -sigma/2}.
```

The horizontal block is positive semidefinite in the mass inner product, because
its quadratic form is:

```text
sum_{i<j} 2 m_i m_j (xi_i-xi_j)^2 / |X_i-X_j|^3.
```

It vanishes only on translations. Therefore `sigma > 0`, and the paired
transverse shape eigenvalue is negative. Since the coefficient multipliers

```text
lambda_n = n(n-3)/9,       n >= 3,
```

are nonnegative, transverse Euler shape modes never create higher-order
resonances. After center-of-mass reduction, the only possible nonhomothetic
Euler resonances are:

```text
lambda_n = sigma,       n >= 5,
```

besides the universal quartic scale resonance. This reduces the arbitrary
ordered Euler normal-form problem to locating the single horizontal shape
eigenvalue `sigma` for each ordered mass triple.

## Ordered Euler Resonance Order Bound

The single ordered-Euler shape eigenvalue has a uniform range:

```text
4/9 < sigma < 32/9.
```

Consequently, because:

```text
lambda_5 = 10/9,
lambda_6 = 2,
lambda_7 = 28/9,
lambda_8 = 40/9,
```

the only possible nonhomothetic ordered-Euler higher resonances are at orders
`n=5,6,7`.

Here is the calculation. For the unscaled ordered Euler shape:

```text
x_1=0, x_2=1, x_3=1+r,
```

let `lambda_c` be the unscaled central multiplier. The horizontal trace is:

```text
tr(L_parallel)
 = 2[(m_1+m_2) + (m_2+m_3)/r^3 + (m_1+m_3)/(1+r)^3].
```

After scaling to `A(C)=-(2/9)C`, the shape eigenvalue is:

```text
sigma = (2/(9 lambda_c)) [tr(L_parallel) - 2 lambda_c].
```

Using Euler's quintic to eliminate `m_3` gives:

```text
sigma =
  8 r^2(1+r)(2r^2+3r+2)(m_1(1+r)+m_2 r)
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1).
```

The upper gap is:

```text
32/9 - sigma =
  8[
    m_1 r^2(r-1)^2(r+2)(2r+1)
    + m_2(2r^6+11r^5+19r^4+22r^3+24r^2+16r+4)
  ]
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1),
```

which is positive for `r>0` and positive masses. For the lower gap:

```text
sigma - 4/9 =
  4(3r^2+3r+1)
  [m_1 r^2(r^2+3r+3)+m_2(r-1)(r+1)(r^2+r+1)]
  -------------------------------------------------
  9((m_1+m_2)r^2+2m_2r+m_2)(r^4+2r^3+r^2+2r+1).
```

If `r>=1`, the bracket is positive. If `0<r<1`, positivity of the eliminated
mass:

```text
m_3 =
  [m_1 r^3(r^2+3r+3)
   + m_2(r-1)(r+1)^2(r^2+r+1)]
  /(3r^2+3r+1)
```

implies:

```text
m_1 r^3(r^2+3r+3)
  > m_2(1-r)(r+1)^2(r^2+r+1),
```

and therefore:

```text
m_1 r^2(r^2+3r+3)
  > m_2(1-r)(r+1)(r^2+r+1).
```

So the lower gap is also positive. This proves the stated range and the finite
resonance-order list.

## Ordered Euler Resonance Surfaces

The finite list can be made explicit. Normalize the ordered Euler line by:

```text
x_1=0, x_2=1, x_3=1+r,       r>0,
```

and write:

```text
q = m_1/m_2,       p = m_3/m_2.
```

Solving Euler's quintic together with the resonance condition:

```text
sigma = lambda_n = n(n-3)/9
```

gives rational one-parameter mass surfaces. Since the preceding section proves
that only `n=5,6,7` can occur, these are all possible ordered-Euler
nonhomothetic higher-resonance surfaces.

For `n=5`:

```text
q_5(r) =
  -(r+1)(3r^5-3r^4-7r^3-15r^2-15r-5)
  ------------------------------------------------
  r^2(3r^4+18r^3+35r^2+18r+3),

p_5(r) =
  (r+1)(5r^5+15r^4+15r^3+7r^2+3r-3)
  ----------------------------------------------
  3r^4+18r^3+35r^2+18r+3.
```

For `n=6`:

```text
q_6(r) =
  -(r+1)(r^5+15r^4+19r^3+27r^2+27r+9)
  ----------------------------------------------
  r^2(r^4-10r^3-31r^2-10r+1),

p_6(r) =
  -(r+1)(9r^5+27r^4+27r^3+19r^2+15r+1)
  ----------------------------------------------
  r^4-10r^3-31r^2-10r+1.
```

For `n=7`:

```text
q_7(r) =
  -(r+1)(3r^5+15r^4+17r^3+21r^2+21r+7)
  ----------------------------------------------
  r^2(3r^4-13r^2+3),

p_7(r) =
  -(r+1)(7r^5+21r^4+21r^3+17r^2+15r+3)
  ----------------------------------------------
  3r^4-13r^2+3.
```

The positive-mass portion of each surface is the set of `r>0` for which both
`q_n(r)>0` and `p_n(r)>0`. Conversely, any ordered Euler mass triple with a
higher nonhomothetic resonance must lie on exactly one of these three surfaces
for its ordered ratio `r`.

At the symmetric point `r=1`, these formulas recover the previously isolated
unequal-mass Euler examples:

```text
n=5: (m_1,m_2,m_3) proportional to (12/11,1,12/11),
n=6: (m_1,m_2,m_3) proportional to (4,1,4),
n=7: (m_1,m_2,m_3) proportional to (24,1,24).
```

Equivalently, after normalizing the endpoint masses to one, the middle masses
are `11/12`, `1/4`, and `1/24`. Thus the arbitrary ordered-Euler resonance
problem has been reduced from all positive mass triples to three explicit
rational curves in positive-mass ratio space. The remaining continuation
problem is the branch construction on those resonant curves, not another
search over resonance orders or masses.

## Cubic-Jet Necessary Condition

The next Laurent coefficient gives a sharper branch filter. Suppose the same
analytic branch has the expansion:

```text
t = tau^3,
q_i(tau) = C_i tau^2 + D_i tau^3 + E_i tau^4 + O(tau^5),
```

with `C` noncollision and already scaled so that `A(C)=-(2/9)C`. Then:

```text
dq_i/dt
  = (2/3) C_i tau^-1 + D_i + (4/3) E_i tau + O(tau^2),
```

and therefore:

```text
d^2q_i/dt^2
  = -(2/9) C_i tau^-4 + (4/9) E_i tau^-2 + O(tau^-1).
```

There is no `tau^-3` term in the physical acceleration obtained from the left
side. On the Newtonian side, smoothness of the acceleration map near the
noncollision shape `C` gives:

```text
A(q(tau))
  = tau^-4 A(C + D tau + E tau^2 + O(tau^3))
  = tau^-4 A(C) + tau^-3 DA(C)[D] + O(tau^-2),
```

where `DA(C)[D]` is the Frechet derivative of the Newtonian acceleration map at
`C` applied to the cubic coefficient `D`. After the second-jet equation cancels
the `tau^-4` terms, the `tau^-3` coefficient of `q''-A(q)` is:

```text
-DA(C)[D].
```

Hence every exact analytic cubic-time total-collision branch must also satisfy:

```text
DA(C)[D] = 0.
```

Equivalently, for the third regularized-time jet
`K = d^3q/dtau^3 at tau=0 = 6D`:

```text
DA(C)[K] = 0.
```

This condition is only necessary. It does not construct a continuation and it
does not replace the missing branch convention. It says that, once the second
jet has selected a scaled central configuration, the third regularized-time jet
must lie in the kernel of the linearized acceleration at that scaled shape.
Uniform translations are in this kernel because they do not change pairwise
differences; after center-of-mass reduction even that inertial direction is
removed. Scaling or rotating the central shape at cubic order is not generally
allowed by this fixed `t=tau^3` normalization, since those directions produce a
nonzero `tau^-3` force coefficient.

## Finite Cubic-Asymptotic Version Of The Cubic-Jet Filter

The cubic-jet filter does not require a full analytic germ. It is already
forced by a finite third-order regularized asymptotic.

Assume, on one punctured side of total collision:

```text
t = T + tau^3,
q_i(tau)=tau^2C_i+tau^3D_i+R_i(tau),
R_i=o(tau^3),
R_i'=o(tau^2),
R_i''=o(tau),
```

where `C` is noncollision and `A(C)=-(2/9)C`. Then:

```text
dq_i/dtau = 2tau C_i + 3tau^2D_i + o(tau^2),
d^2q_i/dtau^2 = 2C_i + 6tau D_i + o(tau).
```

Using:

```text
d^2q_i/dt^2 = q_i''(tau)/(9tau^4) - 2q_i'(tau)/(9tau^5),
```

the `D` terms cancel on the physical-acceleration side:

```text
d^2q_i/dt^2
 = -(2/9)C_i tau^-4 + o(tau^-3).
```

On the Newtonian side:

```text
A_i(q(tau))
 = tau^-4 A_i(C + tau D + o(tau))
 = tau^-4 A_i(C) + tau^-3 DA(C)[D] + o(tau^-3).
```

Since the branch solves Newton's equation for `tau != 0`, multiplying the
residual by `tau^3` and letting `tau -> 0` gives:

```text
DA(C)[D]=0.
```

Thus the first higher branch coefficient is already restricted under a finite
`C^3`-type asymptotic. A full convergent normal-form series is only needed for
constructing and selecting the later branch parameters, not for this necessary
cubic-kernel filter.

## Equal-Mass Equilateral Cubic-Jet Rigidity

For the equal-mass equilateral central configuration, the cubic-jet kernel is
exactly the translation space. Therefore, after center-of-mass reduction, the
third regularized-time jet must vanish.

Write the planar configuration in complex notation:

```text
C_i = s zeta_i,       zeta_i in {1, omega, omega^2},
omega^3 = 1,          1 + omega + omega^2 = 0.
```

Let `l = |C_i-C_j| = sqrt(3)s` be the side length. For equal masses:

```text
A(C) = -(3/l^3) C.
```

The scale used above satisfies `A(C)=-(2/9)C`, so:

```text
3/l^3 = 2/9,
l^3 = 27/2.
```

The derivative of the pair force in direction `D` is:

```text
DA_i(C)[D]
  = sum_{j != i} [
      (D_j-D_i)/|C_j-C_i|^3
      - 3(C_j-C_i)((C_j-C_i) dot (D_j-D_i))/|C_j-C_i|^5
    ].
```

The center-of-mass-zero perturbation space splits into the four real modes:

```text
C,       iC,       conjugate(C),       i conjugate(C).
```

A direct substitution in the derivative formula gives:

```text
DA(C)[C]              = (6/l^3) C              = (4/9) C,
DA(C)[iC]             = -(3/l^3) iC            = -(2/9) iC,
DA(C)[conjugate(C)]   = (3/(2l^3))conjugate(C) = (1/9)conjugate(C),
DA(C)[i conjugate(C)] = (3/(2l^3))i conjugate(C)
                                              = (1/9)i conjugate(C).
```

None of these centered eigenvalues is zero. The only zero modes of `DA(C)` are
the two uniform translations, because translations leave every pair difference
unchanged. Hence the cubic-jet condition:

```text
DA(C)[D] = 0
```

forces `D` to be a uniform translation. In center-of-mass coordinates:

```text
D = 0.
```

Equivalently, for an equal-mass equilateral total-collision branch normalized
by `t=tau^3`:

```text
q(tau) = C tau^2 + D tau^3 + O(tau^4)
```

with center of mass fixed at the collision point, Newton's equation forces:

```text
D = 0.
```

This is still not an arbitrary zero-angular total-collision continuation
theorem. It proves a rigidity property for one important central-configuration
branch: nonhomothetic freedom, if present, cannot enter through the cubic
regularized-time jet around the equal-mass equilateral branch.

## Equal-Mass Equilateral Quartic-Jet Rigidity

The same Laurent balance identifies the next possible centered freedom around
the equal-mass equilateral branch. Suppose:

```text
q(tau) = C tau^2 + D tau^3 + E tau^4 + O(tau^5),
```

where `C` is the scaled equal-mass equilateral central configuration and the
cubic condition has already forced `D` to be a uniform translation. Pairwise
differences do not see that uniform cubic translation, so it does not contribute
to the force expansion. Comparing the `tau^-2` terms gives:

```text
(4/9) E = DA(C)[E].
```

The eigenspace computation above shows that the `4/9` eigenspace is exactly the
scale direction:

```text
E = alpha C.
```

Uniform translations have eigenvalue `0`, rotations have eigenvalue `-2/9`, and
the two shape modes have eigenvalue `1/9`, so none of them can occur at quartic
order. Thus, in a center-of-mass frame:

```text
q(tau) = C tau^2 + alpha C tau^4 + O(tau^5).
```

The allowed quartic coefficient is precisely the homothetic energy direction.
For the energy-parametrized homothetic branch:

```text
q(tau) = C tau^2 u(tau^2),
u(z) = 1 + c_1 z + O(z^2),
```

one has:

```text
E = c_1 C.
```

So the equilateral branch has no centered nonhomothetic freedom through quartic
regularized time; the first allowed quartic motion is the already-known
homothetic energy parameter.

## Equal-Mass Equilateral Quintic And Sextic Rigidity

The next two coefficients continue the same pattern. Work in the center-of-mass
frame and write:

```text
q(tau) = C tau^2 + alpha C tau^4 + F tau^5 + G tau^6 + O(tau^7).
```

The physical acceleration of a monomial is:

```text
d^2/dt^2 (Q_n tau^n)
  = (n(n-3)/9) Q_n tau^(n-6).
```

The `tau^-1` coefficient gives:

```text
(10/9) F = DA(C)[F].
```

But the equilateral linearized spectrum consists of:

```text
0, 0, 4/9, -2/9, 1/9, 1/9.
```

It does not contain `10/9`, so:

```text
F = 0.
```

For the constant term, expand the force around the scaled equilateral shape:

```text
A(C + alpha C tau^2 + G tau^4 + O(tau^5)).
```

The scale part can be computed exactly from homogeneity:

```text
A((1 + alpha tau^2)C)
  = (1 + alpha tau^2)^(-2) A(C)
  = A(C) - 2 alpha A(C) tau^2 + 3 alpha^2 A(C) tau^4 + O(tau^6).
```

Therefore the constant coefficient in Newton's equation is:

```text
2G = DA(C)[G] + 3 alpha^2 A(C).
```

Since `A(C)=-(2/9)C`, this is:

```text
(2I - DA(C))G = -(2/3) alpha^2 C.
```

The operator `2I-DA(C)` is invertible on the full six-dimensional space because
`2` is not in the spectrum above. The right side is in the scale direction, so
`G` must also be in the scale direction. Writing `G=gamma C` and using
`DA(C)[C]=(4/9)C` gives:

```text
(2 - 4/9) gamma C = -(2/3) alpha^2 C,
gamma = -(3/7) alpha^2.
```

Thus:

```text
q(tau)
  = C tau^2
    + alpha C tau^4
    - (3/7) alpha^2 C tau^6
    + O(tau^7).
```

This matches the energy-parametrized homothetic recurrence
`u(z)=1+alpha z-(3/7)alpha^2 z^2+O(z^3)`. Hence the equal-mass equilateral
branch has no centered nonhomothetic freedom through sextic regularized time;
through this order it is forced to agree with the homothetic energy family.

## All-Order Formal Equilateral Rigidity

The finite-order computation is not an accident. In the equal-mass equilateral
case, the formal centered analytic branch is uniquely forced to be the
homothetic energy family once the quartic scale coefficient is chosen.

Write:

```text
q(tau) = C tau^2 + sum_{n>=3} Q_n tau^n,
```

and compare the coefficient of `tau^(n-6)` in Newton's equation. The new
coefficient `Q_n` enters linearly as:

```text
(lambda_n I - DA(C)) Q_n = known lower-order terms,
lambda_n = n(n-3)/9.
```

The equilateral spectrum is:

```text
spec(DA(C)) = {0, 0, 4/9, -2/9, 1/9, 1/9}.
```

The only resonances with `lambda_n` for `n>=3` are:

```text
n=3:  lambda_3 = 0      translation modes,
n=4:  lambda_4 = 4/9    scale mode.
```

For every `n>=5`, `lambda_n` is not in the spectrum, so the linear equation is
invertible.

After center-of-mass reduction the `n=3` translation resonance is removed. The
`n=4` resonance leaves exactly one scalar parameter:

```text
Q_4 = alpha C.
```

Inductively suppose all lower coefficients agree with a scale-only even
series:

```text
q(tau) = C tau^2 u(tau^2) + O(tau^n).
```

Then the lower-order force terms are obtained from:

```text
A(C u(tau^2)) = u(tau^2)^(-2) A(C),
```

so they are also scale-only and contain only even powers of `tau`. Therefore:

- for odd `n>=5`, the known forcing is zero and invertibility gives `Q_n=0`;
- for even `n>=6`, the known forcing is a scalar multiple of `C`, and
  invertibility gives `Q_n` as a scalar multiple of `C`.

The resulting scalar coefficients obey the same recurrence as the homothetic
energy equation:

```text
(u + z u')^2 = 1/u + (9/2) epsilon z,
z = tau^2.
```

Thus the centered formal Newton branch with equal-mass equilateral second jet
has the unique form:

```text
q(tau) = C tau^2 u_epsilon(tau^2),
```

where `u_epsilon` is the homothetic energy-series solution and `epsilon` is
determined by `alpha = (9/10)epsilon`.

Since the homothetic energy branch is analytic by the Briot-Bouquet argument,
any genuinely analytic branch with this formal Taylor series agrees with it in
a neighborhood of `tau=0`. This proves local analytic uniqueness for centered
equal-mass equilateral cubic-time total-collision branches. It is still not the
arbitrary zero-angular total-collision theorem, because other central
configurations, unequal masses, and non-equilateral collision-manifold behavior
remain separate problems.

## Arbitrary-Mass Equilateral Resonance Classification

The equilateral argument extends to arbitrary positive masses, but with one
important exceptional mass ratio. Let:

```text
M = m_1 + m_2 + m_3,
beta = (m_1 m_2 + m_1 m_3 + m_2 m_3) / M^2.
```

For positive masses:

```text
0 < beta <= 1/3,
```

with equality only for equal masses. Put the equilateral triangle in its
mass-centered frame, scale it so that:

```text
A(C) = -(2/9) C.
```

Equivalently, if `l` is the side length after scaling, then:

```text
l^3 = (9/2) M.
```

A direct mass-weighted shape-coordinate calculation gives the linearized
acceleration spectrum:

```text
spec(DA(C)) =
  { 0, 0, 4/9, -2/9,
    1/9 + (1/3) sqrt(1 - 3 beta),
    1/9 - (1/3) sqrt(1 - 3 beta) }.
```

The first two eigenvalues are translations. The `4/9` eigenvalue is the scale
direction, because homogeneity gives:

```text
DA(C)[C] = -2 A(C) = (4/9) C.
```

The `-2/9` eigenvalue is the rotation direction, because rotational covariance
gives:

```text
DA(C)[iC] = i A(C) = -(2/9) iC.
```

The two remaining shape eigenvalues are the roots of:

```text
nu^2 - (2/9)nu + (27 beta - 8)/81 = 0.
```

Hence they are exactly:

```text
nu_pm = 1/9 +/- (1/3) sqrt(1 - 3 beta).
```

Now compare with the regularized coefficient multipliers:

```text
lambda_n = n(n-3)/9,       n >= 3.
```

For `n>=5`, `lambda_n >= 10/9`, while every nontranslation equilateral
linearized eigenvalue is at most `4/9`. Thus there are no resonances at order
`n>=5`. At `n=4`, the only positive-mass resonance with `lambda_4=4/9` is the
scale direction. At `n=3`, translations always resonate with `lambda_3=0`; the
lower shape eigenvalue also resonates exactly when:

```text
1/9 - (1/3) sqrt(1 - 3 beta) = 0,
beta = 8/27.
```

Therefore:

- If `beta != 8/27`, center-of-mass reduction removes the only cubic
  resonances, and the quartic scale resonance is the homothetic energy
  parameter. The same homogeneity induction as above proves local analytic
  uniqueness: every analytic centered arbitrary-mass equilateral branch with
  this second jet is the homothetic energy branch.
- If `beta = 8/27`, the cubic-jet condition `DA(C)[D]=0` has a centered
  nontranslation shape solution. The homothetic rigidity proof genuinely fails
  at cubic order. This does not by itself construct a nonhomothetic
  continuation, but it identifies a real resonant mass surface that any
  arbitrary-mass zero-angular triple-collision theorem must handle separately.

For example, masses proportional to:

```text
(1, 1, 5/2)
```

have:

```text
beta = (1 + 5/2 + 5/2) / (9/2)^2 = 8/27.
```

So the arbitrary-mass equilateral branch is almost covered by the same
homothetic rigidity argument, but not completely: the exceptional
`beta=8/27` resonance is the local normal-form problem handled next.

## Resonant Arbitrary-Mass Equilateral Cubic Branch

The exceptional surface is not merely a bookkeeping artifact. For the concrete
resonant masses:

```text
(m_1, m_2, m_3) = (1, 1, 5/2),
```

the cubic resonance survives the next Laurent balance.

Use the unscaled mass-centered equilateral shape:

```text
hat C =
  ( 7/6,  sqrt(3)/6),
  (-1/3,  2sqrt(3)/3),
  (-1/3, -sqrt(3)/3),
```

and set `C=s hat C`, where `s` is chosen so that `A(C)=-(2/9)C`. A centered
nontranslation cubic-kernel vector is:

```text
D =
  (-1,       sqrt(3)),
  ( 2,       0),
  (-2/5,    -2sqrt(3)/5),
```

up to scalar multiple. It satisfies:

```text
sum_i m_i D_i = 0,
DA(C)[D] = 0.
```

Write the regularized branch as:

```text
q(tau) = C tau^2 + a D tau^3 + E tau^4 + O(tau^5).
```

Let `B_C(D,D)` denote the coefficient of `eta^2` in the force expansion:

```text
A(C + eta D) = A(C) + eta DA(C)[D] + eta^2 B_C(D,D) + O(eta^3).
```

For one pair term `r/|r|^3` and perturbation `u`, this second coefficient is:

```text
-3 u (r.u)/|r|^5
+ r (-(3/2)|u|^2/|r|^5 + (15/2)(r.u)^2/|r|^7).
```

The quartic coefficient equation is:

```text
(4/9 I - DA(C)) E = a^2 B_C(D,D).
```

The operator on the left has only the scale kernel. Because `DA(C)` is
self-adjoint for the mass inner product, the range condition is:

```text
sum_i m_i C_i . B_C(D,D)_i = 0.
```

Direct substitution of the displayed `C` and `D` gives exactly this
orthogonality. Hence a quartic coefficient exists:

```text
E = a^2 E_0 + alpha C,
```

where `E_0` is any particular solution and `alpha` is the homothetic energy
parameter.

For this mass choice there are no further resonances after the cubic and scale
ones: `lambda_n=n(n-3)/9` is not in the spectrum for every `n>=5`. Therefore,
once `a`, `alpha`, and a range-compatible `E_0` are chosen, every later formal
coefficient is uniquely determined by the previous ones.

To see convergence, write `q(tau)=tau^2 S(tau)`. Newton's equation is
equivalent, for `tau != 0`, to the regular-singular analytic equation:

```text
tau^2 S'' + 2 tau S' - 2S = 9 A(S).
```

The force map `A(S)` is analytic in a fixed ball around `C`, because `C` is a
noncollision shape. After the resonant coefficients `S_1=aD` and
`S_2=a^2E_0+alpha C` are fixed, the coefficient equation for `S_k`, `k>=3`,
has inverse norm bounded by `const / k^2`: the scalar multiplier is
`(k+2)(k-1)/9`, and its distance from the finite spectrum of `DA(C)` grows
quadratically. The nonlinear coefficient at order `k` is a finite convolution
of lower coefficients with the Taylor coefficients of the analytic force map.
A scalar majorant series with quadratic divisors therefore dominates the
recurrence and has positive radius of convergence. The formal branch is
convergent.

The constructive recurrence can be written explicitly. If:

```text
S(tau) = sum_{k>=0} S_k tau^k,
S_0 = C,
S_1 = aD,
S_2 = a^2E_0 + alpha C,
```

and `A(S(tau)) = sum_{k>=0} F_k tau^k`, then for `k>=3`:

```text
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0).
```

The right side is computed by substituting the already known lower coefficients
into the analytic pair-force series. Since the operator is invertible for
every `k>=3`, this recursively constructs the nonhomothetic branch.

Thus the resonant mass case has a genuine nonhomothetic local analytic
total-collision family, at least near the displayed equilateral central
configuration. Since the regularized curve has `q(tau)=O(tau^2)` and solves
Newton's equation for `tau != 0`, its angular momentum is conserved away from
collision and tends to zero as `tau -> 0`; the continuation is therefore a
zero-angular total-collision branch.

This sharpens the arbitrary-mass obstruction: a central-configuration second
jet plus homothetic energy is not enough on the resonant surface. A full
zero-angular triple-collision continuation theorem must include the extra
resonant cubic branch parameter or prove a separate selection rule for it.

## Full Equilateral Beta Resonance Surface

The preceding construction is not confined to the displayed mass triple. It
holds for every positive mass triple with:

```text
beta = (m_1m_2+m_1m_3+m_2m_3)/(m_1+m_2+m_3)^2 = 8/27.
```

At such a mass triple the scaled mass-centered equilateral spectrum is:

```text
spec(DA(C)) = {0, 0, 0, 2/9, -2/9, 4/9}.
```

The three zero eigenvalues consist of two translations and one centered
nontranslation shape mode. Let `D` be that centered shape mode:

```text
sum_i m_i D_i = 0,
DA(C)[D] = 0.
```

The only possible obstruction to continuing the cubic resonant coefficient is
again the quartic equation:

```text
(4/9 I - DA(C))E = a^2 B_C(D,D).
```

This solvability condition is automatic. Let:

```text
<X,Y>_m = sum_i m_i X_i . Y_i.
```

The acceleration map is the mass-gradient of the Newtonian potential, so the
third derivative tensor:

```text
T(X,Y,Z)=<X,D^2A(C)[Y,Z]>_m
```

is symmetric in all three slots. Homogeneity of degree `-2` gives:

```text
DA(C)[C] = -2A(C),
```

and differentiating this identity in the direction `D` gives:

```text
D^2A(C)[D,C] = -3 DA(C)[D] = 0.
```

Therefore:

```text
<C,D^2A(C)[D,D]>_m
 = T(C,D,D)
 = T(D,C,D)
 = <D,D^2A(C)[C,D]>_m
 = 0.
```

Since `B_C(D,D)` is one half of `D^2A(C)[D,D]`, it is mass-orthogonal to the
scale kernel `C`. The quartic equation is therefore solvable for every mass
triple on the `beta=8/27` surface:

```text
E = a^2 E_0 + alpha C.
```

No further resonance occurs. For `k>=3`, the coefficient multiplier:

```text
((k+2)(k-1))/9
```

is at least `10/9`, while every eigenvalue of `DA(C)` is at most `4/9`.
Thus the coefficient recurrence is invertible at every later order:

```text
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0).
```

The same analytic majorant argument gives a convergent local branch for each
choice of cubic amplitude `a` and homothetic energy parameter `alpha`. Hence
the entire positive `beta=8/27` equilateral mass surface, not merely the
example `(1,1,5/2)`, carries local nonhomothetic zero-angular total-collision
branches.

## All-Order Formal Equal-Mass Euler Rigidity

The same all-order mechanism also closes the centered equal-mass Euler branch.
Take the unscaled collinear configuration:

```text
X = (-1, 0, 1)
```

on the horizontal axis. For equal masses:

```text
A(X) = -(5/4)X.
```

After scaling `C=sX` with:

```text
s^3 = 45/8,
```

the normalized collision equation becomes:

```text
A(C) = -(2/9)C.
```

The derivative splits into horizontal and transverse blocks. At the unscaled
configuration `X`, the horizontal block is:

```text
L_parallel =
  [  9/4   -2   -1/4 ]
  [   -2    4     -2 ]
  [ -1/4   -2    9/4 ],
```

with eigenvectors/eigenvalues:

```text
(1, 1, 1)      -> 0,
(-1, 0, 1)     -> 5/2,
(1, -2, 1)     -> 6.
```

The transverse block is:

```text
L_perp =
  [ -9/8    1    1/8 ]
  [    1   -2      1 ]
  [  1/8    1   -9/8 ],
```

with eigenvectors/eigenvalues:

```text
(1, 1, 1)      -> 0,
(-1, 0, 1)     -> -5/4,
(1, -2, 1)     -> -3.
```

Since `DA` is homogeneous of degree `-3`, scaling by `s` multiplies these
eigenvalues by:

```text
s^-3 = 8/45.
```

Thus:

```text
spec(DA(C)) = {0, 0, 4/9, -2/9, 16/15, -8/15}.
```

As before, a coefficient `Q_n tau^n` enters Newton's equation through:

```text
(lambda_n I - DA(C)) Q_n = known lower-order terms,
lambda_n = n(n-3)/9.
```

For `n>=3`, the only resonances are:

```text
n=3:  lambda_3 = 0      translation modes,
n=4:  lambda_4 = 4/9    scale mode.
```

Indeed, `lambda_n>=10/9` for every `n>=5`, so it cannot equal `-8/15`,
`-2/9`, `0`, or `4/9`; and it cannot equal `16/15`, since `16/15 < 10/9`.
After center-of-mass reduction the translation resonance disappears, and the
only remaining free coefficient is:

```text
Q_4 = alpha C.
```

The same induction as in the equilateral case then applies. If all lower
coefficients are scale-only and even, homogeneity gives:

```text
A(C u(tau^2)) = u(tau^2)^(-2) A(C),
```

so the known lower-order forcing is scale-only and even. Odd coefficients
therefore vanish, and even coefficients are scalar multiples of `C` determined
by the scalar homothetic recurrence:

```text
(u + z u')^2 = 1/u + (9/2) epsilon z.
```

Consequently, any analytic centered equal-mass Euler cubic-time
total-collision branch with this second jet agrees locally with the
homothetic energy family. This closes the second classical equal-mass central
configuration branch, but still does not prove arbitrary masses, the unequal
mass Euler branches, or general zero-angular collision-manifold continuation.

## Symmetric Unequal-Mass Euler Resonance Classification

The equal-mass Euler proof does not extend uniformly to unequal masses. Already
in the one-parameter symmetric family:

```text
(m_1, m_2, m_3) = (1, mu, 1),
X = (-1, 0, 1),
```

the collinear shape remains central by symmetry, with unscaled central
multiplier:

```text
lambda_c = mu + 1/4.
```

After scaling `C=sX` so that `A(C)=-(2/9)C`, the derivative still splits into
horizontal and transverse blocks. The unscaled horizontal block is:

```text
L_parallel =
  [ 2mu+1/4   -2mu   -1/4    ]
  [ -2          4      -2     ]
  [ -1/4      -2mu    2mu+1/4],
```

with eigenvalues:

```text
0,        2mu + 1/2,        2mu + 4.
```

The unscaled transverse block is:

```text
L_perp =
  [ -mu-1/8    mu      1/8    ]
  [  1         -2      1      ]
  [  1/8       mu     -mu-1/8],
```

with eigenvalues:

```text
0,        -(mu + 1/4),        -(mu + 2).
```

Scaling multiplies all eigenvalues by:

```text
s^-3 = 2 / (9(mu + 1/4)).
```

Therefore:

```text
spec(DA(C)) =
  { 0, 0, 4/9, -2/9,
    16(mu+2)/(9(4mu+1)),
    -8(mu+2)/(9(4mu+1)) }.
```

The first two eigenvalues are translations, `4/9` is the scale direction, and
`-2/9` is the rotation direction. The positive horizontal shape eigenvalue:

```text
h(mu) = 16(mu+2)/(9(4mu+1))
```

can resonate with the regularized multiplier:

```text
lambda_n = n(n-3)/9.
```

Since `h(mu)` decreases from `32/9` to `4/9` as `mu` runs from `0` to
`infinity`, the only higher-order resonances with `n>=5` are:

```text
n=5:  mu = 11/12,
n=6:  mu = 1/4,
n=7:  mu = 1/24.
```

For `n>=8`, `lambda_n >= 40/9 > 32/9`, so no symmetric-Euler horizontal shape
resonance is possible. The transverse shape eigenvalue is negative and cannot
resonate with `lambda_n` for `n>=3`.

Thus the equal-mass Euler all-order homothetic rigidity is a nonresonant
special case, not a theorem for all masses. In the symmetric unequal-mass Euler
family, a zero-angular total-collision normal form must also handle the three
explicit higher-order resonance masses above. These are later than the
equilateral `beta=8/27` cubic resonance, but they are the same kind of
obstruction to a branch convention that only supplies a central configuration
and homothetic energy.

## First Symmetric Euler Resonant Branch

The first symmetric Euler resonance also produces a genuine local
nonhomothetic branch. Take:

```text
(m_1, m_2, m_3) = (1, 11/12, 1).
```

The horizontal shape eigenvalue is:

```text
16(mu+2)/(9(4mu+1)) = 10/9 = lambda_5.
```

Thus the first nonhomothetic coefficient may enter at physical power `tau^5`.
Write again `q(tau)=tau^2 S(tau)` and set:

```text
S_0 = C,
S_1 = 0,
S_2 = alpha C,
S_3 = b H,
```

where the mass-centered horizontal shape eigenvector is:

```text
H = (1, -24/11, 1)
```

on the line. The lower coefficients are scale-only and even, so the coefficient
forcing at `S_3` is zero. Since:

```text
DA(C)[H] = (10/9)H,
```

the `S_3` equation is satisfied for arbitrary scalar `b`.

For every later degree `k>=4`, the operator:

```text
(((k+2)(k-1))/9 I - DA(C))
```

is invertible for this mass choice. Therefore the same shape-series recurrence:

```text
A(S(tau)) = sum_k F_k tau^k,
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0)
```

constructs all later coefficients uniquely. The branch stays collinear because
the force map and derivative blocks preserve the horizontal subspace; hence its
angular momentum is identically zero. The same Cauchy-majorant argument used
above gives convergence of the local regularized series.

This supplies another explicit nonhomothetic zero-angular total-collision
family. It first differs from the homothetic Euler branch at order `tau^5`,
rather than at order `tau^3` as in the resonant equilateral case.

## All Symmetric Euler Resonant Branches

The same construction covers the remaining two symmetric Euler resonance
masses. For:

```text
n=5,  mu=11/12,
n=6,  mu=1/4,
n=7,  mu=1/24,
```

the horizontal shape eigenvalue equals:

```text
lambda_n = n(n-3)/9.
```

Let `r=n-2`, so the resonant shape coefficient is `S_r` in
`q(tau)=tau^2 S(tau)`. Start with the homothetic lower even coefficients, set
all lower nonresonant odd coefficients to zero, and write:

```text
S_r = E_r + bH,
```

where `H=(1,-2/mu,1)` is the mass-centered horizontal shape eigenvector and
`E_r` is any particular solution of the `S_r` coefficient equation. For
`n=5`, the particular part is zero. For `n=6`, it contains the lower
homothetic scale forcing. For `n=7`, the lower odd forcing is again zero after
the nonresonant odd coefficients vanish.

For each of the three masses, no later resonance occurs. Thus all later
coefficients are determined by the same analytic recurrence:

```text
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0).
```

The branches remain collinear because the pair-force series preserves the
horizontal subspace. Hence their angular momentum is identically zero, and the
majorant argument gives local convergence. The symmetric Euler branch
therefore contributes three explicit nonhomothetic zero-angular
total-collision families, first differing from the homothetic branch at
`tau^5`, `tau^6`, and `tau^7`, respectively.

## All Ordered Euler Resonant Branches

The preceding construction is not an artifact of endpoint symmetry. It extends
to every positive-mass point on the ordered-Euler resonance surfaces
`(q_n(r),p_n(r))`.

Fix an ordering:

```text
x_1=0, x_2=1, x_3=1+r,       r>0,
```

and suppose:

```text
(m_1/m_2,m_3/m_2)=(q_n(r),p_n(r)),
q_n(r)>0,
p_n(r)>0,
n in {5,6,7}.
```

Let `C` be the corresponding mass-centered Euler central configuration scaled
by `A(C)=-(2/9)C`, and let `H` be the mass-centered horizontal shape
eigenvector with:

```text
DA(C)[H] = lambda_n H,
lambda_n = n(n-3)/9.
```

The ordered-Euler spectrum reduction gives:

```text
spec(DA(C)) = {0, 0, 4/9, -2/9, lambda_n, -lambda_n/2}.
```

After center-of-mass reduction, the translation eigenvalue is gone. The scale
eigenvalue `4/9` is the homothetic energy parameter at `S_2`, and the only
higher resonance is `lambda_n`, because the resonance surfaces were obtained
from exactly one of the values `n=5,6,7` and the order bound excludes every
`m>=8`.

Write the regularized collision branch in shape form:

```text
q(tau)=tau^2 S(tau),
S(tau)=C + S_1 tau + S_2 tau^2 + S_3 tau^3 + ...
```

The coefficient equation is:

```text
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0).
```

Choose the nonresonant lower coefficients to be the homothetic ones:

```text
S_1=0,
S_2=alpha C,
```

and continue the homothetic scale recurrence until the first resonant index
`k=n-2`. At that index the forcing is already solved by the homothetic
particular coefficient, while the kernel contributes a free scalar:

```text
S_{n-2}=E_{n-2}+bH.
```

For every later `k`, the displayed operator is invertible on the
center-of-mass subspace. Hence the recurrence determines `S_k` uniquely as an
analytic function of the two parameters `alpha` and `b`. The Newtonian
acceleration map is analytic near the noncollision shape `C`, and the
regular-singular recurrence has the same Cauchy-majorant form as the
equilateral and symmetric-Euler constructions. Thus the formal series
converges for sufficiently small `|tau|`.

Because `C`, the homothetic coefficients, and `H` are all horizontal, and the
collinear force map preserves the horizontal subspace, the whole branch remains
collinear. Its angular momentum is therefore identically zero. For `b != 0`,
the branch is nonhomothetic and first differs from the homothetic Euler branch
at physical power `tau^n`.

Thus every positive point on the three ordered-Euler resonance surfaces
contributes a genuine local nonhomothetic zero-angular total-collision family.
The branch-convention problem is no longer merely that some special symmetric
masses are resonant; the full ordered-Euler resonant locus carries a one-scalar
nonhomothetic outgoing parameter.

## Nonresonant Ordered Euler Rigidity

The complement of those three surfaces is rigid. Let `C` be any ordered Euler
central configuration scaled by `A(C)=-(2/9)C`, and assume its horizontal shape
eigenvalue `sigma` is not one of:

```text
lambda_5, lambda_6, lambda_7.
```

The order bound already shows that no other higher resonance is possible.
After center-of-mass reduction, the coefficient equation for:

```text
q(tau)=tau^2 S(tau),       S(tau)=sum_{k>=0} S_k tau^k,
```

is:

```text
(((k+2)(k-1))/9 I - DA(C)) S_k = F_k(S_0,...,S_{k-1},0).
```

The `k=1` equation corresponds to translation and is removed by the
mass-centering condition. The `k=2` equation is the universal scale resonance,
so:

```text
S_2 = alpha C.
```

For every `k>=3`, the multiplier is `lambda_{k+2}`. It cannot equal the
negative rotation or transverse shape eigenvalues, cannot equal the two
translation eigenvalues, cannot equal `4/9`, and by the nonresonance
assumption cannot equal the only positive horizontal shape eigenvalue. Thus
the operator is invertible for all `k>=3`.

Now apply the same homogeneity induction used for the equilateral and
equal-mass Euler branches. If the lower coefficients are scale-only and even,
then:

```text
A(C u(tau^2)) = u(tau^2)^(-2) A(C),
```

so the known forcing is also scale-only and even. Invertibility forces every
odd coefficient to vanish and every even coefficient to be a scalar multiple of
`C`. The scalar coefficients obey the homothetic energy recurrence:

```text
(u+z u')^2 = 1/u + (9/2) epsilon z.
```

Consequently, any analytic centered zero-angular branch with a nonresonant
ordered Euler second jet agrees locally with the homothetic energy branch. The
ordered Euler local normal form is therefore complete: nonresonant mass triples
are homothetic-rigid, while the positive points on the three resonance surfaces
carry exactly the extra horizontal branch parameter described above.

## Three-Body Zero-Angular Local Normal Form

Combining the second-jet dichotomy with the equilateral and ordered-Euler
normal-form calculations gives a complete local classification for analytic
cubic-time total-collision branches whose second shape is noncollision.

After translation, rotation, scale, and possibly reflection, the branch has:

```text
t = tau^3,
q(tau)=tau^2 S(tau),
S_0=C,
A(C)=-(2/9)C.
```

The shape `C` is one of:

```text
1. Equilateral Lagrange,
2. Ordered collinear Euler.
```

There is no third scalene noncollinear case.

For the equilateral case, put:

```text
beta = (m_1m_2+m_1m_3+m_2m_3)/(m_1+m_2+m_3)^2.
```

Then:

```text
beta != 8/27:
  S(tau)=C u(tau^2),
  (u+z u')^2 = 1/u + (9/2) epsilon z.

beta = 8/27:
  S_1 = aD,
  S_2 = a^2E_0 + alpha C,
  S_k determined recursively for k>=3.
```

Here `D` is the centered cubic kernel of `DA(C)`, `a` is the resonant branch
parameter, and `alpha` is the homothetic energy parameter. The quartic
solvability condition for `E_0` holds on the entire positive `beta=8/27`
surface by the homogeneity/self-adjointness argument above, and there are no
later resonances.

For the ordered Euler case, let `r>0` be the unique Euler ratio for the chosen
ordering and let `sigma` be the single positive horizontal shape eigenvalue.
Then:

```text
sigma not in {lambda_5, lambda_6, lambda_7}:
  S(tau)=C u(tau^2),
  (u+z u')^2 = 1/u + (9/2) epsilon z.

sigma = lambda_n, n in {5,6,7}:
  (m_1/m_2,m_3/m_2)=(q_n(r),p_n(r)),
  S_{n-2}=E_{n-2}+bH,
  S_k determined recursively for k>n-2.
```

Here `H` is the centered horizontal shape eigenvector, `b` is the ordered-Euler
branch parameter, and `alpha` remains the homothetic energy parameter carried
inside the lower homothetic coefficients.

Thus, locally, the zero-angular three-body total-collision branch data are:

```text
central-configuration second jet,
homothetic energy parameter alpha,
plus one resonant amplitude only on:
  beta=8/27 equilateral branches, or
  the positive ordered-Euler resonance surfaces n=5,6,7.
```

This is still not the global closed-form theorem. It classifies analytic local
branches after a noncollision central second jet has been chosen. A global
solution must still prove how arbitrary finite-time continuations reach or
avoid such charts, how binary charts are composed, and how the all-future
series recurrence closes.

## Parabolic Scale From A Normalized-Potential Limit

There is one bridge from a physical total-collapse branch into the cubic-time
chart that does not require assuming a full analytic regularized germ.

Let a center-of-mass branch approach total collision at finite time `T`. Define:

```text
I(t) = sum_i m_i |q_i(t)|^2,
U(t) = sum_{i<j} m_i m_j / |q_i(t)-q_j(t)|,
H = K-U.
```

Assume:

```text
I(t) -> 0,
H is finite,
U(t) sqrt(I(t)) -> Gamma,        0 < Gamma < infinity.
```

The Lagrange-Jacobi identity gives:

```text
I'' = 4H + 2U.
```

Thus:

```text
I'' = 2 Gamma I^(-1/2) + o(I^(-1/2)).
```

Sundman's inequality also gives:

```text
(I')^2 <= 4 I K = 4I(H+U) = 4 Gamma sqrt(I) + o(sqrt(I)),
```

so `I'(t)->0`. Since `I''>0` sufficiently near `T` and `I(t)->0`, the branch is
eventually inward: `I'<0`. Put `p=-I'>0`. Along this terminal monotone
interval:

```text
d(p^2)/dI = 2 I''.
```

Using `p->0` at `I=0` and integrating back from the collision gives:

```text
p^2 = 8 Gamma sqrt(I) + o(sqrt(I)).
```

Hence:

```text
-I' = sqrt(8 Gamma) I^(1/4)(1+o(1)).
```

Integrating one more time,

```text
I(t) ~ ((9/2) Gamma)^(2/3) (T-t)^(4/3).
```

Therefore any such collapse has the parabolic inertia scale. If, in addition,
the normalized shape has a collision-free limit:

```text
y_i(t)=q_i(t)/sqrt(I(t)) -> y_i^*,
```

then with signed cubic collision time `tau^3=t-T` on either side,

```text
q_i(t) = tau^2 C_i + o(tau^2),
C_i = ((9/2) Gamma)^(1/3) y_i^*
```

up to the sign convention for `tau^2=|t-T|^(2/3)`. This proves that the
finite normalized-potential limit supplies the same parabolic `tau^2` scale
used by the local normal form.

This still does not prove the full arbitrary zero-angular continuation theorem.
The missing global step is to prove enough regularity and shape convergence at
every zero-angular total collapse to enter the analytic central-configuration
normal form below. The scale estimate removes one degree of freedom: any such
entry must use cubic time and a finite nonzero second regularized-position
limit.

## Shape-Compact Collapse Gives Quotient-Shape Convergence

The collision-free shape-limit assumption can be weakened in one important
direction. If the normalized shape remains in a compact collision-free part of
shape space, then the mutual-distance shape converges to one central
configuration class. This is the McGehee blow-up mechanism specialized to the
three-body case.

Work in center-of-mass coordinates and write:

```text
r = sqrt(I),             q = r y,
<y,y>_m = 1,
V(y) = U(y).
```

Assume finite energy, finite-time total collision, and a normalized pair
distance floor:

```text
r(t)->0,
min_{i<j}|y_i(t)-y_j(t)| >= delta > 0
```

near the collision. By the nonzero-angular exclusion theorem, the centered
angular momentum of such a total collapse is zero. Reduce rotations and use
McGehee time:

```text
ds/dt = r^(-3/2),
nu = r^(1/2) dr/dt,
w = dy/ds
```

on the collision-free reduced shape manifold. The energy identity becomes:

```text
rH = (1/2)nu^2 + (1/2)|w|_m^2 - V(y),
```

and the Lagrange-Jacobi identity gives:

```text
dnu/ds = (1/2)|w|_m^2 + rH.
```

Because `V` is bounded above and below on the compact collision-free shape
region, `nu` is bounded and is eventually negative on an incoming total
collapse. Hence `s -> +infinity` at collision and `r` decays at least
exponentially in `s`; in particular `rH` is integrable in `s`. The displayed
monotonicity identity then implies:

```text
int^infinity |w(s)|_m^2 ds < infinity.
```

The reduced shape equation is a smooth analytic ODE on the same compact
collision-free region. Its omega-limit set is therefore nonempty, compact,
connected, and invariant. On that limit set the monotonicity identity forces
`w=0`; invariance of the reduced shape equation then forces the tangential
gradient of `V` on the inertia sphere to vanish:

```text
grad_S V(y_*) = 0.
```

This is exactly the central-configuration equation in reduced shape space.
Thus every reduced shape accumulation point is a collision-free central
configuration.

For three bodies, the collision-free central configurations in the
rotation/reflection quotient are finite: the Lagrange equilateral class and the
Euler collinear classes. The omega-limit set is connected, so a connected
subset of this finite set is a single point. Therefore the normalized mutual
distances converge:

```text
( |y_2-y_1|, |y_3-y_1|, |y_3-y_2| ) -> d_*,
```

where `d_*` is one of the finitely many central quotient shapes. Consequently:

```text
U(q(t)) sqrt(I(t))
 = sum_{i<j} m_i m_j / |y_i(t)-y_j(t)|
 -> Gamma_*.
```

This proves the normalized-potential limit and the central quotient-shape
selection from shape compactness alone. It still does not give the full
regularized branch entry: one must either fix a rotation gauge and prove the
actual oriented normalized shape converges, or prove directly that the finite
regularized jets used by the selector exist. The theorem below covers the
stronger oriented shape-limit subcase.

## Binary-Degenerate Collapse Reduces To A Jacobi Kepler Contradiction

The shape-compact hypothesis is exactly what fails if the normalized shape
tends to a binary-collision stratum. In three-body Jacobi coordinates this
alternative has a sharper form. Suppose bodies `1` and `2` form the tight pair:

```text
r = q_2-q_1,
rho = q_3 - (m_1q_1+m_2q_2)/(m_1+m_2),
s = |rho|,
m_12 = m_1+m_2,
M = m_1+m_2+m_3.
```

The binary-degenerate total-collapse hypothesis is:

```text
|r| -> 0,
s -> 0,
|r|/s -> 0.
```

The exact Jacobi equations have the asymptotic form:

```text
r''   = -m_12 r/|r|^3 + F_r,
rho'' = -M rho/s^3     + F_rho,
```

where the perturbations satisfy:

```text
|F_r|   <= C |r|/s^3,
|F_rho| <= C |r|/s^3.
```

Indeed, the third-body contribution to the pair equation is the difference of
`x/|x|^3` at `rho + O(r)` and `rho + O(r)`, hence is bounded by the derivative
size `O(s^-3)` times `|r|`. The outer equation is the interaction of body `3`
with the tight binary cluster; replacing the two close bodies by their cluster
center gives `-M rho/s^3`, and the cluster error is again `O(|r|/s^3)`.

Consequently the perturbations are small relative to the two Kepler singular
forces:

```text
|F_r|   / (m_12/|r|^2) <= C (|r|/s)^3 -> 0,
|F_rho| / (M/s^2)     <= C |r|/s       -> 0.
```

Thus a binary-degenerate total collapse would put both Jacobi coordinates in
finite-time perturbed-Kepler collision regimes, with Kepler parameters `m_12`
and `M`. If the standard perturbed-Kepler collision asymptotic holds for those
two coordinates, then, with `u=T-t`,

```text
|r|  ~ (9m_12/2)^(1/3) u^(2/3),
s    ~ (9M/2)^(1/3)    u^(2/3).
```

Dividing gives:

```text
|r|/s -> (m_12/M)^(1/3) > 0,
```

contradicting `|r|/s -> 0`.

Therefore the binary-degenerate entry problem is no longer an unconstrained
shape-compactness gap. It is reduced to the perturbed-Kepler asymptotic lemma
for the two Jacobi coordinates under the displayed lower-order perturbation
bounds. The next lemma supplies that global branch input.

## Perturbed-Kepler Collision Blow-Up Lemma

Here is the precise asymptotic lemma needed by the preceding reduction. Let
`x(t)` be a nonzero vector coordinate on `t<T`, set:

```text
R=|x|,
h=(1/2)|x'|^2 - mu/R,        mu>0,
L=x wedge x',
u=T-t,
```

and suppose:

```text
x'' = -mu x/R^3 + F(t),
R(t) -> 0,
R(t)^2 |F(t)| -> 0.
```

Then:

```text
R(t) h(t) -> 0,
|L(t)|^2/R(t) -> 0,
R(t) ~ (9mu/2)^(1/3) u^(2/3).
```

The point is that the coordinate-energy defect is not an additional
hypothesis. It is forced by the Kepler blow-up.

Write `e=x/R` and introduce the McGehee variables:

```text
nu = R^(1/2) R',
w  = R^(3/2) e',
ds/dt = R^(-3/2).
```

The velocity and scaled angular momentum are:

```text
|x'|^2 = R^(-1)(nu^2+|w|^2),
|L|^2/R = |w|^2,
R h = (1/2)(nu^2+|w|^2)-mu.
```

Resolving the equation into radial and spherical parts gives, with
`epsilon(s)=R(t(s))^2 |F(t(s))|`:

```text
R_s  = R nu,
nu_s = (1/2)nu^2 + |w|^2 - mu + O(epsilon),
D_s w = -(1/2)nu w + O(epsilon),
```

where `D_s` is the covariant derivative on the unit sphere. Since
`epsilon(s)->0`, this is an asymptotically autonomous perturbation of the
Kepler collision blow-up. A terminal collision has `R(s)->0`; after the
standard terminal radial comparison the scaled variables stay bounded and
`s->+infinity`. Therefore every omega-limit point lies on the limiting
collision manifold `R=0`.

On that limiting manifold the equations are:

```text
nu_s = (1/2)nu^2 + |w|^2 - mu,
D_s w = -(1/2)nu w.
```

The omega-limit cannot have positive `nu`, because `R_s=Rnu` would move away
from collision, and it cannot lie in the `nu=0`, `|w|^2=mu` circular set,
because then `int nu ds` would be finite rather than forcing `R(s)->0`.
Thus the invariant omega-limit has `nu<0`. But on such a set:

```text
(|w|^2)_s = -nu |w|^2,
```

so invariance forces `w=0`. The remaining scalar equation gives:

```text
0 = (1/2)nu^2 - mu,
```

and the incoming sign selects:

```text
nu -> -sqrt(2mu),
w -> 0.
```

Consequently:

```text
R h = (1/2)(nu^2+|w|^2)-mu -> 0,
|L|^2/R = |w|^2 -> 0.
```

The scale now follows from the one-coordinate moment `J=R^2`. Indeed:

```text
J''
 = 2|x'|^2 + 2x.x''
 = 2|x'|^2 - 2mu/R + 2x.F
 = 2mu/R + o(1/R)
 = 2mu J^(-1/2) + o(J^(-1/2)).
```

As in the total-collapse inertia argument, `J(t)->0` and
`J'=2R R'=2R^(1/2)nu` is eventually negative. Put `p=-J'>0`; then along the
terminal monotone tail:

```text
d(p^2)/dJ = 2J''.
```

Using `p->0` at `J=0` and integrating back from the collision gives:

```text
p^2 = 8mu J^(1/2) + o(J^(1/2)).
```

Therefore:

```text
-J' = sqrt(8mu) J^(1/4)(1+o(1)).
```

Integrating from `t` to `T` gives:

```text
J(t) ~ ((9/2)mu)^(2/3) (T-t)^(4/3),
```

and taking square roots proves the displayed `R` asymptotic.

Applied to the Jacobi binary-degenerate reduction, the perturbation bounds
already give:

```text
|r|^2|F_r| -> 0,
|rho|^2|F_rho| -> 0.
```

The blow-up lemma supplies the formerly missing coordinate-energy defects:

```text
|r| |h_r| -> 0,
|rho| |h_rho| -> 0.
```

Thus the scale lemma applies to both Jacobi coordinates. With `u=T-t`:

```text
|r|   ~ (9m_12/2)^(1/3) u^(2/3),
|rho| ~ (9M/2)^(1/3)    u^(2/3),
```

so:

```text
|r|/|rho| -> (m_12/M)^(1/3) > 0.
```

This contradicts the binary-degenerate hypothesis `|r|/|rho|->0`. Therefore
finite-energy three-body total collapse cannot approach a binary-collision
shape stratum. The preceding shape-compact quotient-convergence theorem then
applies to every total-collapse branch.

## Finite Reduced Shape Length Gives Oriented Shape Convergence

The remaining rotation gauge is controlled once the reduced shape path has
finite length. This is the precise bridge from quotient-shape convergence to
the oriented normalized-shape convergence used in the cubic-time entry theorem.

Work near one collision-free quotient-shape limit and choose a smooth local
slice `z(s)` for translation, scale, rotation, and reflection, normalized by:

```text
<z,z>_m = 1,
<z,Jz>_m = 0,
```

where `J` is the planar quarter-turn. Write the actual normalized shape as:

```text
y(s) = R(theta(s)) z(s).
```

Assume:

```text
z(s) -> z_*,
int_S^infinity |z_s(s)|_m ds < infinity,
L = 0.
```

The normalized angular momentum is:

```text
<Jy,y_s>_m
 = theta_s <z,z>_m + <Jz,z_s>_m.
```

Since total angular momentum is zero and `r>0` on the punctured branch, this
quantity vanishes. Hence:

```text
theta_s = -<Jz,z_s>_m.
```

By Cauchy's inequality and `<z,z>_m=1`:

```text
|theta_s| <= |z_s|_m.
```

The finite reduced length therefore gives:

```text
int_S^infinity |theta_s| ds < infinity,
```

so `theta(s)` has a finite limit `theta_*`. Thus:

```text
y(s) -> R(theta_*) z_*.
```

This upgrades quotient-shape convergence to an oriented normalized-shape limit
whenever the reduced shape path has finite length.

For the shape-compact total-collision setting above, this finite-length
hypothesis is the natural McGehee-Lojasiewicz condition. The next subsection
spells out the analytic estimate that was previously only invoked as the
standard gradient-like convergence argument.

## Lojasiewicz Damping Gives Finite Reduced Shape Length

Work in a collision-free local reduced-shape slice around the quotient central
configuration selected by the preceding omega-limit argument. Let `z_*` be the
isolated central target, let `V_* = V(z_*)`, and define the nonnegative local
potential gap on the incoming branch by:

```text
Phi(s) = V_* - V(z(s)) >= 0,
Phi(s) -> 0.
```

The sign is chosen so that the incoming damped McGehee branch moves downhill
for `Phi`; reversing the sign covers the opposite local convention. The
analytic Lojasiewicz inequality at the isolated critical point gives constants
`kappa > 0`, `theta in [1/2,1)`, and a neighborhood of `z_*` such that:

```text
|grad V(z)| >= kappa Phi(z)^theta.
```

The positive collision damping in the local reduced McGehee equation supplies
the gradient-like angle inequality on the same tail:

```text
-dPhi/ds >= a |grad V(z(s))| |z_s(s)|,
a > 0.
```

This inequality is the analytic content of the damped shape equation: after
translation, scale, and rotation reduction, the only neutral directions have
been quotiented out, and the incoming collision coefficient `nu` is bounded
away from zero with the damping sign. The integrable `rH` term from the energy
identity can be absorbed by starting the tail later, since `r(s)` decays
exponentially along total collision.

Combining the two displayed inequalities gives:

```text
-d/ds Phi(s)^(1-theta)
 = (1-theta) Phi(s)^(-theta) (-Phi'(s))
 >= (1-theta) a kappa |z_s(s)|.
```

Therefore every later tail has the explicit reduced-length bound:

```text
int_S^infinity |z_s(s)| ds
 <= Phi(S)^(1-theta) / ((1-theta) a kappa).
```

In particular the reduced shape path has finite length and is Cauchy in the
local slice. The rotation-gauge estimate from the previous subsection then
gives:

```text
int_S^infinity |theta_s(s)| ds
 <= int_S^infinity |z_s(s)|_m ds < infinity,
```

so the oriented normalized shape has an actual limit, not merely a quotient
limit. Consequently, any shape-compact zero-angular total collapse whose
isolated central target satisfies this local Lojasiewicz-angle estimate enters
the collision-free oriented shape-limit bridge below and hence the `C^2`
cubic-time entry theorem.

For the three-body collision-free central targets used below, the reduced
hyperbolicity theorem two subsections later gives such a bypass: finite length
and oriented convergence follow from the spectrum after quotienting rotations.
For higher-body targets or unreduced degenerate critical manifolds, one would
still need either the displayed Lojasiewicz-angle estimate or a direct proof of
the required finite regularized selector jets.

## Hyperbolic Reduced McGehee Targets Give Finite Length

There is one important subcase where the finite-length hypothesis follows from
the local analytic dynamics without needing a pointwise gradient-angle
inequality. Keep the same collision-free reduced-shape slice and include the
shape velocity and radial variable in a first-order McGehee state:

```text
X = (xi, eta, r),
xi = z-z_*,
eta = z_s.
```

After translation, scale, rotation, and reflection reduction, the total
collision central target is an equilibrium:

```text
X_*=(0,0,0),
X_s = F(X),
F(0)=0,
```

with `F` analytic on the local collision-free chart. Suppose the quotient
linearization is hyperbolic:

```text
spec DF(0) cap {Re lambda = 0} = empty.
```

If an incoming zero-angular total-collision branch is shape-compact and
converges to this quotient target, then `X(s)->0`. Since the equilibrium is
hyperbolic, the stable-manifold theorem gives a local analytic stable manifold
and constants `C, alpha > 0` such that every branch in it satisfies:

```text
|X(s)| <= C exp(-alpha (s-S))
```

for all sufficiently large `s >= S`. In particular:

```text
int_S^infinity |z_s(s)| ds
 = int_S^infinity |eta(s)| ds
 <= (C/alpha).
```

More precisely, after increasing `S`:

```text
int_S^infinity |z_s(s)| ds
 <= (C/alpha) exp(-alpha S)
```

when the exponential estimate is measured from a fixed origin of McGehee time.
Thus every shape-compact incoming branch whose quotient limit is a hyperbolic
reduced McGehee central target has finite reduced shape length. The
zero-angular rotation-gauge estimate then upgrades quotient convergence to an
oriented normalized-shape limit, so this whole hyperbolic subcase enters the
collision-free `C^2` cubic-time bridge below.

This theorem is deliberately local. It does not decide central targets whose
reduced McGehee linearization has center directions, nor does it supply the
finite resonant or fractional regularized selector jets. Those cases must be
handled by the Lojasiewicz-angle route above, by a center-manifold/Fuchsian
analysis, or by a direct finite-jet theorem.

## Three-Body Collision-Free Central Targets Are Reduced-Hyperbolic

For the planar three-body problem with positive masses, the preceding
hyperbolic bridge is not merely a special case once the binary-degenerate
shape stratum has been excluded. Every collision-free central quotient target
is reduced-hyperbolic.

The link between the central-configuration spectrum and the reduced McGehee
linear rates is already visible in the regular-singular shape equation. If
`DA(C)V=mu V`, then a perturbation `S=C+tau^k V+...` has indicial equation:

```text
(k+2)(k-1)=9mu.
```

In McGehee time, `tau` is an exponential of `s` along the incoming parabolic
scale, so a center direction in the reduced McGehee linearization would require
an indicial root with zero real part. Since the three-body spectra below are
real, this can happen only at the root `k=0`, equivalently:

```text
mu = -2/9.
```

But `mu=-2/9` is exactly the infinitesimal rotation direction. It is removed
when passing to the reduced quotient shape. Translations are removed by
mass-centering, and the scale direction `mu=4/9` has roots `k=2,-3`, hence is
hyperbolic rather than central.

For an arbitrary-mass Lagrange equilateral central configuration, with:

```text
beta=(m_1m_2+m_1m_3+m_2m_3)/M^2,        0 < beta <= 1/3,
```

the scaled spectrum is:

```text
{0,0,4/9,-2/9,
 1/9+(1/3)sqrt(1-3beta),
 1/9-(1/3)sqrt(1-3beta)}.
```

The two zeros are translations. The eigenvalue `4/9` is scale, and `-2/9` is
rotation. The two genuine shape eigenvalues satisfy:

```text
1/9+(1/3)sqrt(1-3beta) > 0,
1/9-(1/3)sqrt(1-3beta) > -2/9,
```

because all masses are positive, so `beta>0`. Thus no reduced equilateral
shape mode has `mu=-2/9`.

For an ordered Euler central configuration, the spectrum reduction gives:

```text
{0,0,4/9,-2/9, sigma, -sigma/2},
```

where the horizontal shape eigenvalue satisfies:

```text
sigma > 4/9.
```

Again `0,0` are translations, `4/9` is scale, and `-2/9` is rotation. The
horizontal shape mode has `sigma>0`, while the transverse shape mode has
`-sigma/2 < -2/9`; neither is a reduced center direction.

Therefore, after translation, scale, rotation, and reflection quotienting,
every positive-mass three-body collision-free central target is hyperbolic.
Combined with the earlier binary-degenerate exclusion and quotient-shape
convergence theorem, every finite-energy zero-angular total-collapse branch
has finite reduced shape length and an oriented normalized-shape limit.

This closes the orientation part of the zero-angular entry problem for three
bodies. It does not by itself prove the finite selector expansion. Hyperbolic
stable eigenrates still have to be converted into the appropriate lifted
Fuchsian or Fuchsian-log coordinates, and resonant or fractional rows still
carry the branch data described below.

## Poincare-Dulac Stable Normal Form Gives Selector Completeness

The remaining hyperbolic entry issue is asymptotic completeness: does an
arbitrary incoming branch that converges to one of the reduced-hyperbolic
central targets actually have the Fuchsian or Fuchsian-log selectors used by
the local continuation theorem? In the three-body collision setting the answer
is yes, locally at such a target.

Work on the analytic stable manifold of the reduced McGehee equilibrium after
translation, scale, rotation, and reflection have been quotiented out. Let the
decaying stable rates be:

```text
alpha_1,...,alpha_m > 0,
```

and include the radial collision coordinate with rate `beta>0`. The stable
eigenvalues lie in a Poincare domain: their convex hull is contained in the
open left half-plane for McGehee time, equivalently the positive rates above
are bounded away from zero. The analytic Poincare-Dulac theorem therefore
gives convergent local coordinates in which the stable equations have a finite
polynomial resonant normal form:

```text
(y_j)_s = -alpha_j y_j + P_j(y_1,...,y_{j-1}),
```

after ordering the coordinates by increasing rate. Only finitely many
resonant monomials can occur, because a resonance for row `j` has:

```text
alpha_j = n_1 alpha_1 + ... + n_m alpha_m,        |n| >= 2,
```

and the right side is a sum of positive terms. Moreover any monomial in row
`j` uses only lower-rate variables: if some factor had rate at least
`alpha_j`, the sum would exceed `alpha_j`.

This triangular finite normal form solves inductively. Suppose every
lower-rate variable already has the form:

```text
y_i(s)=e^(-alpha_i s) P_i(s),
```

with `P_i` a finite polynomial whose constant term is the selector for that
row. Each resonant forcing monomial in row `j` is then
`e^(-alpha_j s)` times a finite polynomial in `s`. Variation of constants
gives:

```text
y_j(s)=e^(-alpha_j s) P_j(s),
```

where the positive powers of `s` are forced by lower selectors and the
constant term is the new free selector. This proves by induction that every
incoming branch on the local stable manifold has a finite list of
log-polynomial selectors in the Poincare-Dulac coordinates.

Convert back to cubic collision time. Along the incoming branch:

```text
s = -(2/beta) log(tau) + O(1),
e^(-alpha_j s) = tau^(k_j) times a nonzero analytic unit,
k_j = 2 alpha_j / beta.
```

The analytic coordinate map from Poincare-Dulac variables to reduced shape
turns the finite polynomials in `s` into convergent Fuchsian-log series:

```text
tau^omega (log tau)^ell,
```

with finitely many base weights and finite log degree at each weight. The
forced positive-log coefficients are determined recursively from lower
selectors, and the constant log coefficients are exactly the selector limits
already used in the Fuchsian-log continuation theorem.

This finite triangular stable-normal-form step is now executable in
`three_body_symmetry/fuchsian.py` through
`construct_stable_log_selector_chain(...)`. Given positive stable rates,
radial decay `beta`, source amplitudes, selector constants, and resonant
monomial couplings, the constructor verifies each resonance
`alpha_j=<n,alpha>`, rejects nontriangular dependencies, multiplies the
already-constructed lower-row polynomials in McGehee time `s`, integrates the
resulting forcing polynomial, and converts `s=-(2/beta)log(tau)` into the
Fuchsian-log coefficients. The returned chain recovers the selector constants
by subtracting the forced positive-log terms from `y_j/tau^(2alpha_j/beta)`.
This makes the finite Poincare-Dulac selector-completeness mechanism
constructive; it still assumes the reduced hyperbolic entry and the analytic
coordinate reduction that produce the finite triangular normal form.

The scalar selector chain can now be pushed through the physical projection
once the reduced mode shapes are supplied. The constructor
`construct_finite_fuchsian_log_branch_from_stable_chain(...)` takes mass
positive `m_i`, a central shape `C`, a scale coefficient `alpha`, a certified
stable chain, and mode shapes `E_j`. It requires

```text
<E_j,C>_m=0,        <E_j,E_j>_m=1,        omega_j=2alpha_j/beta>1.
```

Then the finite Fuchsian-log shape

```text
S(tau)=C+alpha tau^2C+sum_j tau^(omega_j)P_j(log tau)E_j
```

projects by `q=tau^2S`, `t=tau^3` to a punctured physical branch whose
energy has the finite limit `(10/9)alpha<C,C>_m`. Orthogonality removes the
only possible `tau^(omega_j-2)` scale cross-term, while `omega_j>1` makes the
remaining non-scale kinetic and potential terms finite and forces the
centered angular momentum to tend to zero. The selector constants are
recovered by mass projection onto each `E_j` after subtracting the other
finite rows. This proves the local stable-chain-to-branch projection theorem;
it does not prove that an arbitrary incoming zero-angular collision branch
has already entered this finite normal form.

For the positive-mass three-body central targets, the only nondecaying
indicial root is the rotation root removed by quotienting. The unstable roots
are absent because the branch converges to the selected central target. Hence
every finite-energy zero-angular total-collision branch, after the preceding
binary-degenerate exclusion and quotient-convergence theorem, enters the
enlarged Fuchsian/Fuchsian-log selector class on its incoming side.

This is the asymptotic-completeness step for local zero-angular entry. The
only remaining local choice is branch semantics, not hidden analytic data. The
canonical identity-selector rule below assigns each outgoing selector constant
to be the corresponding incoming one while preserving the common energy scale
coefficient.

## Hyperbolic Stable Eigenrates Become Fuchsian Powers

The preceding hyperbolic bridge gives finite reduced length. Under the usual
nonresonant analytic-linearization hypothesis it gives more: the stable
eigenrates themselves become the fractional powers used by the lifted
Fuchsian selector.

Assume the stable part of the reduced McGehee vector field is analytically
linearizable near the selected central target. Thus, on the stable manifold,
there are analytic coordinates:

```text
rho, u_1, ..., u_m
```

with `rho` a radial collision coordinate and:

```text
rho_s = -beta rho,
(u_j)_s = -alpha_j u_j,
beta > 0, alpha_j > 0.
```

The local reduced shape is analytic in these coordinates:

```text
z-z_* = Psi(rho,u),
Psi(0,0)=0,
Psi(rho,u)=rho Z_0 + sum_j u_j D_j + higher monomials.
```

Along an incoming branch:

```text
rho(s)=rho_0 exp(-beta s),
u_j(s)=a_j exp(-alpha_j s).
```

The parabolic scale theorem gives `r = gamma tau^2(1+o(1))` in cubic collision
time, and `rho` differs from `r` by a nonzero analytic factor. Hence, after
rescaling the constants:

```text
rho = tau^2(1+o(1)),
u_j = a_j tau^(k_j)(1+o(1)),
k_j = 2 alpha_j / beta.
```

Because `Psi` is analytic, every monomial `rho^b u^a` becomes a monomial in
the lifted collision variables:

```text
zeta = tau^2,
x_j = tau^(k_j).
```

Consequently the reduced shape has a convergent generalized power expansion:

```text
z(tau)=z_* + zeta Z_0 + sum_j a_j x_j D_j
       + sum_{2b+sum_j a_j k_j > min k_j} Z_{b,a} zeta^b x^a.
```

For a simple selected eigenmode `D_j`, after projecting away lower-weight
modes, the finite Fuchsian selector amplitude is the limit:

```text
a_j D_j = lim_{tau->0+}
  Pi_j (z(tau)-z_* - lower-weight terms) / tau^(k_j).
```

Thus the nonresonant hyperbolic McGehee subcase supplies the missing finite
regularized selector asymptotics in the same lifted coordinates already used
below. If the powers `{2,k_1,...,k_m}` also satisfy the multi-indicial
nonresonance conditions of the Fuchsian recurrence, the constructed lifted
branch is the unique local Newtonian projection with those selected amplitudes.

This is a bridge theorem, not a claim about every central target. If analytic
linearization fails because of stable resonances, the same idea must be
replaced by the resonant Poincare-Dulac normal form, where polynomial factors
or resonant mixed monomials become additional selector data. If the reduced
McGehee equilibrium has center spectrum, the hyperbolic argument does not
apply.

## Resonant Stable Terms Become Fuchsian Log Terms

The first resonant stable normal form can still be lifted explicitly. Keep the
radial equation:

```text
rho_s = -beta rho,
beta > 0,
```

and suppose a stable mode `u` forces another stable mode `v` at exactly the
resonant rate:

```text
u_s = -alpha u,
v_s = -p alpha v + c u^p,
p >= 2.
```

Solving on an incoming branch gives:

```text
u(s)=a exp(-alpha s),
v(s)=exp(-p alpha s)(b + c a^p s).
```

The cubic collision scale again converts the radial variable to:

```text
rho = tau^2,       s = -(2/beta) log(tau) + O(1).
```

After absorbing the harmless additive constant in `s` into `b`, the resonant
mode has the lifted form:

```text
u = a tau^k,
v = tau^(p k) ( b - (2c/beta) a^p log(tau) ),
k = 2 alpha / beta.
```

Thus the resonant selector is not the raw quotient `v/tau^(pk)`, which usually
diverges logarithmically. The finite datum is the log-subtracted quotient:

```text
b = lim_{tau->0+}
    ( v/tau^(pk) + (2c/beta) a^p log(tau) ).
```

In a general resonant Poincare-Dulac stable normal form, each resonant monomial
whose weight matches a target eigenrate contributes a polynomial in `s`; after
`s=-(2/beta)log(tau)+O(1)`, this is a polynomial in `log(tau)` multiplying the
same Fuchsian power. Therefore the lift/construct/project/verify class must
allow monomials:

```text
tau^omega (log tau)^ell
```

whenever stable resonances occur. Once those logarithmic coefficients forced by
lower amplitudes are subtracted, the remaining constant log coefficient is the
finite resonant selector amplitude. This gives the concrete replacement for
analytic linearization in the simplest resonant hyperbolic subcase. Center
spectrum still requires a separate center-manifold analysis.

## Finite Resonant Stable Chains Give Log-Polynomial Selectors

The same calculation extends from one resonant forcing term to any finite
triangular resonant Poincare-Dulac stable block. Fix a stable weight
`omega > 0`, and suppose that, after lower-weight stable coordinates have
already been solved, a resonant block has the form:

```text
(y_0)_s = -omega y_0 + F_0,
(y_1)_s = -omega y_1 + F_1(y_0, lower),
...
(y_N)_s = -omega y_N + F_N(y_0,...,y_{N-1}, lower),
```

where every resonant forcing `F_j` is `exp(-omega s)` times a polynomial in
`s` of finite degree. Then each row solves by one integration:

```text
y_j(s) = exp(-omega s) P_j(s),
```

with `P_j` a finite polynomial. If the largest degree in the forcing for
`y_j` is `d`, then the new polynomial has degree at most `d+1`; its constant
term is the free stable selector for that row, and every positive power of
`s` is forced by lower rows and lower-weight amplitudes.

Using the collision scale:

```text
s = -(2/beta) log(tau) + O(1),
e^(-omega s) = tau^(2 omega/beta),
```

each resonant row becomes:

```text
y_j(tau) = tau^k Q_j(log tau),
k = 2 omega / beta,
```

where `Q_j` is a finite polynomial. The coefficient of the highest log power
is forced first, and descending log degree gives a triangular selector
recovery. After subtracting all positive log powers forced by the lower rows,
the constant coefficient is finite:

```text
b_j = lim_{tau->0+}
  ( y_j/tau^k - sum_{ell=1}^{deg Q_j} q_{j,ell} (log tau)^ell ).
```

Thus stable resonances do not destroy the lifted selector. They enlarge it
from Fuchsian powers to Fuchsian-log finite polynomials, with the free data
still a finite list of constant coefficients after the forced logarithmic
terms have been removed. This is exactly parallel to the log-subtracted escape
endpoint algebra: the lift is richer, but projection and verification remain
algebraic on the punctured physical branch.

## Fuchsian-Log Shape Rows Are Triangular At Resonance

The log-polynomial selectors above are compatible with the regularized shape
equation itself. Linearize the shape equation at a collision-free central
configuration `C`:

```text
tau^2S''+2tau S'-2S=9A(S),
A(C)=-(2/9)C.
```

At a fixed lifted weight `omega > 0`, write a new row as:

```text
H(tau)=tau^omega sum_{ell=0}^d H_ell (log tau)^ell.
```

The scalar differential part acts on one monomial by:

```text
(tau^2 d^2/dtau^2 + 2tau d/dtau - 2)
  [tau^omega (log tau)^ell]
= tau^omega[
    (omega+2)(omega-1)(log tau)^ell
    + ell(2omega+1)(log tau)^(ell-1)
    + ell(ell-1)(log tau)^(ell-2)
  ].
```

Therefore the coefficient of `(log tau)^ell` in the linearized row equation is:

```text
M_omega H_ell
 +(ell+1)(2omega+1)H_{ell+1}
 +(ell+2)(ell+1)H_{ell+2}
 = F_ell,

M_omega=((omega+2)(omega-1)I-9DA(C)).
```

If `M_omega` is invertible, this is the ordinary nonresonant Fuchsian row,
solved by descending log degree. If `M_omega` has a resonant kernel, the same
formula is still triangular after splitting into range and kernel. The range
component of `H_ell` is solved by `M_omega` as usual. The kernel component of
the equation at log degree `ell` is solved by choosing the kernel component of
`H_{ell+1}`, because `omega > 0` implies:

```text
(ell+1)(2omega+1) != 0.
```

Thus a resonant forcing at log degree `ell` raises the row by one log power.
After all positive log powers forced by earlier coefficients have been fixed,
the remaining kernel component of `H_0` is the finite selector amplitude for
that resonant row. Substituting the solved row into
`q(tau)=tau^2(C+...+H(tau))`, `t=tau^3`, the same homogeneity calculation gives
Newton's equation on every punctured side where the Fuchsian-log series
converges.

This proves that the log terms generated by a finite resonant stable normal
form are not merely asymptotic bookkeeping: they are the exact triangular
linear algebra needed by the regular-singular shape equation at resonant
weights.

## Fuchsian-Log Projection And Zero Angular Momentum

The projection calculation used for analytic and Fuchsian branches does not
care whether the lifted shape has powers, logarithms, or both. Let `S(tau)` be
any convergent Fuchsian-log shape branch on one punctured side of `tau=0`,
with collision-free limit `C`, and suppose:

```text
tau^2S'' + 2tau S' - 2S = 9A(S).
```

Project by:

```text
q(tau)=tau^2S(tau),        t=tau^3.
```

For `tau != 0`:

```text
dq/dt = (2/(3tau))S + (1/3)S',
```

and a direct chain-rule calculation gives:

```text
d^2q/dt^2
 = (tau^2S'' + 2tau S' - 2S)/(9tau^4).
```

By homogeneity of the Newtonian acceleration:

```text
A(q)=A(tau^2S)=tau^(-4)A(S).
```

Thus the projected residual is exactly:

```text
q''-A(q)
 = [tau^2S'' + 2tau S' - 2S - 9A(S)]/(9tau^4).
```

Any convergent Fuchsian-log solution of the regularized shape equation
therefore projects to a Newtonian solution on the punctured side.

The same calculation controls angular momentum. Since `q=tau^2S`:

```text
L(tau)=sum_i m_i q_i wedge dq_i/dt
      = (tau^2/3) sum_i m_i S_i wedge S_i'.
```

Every positive-weight Fuchsian-log correction has
`S'=O(tau^(k-1)(log tau)^d)` for some `k>0`, while `S->C`. Hence
`tau^2 S wedge S' -> 0`. Newtonian pair forces conserve angular momentum on
each punctured side, so the constant angular momentum must be zero. This
establishes the construct/project/verify step for convergent local
Fuchsian-log collision branches.

## No Hidden Fuchsian-Log Row Data

The resonant log row also has no hidden coefficients once its selector has
been fixed. Consider one weight `omega` and suppose two Fuchsian-log rows have
the same lower-order forcing `F_ell`, the same forced positive-log
coefficients, and the same kernel projection of `H_0`. Subtracting their row
equations gives the homogeneous triangular system:

```text
M_omega Delta H_ell
 +(ell+1)(2omega+1)Delta H_(ell+1)
 +(ell+2)(ell+1)Delta H_(ell+2)
 = 0.
```

Start at the largest positive log degree. Those coefficients are already fixed
by the lower forcing, so their difference is zero. Descending in `ell`, the
equation reduces to:

```text
M_omega Delta H_ell = 0
```

up to already-vanishing higher-log terms. The range component of
`Delta H_ell` is zero because `M_omega` is invertible on the chosen range
complement. For `ell>0`, the kernel component was also already fixed as part
of the forced positive-log polynomial. At `ell=0`, the remaining kernel
component is exactly the selector, and it is fixed by hypothesis. Therefore
all differences vanish.

Inducting over finitely many resonant rows, a Fuchsian-log normal-form chart
is locally complete once the central shape, energy/scale coordinate, ordinary
Fuchsian amplitudes, and the finite list of log-row constant selectors are
specified. Any different coefficient at a non-selector slot creates a nonzero
coefficient residual in the regularized shape equation.

## Fuchsian-Log Selectors Give A Local Continuation Rule

The preceding row solve is enough to state a genuine local zero-angular
continuation theorem in the hyperbolic/Fuchsian-log subcase. Suppose an
incoming total-collision branch has a convergent finite Fuchsian-log shape
expansion:

```text
q(tau)=tau^2 S_-(|tau|),       t=tau^3,       tau<0,
S_-(sigma)=C + alpha sigma^2 C + R_-(sigma),
```

where `C` is a collision-free central configuration scaled by
`A(C)=-(2/9)C`, `R_-` is a finite Fuchsian-log normal-form expansion in
positive shape weights, and every non-scale leading row is mass-orthogonal to
`C`. Assume the finite triangular recurrence described above solves the
regularized shape equation:

```text
sigma^2 S_-'' + 2 sigma S_-' - 2 S_- = 9A(S_-).
```

Recover the incoming selector list by subtracting the forced positive-log
terms row by row and taking the constant kernel projections. Let a branch rule
`R` assign outgoing selector constants from those incoming constants, while
keeping the same `C` and the same scale coefficient `alpha`.

Then the same finite triangular recurrence constructs a unique outgoing
Fuchsian-log shape branch:

```text
S_+(sigma)=C + alpha sigma^2 C + R_+(sigma),
```

with the prescribed outgoing selectors and the same forced nonselector rows.
The no-hidden row theorem says no further local row data can be chosen without
creating a coefficient residual. Defining:

```text
q(tau)=tau^2 S_+(tau),       t=tau^3,       tau>0,
```

projects to a Newtonian solution on the outgoing side. On the incoming side,
view `S_-(|tau|)` as a function of negative `tau`; then
`tau^2 S''+2tau S'-2S=9A(S)` is unchanged, so the same projection identity
verifies Newton's equation there too.

The finite energy is also matched by the common scale coefficient. For either
side:

```text
H(tau)
 = tau^(-2) [
     (1/18)<2S+tau S_tau, 2S+tau S_tau>_m - U(S)
   ].
```

The central-configuration identity cancels the constant term. The scale row
`alpha tau^2 C` contributes:

```text
(10/9) alpha <C,C>_m.
```

A non-scale Fuchsian-log row of weight `k>1` has the form
`tau^k P(log tau)D`, with `<C,D>_m=0`. Its linear energy variation vanishes,
and every remaining contribution is
`O(tau^(2k-2) |log tau|^d)` after multiplying by `tau^(-2)`, hence tends to
zero. Thus changing Fuchsian-log selector constants changes the outgoing
regularized curve but not the limiting finite energy. The angular-momentum
calculation above gives zero angular momentum on both punctured sides.

Consequently, in this subcase, the zero-angular total-collision continuation
problem has the expected form: the incoming branch supplies `C`, `alpha`, and
finite Fuchsian-log selector constants; an explicit selector rule supplies the
outgoing constants; the lifted recurrence constructs the branch; projection
verifies Newton's equation, zero angular momentum, and finite-energy matching.

## Canonical Identity-Selector Continuation

The preceding pieces now give a concrete zero-angular total-collision
continuation convention for the three-body problem. Use the identity rule on
the lifted Fuchsian-log selector constants.

Let a finite-energy, zero-angular, center-of-mass three-body branch reach total
collision at `t=T`. The binary-degenerate Jacobi argument excludes approach to
a binary shape stratum, so the normalized shape is eventually collision-free.
The quotient McGehee argument gives convergence to one Lagrange or ordered
Euler central quotient target. The reduced-hyperbolicity theorem upgrades this
to finite reduced length and an oriented normalized-shape limit. The
Poincare-Dulac stable normal-form theorem then gives finite incoming
Fuchsian-log selector data:

```text
C, alpha, b_1, ..., b_N,
```

where `C` is the oriented central configuration scaled by `A(C)=-(2/9)C`,
`alpha` is the homothetic energy/scale coefficient, and the `b_j` are the
ordinary Fuchsian amplitudes and constant log-row selectors after all forced
positive-log terms have been subtracted.

The identity-selector convention sets the outgoing selector list equal to the
incoming one:

```text
C_+ = C_-,
alpha_+ = alpha_-,
b_{j,+} = b_{j,-}        for all j.
```

The triangular Fuchsian-log recurrence constructs a unique outgoing lifted
shape branch with exactly those data:

```text
S_+(sigma)=C+alpha sigma^2 C+R_+(sigma),
sigma>0.
```

Define the two-sided local continuation by:

```text
q(tau)=tau^2 S_-(|tau|),        tau<0,
q(0)=0,
q(tau)=tau^2 S_+(tau),          tau>0,
t=T+tau^3.
```

On each punctured side the regularized shape equation is:

```text
tau^2S''+2tau S'-2S=9A(S),
```

so the projection identity gives Newton's equation. The angular momentum on
each punctured side is conserved and has collision limit:

```text
L(tau)= (tau^2/3) sum_i m_i S_i wedge S_i' -> 0,
```

hence it is identically zero on both sides. The common scale coefficient gives
the same finite collision energy on both sides:

```text
H_+ = H_- = (10/9) alpha <C,C>_m
```

plus the already-accounted finite resonant corrections in the analytic
integer-power cases. In the Fuchsian-log rows, non-scale selector changes have
zero finite-energy contribution; the identity rule simply keeps those branch
coordinates continuous in the lifted selector chart.

Thus the zero-angular triple-collision continuation is no longer an undefined
branch choice in this construction. It is:

```text
lift     to reduced McGehee/Fuchsian-log selector coordinates,
construct the outgoing lifted branch with identical selectors,
project   by q=tau^2S(tau), t=T+tau^3,
verify    Newton's equation, zero angular momentum, and energy matching.
```

This is a convention, not a uniqueness theorem for the collapsed physical
point. Other outgoing selector rules can also preserve the usual collision
invariants, as shown by the nonuniqueness examples. The identity rule is the
canonical analytic continuation in the symmetry-rich lifted selector space.

## Finite Compact-Interval Atlas With Identity-Selector Total Collisions

The local identity-selector rule also composes with ordinary and binary charts
on a compact physical-time interval, provided the interval has only finitely
many isolated collision events and every total collision is assigned the
identity-selector convention above.

Let `[a,b]` be a compact interval. Assume:

1. Total-collision times are finite in number, and each is a finite-energy
   zero-angular total-collision branch in the collision-free central-target
   class covered above.
2. At each such time `T_k`, the identity-selector data
   `C_k, alpha_k, b_{k,1},...,b_{k,N_k}` are extracted from the incoming
   branch and used to construct the outgoing Fuchsian-log branch.
3. Any binary collisions away from total collision are separated-third-body
   binary events covered by the Levi-Civita binary chart.
4. The remaining complement is collision-free.

For each total collision, choose a small regularized interval:

```text
|tau| <= epsilon_k,
t = T_k + tau^3,
q = tau^2 S_k(tau),
```

small enough that no other collision event lies in its physical image. On the
punctured sides this chart solves Newton's equation by the projection identity,
has zero angular momentum, and has matching finite energy because the same
scale coefficient `alpha_k` is used on both sides.

For each separated binary collision, choose the corresponding small
Levi-Civita chart. Since the collision set on `[a,b]` is finite, all these
collision neighborhoods can be chosen disjoint. The complement is then a finite
union of compact collision-free intervals. The ordinary compact Taylor-cover
lemma covers each component by finitely many ordinary analytic charts.

At every noncollision boundary between neighboring charts, both projected
curves solve the analytic Newtonian ODE and have the same finite physical state
by construction. Analytic ODE uniqueness glues them on overlaps. At a total
collision boundary itself, the physical velocity may diverge and the ordinary
ODE uniqueness theorem is not applied; the Fuchsian-log lifted chart is the
continuation rule through the singular point. At a binary collision, the
Levi-Civita lifted chart plays the same role for the selected binary pair.

Thus the compact interval is represented by a finite atlas:

```text
ordinary Taylor charts
+ separated-binary Levi-Civita charts
+ identity-selector Fuchsian-log total-collision charts.
```

The verification budget is finite because the chart list is finite. Ordinary
charts contribute their usual Taylor coefficient residuals and Cauchy tails,
binary charts contribute regularized residuals plus projection checks away
from `z=0`, and total-collision charts contribute the Fuchsian-log residual:

```text
tau^2S''+2tau S'-2S-9A(S)
```

plus the projection identity away from `tau=0`. This proves finite
compact-interval composition once the local collision selector data are known.
It still does not prove an all-time recurrence for infinitely many collision
events or decide which global regime arbitrary initial data enter.

## Collision-Free Shape Limit Supplies The Normalized-Potential Limit

The normalized-potential limit in the scale theorem is automatic once the
normalized shape has a collision-free limit. Write:

```text
y_i(t) = q_i(t)/sqrt(I(t)),
sum_i m_i |y_i(t)|^2 = 1.
```

If:

```text
y_i(t) -> y_i^*,
min_{i<j}|y_i^*-y_j^*| > 0,
```

then:

```text
U(t) sqrt(I(t))
 = sum_{i<j} m_i m_j / |y_i(t)-y_j(t)|
 -> Gamma
 = sum_{i<j} m_i m_j / |y_i^*-y_j^*|.
```

The limit is finite and positive. Therefore the Lagrange-Jacobi scale theorem
above applies with this explicit `Gamma`, giving:

```text
I(t) ~ ((9/2)Gamma)^(2/3)(T-t)^(4/3),
q_i(t) = tau^2 C_i + o(tau^2),
C_i = ((9/2)Gamma)^(1/3)y_i^*.
```

So, in the collision-free shape-limit subcase, one no longer needs to assume a
separate normalized-potential limit. The remaining entry assumptions are shape
convergence itself and enough regularized branch information to match one of
the local normal-form series below.

## Collision-Free Shape Convergence Already Forces A Central Limit

In the same collision-free shape-limit subcase, the central-configuration
conclusion also does not require an a priori `C^2` regularized remainder.
Newton's equation supplies the needed second-derivative asymptotic.

Let `u=T-t>0` on an incoming total-collapse branch and write:

```text
r(t)=sqrt(I(t)),
y_i(t)=q_i(t)/r(t).
```

Assume finite energy, total collision, and collision-free shape convergence:

```text
r(t)->0,
y_i(t)->y_i^*,
min_{i<j}|y_i^*-y_j^*|>0.
```

The previous two lemmas give:

```text
Gamma = sum_{i<j} m_i m_j/|y_i^*-y_j^*|,
a = ((9/2)Gamma)^(1/3),
r(T-u) = a u^(2/3)(1+o(1)),
q_i(T-u) = a u^(2/3)y_i^* + o(u^(2/3)).
```

Because the pair distances of `y(t)` stay bounded away from zero near the
cluster limit, the acceleration has the homogeneous asymptotic:

```text
d^2q_i/dt^2 = A_i(q)
 = r(t)^(-2) A_i(y(t))
 = a^(-2) u^(-4/3) A_i(y^*) + o(u^(-4/3)).
```

Now put `p_i(u)=q_i(T-u)`. Since `p_i''(u)=q_i''(T-u)`, the preceding display is:

```text
p_i''(u) = D_i u^(-4/3) + o(u^(-4/3)),
D_i = a^(-2)A_i(y^*).
```

Finite energy gives `|p_i'(u)|=O(u^(-1/3))`, because
`K=H+U=O(r^(-1))=O(u^(-2/3))`. Integrating the last equation once therefore
gives:

```text
p_i'(u) = -3D_i u^(-1/3) + o(u^(-1/3)),
```

where any finite integration constant is lower order. Integrating from the
collision value `p_i(0)=0` gives:

```text
p_i(u) = -(9/2)D_i u^(2/3) + o(u^(2/3)).
```

But the scale lemma already gave `p_i(u)=a u^(2/3)y_i^*+o(u^(2/3))`. Comparing
the two leading coefficients:

```text
a y_i^* = -(9/2)a^(-2)A_i(y^*).
```

Hence:

```text
A_i(y^*) = -(2/9)a^3 y_i^* = -Gamma y_i^*.
```

Equivalently, for the regularized second-position coefficient
`C_i=a y_i^*`:

```text
A(C)=-(2/9)C.
```

So a finite-energy total-collapse branch whose normalized shape converges to a
collision-free limit automatically has a central limiting second shape. The
next two subsections sharpen the same entry bridge to first and second
regularized-time derivatives. A separate `C^2` assumption is not needed merely
to identify the central configuration.

## Collision-Free Shape Convergence Gives The C1 Cubic-Time Entry

The same argument also fixes the first regularized-time derivative. This is a
small but useful narrowing of the branch data: the collision-through branch
does not get a free first jet.

Keep the notation of the preceding section, with `u=T-t`, `p_i(u)=q_i(T-u)`,
`C_i=a y_i^*`, and:

```text
p_i''(u) = D_i u^(-4/3) + o(u^(-4/3)),
D_i = a^(-2)A_i(y^*) = -(2/9)C_i.
```

The once-integrated estimate from the central-limit proof becomes:

```text
p_i'(u) = -3D_i u^(-1/3) + o(u^(-1/3))
        = (2/3)C_i u^(-1/3) + o(u^(-1/3)).
```

Use incoming cubic collision time:

```text
tau = -u^(1/3),        t = T + tau^3.
```

Then `u=-tau^3` and:

```text
dq_i/dtau = dp_i/du * du/dtau
          = p_i'(u)(-3 tau^2)
          = 2 tau C_i + o(tau).
```

Together with `q_i(tau)=tau^2C_i+o(tau^2)`, this gives a `C^1` entry into
cubic collision time:

```text
q_i(0)=0,
dq_i/dtau(0)=0,
q_i(tau)/tau^2 -> C_i,
(dq_i/dtau)/tau -> 2C_i.
```

This still does not select the analytic normal-form branch. It rules out an
extra first-jet choice and shows that, under collision-free shape convergence,
any remaining branch freedom starts in higher regularized jets or resonant
normal-form parameters.

## Collision-Free Shape Convergence Gives The C2 Cubic-Time Entry

The previous two consequences actually give the `C^2` regularized asymptotic
needed by the central-shape entry theorem. No additional regularized
differentiability hypothesis is needed in the collision-free shape-limit
subcase.

For `t=T+tau^3`, the chain rule gives, on the punctured incoming branch:

```text
dq_i/dtau = 3 tau^2 dq_i/dt,
d^2q_i/dtau^2 = 6 tau dq_i/dt + 9 tau^4 d^2q_i/dt^2.
```

Equivalently:

```text
d^2q_i/dtau^2 = 2 (dq_i/dtau)/tau + 9 tau^4 A_i(q).
```

The `C^1` entry gives `(dq_i/dtau)/tau -> 2C_i`. The parabolic scale and
collision-free shape convergence give:

```text
q_i(tau)=tau^2C_i+o(tau^2),
tau^4 A_i(q(tau)) -> A_i(C).
```

The central-limit result has already shown `A(C)=-(2/9)C`. Therefore:

```text
d^2q_i/dtau^2 -> 4C_i + 9A_i(C) = 2C_i.
```

Thus, writing `R_i(tau)=q_i(tau)-tau^2C_i`, one has:

```text
R_i=o(tau^2),
R_i'=o(tau),
R_i''=o(1).
```

Collision-free shape convergence therefore supplies the full `C^2` cubic-time
entry used below. What remains open is not this regularity level; it is the
higher-order regularized branch or jet convergence needed to identify the
selected normal-form parameters, especially on resonant central-configuration
surfaces.

## Central Shape From A C2 Cubic-Time Asymptotic

The central-configuration conclusion does not require a full analytic germ.
It already follows from a `C^2` asymptotic in cubic collision time.

Let `t=T+tau^3` and suppose, on one punctured side of total collision:

```text
q_i(tau) = tau^2 C_i + R_i(tau),
R_i(tau) = o(tau^2),
R_i'(tau) = o(tau),
R_i''(tau) = o(1),
```

where the limiting shape `C` is collision-free. Then:

```text
dq_i/dt = q_i'(tau)/(3 tau^2),
d^2q_i/dt^2 = q_i''(tau)/(9 tau^4) - 2q_i'(tau)/(9 tau^5).
```

Substituting `q_i=tau^2C_i+R_i` gives:

```text
tau^4 d^2q_i/dt^2
 = -(2/9)C_i + (tau R_i'' - 2R_i')/(9 tau)
 -> -(2/9)C_i.
```

Since `C` is collision-free and Newtonian acceleration is homogeneous of
degree `-2`,

```text
tau^4 A_i(q(tau))
 = A_i(C + R/tau^2)
 -> A_i(C).
```

If the projected branch satisfies Newton's equation for `tau != 0`, the two
limits agree, so:

```text
A(C) = -(2/9)C.
```

Thus a physical total-collapse branch that has the parabolic scale plus this
`C^2` regularized remainder already enters the central-configuration part of
the local normal form. This remains useful as an independent entry criterion,
but under collision-free shape convergence the preceding double-integration
argument already identifies the central limiting shape without assuming this
`C^2` remainder. The remaining global work is sharper: prove collision-free
shape convergence when it should hold, and prove enough regularized branch or
jet matching to select the finite branch parameters described below.

## Indicial Shape Modes Show Finite Selector Jets Are Extra Data

The `C^2` entry is not enough to force the finite integer jets used by the
analytic selector. This is a regular-singular obstruction, not a certificate
bookkeeping issue.

Write the shape equation:

```text
tau^2S'' + 2tau S' - 2S = 9A(S),
S(0)=C,
A(C)=-(2/9)C.
```

Linearize at `C` and insert a perturbation:

```text
S(tau)=C+tau^k V.
```

The indicial equation is:

```text
((k+2)(k-1)I - 9DA(C))V=0.
```

Thus every eigenvalue `mu` of `DA(C)` gives possible exponents:

```text
(k+2)(k-1)=9mu.
```

The analytic normal form above keeps only the nonnegative integer exponents.
But the equal-mass equilateral central configuration already has centered shape
eigenvalues `mu=1/9`. The corresponding positive exponent is:

```text
k = (-1+sqrt(13))/2,
1 < k < 2.
```

The perturbation:

```text
q(tau)=tau^2C + epsilon tau^(2+k)V
```

still has the same `C^2` cubic-time entry:

```text
q/tau^2 -> C,
q_tau/tau -> 2C,
q_tautau -> 2C,
```

and its cubic coefficient is zero because `k>1`. But the quartic quotient

```text
(q-tau^2C)/tau^4 = epsilon tau^(k-2)V
```

diverges because `k<2`. So the finite quartic jet used to read the homothetic
energy direction is not forced by oriented shape convergence or by the `C^2`
entry alone.

This narrows the zero-angular gap in the opposite direction from a naive
bootstrap: one cannot prove arbitrary entry into the analytic cubic-time normal
form from `C^2` data alone. A full theorem must either prove that the
noninteger Fuchsian shape modes are absent for the branch being continued, or
enlarge the lifted function class beyond analytic integer-power series to
include the relevant regular-singular powers and then project/verify those
generalized branches.

## Fractional Fuchsian Branch In The Lifted Shape Variable

The preceding obstruction also points to a constructive enlargement of the
lifted space. For the equal-mass equilateral branch and the centered
shape-mode exponent:

```text
k = (-1+sqrt(13))/2,
DA(C)[V] = (1/9)V,
```

introduce the fractional collision coordinate:

```text
x = tau^k,        tau > 0,
S(tau)=Phi(x).
```

The regular-singular shape equation becomes an analytic Fuchsian equation in
`x`:

```text
k^2 x^2 Phi'' + k(k+1)x Phi' - 2Phi = 9A(Phi).
```

Seek a one-sided branch:

```text
Phi(x)=C+a x V+sum_{n>=2} Phi_n x^n.
```

The coefficient `Phi_1=aV` is allowed because the indicial equation is exactly
the eigenvalue equation for `V`. For `n>=2`, the coefficient equation is:

```text
(((nk+2)(nk-1)/9)I - DA(C)) Phi_n
  = [A(C+Phi_1x+...+Phi_{n-1}x^(n-1))]_n.
```

The operator is invertible for every `n>=2`. Indeed the equal-mass
equilateral spectrum is:

```text
{0,0,4/9,-2/9,1/9,1/9}.
```

If `((nk+2)(nk-1)/9)` hit one of these eigenvalues, then `nk` would have to be
one of the corresponding indicial roots. The only positive noninteger root
equal to a multiple of `k` in the centered shape spectrum is `k` itself, which
is the already chosen `n=1` mode. The roots for `0`, `4/9`, and `-2/9` are
rational or nonpositive, so irrationality of `k` excludes them for `n>=2`.

Thus the recurrence determines every later `Phi_n` uniquely after the
fractional amplitude `a` is chosen. Since the force map is analytic near the
collision-free central configuration and the divisors grow quadratically in
`n`, the usual Cauchy-majorant proof for analytic Fuchsian recurrences gives a
positive radius of convergence in `x`.

Project back by:

```text
q(tau)=tau^2 Phi(tau^k),       t=tau^3.
```

For every `tau>0` inside the convergence interval, the Fuchsian equation is
identical to:

```text
tau^2S''+2tau S'-2S=9A(S),
```

and the homogeneity calculation used above gives `q''=A(q)`. This is exactly
the lift/construct/project/verify pattern in a larger function class: lift to
`x=tau^k`, construct `Phi(x)` by an analytic recurrence, project by
`q=tau^2Phi`, and verify Newton's equation algebraically.

This does not yet solve the arbitrary zero-angular continuation problem. It
constructs one class of one-sided regular-singular local branches and shows
that the final theorem's closed-form class must be broad enough to include
such Fuchsian powers, or must prove a selection principle that excludes their
amplitudes.

## Full Equal-Mass Fractional Eigenspace And Two-Sided Projection

The preceding construction is not tied to one chosen real eigenvector. For the
equal-mass equilateral branch, the centered noninteger eigenspace is two
dimensional:

```text
E_k = span{conjugate(C), i conjugate(C)},
DA(C)[V] = (1/9)V        for every V in E_k.
```

Choose any vector `D in E_k`, and set:

```text
Phi(x)=C+xD+sum_{n>=2} Phi_n x^n.
```

The same recurrence applies:

```text
(((nk+2)(nk-1)/9)I - DA(C)) Phi_n
  = [A(C+xD+Phi_2x^2+...+Phi_{n-1}x^(n-1))]_n.
```

Its invertibility for every `n>=2` is unchanged, because it depends only on
the spectrum of `DA(C)` and on the irrationality of the indicial exponent
`k`, not on the direction of `D` inside the eigenspace. Hence each small
fractional amplitude vector `D in E_k` determines a convergent local Fuchsian
shape branch in the lifted variable `x`.

This gives an actual collision-through convention in the enlarged one-sided
Fuchsian class. Pick possibly different incoming and outgoing fractional
amplitudes:

```text
D_- in E_k,       D_+ in E_k,
Phi_-(x),         Phi_+(x)
```

constructed by the recurrence above. Define, for `tau != 0`,

```text
S(tau) =
  Phi_+ ( tau^k)       if tau > 0,
  Phi_- ((-tau)^k)     if tau < 0,

q(tau)=tau^2 S(tau),       t=tau^3.
```

On either side, putting `x=|tau|^k` gives:

```text
tau^2 d^2/dtau^2 Phi(x) + 2 tau d/dtau Phi(x)
  = k^2 x^2 Phi''(x) + k(k+1)x Phi'(x),
```

so the same Fuchsian equation implies the regularized shape equation on
`tau>0` and on `tau<0`. Therefore the projected curve satisfies Newton's
equation for every `tau != 0` on both sides of total collision.

The branch is generally not an analytic integer-power germ at `tau=0`; since
`1<k<2`, it has a `C^2` collision entry with a fractional correction
`tau^2 |tau|^k D_\pm`, while the quartic quotient usually diverges. That is
precisely why the integer-power selector cannot recover these amplitudes from
finite analytic jets. Nevertheless the Newtonian verification is exact away
from collision, and the angular-momentum argument still applies:

```text
L(tau) = (tau^2/3) sum_i m_i S_i(tau) x S_i'(tau) -> 0.
```

Since Newtonian pair forces conserve angular momentum on each punctured side,
the continuation has zero angular momentum on both sides. Thus the enlarged
local zero-angular continuation data for this equal-mass equilateral Fuchsian
subcase are:

```text
C,        D_-,        D_+,
```

with `D_\pm` lying in the two-dimensional fractional eigenspace. This still
does not prove that every zero-angular total collapse reaches such a
Fuchsian chart. It proves a concrete missing continuation class that the final
closed-form representation must either include or exclude by an additional
selection theorem.

## General Nonresonant Fuchsian Eigenmode Lemma

The fractional construction is not special to the equal-mass spectrum. It is a
local theorem for any real central-configuration eigenmode satisfying a
nonresonance condition.

Let `C` be any mass-centered noncollision central configuration, scaled by:

```text
A(C)=-(2/9)C.
```

Let `E_mu` be a real eigenspace of the mass-self-adjoint linear map `DA(C)`:

```text
DA(C)[D] = mu D,        D in E_mu,
sum_i m_i D_i = 0.
```

Choose the positive indicial root:

```text
k = (-1 + sqrt(9+36mu))/2,
```

so that:

```text
(k+2)(k-1) = 9mu.
```

Assume `k>0` and the multiples of this exponent do not hit the spectrum after
the first coefficient:

```text
((nk+2)(nk-1)/9) notin spec(DA(C)),        n >= 2.
```

Then every sufficiently small amplitude `D in E_mu` determines a convergent
one-sided Fuchsian branch:

```text
Phi(x)=C+xD+sum_{n>=2} Phi_n x^n,
```

solving:

```text
k^2 x^2 Phi'' + k(k+1)x Phi' - 2Phi = 9A(Phi).
```

The recurrence is exactly:

```text
(((nk+2)(nk-1)/9)I - DA(C)) Phi_n
  = [A(C+xD+Phi_2x^2+...+Phi_{n-1}x^(n-1))]_n.
```

The nonresonance hypothesis makes the operator invertible for every `n>=2`.
The Newtonian force is analytic near the noncollision shape `C`, and the
inverse norms are `O(1/n^2)`, so the standard Cauchy-majorant argument gives
positive convergence radius in `x` for small `D`.

As in the equal-mass case, independent incoming and outgoing amplitudes
`D_-`, `D_+` in the same eigenspace give a two-sided collision-through
projection:

```text
S(tau) = Phi_-((-tau)^k)    for tau < 0,
S(tau) = Phi_+( tau^k)      for tau > 0,
q(tau)=tau^2S(tau),         t=tau^3.
```

For every `tau != 0` in the local interval, `q` solves Newton's equation by
the same homogeneity calculation. Since `k>0`, `S` is bounded and
`tau^2 sum_i m_i S_i wedge S_i' -> 0`; conservation of angular momentum on
each punctured side then forces zero angular momentum. Thus the Fuchsian
eigenmode data are legitimate local zero-angular continuation parameters in
the enlarged function class.

This theorem is conditional on the displayed nonresonance. It does not replace
the integer-power resonant normal form; it says that whenever the indicial
semigroup is nonresonant, the lift/construct/project/verify method constructs
the corresponding fractional branch directly.

## Multi-Indicial Fuchsian Lift

A single fractional coordinate is still not the full local picture when
several noninteger indicial modes are present. The natural lifted space uses
one coordinate for each selected exponent:

```text
x_j = tau^(k_j),        j=1,...,r.
```

Let `D_j` lie in centered eigenspaces of `DA(C)` with eigenvalues `mu_j`, and
let:

```text
(k_j+2)(k_j-1) = 9mu_j,
k_j > 0.
```

For a multi-index `alpha=(alpha_1,...,alpha_r)`, write:

```text
|alpha|_k = alpha_1 k_1 + ... + alpha_r k_r,
x^alpha = x_1^alpha_1 ... x_r^alpha_r.
```

The chain rule is governed by the Euler operator:

```text
E = sum_j k_j x_j partial_{x_j}.
```

If `S(tau)=Phi(tau^(k_1),...,tau^(k_r))`, then:

```text
tau^2S''+2tau S'-2S = (E^2+E-2)Phi.
```

Thus the lifted Fuchsian equation is:

```text
(E^2+E-2)Phi = 9A(Phi).
```

Seek:

```text
Phi(x)=C+sum_j x_j D_j + sum_{|alpha|>=2} Phi_alpha x^alpha.
```

For every multi-index `alpha` not one of the selected basis indices `e_j`, the
coefficient equation is:

```text
(((|alpha|_k+2)(|alpha|_k-1)/9)I - DA(C)) Phi_alpha
  = [A(Phi with Phi_alpha set to 0)]_alpha.
```

Assume the semigroup nonresonance condition:

```text
((|alpha|_k+2)(|alpha|_k-1)/9) notin spec(DA(C))
```

for every `alpha` with `|alpha|>=2`. Then the recurrence determines all
`Phi_alpha` in increasing total multi-degree. The force map is analytic near
the noncollision shape `C`, and the scalar multiplier grows quadratically in
`|alpha|_k`, so a polydisc Cauchy-majorant argument gives convergence for
sufficiently small amplitudes `D_j`.

Projection is again algebraic. For `tau>0`, set:

```text
q(tau)=tau^2 Phi(tau^(k_1),...,tau^(k_r)),
t=tau^3.
```

The displayed lifted equation is exactly the regularized shape equation, and
homogeneity of `A` gives Newton's equation for every `tau != 0` in the
one-sided interval. The same signed construction with `x_j=|tau|^(k_j)` gives
incoming and outgoing branches with independent selected amplitudes. This is
the appropriate lifted function class for simultaneous fractional modes:
construct in a convergent multivariable Fuchsian series, project to physical
coordinates, and verify by the Euler-operator identity.

## Adding The Homothetic Energy Coordinate To The Fuchsian Lift

The multivariable lift can include the ordinary homothetic energy coordinate
in the same recurrence. The scale direction is always an eigenmode:

```text
DA(C)[C] = -2A(C) = (4/9)C,
```

so its positive indicial exponent is:

```text
k_0 = 2.
```

Set:

```text
z = tau^2,
x_j = tau^(k_j),
```

and seek:

```text
Phi(z,x)=C+z alpha C+sum_j x_jD_j
          + higher multi-index terms.
```

The same Euler operator formula applies, now with:

```text
E = 2z partial_z + sum_j k_j x_j partial_{x_j}.
```

At the basis index for `z`, the indicial equation is the scale eigenvalue
equation. For every other multi-index, including mixed terms such as
`z x_j` and `x_i x_j`, the coefficient equation is:

```text
(((|alpha|_k+2)(|alpha|_k-1)/9)I - DA(C)) Phi_alpha
  = [A(Phi with Phi_alpha set to 0)]_alpha,
```

where `|alpha|_k` now includes the `2 alpha_0` contribution from `z`. Under
the same semigroup nonresonance condition away from the selected basis
indices, the recurrence is invertible and convergent.

This matters for continuation data. The homothetic energy parameter and the
fractional Fuchsian amplitudes are not separate charts that must be glued by
hand; they live in one lifted analytic chart. After constructing `Phi`, the
projection:

```text
q(tau)=tau^2 Phi(tau^2,tau^(k_1),...,tau^(k_r)),
t=tau^3
```

again satisfies Newton's equation for every `tau != 0`, and the conserved
angular momentum is zero on each side. Thus the enlarged local zero-angular
branch parameters may include the central shape, the homothetic energy
coordinate, and any nonresonant fractional amplitudes simultaneously.

## Energy Limit In The Mixed Fuchsian Chart

The mixed chart also has the expected finite energy selector. Write:

```text
S(tau)=Phi(tau^2,tau^(k_1),...,tau^(k_r)),
q(tau)=tau^2S(tau),
t=tau^3,
```

and assume all non-scale selected exponents satisfy `k_j>1`. The physical
velocity is:

```text
dq/dt = (2/(3tau))S + (1/3)S_tau.
```

Thus the Newtonian energy is:

```text
H(tau)
 = tau^-2 [
     (1/18)<2S+tau S_tau, 2S+tau S_tau>_m - U(S)
   ].
```

At `S=C`, the bracket vanishes because:

```text
U(C) = (2/9)<C,C>_m.
```

For a perturbing coefficient `Y tau^rho`, the linear coefficient of the
bracket is:

```text
(2/9)(rho+3)<C,Y>_m tau^rho.
```

Every fractional eigenmode is mass-orthogonal to the scale direction `C`,
because `DA(C)` is self-adjoint for the mass inner product and its eigenvalue
differs from the scale eigenvalue `4/9`. Hence the fractional linear terms
vanish. Fractional quadratic and mixed fractional-scale terms have exponent
strictly larger than `2`, since `k_j>1`. Therefore they disappear after the
outer `tau^-2` factor.

Only the scale coefficient:

```text
S(tau)=C+alpha C tau^2+...
```

contributes to the finite collision energy limit:

```text
H -> (10/9) alpha <C,C>_m.
```

So in the nonresonant mixed Fuchsian chart, the homothetic energy coordinate
continues to select the finite energy, while the fractional amplitudes are
additional zero-angular branch data that do not alter that limiting energy.

## Two-Sided Energy Matching In The Mixed Fuchsian Chart

The same calculation gives the local energy-matching rule for a
collision-through convention in the enlarged Fuchsian class. Let `u=|tau|`.
On the incoming and outgoing sides choose possibly different lifted branches:

```text
S_-(tau)=Phi_-(u^2,u^(k_1),...,u^(k_r)),  tau<0,
S_+(tau)=Phi_+(u^2,u^(k_1),...,u^(k_r)),  tau>0,
```

with the same central configuration `C`, non-scale exponents `k_j>1`, and
scale coefficients:

```text
Phi_-(z,x)=C+alpha_- z C+sum_j x_j D_{j,-}+...
Phi_+(z,x)=C+alpha_+ z C+sum_j x_j D_{j,+}+...
```

Project each side by:

```text
q(tau)=tau^2 S_\pm(tau),   t=tau^3.
```

For `tau != 0` the projected branches solve Newton's equation by the
Fuchsian recurrence. Their angular momentum is conserved on each punctured
side and tends to zero at the collision, since `q=O(tau^2)` and
`dq/dt=O(|tau|^-1)`. Thus both sides have zero angular momentum.

The velocity on either side is:

```text
dq/dt = sign(tau) [ (2/(3u))S + (1/3) dS/du ].
```

The sign disappears from the kinetic energy, so the energy expansion is the
same bracket as above with `u` in place of `tau`. Orthogonality to the scale
direction removes the fractional linear terms, and the assumptions `k_j>1`
put every fractional quadratic or scale-fractional mixed term above exponent
`2`. Therefore:

```text
H_- = (10/9) alpha_- <C,C>_m,
H_+ = (10/9) alpha_+ <C,C>_m.
```

Hence a two-sided zero-angular total-collision continuation in this class
matches finite physical energy if and only if:

```text
alpha_-=alpha_+.
```

The incoming and outgoing fractional amplitudes `D_{j,-}` and `D_{j,+}` may
still be chosen independently without changing that limiting energy. If the
scale coefficients differ, the energy jump is exactly:

```text
H_+ - H_- = (10/9)(alpha_+-alpha_-)<C,C>_m.
```

This is the branch-semantics content of the zero-angular total-collision
obstruction in the mixed Fuchsian chart: energy preservation selects the
common homothetic scale coordinate, but it does not select the noninteger
Fuchsian branch amplitudes. Those amplitudes remain extra continuation data,
or must be supplied by an explicit convention.

## Finite Regularized Jets Do Not Select Fuchsian Amplitudes

The previous theorem also explains why the missing zero-angular branch datum
cannot be recovered from ordinary finite regularized jets. In the same mixed
chart, write `u=|tau|` and:

```text
S_\pm(u)
 = C + alpha C u^2 + sum_j D_{j,\pm} u^(k_j) + higher terms,
q_\pm(tau)=tau^2 S_\pm(|tau|),
```

with every non-scale exponent `k_j>1`. A monomial `Y u^rho` in `S` contributes
`Y u^(rho+2)` to `q`. Therefore its first three `tau`-derivatives are:

```text
d^n/dtau^n [Y u^(rho+2)]
  = sign(tau)^n (rho+2)(rho+1)...(rho+3-n) Y u^(rho+2-n),
  n=0,1,2,3.
```

For each fractional exponent `rho=k_j>1`, the third derivative still contains
the positive power `u^(k_j-1)`. Hence the limits through total collision are:

```text
q(0)=0,
q_tau(0)=0,
q_tautau(0)=2C,
q_tautautau(0)=0,
```

independently of all fractional amplitudes `D_{j,\pm}`. The homothetic energy
coefficient `alpha` first appears in the fourth regularized derivative through
the integer scale term `alpha C tau^4`; a fractional mode with `1<k_j<2`
does not even have a finite ordinary fourth derivative, because it contributes
`u^(k_j-2)`.

The fractional branch data are visible only in fractional asymptotic quotients.
If the selected exponents are ordered and `P_j` is the mass-orthogonal
projection onto the eigenspace of `D_j`, then:

```text
D_{j,\pm}
 = lim_{u->0+} u^(-k_j)
     P_j[ S_\pm(u) - C - alpha C u^2
          - sum_{k_l<k_j} D_{l,\pm}u^(k_l) ].
```

Thus a convention based only on the finite regularized collision jet through
order three cannot select the mixed Fuchsian continuation branch. A valid
zero-angular continuation convention for this enlarged class must supply
the fractional amplitudes, supply these fractional quotient limits from the
incoming side, or impose an external branch-selection rule for them.

## Fractional-Quotient Selector Gives A Local Mixed Fuchsian Continuation

The fractional quotient data are enough to turn the mixed Fuchsian
construction into a local continuation theorem. Assume a one-sided incoming
zero-angular total-collision branch has a mixed Fuchsian expansion on
`tau<0`:

```text
S_-(u)=C+alpha C u^2+sum_j D_{j,-}u^(k_j)+higher terms,
q_-(tau)=tau^2S_-(|tau|),    t=tau^3,
```

where `u=|tau|`, `A(C)=-(2/9)C`, every selected non-scale exponent satisfies
`k_j>1`, and the semigroup nonresonance condition from the multivariable
Fuchsian lift holds. Suppose the incoming fractional quotient limits exist:

```text
D_{j,-}
 = lim_{u->0+} u^(-k_j)
     P_j[ S_-(u) - C - alpha C u^2
          - sum_{k_l<k_j} D_{l,-}u^(k_l) ],
```

with `P_j` the mass-orthogonal projection onto the selected eigenspace. A
local branch convention is then any rule `R` that assigns outgoing amplitudes:

```text
(D_{1,+},...,D_{r,+}) = R(D_{1,-},...,D_{r,-}).
```

The homothetic scale coefficient must be the same `alpha` if the continuation
is to preserve the incoming finite energy. With these data fixed, the
multivariable recurrence constructs a unique convergent lifted branch:

```text
Phi_+(z,x)=C+alpha z C+sum_j x_jD_{j,+}+sum_{|beta|>=2} Phi_beta x^beta,
```

where `z=u^2` and `x_j=u^(k_j)`. Define the outgoing physical branch by:

```text
S_+(u)=Phi_+(u^2,u^(k_1),...,u^(k_r)),
q_+(tau)=tau^2S_+(tau),   tau>0,   t=tau^3.
```

By the Fuchsian recurrence, `S_+` satisfies the regular-singular shape
equation, so `q_+` solves Newton's equation for every `tau>0`. Its angular
momentum is conserved on the punctured side and tends to zero at the
collision, hence is identically zero. The energy calculation above gives:

```text
H_+ = (10/9)alpha<C,C>_m = H_-.
```

Thus the data:

```text
C, alpha, (D_{j,-}), R
```

define a local zero-angular total-collision continuation in the enlarged
mixed Fuchsian class. The identity rule `R(D_-)=D_-` is the direct analogue of
analytic-germ continuation for noninteger branches: it matches the incoming
fractional quotient limits on the outgoing side. Other rules are possible,
but they are real branch conventions; they are not consequences of the
ordinary finite collision jet or of energy conservation alone.

The finite nonresonant part of this construction is now executable in
`three_body_symmetry/fuchsian.py`. The constructor
`construct_fuchsian_shape_branch(...)` takes the scaled central shape `C`, the
lifted powers `(2,k_1,...,k_r)`, and selected coefficients
`alpha C,D_1,...,D_r`; it solves each nonselected multi-index by

```text
(((|beta|_k+2)(|beta|_k-1)/9)I-DA(C)) Phi_beta
  = [A(Phi with Phi_beta=0)]_beta,
```

rejecting resonant rows with a small singular-value floor. The returned branch
evaluates `S`, `q=tau^2S`, physical velocities, the regularized shape
residual, the scaled Newton residual, the finite energy contribution
`(10/9)alpha<C,C>_m`, and the zero-angular collision-limit expression. The
companion `construct_fuchsian_selector_continuation(...)` builds incoming and
outgoing branches from selector data; with no explicit outgoing data it uses
the identity rule. This is an executable local continuation theorem for the
finite nonresonant mixed-Fuchsian class, not a proof that every arbitrary
zero-angular total collision enters such a nonresonant chart.

The first resonant Fuchsian-log row is executable as well. For one row of
weight `omega`, let:

```text
M = ((omega+2)(omega-1)I - 9 DA(C)).
```

The constructor `construct_fuchsian_log_row_solution(...)` solves the
triangular equations:

```text
M H_l +(l+1)(2omega+1)H_{l+1} +(l+2)(l+1)H_{l+2}=F_l
```

by descending log degree. At each row it projects the unsolved resonant kernel
part of the forcing into `H_{l+1}` and solves the remaining range equation for
`H_l`; the constant kernel component of `H_0` is the free selector. The
returned `FuchsianLogRowSolution` records the log coefficients, row residuals,
and recovered selector, while `FuchsianLogBranch` projects
`S=C+alpha sigma^2C+sigma^omega sum_l H_l(log sigma)^l` by
`q=tau^2S`, `t=tau^3` and evaluates the projection identity, finite-energy
limit, and angular-momentum collision-limit expression. This closes the
single-row resonant mechanism used by the Fuchsian-log selector theorem; it is
still not a complete finite triangular Fuchsian-log normal-form constructor for
all coupled resonant rows.

Finite row composition is executable too. A `FuchsianLogTerm` stores one row
`sigma^omega P(log sigma)`, and `FiniteFuchsianLogBranch` forms:

```text
S(sigma)=C+alpha sigma^2 C+sum_a sigma^(omega_a) P_a(log sigma).
```

It evaluates `q=tau^2S`, physical velocities, the projection identity, the
finite energy limit `(10/9)alpha<C,C>_m`, the angular-momentum collision-limit
expression, and selector recovery for each row after subtracting the other
known rows. `FiniteFuchsianLogContinuation` compares incoming and outgoing
finite-row branches and certifies the identity-selector case when all row
coefficients and the energy scale match. This gives an executable projected
branch for a finite selected Fuchsian-log germ; the remaining hard step is
still deriving that germ from arbitrary zero-angular dynamics.

In the stable-chain case the row list no longer has to be written by hand.
`construct_finite_fuchsian_log_branch_from_stable_chain(...)` converts each
mode in a certified triangular stable selector chain into a
`FuchsianLogTerm`, using the supplied mass-orthogonal vector mode shape as the
selector basis and the chain's converted log polynomial as that row's
coefficient polynomial. The regression checks selector recovery across all
five rows of a coupled resonant chain, the projection identity, finite-energy
convergence to `(10/9)alpha<C,C>_m`, angular-momentum decay, identity
continuation, and rejection of missing, scale-contaminated, or low-power mode
data.

## No Hidden Higher Mixed Fuchsian Branch Data

Once the central shape, energy coordinate, and selected fractional amplitudes
are fixed, the nonresonant mixed Fuchsian chart has no further local branch
parameters. Let `Phi` and `Psi` be two convergent lifted solutions in the same
mixed chart:

```text
Phi(z,x)=C+alpha z C+sum_j x_jD_j+...
Psi(z,x)=C+alpha z C+sum_j x_jD_j+...
```

Assume both satisfy:

```text
(E^2+E-2)Theta=9A(Theta)
```

and both use the same selected basis indices for `z` and the `x_j`. If
`Phi != Psi`, choose the first multi-index `beta`, ordered by total degree and
then by any fixed tie-breaker, where their coefficients differ. All lower
coefficients agree, so the beta coefficient of the nonlinear force with the
unknown beta coefficient removed is the same for both branches. Subtracting
the two coefficient equations gives:

```text
(((|beta|_k+2)(|beta|_k-1)/9)I - DA(C))
  (Phi_beta-Psi_beta) = 0.
```

The semigroup nonresonance hypothesis says this operator is invertible for
every non-basis multi-index. Therefore `Phi_beta=Psi_beta`, contradicting the
choice of `beta`. Hence:

```text
C, alpha, D_1,...,D_r
```

determine the full lifted germ uniquely. The fractional-quotient selector is
therefore complete inside the nonresonant mixed Fuchsian class: after an
outgoing amplitude rule has selected the `D_j`, no additional higher-order
choices remain.

## Regularized Smoothness Excludes Low Fuchsian Modes

The finite-jet obstruction has a useful converse as a selection principle.
Suppose a mixed Fuchsian branch has a nonzero selected shape term:

```text
q(tau) = tau^2 C + alpha C tau^4 + D |tau|^(k+2) + higher terms,
```

with `1<k<2`. Its fourth regularized-time derivative contains:

```text
(k+2)(k+1)k(k-1)D |tau|^(k-2).
```

Since `k-2<0`, this term is unbounded unless its amplitude vanishes after
projection onto the corresponding eigenspace. Higher Fuchsian monomials have
larger weighted exponent in the nonresonant construction and cannot cancel
the first nonzero low-exponent eigenspace projection. Therefore:

```text
q in C^4 at tau=0  =>  D_j=0 for every selected 1<k_j<2.
```

More generally, if `q` has a finite `m`-th regularized derivative at total
collision, then every selected noninteger mode with `k_j+2<m` must have zero
amplitude. Thus stronger regularized smoothness can be a genuine branch
selector: it collapses the mixed Fuchsian continuation back toward the
integer-power normal form by excluding the low noninteger modes. It still does
not follow from the Newton equations alone; it is an additional regularity
hypothesis on the incoming collision germ.

## C4 Mixed Fuchsian Branches Collapse To The Homothetic Branch When All Shape Exponents Are Low

Assume now that the selected non-scale exponents in the mixed Fuchsian chart
all satisfy:

```text
1 < k_j < 2.
```

If the incoming regularized branch is `C^4` at total collision, the previous
smoothness selector forces every selected fractional amplitude to vanish:

```text
D_j=0 for all j.
```

With no fractional seed, the mixed recurrence has only the scale coordinate
`z=tau^2`. Every coefficient carrying any `x_j` index is then forced to be
zero. Indeed, take the first such mixed coefficient. Its lower mixed
coefficients vanish, so its force coefficient is zero; the nonresonant
operator at that multi-index is invertible, hence the coefficient is zero.
Induction removes all `x`-dependent terms.

The remaining `z`-only branch has the form:

```text
Phi(z)=u(z)C,        u(0)=1,        u'(0)=alpha.
```

Since `A(uC)=u^(-2)A(C)=-(2/9)u^(-2)C`, the lifted shape equation reduces to
the scalar homothetic energy recurrence. Equivalently:

```text
(u+z u')^2 = 1/u + (9/2) epsilon z,
epsilon = (10/9)alpha.
```

Thus, in any nonresonant mixed Fuchsian chart whose selected noninteger shape
exponents all lie between `1` and `2`, the extra Fuchsian branch freedom is
exactly the obstruction to `C^4` regularity. Imposing `C^4` selects the
ordinary homothetic energy branch with the same central shape and finite
energy.

## Integer-Power Analytic Germs Exclude All Noninteger Fuchsian Modes

The same smoothness argument gives the clean analytic selector. Let a mixed
Fuchsian branch contain a first nonzero selected noninteger shape mode:

```text
D_j |tau|^(k_j+2),      k_j notin Z.
```

Choose:

```text
m_j = floor(k_j+2)+1.
```

Then `m_j` is the first integer derivative order strictly larger than
`k_j+2`, and:

```text
d^(m_j)/dtau^(m_j) [D_j |tau|^(k_j+2)]
 = c_j D_j |tau|^(k_j+2-m_j),
c_j != 0,
```

with a negative exponent `k_j+2-m_j`. Projecting onto the corresponding
mass-orthogonal eigenspace isolates this first nonzero mode; higher weighted
terms have larger exponents and lower noninteger modes have already been
removed by the ordering. Thus a finite `m_j`-th regularized derivative is
impossible unless `D_j=0`.

Consequently, if the incoming regularized collision germ is ordinary
integer-power analytic in `tau`, or merely `C^m` for all required
`m_j`, every selected noninteger Fuchsian amplitude vanishes. In a
nonresonant mixed chart whose only selected non-scale modes are noninteger,
ordinary regularized analyticity therefore leaves only the integer-power
normal-form coordinates, in particular the homothetic energy coordinate in
the pure equilateral mixed chart above.

## Branch-Parameter Continuation Through Total Collision

The integer-power local normal form does give a genuine zero-angular
continuation theorem once the branch parameters have been selected. Let
`S(tau)` be any one of the analytic shape series from that normal form,
convergent for `|tau|<rho`, with noncollision
initial shape `S(0)=C`. After decreasing `rho` if necessary, all pair
differences `S_i(tau)-S_j(tau)` remain nonzero for `0<|tau|<rho`. Define:

```text
t = tau^3,
q_i(tau)=tau^2 S_i(tau).
```

The constructed shape series satisfy the regular-singular shape equation:

```text
tau^2 S_i'' + 2 tau S_i' - 2 S_i = 9 A_i(S).
```

Indeed, this is exactly the coefficient recurrence used above, with the finite
resonant parameters fixed where resonance occurs. For `tau != 0`:

```text
dq_i/dt = (1/(3 tau^2)) d/dtau (tau^2 S_i)
        = (2/(3 tau)) S_i + (1/3) S_i',
```

and:

```text
d^2q_i/dt^2
  = tau^-4 (tau^2 S_i'' + 2 tau S_i' - 2 S_i)/9.
```

Since Newtonian acceleration is homogeneous of degree `-2`,

```text
A_i(q(tau)) = A_i(tau^2 S(tau)) = tau^-4 A_i(S(tau)).
```

Thus the shape equation implies `q_i''(t)=A_i(q(t))` for every `tau != 0`, on
both the incoming and outgoing sides. The map `tau -> t=tau^3` is monotone on
each side and has the same collision value at `tau=0`, so the selected
regularized branch gives a collision-through-ejection continuation in physical
time away from the collision instant.

The continuation is automatically zero-angular. The physical angular momentum
is:

```text
L(tau)
 = sum_i m_i q_i x dq_i/dt
 = (tau^2/3) sum_i m_i S_i(tau) x S_i'(tau).
```

The right side tends to zero as `tau -> 0`. On each side of the collision,
Newton's pair forces conserve angular momentum, so `L` is constant there; its
limit is zero, hence the constant is zero on both sides.

Therefore the local branch parameters listed in the normal form are not merely
formal coefficients. They are a local continuation convention:

```text
central-configuration second jet
+ homothetic energy parameter
+ resonant amplitude when the listed resonance locus is present
```

select a unique analytic regularized branch, and the same branch projects to a
zero-angular Newtonian solution for negative and positive physical time. This
still does not prove that an arbitrary zero-angular solution reaching total
collision has been matched to one of these local branches by a global
collision-manifold construction. It does close the local continuation part once
that branch-selection datum is supplied.

## Spatial Orthogonal Embeddings Preserve Local Collision Continuations

The local zero-angular branch construction is written in the reduced collision
plane, but it is not intrinsically planar. Let `E:R^2 -> R^d`, `d>=2`, be a
fixed isometric embedding:

```text
E^T E = I.
```

Suppose a selected local branch in the plane is:

```text
t = tau^3,
q_i(tau)=tau^2 S_i(tau),
```

and solves Newton's equation for every `tau != 0`. Define the spatial branch:

```text
Q_i(tau)=E q_i(tau).
```

The pair distances are unchanged:

```text
|Q_j-Q_i|=|E(q_j-q_i)|=|q_j-q_i|.
```

Therefore the spatial Newtonian acceleration is equivariant:

```text
A_d(Q)_i
 = sum_{j != i} m_j (Q_j-Q_i)/|Q_j-Q_i|^3
 = E sum_{j != i} m_j (q_j-q_i)/|q_j-q_i|^3
 = E A_2(q)_i.
```

Since `Q_i''=E q_i''`, the projected spatial branch satisfies:

```text
Q_i''(t)=A_d(Q)_i
```

on both punctured sides of the collision. The center-of-mass and linear
momentum equations are also carried by `E`.

Energy is preserved because `E` is an isometry. The angular momentum bivector
is the exterior-square image of the planar angular momentum:

```text
sum_i m_i Q_i wedge dQ_i/dt
 = (E wedge E) sum_i m_i q_i wedge dq_i/dt.
```

Thus any planar selected branch whose scalar angular momentum is zero becomes a
spatial branch whose full angular-momentum bivector is zero. Collinear Euler
branches are included as the one-dimensional subcase inside the same embedded
plane.

Consequently, the local zero-angular continuation theorem is not restricted to
the coordinate plane used for the normal-form calculation. The branch datum in
spatial dimension is the same finite local datum as before, plus a fixed
orthonormal collision-plane frame. This still does not prove that arbitrary
spatial total-collision approaches enter one of these normal forms; it proves
that once the reduced branch is selected, projection back to any spatial
orientation preserves Newton's equation, energy, and zero full angular
momentum.

## Incoming Analytic Germ Selects The Outgoing Branch

The previous statement can be read constructively from the incoming side. Let a
Newtonian branch be defined for negative physical time near a total collision,
and write `t=tau^3` with `tau<0`. Suppose the branch has a convergent
regularized expansion:

```text
q_i(tau)=tau^2 S_i(tau),
S(tau) analytic at tau=0,
S_i(0) != S_j(0).
```

Then the same analytic germ gives the local outgoing continuation. Indeed, on
the incoming side `tau<0`, the change of variables used above gives:

```text
tau^2 S_i'' + 2 tau S_i' - 2 S_i = 9 A_i(S).
```

Because the limiting shape `C=S(0)` is noncollision, the right side is analytic
in `S` for all sufficiently small `|tau|`. The left and right sides are
therefore two analytic functions of `tau` that agree on the interval
`(-rho,0)`. By the identity theorem they agree on a full two-sided interval
`|tau|<rho` after possibly shrinking `rho`. Hence the same formula:

```text
q_i(tau)=tau^2 S_i(tau),       t=tau^3,
```

projects to a Newtonian solution for `tau>0` as well as for `tau<0`.

Angular momentum is forced to be zero. In regularized time:

```text
L(tau)=sum_i m_i q_i x dq_i/dt
      =(tau^2/3) sum_i m_i S_i(tau) x S_i'(tau),
```

so `L(tau)->0` at collision. Since Newtonian angular momentum is conserved on
each punctured side, the incoming and outgoing constants are both zero.

Thus, for analytic cubic-time total-collision germs with noncollision second
shape, the missing local continuation datum is not an infinite future choice.
It is the finite normal-form coordinate system already identified:

```text
central-configuration second jet,
homothetic energy parameter,
and, only on the listed resonance loci, the resonant amplitude.
```

The incoming analytic germ supplies these coordinates through its finite
regularized jets. Substituting them into the local normal-form recurrence gives
the unique analytic germ agreeing with the incoming branch, and the identity
theorem carries that same germ to the outgoing side.

This still leaves the global theorem with two separate obligations: prove that
an arbitrary zero-angular finite-time approach to total collision admits such a
regularized analytic germ with noncollision limiting shape, and prove how the
ordinary, binary, and triple-collision charts are reached by the all-time
construction. The statement here closes the local continuation once that
analytic incoming germ is available.

## Energy Matching For A Selected Branch

The selected normal-form coefficients also determine the physical energy that
must be matched by any incoming branch. Use the mass inner product:

```text
<X,Y>_m = sum_i m_i X_i dot Y_i.
```

Write:

```text
S(tau)=C + D tau + E tau^2 + O(tau^3),
q(tau)=tau^2 S(tau),
t=tau^3.
```

For the selected local branches, `A(C)=-(2/9)C`, and the cubic coefficient
`D` is mass-orthogonal to `C`. In the nonresonant and ordered-Euler resonant
branches `D=0`; in the equilateral `beta=8/27` branch this orthogonality follows
from self-adjointness of `DA(C)` and the fact that `D` and `C` are eigenvectors
with distinct eigenvalues `0` and `4/9`.

The physical velocity is:

```text
dq/dt = (2/(3 tau))S + (1/3)S'
      = (2/(3 tau))C + D + (4/3)E tau + O(tau^2).
```

Hence:

```text
K = (1/2)<dq/dt,dq/dt>_m
  = (2/9)<C,C>_m tau^-2
    + (2/3)<C,D>_m tau^-1
    + (1/2)<D,D>_m + (8/9)<C,E>_m
    + O(tau).
```

The Newtonian potential `U(q)=sum_{i<j}m_i m_j/|q_i-q_j|` has:

```text
U(tau^2S(tau)) = tau^-2 U(S(tau)).
```

Since `grad U(C)=m_i A_i(C)` in the mass inner product notation,

```text
U(C) = (2/9)<C,C>_m,
dU(C)[D] = <A(C),D>_m = -(2/9)<C,D>_m,
dU(C)[E] = -(2/9)<C,E>_m,
d^2U(C)[D,D] = <D,DA(C)[D]>_m.
```

Therefore the singular terms in `H=K-U` are:

```text
tau^-2:  (2/9)<C,C>_m - U(C) = 0,
tau^-1:  (2/3)<C,D>_m + (2/9)<C,D>_m = 0,
```

where the second cancellation uses `<C,D>_m=0`. The finite energy selected by
the branch is:

```text
H =
  (1/2)<D,D>_m
  - (1/2)<D,DA(C)[D]>_m
  + (10/9)<C,E>_m.
```

For nonresonant and ordered-Euler resonant branches `D=0`, so this reduces to:

```text
H = (10/9)<C,E>_m.
```

If `E=alpha C + ...`, the homothetic part contributes
`(10/9)alpha <C,C>_m`. For the equilateral `beta=8/27` branch,
`DA(C)[D]=0`, and the energy is an even function of the resonant amplitude
because `D` enters quadratically while the odd coefficients change sign under
`D -> -D`.

Thus matching a physical incoming energy is not an extra opaque convention: it
is an explicit scalar equation on the selected branch parameters.

## Solving The Energy-Matching Parameter

The scalar equation is also nondegenerate. In every branch family above the
quartic coefficient has the form:

```text
E = E_base + alpha C,
```

where `alpha` is the homothetic energy parameter and `E_base` is fixed by the
chosen central configuration and any resonant amplitude. Substituting into the
energy formula gives:

```text
H(alpha) = H_base + (10/9) alpha <C,C>_m.
```

Because `C` is a noncollision central configuration, `<C,C>_m > 0`. Hence, for
any prescribed finite incoming energy `H_in`, there is a unique local value:

```text
alpha =
  (9/(10 <C,C>_m)) (H_in - H_base).
```

This proves that energy matching does not obstruct the selected local
zero-angular continuation. After the central-configuration branch and any
resonant amplitude are selected, the homothetic energy parameter is fixed
uniquely by the incoming energy invariant. At resonant loci, energy still does
not determine the resonant amplitude itself; it only fixes `alpha` once that
amplitude has been chosen.

## Resonant Amplitude As A Finite Jet Coordinate

The remaining local branch datum at a resonance is also explicit. It is not a
hidden infinite choice; it is one scalar projection of the first resonant
regularized jet.

For the equilateral `beta=8/27` surface, choose the centered cubic kernel `D`
with:

```text
DA(C)[D]=0,
<D,D>_m=1,
<C,D>_m=0.
```

The branch has:

```text
S_1 = aD.
```

Since `q(tau)=tau^2S(tau)`, the third regularized-time derivative is:

```text
q'''(0) = 6S_1.
```

Therefore the resonant amplitude is recovered by:

```text
a = <S_1,D>_m = (1/6)<q'''(0),D>_m.
```

For an ordered-Euler resonance at `lambda_n`, `n in {5,6,7}`, let `H` be the
mass-normalized centered horizontal eigenvector and let `E_{n-2}` be the
particular coefficient forced by the already selected lower coefficients
including the energy-matching `alpha`. The first resonant coefficient is:

```text
S_{n-2}=E_{n-2}+bH,
<H,H>_m=1.
```

The corresponding regularized-time jet of `q` is:

```text
q^(n)(0) = n! S_{n-2}.
```

Thus:

```text
b = <S_{n-2}-E_{n-2},H>_m
  = <q^(n)(0)/n! - E_{n-2},H>_m.
```

After the central configuration, ordering, resonant surface point, and energy
have been fixed, this projection is the missing local branch-selection
coordinate. The sign ambiguities above are exactly the sign of this finite-jet
coordinate.

## Finite-Jet Selector Gives A Local Continuation Convention

The incoming analytic-germ theorem above is stronger than what is needed to
define the outgoing local branch. Analyticity of the whole incoming germ proves
that the same germ continues by the identity theorem. For a continuation
convention, however, the local normal form only needs the finite data that enter
the branch recurrence.

Assume a one-sided zero-angular total-collision approach has a noncollision
central limiting second shape `C`, and suppose the required finite
regularized-time asymptotic coefficients exist on the incoming side:

```text
q(tau)=tau^2C+tau^3S_1+tau^4S_2+...+tau^rS_{r-2}+o(tau^r),
tau<0,
```

where `r=4` in the nonresonant cases, `r=4` for the equilateral
`beta=8/27` resonance after the cubic coefficient is read, and
`r=n in {5,6,7}` on the ordered-Euler resonance surfaces. The lower
coefficients are interpreted in the normal-form coordinates listed above:

```text
C                      central configuration and ordering,
H_in                   conserved incoming energy,
a=<S_1,D>_m            equilateral beta=8/27 cubic amplitude, when present,
b=<S_{n-2}-E_{n-2},H>_m ordered-Euler resonant amplitude, when present.
```

Here `D` is the mass-normalized equilateral cubic kernel, `H` is the
mass-normalized ordered-Euler horizontal resonant eigenvector, and `E_{n-2}` is
the particular lower-forced coefficient computed after the central
configuration, energy parameter, and earlier coefficients have been fixed. The
homothetic energy parameter is then selected by the nondegenerate scalar
formula:

```text
alpha = 9(H_in-H_base)/(10<C,C>_m).
```

These are limits of finite one-sided jets. They do not require a full
convergent incoming series. Once the numbers `C`, `H_in`, and any resonant
amplitude have been supplied, the local normal-form recurrence constructs a
unique analytic shape series `S_*(tau)` with those parameters. The projected
curve:

```text
q_*(tau)=tau^2S_*(tau),       t=tau^3,
```

solves Newton's equation for every `tau != 0` in a two-sided punctured
neighborhood and has zero angular momentum by the same collision-limit argument
as before. Its finite incoming jets match the selected incoming data, and its
energy matches `H_in`.

This is a genuine local continuation convention under finite asymptotic
hypotheses: the selector is a finite list of central-configuration, energy, and
resonant-jet projections. It is weaker than the incoming analytic-germ theorem,
because it does not assert that an arbitrary nonanalytic incoming branch equals
the selected analytic normal-form branch to all orders. It is also not yet the
full zero-angular theorem, because the global work still has to prove that
arbitrary total-collision approaches supply a collision-free central limiting
shape and the required finite resonant jets.

The finite-jet entry step is now executable. Given regularized coefficients

```text
q(tau)=sum_d C_d tau^(d+2),
```

and constructor-supplied forced reference rows, the certificate recovers each
finite selector coordinate by mass projection:

```text
s_j = <C_{d_j}-R_{d_j}, B_j>_m / <B_j,B_j>_m.
```

The selected outgoing branch is accepted only if its coefficient rows recover
the same selector list, the finite energy limit

```text
H = 1/2<S_1,S_1>_m - 1/2<S_1,DA(C)S_1>_m + (10/9)<C,S_2>_m
```

matches the incoming value, and sampled punctured states satisfy both the
cubic-time Newton residual formula and zero angular momentum. This is
implemented as `certify_finite_jet_identity_selector_entry(...)` in
`three_body_symmetry/zero_angular_entry.py`. The constructor does not prove
the existence of the finite incoming limits; it turns those finite limits,
once supplied by an entry theorem, into checked identity-selector continuation
data.

The selected outgoing coefficient list is no longer merely a test fixture.
`construct_finite_jet_selected_branch(...)` solves the finite regularized
recurrence degree by degree. At row `d` it uses

```text
(((d+2)(d-1))/9 I - DA(C_0)) C_d = known_d,
```

where `known_d` is computed from the already-solved lower rows of
`A(sum C_l tau^l)`. If selector specs occur at degree `d`, the constructor
adds the mass-orthogonal correction in their span so that every
`<C_d-R_d,B_j>_m/<B_j,B_j>_m` equals the incoming selector value. Certification
then checks two algebraic preconditions before accepting the row. First, each
selector basis must lie in the kernel of the row operator above, up to the
certificate tolerance. Second, the mass Gram matrix of selector bases at that
degree must have positive singular-value floor, so the selector coordinates are
independent. It then recomputes the coefficient recurrence residuals and
rejects nonresonant or inconsistent selector rows.
`derive_finite_jet_identity_selector_entry_from_incoming(...)`
composes this solver with the selector-coordinate recovery above, giving an
executable finite-jet identity continuation for the resonant finite-jet
subcases while still leaving arbitrary incoming finite-jet existence as a
separate theorem. This finite-jet solver is not yet a standalone analytic
chart: a new `ValidatedAtlasSolution` for general finite-jet branches would
also need a tail/convergence majorant for the omitted rows.

## Finite-Jet Identity-Selector Compact Atlas

The compact identity-selector atlas above can therefore be weakened. It does
not need the full incoming Fuchsian-log germ at every total collision; it is
enough to know the finite selector jets from the previous theorem.

Let `[a,b]` be compact, and assume the collision set in `[a,b]` is finite.
At each total-collision time `T_k`, assume the incoming side is a
finite-energy zero-angular approach with a collision-free central limiting
shape and the finite selector limits:

```text
C_k,        H_k,
a_k        on the equilateral beta=8/27 resonance, if present,
b_k        on an ordered-Euler n in {5,6,7} resonance, if present.
```

In the nonresonant cases the resonant list is empty. In the resonant cases the
amplitudes are the finite jet projections:

```text
a_k = <q'''(0)/6, D_k>_m,
b_k = <q^(n)(0)/n! - E_{k,n-2}, H_k^res>_m.
```

The energy-matching parameter is then fixed by:

```text
alpha_k = 9(H_k-H_{base,k})/(10<C_k,C_k>_m).
```

Use the identity-selector rule on these finite selector coordinates: the
outgoing side has the same `C_k`, the same `H_k`, the same `alpha_k`, and the
same resonant amplitudes. The local normal-form recurrence constructs the
unique outgoing analytic branch with those finite data. Projection by

```text
q=tau^2S_k(tau),        t=T_k+tau^3
```

solves Newton's equation on the two punctured sides, has zero angular momentum
by the collision-limit formula, and has the same finite energy `H_k`.

For each total collision choose a small `tau`-neighborhood for this selected
branch. For each separated binary collision choose its Levi-Civita chart. The
finite collision neighborhoods can be taken disjoint. The complement is a
finite union of compact collision-free intervals, so the ordinary compact
Taylor-cover lemma supplies finitely many ordinary charts. Consequently the
whole compact interval has a finite atlas:

```text
ordinary Taylor charts
+ separated-binary Levi-Civita charts
+ finite-jet identity-selector total-collision charts.
```

All verification budgets are finite sums over this finite list. The
total-collision chart residual is the selected normal-form residual

```text
tau^2S''+2tau S'-2S-9A(S),
```

and projection verifies Newton's equation away from `tau=0`. This closes the
compact zero-angular identity-selector composition under finite selector-jet
hypotheses. It still does not prove that arbitrary zero-angular total
collapses have those finite selector limits, nor that arbitrary all-time
branches have only finitely many such events.

The theorem pipeline now exposes exactly this scoped composition as
`construct_zero_angular_compact_finite_atlas_with_selector(...)`. The
constructor consumes a proof-certified finite `ValidatedAtlasSolution` plus
constructor-derived selector-entry certificates, requires explicit selector
coverage for every total-collision chart, and rejects atlases whose collision
policy does not name selector continuation. Thus a compact zero-angular
finite-events branch can enter the shared `GeneralSolutionTheoremCertificate`
surface without a hand-supplied global boolean. The remaining analytic gap is
unchanged: one still needs an entry theorem proving that arbitrary incoming
zero-angular total-collision branches supply the finite selector data, and a
global classification theorem proving that an arbitrary all-time branch falls
into one of the certified regimes.

The first native total-collision `ValidatedAtlasSolution` chart is the exact
parabolic homothetic case. If `A(Q)=-(2/9)Q`, then

```text
q(tau)=tau^2 Q,        t=T+tau^3
```

gives

```text
d^2q/dt^2 = -(2/9)Q/tau^4 = A(tau^2 Q)
```

on both punctured sides. When `Q` is mass-centered, the same chart has zero
center of mass, zero linear momentum, zero angular momentum, and zero energy.
The adapter
`validated_atlas_from_parabolic_homothetic_total_collision_branch(...)` records
this as a finite-jet identity-selector total-collision `ValidatedChart` with
zero omitted tail and an explicit selector collision policy. It refuses the
nonzero-energy homothetic family because that family needs the analytic
`u(tau^2)` scalar tail bound before it can honestly be treated as a zero-tail
finite chart.

That scalar tail is now constructor-backed for the homothetic family.  Write
`z=tau^2` and

```text
q(tau)=tau^2 u(z)Q,        u(0)=1.
```

The energy identity is:

```text
(u+z u')^2 = 1/u + alpha z,        alpha=(9/2)E/I.
```

Equivalently, the collision-time integral gives an analytic equation
`H(u,z)=0` with linear part `(3/2)(u-1)` at `(u,z)=(1,0)`. The constructor
`certify_homothetic_total_collision_scalar_majorant(...)` checks a Rouche
inequality on `|u-1|=r`:

```text
(3/2)r > base_nonlinear_bound + energy_perturbation_bound.
```

When this holds, a unique analytic scalar branch exists for `|z|<R`, with
Cauchy majorant `1+r`, giving an explicit omitted-tail bound for
`u(tau^2)`. The adapter
`validated_atlas_from_homothetic_total_collision_branch(...)` now promotes this
majorant into a nonzero-energy total-collision `ValidatedAtlasSolution` with
inflated initial/target state intervals, Newton residual and invariant ledgers,
tail budget, target-time containment, and explicit selector collision policy.

The matching theorem constructor
`construct_zero_angular_parabolic_homothetic_total_collision_atlas(...)` now
performs the whole scoped proof-pipeline assembly. It derives the initial
noncollision state from the incoming punctured side, builds the native
homothetic total-collision `ValidatedAtlasSolution`, recovers the finite
homothetic selector coordinate from the branch coefficients, and feeds both
certificates into the compact zero-angular finite-events-with-selector regime.
Thus this exact subcase no longer needs a hand-supplied selector envelope. It
still leaves the arbitrary zero-angular entry/classification theorem as a
separate obligation.

## Automatic Three-Body Selector Compact Atlas For Finite Event Sets

For positive-mass three-body motion, the selector limits are not an additional
local hypothesis once the earlier entry theorems apply. They are a consequence
of the total-collision entry analysis.

Let `[a,b]` be compact. Assume the branch has finite energy, zero centered
angular momentum, finitely many total collisions in `[a,b]`, and that any
binary collisions away from total collision are separated-third-body
Levi-Civita events. At each total-collision time, translate the event to
`t=0`. The binary-degenerate Jacobi theorem rules out approach to a binary
shape stratum. Hence the normalized shape is eventually in a compact
collision-free region. The McGehee monotonicity and finite central-target
classification give convergence in quotient shape to a Lagrange or ordered
Euler target. The reduced-hyperbolicity theorem for all positive-mass
three-body collision-free central targets gives finite reduced shape length and
an oriented normalized-shape limit. Finally, the Poincare-Dulac stable
normal-form theorem gives a finite Fuchsian-log selector expansion on the
incoming side.

Thus each total collision automatically supplies the data needed by the local
identity-selector rule:

```text
C, alpha, b_1, ..., b_N
```

where `C` is the oriented central target, `alpha` is the energy/scale
coordinate fixed by the finite energy, and the finite list `b_j` contains the
ordinary Fuchsian amplitudes and constant log-row selectors. The identity rule
sets the outgoing list equal to the incoming list. The triangular
Fuchsian-log recurrence constructs the outgoing branch, and projection by
`q=tau^2S(tau)`, `t=T+tau^3` verifies Newton's equation away from `tau=0`,
zero angular momentum, and matching finite energy.

Since the total-collision set is finite, choose disjoint Fuchsian-log
neighborhoods for those selected branches. Choose disjoint Levi-Civita
neighborhoods for the finitely many separated binary events. The remaining
collision-free complement is a finite union of compact intervals, and the
ordinary compact Taylor-cover lemma covers it by finitely many ordinary charts.
Therefore the compact interval has a finite automatically selected atlas:

```text
ordinary Taylor charts
+ separated-binary Levi-Civita charts
+ automatic identity-selector Fuchsian-log total-collision charts.
```

The only hypotheses not supplied by local total-collision analysis are global
ones: that the compact interval contains only finitely many collision events,
that separated binary events satisfy the Levi-Civita separation assumptions,
and that the branch has the finite-energy zero-angular total-collision
semantics being continued. This theorem removes "supply the selector limits"
as a separate local assumption for positive-mass three-body total collisions;
it does not prove all-time finiteness of such events or arbitrary-data regime
classification.

## Resonant Branches Are Not Selected By Energy

These resonant families show that the missing continuation datum is not
recoverable from the usual collision invariants. For the symmetric Euler
branches, the resonant coefficient enters at `S_r` with `r>=3`. The finite
energy limit depends only on `S_0`, `S_1`, and `S_2`. Since these branches have:

```text
S_0 = C,
S_1 = 0,
S_2 = alpha C,
```

their energy is:

```text
E = (10/9) alpha sum_i m_i |C_i|^2,
```

independent of the resonant amplitude `b`. The sign choices `+b` and `-b`
therefore have the same collapsed collision point, the same mass-centered
second jet, the same homothetic energy parameter, zero angular momentum, and
the same energy, but different outgoing regularized curves.

For the equilateral `beta=8/27` branch the resonant coefficient enters earlier,
at `S_1`; the finite energy can depend on `b^2`. Even there, the two sign
choices `+b` and `-b` have the same energy while producing different outgoing
curves. The pair-force recurrence is odd/even symmetric under `b -> -b`:
the coefficients with odd index change sign and the coefficients with even
index do not. The second jet `S_0=C` and quartic energy data are therefore the
same for both signs.

Thus neither energy nor the second regularized-time jet removes the need for a
resonant branch-selection parameter.

## Proof

For the exact homothetic branch and `tau != 0`, set `t = tau^3`. Since:

```text
dt/dtau = 3 tau^2,
q_i(tau) = C_i tau^2,
```

we get:

```text
dq_i/dt
  = (dq_i/dtau) / (dt/dtau)
  = (2 C_i tau) / (3 tau^2)
  = (2/3) C_i / tau.
```

Differentiating once more with respect to physical time:

```text
d^2 q_i/dt^2
  = (1 / (3 tau^2)) d/dtau ((2/3) C_i tau^-1)
  = -(2/9) C_i tau^-4.
```

The Newtonian acceleration is homogeneous of degree `-2`, so:

```text
A_i(q(tau)) = A_i(C tau^2) = tau^-4 A_i(C).
```

Therefore `q'' = A(q)` for all `tau != 0` if and only if:

```text
A_i(C) = -(2/9) C_i.
```

Since `J = 2C`, homogeneity also gives:

```text
A(J) = A(2C) = (1/4) A(C) = -(1/18) C = -(1/36) J.
```

This proves that the second `tau`-jet must be a scaled central configuration.

## Consequence

Both the equal-mass equilateral and Euler central configurations can be scaled
to satisfy `A(C)=-(2/9)C`. Their regularized collision curves have:

```text
q(0) = 0,
dq/dtau(0) = 0,
```

but different values of:

```text
d^2q/dtau^2(0).
```

So a continuation theorem that only says "continue through total collision" is
not yet mathematical data. The missing branch selector can be stated precisely:
it is a normalized second regularized-time jet satisfying the central
configuration equation above, together with whatever convention chooses among
the admissible central-configuration branches.

This does not close the full zero-angular-momentum triple-collision theorem.
It closes a smaller analytic obligation: it identifies the exact local datum
that the continuation convention must provide for the parabolic homothetic
total-collision branches.

The closed-form witness layer now recognizes this as the
`regularized_second_jet_branch` convention. That convention is certified only
when the regularized time parameter, terminal collision value, and branch
selection rule are each supplied; merely naming the convention is not enough.
