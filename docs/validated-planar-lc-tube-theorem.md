# Validated Planar Levi-Civita Tube Theorem

## Statement implemented now

Fix positive masses, a selected planar pair, and a finite serialized polynomial
in the lifted variables

\[
 X=(z,w,h,R,U,y,V,t)\in\mathbb R^{14}.
\]

Here \(z,w,R,U,y,V\in\mathbb R^2\), \(h,t\in\mathbb R\), the selected
relative position is \(Q(z)=(z_1^2-z_2^2,2z_1z_2)\), and
\(dt/ds=\rho=|z|^2\).

Assuming the interval operations and interval automatic differentiation used
by the checker are outward-rounded, acceptance has the following uniform
meaning. For every exact anchor value \(x_a\) in the certified anchor ball,
the analytic regularized LC IVP \(X(a)=x_a\) has a unique solution on the
certified parameter interval, and its infinity-norm distance from the
polynomial is bounded by `gronwall_error_bound`. This conclusion remains valid
when \(z=0\).

It is a lifted-ODE theorem.  By itself it does not yet assert that the lifted
solution lies on the physical pair-energy constraint, corresponds to a
specified Newtonian IVP, or has a finite physical-velocity projection at the
collision parameter.

When `require_pair_energy_constraint=True`, the checker additionally evaluates

\[
 2|w|^2-(m_i+m_j)-|z|^2h
\]

at the polynomial anchor using exact rational arithmetic over the serialized
binary-float coefficients and requires it to vanish exactly. A numerically
small residual is deliberately rejected. This constrains the enclosed exact
solution only when its anchor is that polynomial center (in particular, when
the certified initial-error allowance is zero). With a positive anchor ball,
off-constraint exact anchors are also enclosed, so the standalone tube is not
promoted to a constrained family. The result property
`constrained_newtonian_lift_certified` enforces this distinction.

The constraint is invariant under the displayed lifted field. Indeed, with
\(C=2|w|^2-M-\rho h\) and \(\rho'=2z\cdot w\),

\[
 C'=4w\cdot w'-\rho'h-\rho h'
 =2h z\cdot w+\rho(Lw)\cdot P
 -2h z\cdot w-\rho(Lw)\cdot P=0.
\]

## Punctured LC/Newton projection-equivalence lemma

The following is a conventional algebraic lemma for the exact vector field
displayed below; `scripts/verify_lc_projection_identities.py` independently
reduces its polynomial identities to zero over the rationals.

Let \(X(s)\) be a constrained lifted solution, let \(\rho>0\), and use physical
time \(dt/ds=\rho\). Put

\[
 q=Q(z),\qquad v=q_t=\frac{L(z)w}{\rho}.
\]

The identities

\[
 L^TL=LL^T=4\rho I,\quad Lz=2q,
 \quad \rho L(w)w-\rho_sL(z)w=-2|w|^2q
\]

and the lifted equation for \(w_s\) give

\[
 q_{tt}=\frac{(-2|w|^2+\rho h)q}{\rho^3}+P.
\]

On the invariant constraint
\(C=2|w|^2-M-\rho h=0\), this is precisely

\[
 q_{tt}=-\frac{M q}{|q|^3}+P.
\]

The constraint is essential: off it the effective singular coefficient is
\(-(M+C)q/\rho^3\). Meanwhile \(R_s=\rho U\), \(U_s=\rho B\),
\(y_s=\rho V\), and \(V_s=\rho Y\) yield
\(R_t=U,R_{tt}=B,y_t=V,y_{tt}=Y\). Substitution of the definitions of
\(B,Y,P\) then reconstructs all three Newton equations. Therefore every
constrained lifted solution projects on each punctured component to a
classical Newtonian solution, provided the two third-body denominators remain
nonzero.

Conversely, a punctured Newtonian solution together with a continuous local
choice of the square root \(Q(z)=q\), and the displayed definitions of
\(w,h,R,U,y,V\), satisfies the lifted equations. This converse is local to a
chosen LC branch.

This lemma proves punctured equivalence, not collision occurrence for an
arbitrary tube. A validated generalized collision continuation must separately
identify a parameter \(s_*\) where the selected exact solution has \(z(s_*)=0\)
and prove entry/exit ordering. If \(C=0\) at such a point, then
\(2|w(s_*)|^2=M>0\), so the zero is simple and isolated, and
\(t-t_*\sim |w(s_*)|^2(s-s_*)^3/3\).

### Implemented exact collision anchor

`check_planar_lc_exact_collision_anchor(...)` supplies that missing fact for a
specific lifted IVP. It requires the collision parameter to equal the tube
anchor, the initial-error allowance to be exactly zero, and exact rational
evaluation of the serialized anchor to give

\[
 z(s_*)=0,\qquad 2|w(s_*)|^2=M,
\]

with positive masses and a certified separated-third-body LC tube. The tube
existence theorem then applies to the unique exact IVP whose anchor is exactly
that serialized state. Since \(|w(s_*)|^2=M/2>0\), the exact solution has a
simple, isolated selected-pair collision at \(s_*\); physical time is strictly
increasing locally across it and has the stated cubic leading behavior.

The zero-error condition is indispensable. A positive ball centered at
\(z=0\) includes exact anchors with \(z\ne0\), so neither collision occurrence
nor its time can be inferred merely from tube containment.

## Weighted ordinary endpoint tubes

Near collision, LC projection naturally gives position error much smaller than
velocity error. Applying one scalar state radius to both blocks can falsely
destroy the collision-free position domain. The endpoint checker therefore
uses the weighted norm

\[
 \|(\Delta q,\Delta v)\|_W=
 \max\{\|\Delta q\|_\infty/R_q,\|\Delta v\|_\infty/R_v\}.
\]

If \(A\) bounds the infinity norm of the Newton acceleration Jacobian on the
position tube, the full first-order field has weighted Lipschitz bound

\[
 L_W=\max\{R_v/R_q,\; A R_q/R_v\}.
\]

For block defects \(\delta_q,\delta_v\) and anchor errors
\(\varepsilon_q,\varepsilon_v\), define

\[
 \beta=\max\{\delta_q/R_q,\delta_v/R_v\},\qquad
 \eta_0=\max\{\varepsilon_q/R_q,\varepsilon_v/R_v\}.
\]

`check_weighted_ordinary_aposteriori_tube(...)` recomputes these quantities,
requires the position tube alone to remain collision free, and accepts only if

\[
 E_W=e^{L_Wh}\eta_0+\beta\frac{e^{L_Wh}-1}{L_W}<1.
\]

It then proves position error at most \(R_qE_W\) and velocity error at most
\(R_vE_W\). This is the same first-exit/Gronwall theorem as the scalar checker
under a diagonal change of norm; equal radii recover the scalar case.

## Implemented two-sided generalized binary-collision passage

`check_planar_lc_two_sided_collision_passage(...)` composes:

1. one certified exact collision anchor;
2. two source parameters strictly ordered on opposite sides of it;
3. positive \(\rho\) enclosures at both endpoints;
4. the punctured LC/Newton projection-equivalence lemma; and
5. complete containment of each projected endpoint enclosure in a certified
   elapsed-time ordinary tube, using separate position and velocity allowances.

Both ordinary branches therefore belong to the same unique exact lifted IVP,
not merely to overlapping state boxes. The exact collision anchor makes
\(z\) a nonzero analytic function, so \(\int |z|^2ds>0\) on every
nondegenerate subinterval. Thus the left endpoint, collision, and right
endpoint have the same strict order in physical time even when coarse numeric
time enclosures overlap.

The regression fixture supplies positive masses, a separated third body, an
exact constrained collision state, a certified LC tube across the collision,
and weighted ordinary tubes on both punctured sides. Its accepted result is a
validated local planar generalized binary-collision continuation for that
supplied IVP. This is a local constructive theorem, not a proof that arbitrary
three-body initial data encounter a binary collision or that a global atlas is
complete.

## Regularized vector field

Let the selected pair be \((i,j)\), the remaining body be \(k\),
\(M=m_i+m_j\), \(\alpha=m_j/M\), and \(\beta=m_i/M\).  Define

\[
 d_i=y+\alpha Q(z),\qquad d_j=y-\beta Q(z),
 \qquad f_i=\frac{d_i}{|d_i|^3},\quad
 f_j=\frac{d_j}{|d_j|^3}.
\]

The analytic accelerations used by the checker are

\[
 B=\frac{m_k}{M}(m_i f_i+m_j f_j),
 \quad A_k=-m_i f_i-m_j f_j,
 \quad Y=A_k-B,
 \quad P=m_k(f_j-f_i).
\]

With

\[
 L(z)=2\begin{pmatrix}z_1&-z_2\\z_2&z_1\end{pmatrix},
\]

the lifted equations are

\[
\begin{aligned}
z'&=w,\\
w'&=\tfrac12hz+\tfrac14\rho L(z)^TP,\\
h'&=(L(z)w)\cdot P,\\
R'&=\rho U, & U'&=\rho B,\\
y'&=\rho V, & V'&=\rho Y,\\
t'&=\rho.
\end{aligned}
\]

There is no division by \(z\) or \(\rho\).  The only denominators are
\(|d_i|^3\) and \(|d_j|^3\), so the vector field is analytic at binary
collision whenever the third body remains separated.

## Checker argument

The checker performs interval polynomial evaluation of \(X\) and \(X'\).  It
evaluates the displayed vector field with interval forward automatic
differentiation.  One pass therefore gives both:

- a direct interval bound \(\delta\) on \(X'-F(X)\); and
- every entry of an interval enclosure of \(DF\) on the radius-\(r\) tube.

The infinity operator norm bound is the maximum interval absolute row sum.
The same evaluation proves positive lower bounds on \(|d_i|\) and \(|d_j|\)
throughout the inflated tube.  No punctured \(\rho>0\) assumption is used.

For anchor error \(\varepsilon\), half-width \(\ell\), and Lipschitz bound
\(K\), the checker evaluates

\[
 E=e^{K\ell}\varepsilon+\delta\frac{e^{K\ell}-1}{K}
\]

(with the continuous \(K=0\) case) and requires the strict bootstrap condition
\(E<r\).  The proof is the same first-exit/Grönwall argument as in
`validated-ordinary-ivp-chain-theorem.md`, now applied to the analytic lifted
field.

## Why this checker differs from the legacy LC checker

The legacy `check_planar_levi_civita_binary_chart(...)` combines lifted
regularized checks with physical Newton residual checks on a slab where
\(\rho\) has a positive lower bound.  It therefore cannot accept an interval
containing \(z=0\), because physical velocity projection contains division by
\(\rho\).

The new tube checker deliberately separates the claims:

- lifted analytic existence is checked on the full tube, including \(z=0\);
- physical position and time projection can be checked through collision;
- physical velocities and Newton residuals are checked only on punctured
  subintervals with \(\rho>0\).

Tests include an exact-collision polynomial that the legacy projected checker
rejects but the lifted tube accepts, and a corrupted tube whose third-body
offset reaches a force denominator; the latter is rejected.

## Obligations before Newtonian continuation can be claimed

### Implemented ordinary-to-LC entry

`check_ordinary_to_planar_lc_enclosure_transition(...)` now provides a
conservative first implementation of the entry obligation away from selected
binary collision. It:

1. evaluates the source ordinary polynomial exactly at the handoff;
2. enlarges it by the source chart's proven exact-solution error;
3. lifts the whole physical state box through the finite principal LC branch
   atlas;
4. recomputes \(z,w,h,R,U,y,V\), rather than trusting supplied lift values;
5. requires every possible branch box to lie inside the target LC tube's
   anchor allowance; and
6. evaluates both physical times exactly over the serialized coefficients and
   requires their discrepancy to be no larger than the declared
   `max_time_gap` (exact equality requires a zero cap).

It also requires the binary64 values actually used for \(M,\alpha,\beta\), and
(m_k/M) to equal the corresponding exact rational expressions in the
serialized binary masses. This is restrictive but prevents a rounded LC field
from being identified with Newton's equations for a nearby mass system.
Interval-enclosing these coefficients is the preferred future generalization.

For every exact physical state in the source enclosure, the lift formulas

\[
 Q(z)=q,\qquad w=\tfrac14L(z)^Tv,
 \qquad h=\tfrac12|v|^2-M/|q|
\]

give a lifted state satisfying the pair-energy constraint identically. Thus
the target LC tube contains constrained candidate anchors lifted from the
source enclosure. Identifying their LC evolution with the same Newtonian
branch additionally requires the punctured LC/Newton projection-equivalence
lemma listed below. Requiring
the entire interval lift atlas to fit is deliberately strong; a future
parametric Krawczyk partition can certify smaller branch images without
weakening the theorem.

The transition is impossible at \(q=0\), as it should be: physical velocity
does not determine an LC lift there. Entry occurs on a punctured noncollision
slab before the chart crosses \(z=0\).

The checker also records a positive lower bound for \(|z|^2\) on the entry
lift box. Therefore the exact analytic lifted solution has \(z\not\equiv0\).
Since

\[
 t(s_2)-t(s_1)=\int_{s_1}^{s_2}|z(s)|^2\,ds,
\]

and the zeros of a nonzero analytic \(z\) are isolated, this integral is
strictly positive whenever \(s_2>s_1\). Thus physical time is strictly
increasing through an isolated binary collision even though \(t'=0\) at the
collision parameter. The transition result exposes this conclusion as
`physical_time_strictly_monotone_certified`.

### Implemented punctured LC-to-ordinary exit

`check_planar_lc_to_ordinary_enclosure_transition(...)` evaluates the LC tube
at a declared punctured exit, requires a positive lower bound for \(\rho\),
projects the complete lifted enclosure into physical position and velocity,
and requires an elapsed-time ordinary tube to contain that projection. It also
returns an enclosure of the absolute physical-time origin. Its safe conclusion
is `projected_exit_enclosure_certified`; it does not by itself certify a
Newtonian collision continuation.

### Status and remaining improvements

The production two-sided passage checker now implements the punctured
projection/equivalence identities, endpoint time ordering, collision
isolation, and explicit LC generalized-continuation semantics. The remaining
items are improvements to strength or trusted-base assurance, not missing
logical links in the currently stated conditional theorem:

1. Replace the conservative interval square-root atlas with optional exact-
   dyadic parametric Krawczyk branch records for practical larger boxes.
2. Add a certified physical-time image/inversion procedure beyond the current
   endpoint ordering and strict-monotonicity result.
3. Audit interval automatic differentiation and outward rounding as part of
   the trusted kernel; until then their soundness is an explicit theorem
   hypothesis, not a conclusion of the Python implementation alone.

Subject to that explicit trusted-kernel hypothesis, the accepted
ordinary--LC--ordinary chain is a validated local Newtonian binary-collision
continuation in the LC generalized sense; it is not a classical solution at
the collision instant itself.
