# A Validated Planar Binary-Collision Passage

## Scope

This note records the strongest presently defensible result in this repository:
a computer-checked, local, two-sided continuation of one specified planar
three-body solution through an isolated binary collision. The continuation is
selected by a single analytic Levi-Civita (LC) initial-value problem and is
matched on both punctured sides to validated ordinary Newtonian solution
tubes.

This is not a closed form for the three-body problem, a global continuation
theorem, an atlas-completeness result, or a claim about arbitrary initial data.

## Conditional computer-assisted theorem

Fix positive masses and a selected pair \((i,j)\). Let a serialized polynomial
\(\bar X(s)\) approximate the 14-dimensional lifted LC state

\[
 X=(z,w,h,R,U,y,V,t), \qquad \frac{dt}{ds}=\rho=|z|^2,
\]

on \([s_-,s_+]\), where \(s_-<0<s_+\). Suppose the certificate checker
accepts:

1. an LC a-posteriori tube centered at \(\bar X\), anchored with zero error at
   \(s=0\);
2. the exact collision-anchor obligations at \(s=0\);
3. positive-\(\rho\) projected endpoint enclosures at \(s_-\) and \(s_+\); and
4. weighted ordinary Newtonian tubes containing the complete projected
   endpoint enclosures.

Assume the interval arithmetic and interval automatic differentiation used by
the checker are inclusion isotonic and outward rounded. Then there is a unique
analytic lifted LC solution \(X(s)\) throughout \([s_-,s_+]\), with an isolated
selected-pair collision at \(s=0\). On each punctured side its LC projection is
a classical solution of the planar Newtonian three-body equations. The two
classical branches, and the accepted ordinary tubes that continue them, are
projections of the same exact lifted IVP. Physical time is strictly ordered
across the collision. Thus the accepted data certify one local generalized
binary-collision continuation in the standard LC sense.

The physical trajectory is classical only away from the collision instant;
the analytic lifted solution selects its continuation through that instant.

## Certificate hypotheses checked

The checker does not infer collision passage from overlapping numerical
boxes. Its gates include the following.

- Identifiers, masses, planar dimension, charts, tubes, and endpoint
  parameters must agree; the declared parameters must strictly straddle the
  collision anchor.
- The binary64 computations of \(M=m_i+m_j\),
  \(\alpha=m_j/M\), \(\beta=m_i/M\), and \(m_k/M\) must equal their exact
  rational values for the serialized binary masses. This prevents rounded
  center-of-mass coefficients from being silently identified with a different
  Newtonian mass system. The present fixture satisfies this restrictive gate;
  a future generic implementation should interval-enclose these ratios.
- The LC tube directly bounds the defect \(\bar X'-F(\bar X)\), encloses the
  Jacobian \(DF\) by interval forward automatic differentiation, proves both
  third-body denominators stay positive, and satisfies a strict Gronwall
  bootstrap inside its declared radius.
- The collision parameter equals the tube anchor and the anchor uncertainty is
  exactly zero. Exact rational evaluation of the serialized binary-float
  coefficients must give

  \[
  z(0)=0,\qquad 2|w(0)|^2=m_i+m_j.
  \]

  Hence \(w(0)\ne0\), the zero of \(z\) is simple, and the collision is
  isolated. A merely small constraint residual is rejected.
- At both endpoint parameters, interval projection of the complete LC
  enclosure must have a positive lower bound for \(\rho\), and the projected
  position and velocity boxes must fit inside the respective ordinary-tube
  anchor allowances.
- Each ordinary tube independently bounds its Newtonian defect and Lipschitz
  constant. Separate position and velocity scales are used because LC
  projection near collision produces errors of very different sizes in these
  blocks.

## Proof outline

The regularized vector field is analytic at binary collision provided the
third body remains separated: its only denominators are the two third-body
distances. Direct interval residual and Jacobian bounds, followed by a
first-exit/Gronwall argument, produce a unique exact lifted solution in the
accepted tube.

At the zero-error anchor, exact rational identities impose the pair-energy
constraint

\[
 C=2|w|^2-(m_i+m_j)-|z|^2h=0.
\]

The implemented field preserves \(C\). Since \(z(0)=0\) and
\(|w(0)|^2=(m_i+m_j)/2>0\), analyticity gives a simple isolated zero, while

\[
 t(s)-t(0)\sim \frac{|w(0)|^2}{3}s^3.
\]

Consequently physical time has the same strict ordering as the selected
parameters across the collision.

On \(\rho>0\), put

\[
 q=(z_1^2-z_2^2,2z_1z_2),\qquad v=\frac{L(z)w}{\rho}.
\]

The LC algebra and \(C=0\) reduce the projected relative acceleration to
\(- (m_i+m_j)q/|q|^3+P\); the remaining lifted equations reconstruct the other
Newton equations. Seven exact symbolic identities supporting this reduction
are checked separately. Endpoint containment then binds both ordinary
solutions to this one exact lifted branch, rather than merely showing that
unrelated enclosures intersect.

For an ordinary tube with position and velocity radii \(R_q,R_v\), the checker
works in

\[
 \|(\Delta q,\Delta v)\|_W=
 \max\{\|\Delta q\|_\infty/R_q,\|\Delta v\|_\infty/R_v\}.
\]

Its accepted bound is the usual scaled defect/Lipschitz Gronwall bound, with a
strict normalized error below one.

## Reproduction

From the repository root, run the production example:

```bash
python scripts/certify_planar_binary_collision_passage.py
```

Verify the exact LC projection algebra and focused regression:

```bash
python scripts/verify_lc_projection_identities.py
python -m pytest -q tests/test_certificate_checker.py \
  -k 'planar_lc or weighted_ordinary or two_sided'
```

The current fixture uses masses \((1,1,1.2)\), pair \((0,1)\), LC endpoints
\(s_\pm=\pm10^{-2}\), \(z(0)=0\), \(w(0)=(1,0)\), and a separated third body.
The production script currently reports:

```json
{"collision_certified":true,"passage_certified":true,
 "rho_floors":{"left":9.995979933166368e-05,
               "right":9.995979933166368e-05},
 "projection_gaps":{"left":0.06052842871002895,
                    "right":0.06052842871002895},
 "weighted_tube_errors":{
   "left":{"position":5.003127995873451e-06,
           "velocity":0.500312799587345},
   "right":{"position":5.003127995873451e-06,
            "velocity":0.500312799587345}}}
```

These are rigorous-enclosure outputs conditional on the trusted kernel, not
estimates of truncation error inferred from sampling.

## Trusted kernel and limitations

The mathematical certificate architecture recomputes direct interval defects,
Jacobian row-sum bounds, denominator floors, exact rational anchor identities,
endpoint containment, and Gronwall inequalities. Its present trusted kernel is
the Python/NumPy binary64 interval implementation, directed `nextafter`
inflations, elementary interval operations, high-precision Decimal exponential
upper bounds, interval automatic differentiation, and the semantics of the
serialized checker code. An independent audit or migration to a small verified
interval library is required before treating the executable alone as a fully
foundational proof artifact.

The fixture is deliberately local and planar. It neither covers spatial KS
regularization nor simultaneous/total collision, proves no global atlas
termination or completeness, and does not establish a general collision
classification. Its potential mathematical value is narrower but real: it is
an explicit, falsifiable certificate format that joins an analytic
regularization, exact collision anchoring, punctured projection equivalence,
anisotropic Newtonian validation, and same-IVP branch identity in one
reproducible passage. A publishable next step would replace the trusted
interval layer, generate nontrivial families of independently reproducible
certificates, and compare the construction with established validated
regularization methods.
