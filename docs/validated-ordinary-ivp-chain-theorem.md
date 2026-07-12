# Validated Ordinary IVP Chain Theorem

## Scope

This note states the first exact-trajectory theorem implemented by the revised
certificate kernel.  It concerns collision-free ordinary Newtonian charts
only.  It does not regularize binary collision, prove that a constructor will
find a chain, or establish the earlier global closed-form claim.

All norms below are infinity norms.  Serialized inputs are finite binary64
numbers. IVP and handoff polynomial values are evaluated as exact rationals of
those serialized floats. Lipschitz arithmetic uses exact rational
intermediates, and the exponential uses 80-digit upward-rounded decimal
evaluation before outward binary64 conversion. The remaining interval
operations still require a separately audited directed-rounding backend for
publication.

## Newtonian phase equation

For positive masses \(m_1,m_2,m_3\) in dimension \(d\in\{2,3\}\), write

\[
 y=(q,v),\qquad y'=F(y)=(v,a(q)),
\]

where

\[
 a_i(q)=\sum_{j\ne i}m_j\frac{q_j-q_i}{|q_j-q_i|^3}.
\]

Let \(p(t)=(Q(t),V(t))\) be a finite polynomial model on an interval \(I\),
and let

\[
 \delta=\sup_{t\in I}|p'(t)-F(p(t))|_\infty.
\]

The checker bounds this expression by direct interval evaluation of the
polynomial and the Newton vector field.  A coefficient recurrence residual is
not used as a substitute for \(\delta\).

## Collision-free tube and Lipschitz bound

Suppose interval evaluation proves that every nominal pair separation has
Euclidean lower bound \(\rho_0\).  A phase-space infinity tube of radius
\(R\) changes either endpoint of a pair vector by at most \(R\) per
coordinate.  Therefore every pair separation throughout the tube is at least

\[
 \rho=\rho_0-2\sqrt d\,R.
\]

The checker requires \(\rho>0\).

For \(g(x)=x/|x|^3\),

\[
 Dg(x)=|x|^{-3}I-3|x|^{-5}xx^T.
\]

The infinity operator norm obeys

\[
 \|Dg(x)\|_\infty
 \leq \frac{1+3\sqrt d}{|x|^3}.
\]

Accounting for perturbations of both bodies in each pair gives the valid
phase-vector-field Lipschitz bound

\[
 L=\max\left\{1,
 \max_i 2\sum_{j\ne i}m_j
 \frac{1+3\sqrt d}{\rho^3}\right\}.
\]

The checker recomputes this bound; the certificate's Lipschitz value is only
an admissible upper cap.

## Local enclosure lemma

Fix an anchor \(t_a\in I\).  Let an exact initial state \(y_a\) satisfy

\[
 |y_a-p(t_a)|_\infty\leq\varepsilon.
\]

For

\[
 h=\max_{t\in I}|t-t_a|,
\]

define

\[
 E=e^{Lh}\varepsilon+\delta\frac{e^{Lh}-1}{L}.
\]

Here \(L\geq1\), as enforced by the checker.  If \(E<R\), the unique
Newtonian solution with initial value \(y_a\) exists throughout \(I\) and
satisfies

\[
 \sup_{t\in I}|y(t)-p(t)|_\infty\leq E.
\]

### Proof

Local existence and uniqueness hold while the solution remains in the
collision-free tube.  On any maximal subinterval on which it does, variation
of constants and the Lipschitz estimate give

\[
 |y(t)-p(t)|_\infty
 \leq \varepsilon+\delta|t-t_a|
 +L\int_{t_a}^{t}|y(s)-p(s)|_\infty\,ds.
\]

Grönwall's inequality gives the stated bound \(E\).  If the solution left the
closed radius-\(R\) tube before reaching an endpoint of \(I\), continuity and
the strict inequality would contradict the first exit.  Because the closed
tube remains a positive distance from collision,
the vector field is smooth and continuation reaches both endpoints.  Uniqueness
holds on the tube.  ∎

## Enclosure-aware transition lemma

Let a source chart enclose an exact solution at a common handoff time \(t_h\)
with error at most \(E_s\).  Let the source and target polynomial states at
that time differ by at most \(G\).  Then

\[
 |y(t_h)-p_{\rm target}(t_h)|_\infty\leq E_s+G.
\]

Consequently, if the target tube's anchor allowance satisfies

\[
 \varepsilon_{\rm target}\geq E_s+G,
\]

the local enclosure lemma applied with exact initial value \(y(t_h)\) produces
a target-chart solution.  Local uniqueness identifies it with the continuation
of the source branch.  This is the inequality recomputed by
`check_ordinary_enclosure_transition(...)`.  Arbitrary finite transition
tolerances play no role.

## Finite-chain theorem

Consider a finite ordered family of ordinary polynomial charts
\(p_0,\ldots,p_N\), with:

1. a checked IVP binding for \(p_0\);
2. a checked collision-free a-posteriori tube for every \(p_k\);
3. a checked enclosure-aware transition from \(p_k\) to \(p_{k+1}\);
4. common masses and dimension;
5. forward physical-time images whose frontiers do not retreat and whose
   consecutive intervals touch or overlap; and
6. a requested interval from the IVP time contained in their union.

Then there exists a unique Newtonian solution of the bound IVP throughout the
requested interval, and on each chart it lies in that chart's checked tube.

### Proof

The local enclosure lemma proves the claim on the first chart.  Apply the
transition lemma and local enclosure lemma successively.  Induction shows that
every chart encloses the same exact branch.  The ordered, gap-free physical
images cover the requested interval.  Uniqueness at every handoff makes the
assembled branch the unique IVP solution.  ∎

The executable counterpart is
`check_validated_ordinary_ivp_chain(...)`.  Its result property
`exact_ivp_enclosure_certified` refers only to this finite ordinary theorem.
For the requested endpoint it also evaluates the selected chart polynomial as
an exact rational function of serialized binary-float data and returns
componentwise position and velocity intervals enlarged by the proven uniform
tube error. `target_state_enclosure_certified` is true only when that output is
present and the whole chain is certified.

## Remaining trust and extension obligations

- Audit all outward rounding, especially `exp`, as a proof-grade numerical
  backend rather than relying only on local `nextafter` calls.
- Canonically serialize and hash the IVP, charts, tubes, and transitions.
- Make validated-chain acceptance a distinct API from the legacy
  bounded-defect polynomial bundle verifier.
- Compute final target-state interval output explicitly.
- Extend the local enclosure lemma to the lifted LC and KS vector fields.
- Prove projection and collision-passage compatibility for regularized tubes.
- Do not infer constructor completeness from this supplied-chain theorem.
