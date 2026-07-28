# Proof-grade supplied planar-chain theorem

## Status and scope

This note isolates the mathematical theorem that can be supported by the
direct-defect tube, binding, entry, exit, and containment checks already present
in the independent Rust replay. It deliberately does not use a serialized
Taylor `tail_bound`, a coefficient-recurrence ledger, or a claimed chart
residual as evidence that an exact solution exists near a chart polynomial.
The finite polynomial itself is the approximate trajectory; its directly
recomputed ODE defect is the relevant evidence.

The strongest conclusion is conditional soundness for a **supplied, admitted,
finite planar chain**. If every proof-grade obligation below succeeds, the
chain carries one exact branch from the bound point IVP through all accepted
ordinary pieces and regularized planar binary-collision passages. At a terminal
physical time covered by the final chart, the replayed terminal interval box
contains that branch's state. This is not a theorem that a producer finds such
a chain for every IVP or target time.

The raw-v1 wire language and frozen compatibility behavior remain specified in
[`raw-v1-certificate-specification.md`](raw-v1-certificate-specification.md).
This note proposes a separately named proof-grade theorem surface; it does not
change the meaning of a frozen profile.

All finite-dimensional norms below are infinity norms unless another norm is
shown explicitly.

## A two-sided a-posteriori tube theorem

### Theorem 1 (finite-dimensional tube enclosure)

Let \(I=[a_-,a_+]\subset\mathbb R\) be compact, let \(s_0\in I\), let
\(U\subset\mathbb R^n\) be open, and let \(F:U\to\mathbb R^n\) be locally
Lipschitz. Let \(P:I\to\mathbb R^n\) be continuously differentiable. For
\(r>0\), define the complete closed tube

\[
 K_r=\{P(s)+u:s\in I,\ \|u\|_\infty\le r\}.
\]

Assume:

1. \(K_r\subset U\);
2. \(F\) is \(L\)-Lipschitz on \(K_r\), for some \(L\ge0\);
3.
   \[
   \sup_{s\in I}\|P'(s)-F(P(s))\|_\infty\le\delta
   \]
   for some \(\delta\ge0\);
4. the exact anchor \(x_0\in U\) obeys

   \[
   \|x_0-P(s_0)\|_\infty\le\epsilon;
   \]
5. with \(h=\max\{s_0-a_-,a_+-s_0\}\),

   \[
   E=
   \begin{cases}
   e^{Lh}\epsilon+\delta\dfrac{e^{Lh}-1}{L},&L>0,\\[6pt]
   \epsilon+\delta h,&L=0
   \end{cases}
   <r.
   \]

Then the IVP

\[
 x'=F(x),\qquad x(s_0)=x_0
\]

has a unique solution on all of \(I\), and

\[
 \|x(s)-P(s)\|_\infty\le
 \begin{cases}
 e^{L|s-s_0|}\epsilon+\delta\dfrac{e^{L|s-s_0|}-1}{L},&L>0,\\[6pt]
 \epsilon+\delta|s-s_0|,&L=0
 \end{cases}
 \le E
\]

for every \(s\in I\).

The hypotheses require local existence and uniqueness only: in finite
dimensions, local Lipschitz continuity of \(F\) on the open set \(U\) supplies
that premise by the Picard--Lindelöf theorem. No global existence premise is
being hidden in the statement.

### Proof

Local existence and uniqueness give a maximal solution on an open parameter
interval containing \(s_0\). Put \(e=x-P\). Since
\(\|e(s_0)\|_\infty\le\epsilon\le E<r\), there is a nonempty connected forward
interval \(J_+\) beginning at \(s_0\) on which
\(\|e(s)\|_\infty<r\). Take \(J_+\) maximal with this property within both the
maximal solution interval and \([s_0,a_+]\).

For every \(s\in J_+\), the two contemporaneous states \(x(s)=P(s)+e(s)\) and
\(P(s)\) belong to \(K_r\). Thus the Lipschitz hypothesis applies to
\(F(x(s))-F(P(s))\). With \(R=P'-F(P)\), the integral equation gives

\[
 \|e(s)\|_\infty
 \le \epsilon+\delta(s-s_0)
    +L\int_{s_0}^{s}\|e(\sigma)\|_\infty\,d\sigma.
\]

Grönwall's inequality yields the asserted forward bound on \(J_+\), in
particular \(\|e(s)\|_\infty\le E<r\). If the right endpoint of \(J_+\) were a
solution time strictly before \(a_+\), continuity would retain the strict
inequality there and extend \(J_+\), contradicting maximality. Equivalently,
there can be no first contemporaneous contact
\(\|x(s)-P(s)\|_\infty=r\).

The remaining possibility is that the maximal solution itself ends before
\(a_+\). This is also impossible. The bound just obtained keeps
\(x(s)=P(s)+e(s)\) in the compact set \(K_r\) as that endpoint is approached.
Because \(K_r\subset U\), the standard continuation theorem extends the
solution past such a finite endpoint. The bound passes to the endpoint by
continuity and remains strictly below \(r\), again extending \(J_+\).
Consequently the solution and the error bound reach \(a_+\).

Apply the same argument backward from \(s_0\), equivalently after the change
of variable \(\tau=s_0-s\), to reach \(a_-\). The compactness used above follows
because \(K_r\) is the continuous image under
\((s,u)\mapsto P(s)+u\) of the compact set
\(I\times[-r,r]^n\). Local uniqueness patches along the compact interval and
gives uniqueness on all of \(I\).

When \(L=0\), the integral inequality directly gives
\(\|e(s)\|_\infty\le\epsilon+\delta|s-s_0|\), which is also the continuous
\(L\to0\) limit of the displayed expression. \(\square\)

### What a finite replay must establish

A checker need not compute the best \(\delta\), \(L\), or exponential. It must
compute outward bounds \(\widehat\delta\ge\delta\), \(\widehat L\ge L\), and
\(\widehat E\ge E\), and prove the strict comparison \(\widehat E<r\). Supplied
maximum-defect and maximum-Lipschitz fields are only caps on recomputed values;
they are not witnesses. Equality with \(r\) is insufficient for the first-exit
bootstrap.

## Ordinary planar Newton instantiation

For three positive masses in the plane, use the 12-state order \(x=(q,v)\),
where \(q,v\in\mathbb R^6\), and

\[
 F_N(q,v)=\left(v,
 \left(\sum_{j\ne i}m_j\frac{q_j-q_i}{|q_j-q_i|^3}\right)_{i=1}^3
 \right).
\]

Its open domain is the complement of the three binary-collision diagonals.
On that domain it is analytic, hence locally Lipschitz.

Let \(P=(Q,V)\) be the exact-dyadic polynomial obtained from an admitted
ordinary chart. Suppose interval evaluation of \(Q(I)\) gives a common lower
bound \(d_{\rm nominal}\) on all nominal Euclidean pair distances. A state
infinity ball of radius \(r\) moves either body by at most \(\sqrt2r\) in
Euclidean norm, so every pair distance in the complete tube is at least

\[
 d_{\rm tube}=d_{\rm nominal}-2\sqrt2\,r.
\]

The strict condition \(d_{\rm tube}>0\) proves that the complete tube is inside
the ordinary field domain.

For \(g(y)=y/|y|^3\),

\[
 Dg(y)=|y|^{-3}I-3|y|^{-5}yy^T,
 \qquad
 \|Dg(y)\|_\infty\le\frac{1+3\sqrt2}{|y|^3}.
\]

Accounting separately for the derivative with respect to the other body and
the summed derivative with respect to the accelerated body gives

\[
 L_N=\max\left\{1,
 \max_i\frac{2(1+3\sqrt2)\sum_{j\ne i}m_j}{d_{\rm tube}^3}
 \right\}.
\]

The \(1\) is the infinity row sum of the \(q'=v\) block. Consequently Theorem
1 applies once the direct residual

\[
 \delta_N\ge\sup_{s\in I}\|P'(s)-F_N(P(s))\|_\infty
\]

and the anchor error have been bounded.

### Mapping to the Rust replay

The present arithmetic path matches these hypotheses as follows.

- `replay_ordinary_tube_exact_rational_v04` in
  [`ordinary_tube.rs`](../verifiers/rust-v1/src/ordinary_tube.rs) evaluates the
  six ordered `ORDINARY_TUBE_OBLIGATION_IDS`.
- `evaluate_planar_three_body_ordinary_polynomial_defect` in
  [`ordinary_defect.rs`](../verifiers/rust-v1/src/ordinary_defect.rs) evaluates
  \(Q'-V\) and \(V'-a(Q)\) over the whole parameter interval. It does not add
  the chart's `tail_bound`.
- The helpers `tube_pair_distance_lower` and `analytic_lipschitz_upper`
  implement the complete-tube floor and the displayed \(L_N\).
- `gronwall_upper` uses the recomputed defect and Lipschitz values and
  `strictly_inside_radius` enforces the strict bootstrap.
- `replay_validated_ordinary_root_exact_rational_v04` in
  [`ordinary_binding.rs`](../verifiers/rust-v1/src/ordinary_binding.rs) combines
  exact IVP binding with the tube and checks the actual polynomial-to-IVP
  anchor gap. Its proof-oriented result intentionally does not consume the
  claimed-tail chart ledger.
- `replay_carried_ordinary_bridge_exact_rational_v04` in
  [`ordinary_bridge.rs`](../verifiers/rust-v1/src/ordinary_bridge.rs) evaluates
  and inflates the complete 12-component source endpoint and requires its
  containment in the target initial-error ball.

The name `ordinary_autonomous_uniqueness_bridge_kernel_v1` identifies the
mathematical handoff lemma that must be proved: endpoint containment supplies
the target anchor premise of Theorem 1, and uniqueness of the autonomous
Newton IVP identifies the target solution with the already carried source
solution. The identifier itself is not a proof of that lemma.

## Planar Levi-Civita instantiation

### Lifted state and regular domain

For a selected ascending pair \((i,j)\), the implemented lifted state has 14
components in the order

\[
 X=(z,w,h,R,U,y,V,t),
\]

with \(z,w,R,U,y,V\in\mathbb R^2\) and \(h,t\in\mathbb R\). The selected pair
relative position is \(Q(z)=(z_1^2-z_2^2,2z_1z_2)\). Write

\[
 \Lambda(z)=
 \begin{pmatrix}
 z_1&-z_2\\
 z_2& z_1
 \end{pmatrix},
 \qquad L(z)=2\Lambda(z),
 \qquad \rho=|z|^2.
\]

Thus \(\Lambda\) is the base matrix used by the component formulas, while
\(L=DQ=2\Lambda\) is the doubled matrix used in the specification. The
punctured relative-velocity projection is
\(2\Lambda(z)w/\rho=L(z)w/\rho\). In particular, the selected binary collision
is \(z=0\).

The regularized field \(F_{LC}\) contains no division by \(z\) or \(\rho\).
Its only force denominators are the distances from the two selected-pair bodies
to the third body. Let these displacement vectors be \(d_i(X)\) and \(d_j(X)\),
and define

\[
 U_{LC}=\{X:|d_i(X)|>0\text{ and }|d_j(X)|>0\}.
\]

The implemented formulas are analytic on \(U_{LC}\), including at \(z=0\).
Thus a complete lifted tube may cross a selected binary collision; it may not
touch a third-body denominator zero.

### Lifted tube theorem

Let \(\bar X:I\to\mathbb R^{14}\) be the admitted finite LC polynomial, let
\(X_0\) be an exact lifted anchor, and assume:

1. \(\bar X(I)+[-r,r]^{14}\subset U_{LC}\);
2. direct interval evaluation proves

   \[
   \sup_I\|\bar X'-F_{LC}(\bar X)\|_\infty\le\delta_{LC};
   \]
3. interval differentiation encloses \(DF_{LC}\) on the complete inflated
   tube and proves an infinity row-sum bound \(L_{LC}\);
4. \(\|X_0-\bar X(s_0)\|_\infty\le\epsilon_{LC}\);
5. the corresponding two-sided Grönwall bound is strictly less than \(r\).

Then Theorem 1 produces a unique exact lifted solution throughout \(I\), even
if its selected-pair coordinate \(z\) vanishes inside \(I\).

`replay_planar_lc_tube_exact_rational_v04` in
[`planar_lc_tube.rs`](../verifiers/rust-v1/src/planar_lc_tube.rs) recomputes the
nine `PLANAR_LC_TUBE_OBLIGATION_IDS`. The direct defect is implemented by
`evaluate_planar_lc_polynomial_defect` in
[`planar_lc_defect.rs`](../verifiers/rust-v1/src/planar_lc_defect.rs). The
complete interval field and Jacobian are produced by
`evaluate_planar_lc_field` in
[`planar_lc_field.rs`](../verifiers/rust-v1/src/planar_lc_field.rs). In that
function, the `inverse_cube_vector` calls are exactly the two excluded
third-body denominators, while the final ODE component is \(t'=\rho\).

The optional obligation
`planar_lc_tube_pair_energy_constraint_when_required` checks the polynomial
center at the anchor. It is not, by itself, a proof that every point in a
positive initial ball is constrained. Conversely, a chain may carry a
constrained exact anchor even when this optional center flag is false. The
chain theorem therefore obtains constrainedness from the entry lift of the
actual carried state, not from a rectangular tube assertion.

## Lift, constraint, clock, deck, and projection lemmas

The numerical entry and exit computations require the following mathematical
lemmas. They must appear as proved lemmas in the theorem artifact or be made
explicit assumptions. Merely returning a constant implementation ID does not
establish them.

### Lemma 2 (complete square-root lift cover)

Subject to its stated arithmetic and source-to-binary trust boundary, the
[canonical planar LC square-root lift-cover audit](canonical-planar-lc-lift-cover-audit.md)
closes the source-level finite-cover and deck-parity argument for this
construction. Its fixed-resolution square-root limitation remains a
fail-closed completeness limitation.

Let a Cartesian source box exclude selected-pair collision. The canonical
one- or two-patch square-root construction in
`replay_planar_lc_lift_cover_exact_rational_v04`, implemented in
[`planar_lc_lift.rs`](../verifiers/rust-v1/src/planar_lc_lift.rs), must cover
every selected relative position in that box. For the one actual Cartesian
source state, at least one patch contains a lift satisfying

\[
 Q(z)=q_j-q_i,\qquad
 w=\tfrac12\Lambda(z)^T(v_j-v_i)
   =\tfrac14L(z)^T(v_j-v_i),
\]

together with the implemented definitions of \(h,R,U,y,V\). That exact lift
satisfies the pair-energy constraint

\[
 C(X)=2|w|^2-(m_i+m_j)-|z|^2h=0.
\]

Indeed \(\Lambda(z)\Lambda(z)^T=\rho I\), so the implemented
\(w=\tfrac12\Lambda(z)^Tv\) gives
\(2|w|^2=\tfrac12\rho|v|^2\).
The implemented
\(h=\tfrac12|v|^2-(m_i+m_j)/\rho\) then makes the displayed constraint
identically zero for the exact lifted state.

The statement is existential for the actual state. It does not assert that
every point of an interval patch satisfies \(C=0\).

The patch overlap parity equations and the two global complementary gauge
assignments must implement the deck transformation
\((z,w)\mapsto(-z,-w)\) without changing the physical state.

The elementary projection part is immediate from the displayed conventions:

\[
 Q(-z)=Q(z),\qquad
 \Lambda(-z)(-w)=\Lambda(z)w,\qquad
 L(-z)(-w)=L(z)w,\qquad
 |-z|^2=|z|^2.
\]

Thus relative position, punctured relative velocity, and the physical clock
rate are deck invariant. The nontrivial finite assertion is that the finite
interval patch construction is a complete cover and that its parity graph
selects one coherent representative for every actual source state.
The linked canonical lift-cover audit establishes that assertion at source
level, subject to its stated arithmetic and source-to-binary trust boundary.
`planar_lc_constrained_lift_deck_gauge_kernel_v1` names this intended lemma; it
does not numerically recheck the identities for every real point in a patch.

### Lemma 3 (constraint invariance)

For the exact implemented lifted field,

\[
 D C(X)\,F_{LC}(X)=0
\]

on \(U_{LC}\). Hence an exact lifted solution whose entry state satisfies
\(C=0\) remains constrained throughout its interval of existence. This is an
algebraic identity against the exact field formulas, not a consequence of a
small numerical residual.

Let \(P=(p_x,p_y)\) denote the perturbing acceleration vector computed by
`evaluate_planar_lc_field`. In the base-matrix convention, the relevant exact
equations are

\[
 z'=w,\qquad
 w'=\frac12hz+\frac12\rho\Lambda(z)^TP,\qquad
 h'=2(\Lambda(z)w)\mathbin{\cdot}P.
\]

Since \(\rho'=2z\cdot w\), direct differentiation gives

\[
\begin{aligned}
 C'
 &=4w\cdot w'-\rho'h-\rho h'\\
 &=2h\,w\cdot z+2\rho\,w\cdot\Lambda(z)^TP
   -2h\,z\cdot w-2\rho\,(\Lambda(z)w)\cdot P\\
 &=0,
\end{aligned}
\]

where \(w\cdot\Lambda(z)^TP=(\Lambda(z)w)\cdot P\). The Boolean
`carried_lc_exit_constraint_invariance_kernel` in
[`planar_lc_exit.rs`](../verifiers/rust-v1/src/planar_lc_exit.rs) records that
the premise is intended to be used; it is not an independent numerical proof
of the identity.

### Lemma 4 (strict physical clock)

Along a lifted solution, \(t'=\rho=|z|^2\ge0\). If \(z\) is not identically
zero, analyticity implies that its zeros are isolated. Therefore, for every
\(s_1<s_2\),

\[
 t(s_2)-t(s_1)=\int_{s_1}^{s_2}|z(s)|^2\,ds>0.
\]

The entry transition proves a positive \(\rho\) lower bound on its punctured
lift patches, so the selected exact branch has \(z\ne0\) at entry and is not
identically zero. The clock is consequently strictly increasing across any
isolated selected binary collision even though \(t'=0\) at the collision.

This proves ordering, not occurrence. A chain with no zero of \(z\) remains a
valid LC passage and does not thereby certify a collision event.
`carried_lc_exit_strict_physical_clock_kernel` names this implication but does
not replace its analytic proof.

### Lemma 5 (punctured deck-equivariant Newton projection)

On the constrained set \(C=0\) with \(\rho>0\), the implemented projection from
\((z,w,h,R,U,y,V)\) to the three Cartesian positions and velocities must:

1. be invariant under the deck transformation;
2. map integral curves of \(F_{LC}\), after the time change \(dt/ds=\rho\), to
   integral curves of the planar Newton field with the same exact masses; and
3. agree with the entry lift formulas.

Here is the full mass-weighted projection proof. Put

\[
 M=m_i+m_j,\qquad
 \alpha=\frac{m_j}{M},\qquad
 \beta=\frac{m_i}{M},
\]

and, with \(q=Q(z)\), define the two third-body displacement and force blocks

\[
\begin{aligned}
 d_i&=y+\alpha q,& f_i&=\frac{d_i}{|d_i|^3},\\
 d_j&=y-\beta q,& f_j&=\frac{d_j}{|d_j|^3}.
\end{aligned}
\]

The exact lifted equations use

\[
\begin{aligned}
 P&=m_k(f_j-f_i),\\
 B&=\frac{m_k}{M}(m_i f_i+m_j f_j),\\
 A_k&=-m_i f_i-m_j f_j,\qquad Y=A_k-B.
\end{aligned}
\]

Constraint invariance and the relative LC calculation give, on \(\rho>0\),

\[
 \ddot q=-M\frac{q}{|q|^3}+P.
\]

Differentiating
\(q_i=R-\alpha q\), \(q_j=R+\beta q\), and \(q_k=R+y\) in
physical time therefore gives

\[
\begin{aligned}
 \ddot q_i
 &=B-\alpha\ddot q
 =m_j\frac{q}{|q|^3}+m_kf_i,\\
 \ddot q_j
 &=B+\beta\ddot q
 =-m_i\frac{q}{|q|^3}+m_kf_j,\\
 \ddot q_k
 &=B+Y=A_k=-m_if_i-m_jf_j.
\end{aligned}
\]

For example, the noncentral force in the first equality reduces as

\[
 \frac{m_k}{M}(m_if_i+m_jf_j)
 -\frac{m_jm_k}{M}(f_j-f_i)=m_kf_i;
\]

the second equality is analogous. Since
\(d_i=q_k-q_i\) and \(d_j=q_k-q_j\), the three displayed equations are exactly
the three planar Newton acceleration equations.

Deck equivariance also follows for the complete force blocks, not only for the
relative kinematics. The identities \(Q(-z)=Q(z)\) and unchanged \(y\) imply
that \(d_i,d_j\), hence \(f_i,f_j,P,B,Y\), are unchanged. Together with the
deck identities in Lemma 2, this proves that the punctured Cartesian projection
is deck invariant and that deck-related lifted solutions have the same
physical projection.

The displayed algebra is the proof. The exact symbolic checks in
[verify_lc_projection_identities.py](../scripts/verify_lc_projection_identities.py)
are regression checks against transcription or convention drift; they are not
a foundational proof system.

The exact-derived mass profile used by field, lift, and projection must denote
one and the same positive mass problem. The finite interval projection is
implemented by `project_planar_lc_full_state_exact_rational` in
[`planar_lc_projection.rs`](../verifiers/rust-v1/src/planar_lc_projection.rs).
The obligation `carried_lc_exit_deck_equivariant_newton_projection_kernel`
marks use of this mathematical lemma. The code's Boolean dependency alone is
not its proof. The source-level
[Rust LC formula-correspondence and interval-inclusion audit](rust-lc-formula-and-interval-inclusion-audit.md)
closes the line-by-line field/lift/projection/mass correspondence and proves
the rational-interval inclusion contract, including complete
positive-\(\rho\) slice projection, conditional on its stated Rust and
`num-bigint`/`num-rational` arithmetic trust boundary. The remaining
implementation gap is foundational verification of that boundary, not an
unidentified formula or interval-inclusion step.

### Lemma 6 (LC passage handoff)

Assume a parent-carried ordinary solution lies in the source ordinary tube.
The entry replay reconstructs its complete endpoint box. By Lemma 2, its actual
endpoint has a constrained lifted representative in one complete patch. If one
global gauge assignment places every complete transformed patch inside the LC
tube's initial ball, that representative satisfies the anchor premise of the
lifted tube theorem. It therefore generates a unique lifted solution in the LC
tube.

This establishes branch identity, rather than mere intersection of enclosures,
as follows. At the punctured entry, projection of the chosen exact lift is
exactly the actual source endpoint state, and its fourteenth component is the
same physical time carried by the parent clock. On the connected incoming
punctured component, Lemma 5 makes the LC projection a solution of the same
autonomous Newton equation as the source ordinary branch. They have the same
state at the entry time, so local Newton IVP uniqueness makes them equal near
entry; continuation of that uniqueness along the connected component makes
them equal up to the collision boundary.

If \(z\) vanishes, no uniqueness of the singular Cartesian Newton equation is
invoked at that instant. The unique analytic lifted IVP itself crosses the
regularized point and selects the outgoing continuation. Lemma 3 keeps this
one lifted solution constrained, Lemma 4 strictly orders its physical clock,
and Lemma 5 projects its outgoing punctured component to Newton.

Under the audited arithmetic trust boundary, at an exit whose complete slice
has \(\rho>0\), the interval projection contains the exact
projected state of this same lifted solution. If the complete projected box is
contained in the target ordinary tube's initial ball, instantiate Theorem 1
with that exact projected exit state as the target anchor state. The resulting
target ordinary solution and the outgoing LC
projection solve the same Newton IVP at the same physical time, so ordinary
local uniqueness identifies them near the handoff and hence throughout their
common connected domain. Complete containment is what supplies the actual
anchor premise; overlap of two boxes would not suffice. The fourteenth lifted
component and the exact clock cocycle carry the physical-time identity through
both handoffs.

The entry arithmetic is in
`replay_carried_planar_lc_entry_exact_rational_v04` in
[`planar_lc_entry.rs`](../verifiers/rust-v1/src/planar_lc_entry.rs). The full
14-component exit slice, positive exit \(\rho\), projection, target containment,
and clock derivation are in
`replay_carried_planar_lc_exit_exact_rational_v04` in
[`planar_lc_exit.rs`](../verifiers/rust-v1/src/planar_lc_exit.rs).

## Finite supplied-chain theorem

### Carried invariant

At each accepted ordinary vertex retain:

1. one exact branch solving the ordinary Newton IVP on the current chart's
   physical-time image;
2. the proof that its state lies in the freshly recomputed ordinary tube;
3. an exact rational interval \(B\) containing its clock origin in
   \(t=s+b\), \(b\in B\); and
4. identity with the branch bound at the root, through the already accepted
   ordinary bridges and LC passages.

For an LC passage the invariant temporarily carries the exact constrained
lifted solution, its strictly increasing physical clock, and its identity with
the ordinary branch on both punctured ends.

### Theorem 7 (finite chain soundness)

Let a finite raw planar chain have:

1. an admitted exact point-IVP binding and a proof-grade initial ordinary tube;
2. positive common masses and consistent planar problem identity everywhere;
3. a globally coherent namespace and exact endpoint/anchor identities;
4. for every ordinary bridge, two proof-grade ordinary tubes, complete source
   endpoint reconstruction, target initial-ball containment, and exact clock
   cocycle;
5. for every LC passage, a proof-grade source ordinary tube, complete entry
   lift cover and target containment, a proof-grade 14-dimensional LC tube,
   the hypotheses of Lemmas 2--5, complete punctured exit projection, target
   ordinary initial-ball containment, and exact clock derivation; and
6. transactional replay, so no state, branch, clock, or segment count from a
   failed segment is committed.

Then every accepted prefix carries one exact branch from the bound initial
state. On ordinary pieces it is the unique classical planar Newton solution.
On each LC piece it is the unique exact lifted solution; away from zeros of
\(z\), its projection is the same classical Newton branch. At isolated zeros of
\(z\), it supplies the selected Levi-Civita generalized binary-collision
continuation. The theorem does not assert that an LC piece actually contains a
zero unless a separate event obligation proves one.

### Proof

The exact root binding supplies the anchor state for Theorem 1 and establishes
the carried invariant at vertex zero. For an ordinary bridge, complete endpoint
containment supplies the next anchor premise, and autonomous uniqueness shows
that the next tube encloses the same branch. For an LC passage, Lemma 6 carries
the invariant through the constrained lift and punctured projection. Exact
clock cocycles preserve the relation between local parameter and physical
time. Induction over the finite segment word proves the conclusion. Because a
failed segment is not committed, the same proof applies to the longest
accepted prefix. \(\square\)

### Theorem 8 (terminal fixed-time enclosure)

Assume Theorem 7 has carried a final ordinary chart with parameter interval
\(I=[a,b]\), clock-origin interval \(B=[B_-,B_+]\), polynomial \(P\), and fresh
tube error \(E\). For an exact requested physical time \(T\), define

\[
 J=T-B=[T-B_+,T-B_-].
\]

If the clock ledger proves that the actual clock origin lies in \(B\) and
\(J\subset I\), then the actual local parameter at physical time \(T\) belongs
to \(J\), and

\[
 x(T)\in \mathcal P(J)+[-E,E]^{12},
\]

where \(\mathcal P(J)\) is an outward interval evaluation of all 12 polynomial
components. Thus the complete inflated box is a rigorous terminal enclosure.
An optional maximum-component-width comparison changes whether a requested
accuracy was met; it does not strengthen or weaken containment in the box.

The current fixed-time arithmetic appears in
`replay_raw_ordinary_only_chain_exact_rational_v04` in
[`ordinary_chain.rs`](../verifiers/rust-v1/src/ordinary_chain.rs) and in the
mixed fold in [`mixed_chain.rs`](../verifiers/rust-v1/src/mixed_chain.rs).

## Why recurrence and claimed tails are nondecisive

Theorem 1 accepts any finite \(C^1\) approximate trajectory with a direct,
rigorously enclosed defect. It neither requires nor concludes that the
polynomial coefficients came from a Taylor recurrence. It also does not use a
Taylor-series convergence radius or an omitted-series remainder: the
polynomial is evaluated as a finite function, and its deviation from the ODE
is measured directly over the whole interval.

Therefore the profiles
`exact_rational_ordinary_chart_claimed_tail_v04` and
`exact_rational_planar_lc_chart_claimed_tail_v04` may be useful compatibility
diagnostics, but their unproved serialized `tail_bound` values supply no premise
of the proof-grade theorem. Requiring such a diagnostic as an additional
filter cannot make an otherwise valid direct-tube conclusion false, but naming
it “fresh chart certification” obscures the proof dependency and can reject a
sound tube for an irrelevant reason.

A separately named proof-grade chain profile should:

- require finite polynomial/schema admission, dimensions, identifiers, masses,
  parameter intervals, and endpoint anchors;
- make direct ordinary and LC tube replays decisive;
- make exact binding, complete handoff, lift, constraint, clock, projection,
  and containment obligations decisive;
- retain recurrence and claimed-tail results only in explicitly diagnostic
  fields; and
- use new obligation and profile identifiers rather than silently changing a
  frozen compatibility profile.

The local
`exact_rational_proof_grade_carried_planar_lc_entry_v04` profile is now
implemented in [`planar_lc_entry.rs`](../verifiers/rust-v1/src/planar_lc_entry.rs)
with a 19-row decisive entry ledger. Its source and target claimed-tail chart
`Result` values are retained only as diagnostic, nondecisive outputs and do
not feed `conditional_profile_satisfied()`.

The matching local
`exact_rational_proof_grade_carried_planar_lc_exit_v04` profile is implemented
in [`planar_lc_exit.rs`](../verifiers/rust-v1/src/planar_lc_exit.rs) with a
21-row decisive exit ledger. It nests the proof-grade entry chart diagnostics
without making them decisive, while freshly replayed direct slice, projection,
containment, time, and clock evidence remains decisive. These entry and exit
kernels are tested independently of the frozen compatibility profiles.

The separately named
`exact_rational_proof_grade_raw_mixed_planar_chain_v04` profile is now
implemented in
[`proof_grade_mixed_chain.rs`](../verifiers/rust-v1/src/proof_grade_mixed_chain.rs).
Its eight decisive rows fold the exact root, direct local ordinary or LC
evidence, exact clock/preimage derivation, direct terminal evaluation, and the
requested width bound. The root ordinary claimed-tail replay is retained as a
typed, nondecisive diagnostic. Proof-grade LC exits are committed
transactionally: an LC passage changes the carried chart and clock only after
its proof-grade exit ledger succeeds. On an uncommitted LC passage the profile
retains a proof-only lifted LC-right frontier when the direct evidence supports
one; this retained frontier is diagnostic coverage, not a terminal acceptance.

The separately named admitted outcome
`exact_rational_proof_grade_admitted_raw_v1_outcome_v04` is now implemented in
[`raw_v1_proof_grade_outcome.rs`](../verifiers/rust-v1/src/raw_v1_proof_grade_outcome.rs).
It is deliberately a profile-local semantic result *after* strict opaque
`CanonicalRawV1Admission`; it is not a parser result, CLI envelope, or
cross-verifier result. Its ordered 13-row ledger is:

1. `proof_grade_raw_planar_chain_outer_schema_exact`;
2. `proof_grade_raw_planar_chain_global_identifier_namespace_unique`;
3. `proof_grade_raw_planar_chain_canonical_evidence_serializable`;
4. `proof_grade_raw_planar_chain_requested_target_finite`;
5. `proof_grade_raw_planar_chain_requested_width_admissible`;
6. `proof_grade_raw_planar_chain_root_exact_point_left_anchor`;
7. `proof_grade_raw_planar_chain_root_direct_tube_and_binding_certified`;
8. `proof_grade_raw_planar_chain_all_segments_direct_evidence_folded`;
9. `proof_grade_raw_planar_chain_target_not_before_current_left_clock`;
10. `proof_grade_raw_planar_chain_fixed_time_preimage_exactly_derived`;
11. `proof_grade_raw_planar_chain_fixed_time_preimage_inside_forward_current_domain`;
12. `proof_grade_raw_planar_chain_target_state_directly_evaluated_and_inflated`;
13. `proof_grade_raw_planar_chain_final_component_width_within_requested_bound`.

Rows 1--5 are the admitted outer checks: possession of the immutable
canonical-and-typed token, namespace replay, the token's canonical byte
ownership, finite binary64 target decoding, and nonnegative requested width.
Rows 6--13 are exactly the eight decisive rows of the proof-grade mixed
replay. `CERTIFIED_TO_T` means that all thirteen rows are true; otherwise the
result is `UNRESOLVED` and the first failure records the outer row or the
root/segment-local decisive row that failed. This is a total Boolean result
only once strict admission and the bounded replay have both returned normally;
it does not turn malformed bytes, admission failure, or resource/error paths
outside that execution envelope into a portable semantic classification.

The portable projection has its own schema,
`raw-v1-rust-proof-grade-semantic-outcome-v1`, rather than reusing the
compatibility semantic-outcome schema. It hashes precisely the immutable
canonical admission bytes with SHA-256 and emits that lower-case hexadecimal
digest as `evidence_sha256`; for the canonical success input the digest is
`ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11`.
Its deterministic compact UTF-8 JSON has no trailing newline and contains the
request as exact rational numerator/denominator strings, the full top-level
ledger, status and first failure, clock ledger, current chart/clock, coverage
and preimage intervals, width, and source-derived segment metadata. A segment
metadata record gives its index, kind, profile, and (for an LC passage) the
canonical pair. In the canonical five-segment success this is one
`exact_rational_carried_ordinary_bridge_v04` segment followed by four
`exact_rational_proof_grade_carried_planar_lc_exit_v04` segments for pairs
`[0,1]`, `[0,2]`, `[1,2]`, and `[0,1]`. The metadata is descriptive only; the
owned proof-grade mixed replay remains the certification authority.

For a successful terminal replay, the portable object includes the direct
12-component planar Cartesian fixed-physical-time enclosure. An unresolved
replay that fails before target-state evaluation has no such final enclosure;
it can instead retain a certified ordinary right frontier or a 14-component
lifted LC-right frontier, with its clock/time interval, chart, pair, and
certified/failed segment counts. If all target-state rows succeed and only the
requested-width row fails, the computed fixed-time enclosure remains present
and is also identified as the retained ordinary fixed-time region. Every
retained region reports the strongest committed direct evidence, not
acceptance under all thirteen rows. Claimed-tail results, recurrence data,
typed diagnostics, resource limits, and diagnostic error strings are
deliberately omitted from this portable JSON. Tests mutate all claimed tails
and trigger a claimed-tail resource diagnostic while preserving the decisive
ledger, clocks, terminal/retained evidence, and diagnostic-free projection
(apart from the intentionally changed evidence hash).

The byte-level execution checkpoint is separately implemented in
[`raw_v1_proof_grade_execution.rs`](../verifiers/rust-v1/src/raw_v1_proof_grade_execution.rs)
as `exact_rational_proof_grade_raw_v1_execution_v04`, with its own
`raw-v1-rust-proof-grade-execution-v1` schema. It strictly admits the supplied
raw-v1 bytes canonically before evaluating the admitted 13-row outcome. Its
portable envelope gives the stable, diagnostic-free classifications
`REJECT`/`CANONICAL_WIRE`/`NOT_RUN` or `REJECT`/`SCHEMA`/`NOT_RUN` for failed
admission; `ACCEPT`/`RESULT` with the exact nested proof-grade semantic
projection for an admitted replay result; and `ACCEPT`/`ERROR` with exactly
`NAMESPACE` or `MIXED_REPLAY` for a returned evaluation error. It emits no
private source errors, typed diagnostics, tail values, resource details, or
error strings.

The corresponding CLI binary is
[`raw_v1_proof_grade_verify.rs`](../verifiers/rust-v1/src/bin/raw_v1_proof_grade_verify.rs),
invoked as `raw_v1_proof_grade_verify <raw-v1-path>` (or through Cargo with
`--bin raw_v1_proof_grade_verify -- <raw-v1-path>`). It writes the compact
execution JSON without a newline and exits 0 for a semantic result or a stable
rejection, 2 for a classified evaluation error, 1 for I/O, allocation, input
limit-representation, or serialization failure, and 64 for misuse. Its input
reader reserves and reads at most `DEFAULT_WIRE_JSON_LIMITS.max_input_bytes +
1` bytes, so an oversized file is bounded and then reaches canonical-wire
rejection rather than being read without limit. The execution tests include an
actual 4,097-coefficient initial ordinary-chart mutation, which is admitted
but produces the stable `ACCEPT`/`ERROR`/`MIXED_REPLAY` envelope (and CLI exit
2) without exposing its resource diagnostic.

This closes the current bounded `Result` paths only: it is not a proof of
total behavior under process OOM, panic, hostile operating-system failure, or
other termination outside the implemented reader/admission/replay/
serialization paths. It remains an important local proof-grade checkpoint,
not the v0.4 release gate. A versioned proof-grade corpus and cross-verifier
comparator, Python proof-grade parity/agreement, two isolated
pinned-environment replays, and review-paper/release integration are still
missing. No corpus agreement, independence, or release claim follows from
this execution profile.

## Exact remaining proof and code gates

Before the assembled proof-grade surface can support a publication-grade claim,
the following gates should be closed explicitly.

### Mathematical lemmas

- Record Theorem 1, including its backward-time and \(L=0\) cases, as a named
  theorem dependency of both tube profiles.
- Prove the ordinary complete-tube pair-floor and analytic infinity-row-sum
  bound against the exact implemented coordinate order.
- Prove that the exact LC formulas in `evaluate_planar_lc_field` are analytic
  whenever the two third-body denominators are nonzero.
- The [canonical planar LC square-root lift-cover audit](canonical-planar-lc-lift-cover-audit.md)
  closes the source-level proof of every canonical square-root-cover case and
  its parity graph under its stated trust boundary. The fixed-resolution
  square-root limitation remains a fail-closed completeness limitation.
- Regression-check the displayed proof of \(D C\,F_{LC}=0\) against any future
  change to the implemented field normalization.
- Regression-check the displayed deck-equivariant, relative, and full-body
  projection proof against future changes in lift, field, or mass conventions;
  the current Rust correspondence and interval-inclusion argument is recorded
  in the source audit linked from Lemma 5.
- Prove the strict-clock lemma from positive entry \(\rho\), analyticity, and
  \(t'=|z|^2\).
- Record the displayed ordinary and LC same-branch handoff arguments and finite
  induction as named theorem dependencies, rather than treating implementation
  IDs as proof objects.

### Arithmetic and implementation gates

- Retain the proved source-level inclusion contract for rational intervals,
  interval Horner evaluation, interval dual differentiation, and dyadic
  square-root enclosures; separately close its Rust/`num-bigint`/
  `num-rational` trust boundary. The analogous exponential-enclosure and
  complete ordinary/LC Grönwall contract is proved in the
  [Rust exponential and Grönwall source audit](rust-exp-gronwall-source-audit.md),
  conditional on the same explicitly retained arithmetic, compiler, and
  source-to-binary trust boundary; resource exhaustion fails closed on these
  audited paths.
- Retain the source-level
  [mass identity and flow audit](rust-mass-identity-and-flow-audit.md), which
  establishes that ordinary/LC defect, tube, field, series, lift, projection,
  entry, exit, and mixed-chain handoffs use one exact serialized positive mass
  triple and freshly validated exact-derived LC profiles. Its trust boundary
  remains open. The profile's tight outward binary64 enclosures are validated
  witnesses; exact rational profile members, rather than those endpoints, are
  the scalars propagated by the exact-rational verifier.
- Extend the current bounded proof-grade execution envelope into its release
  gate: a versioned proof-grade corpus and cross-verifier comparator, explicit
  Python proof-grade parity/agreement, two isolated pinned-environment
  replays, and paper/release integration. Retain the bounded-reader and
  process-failure boundary explicitly, and keep every claimed-tail
  `conditional_profile_satisfied()` value nondecisive.
- Preserve claimed-tail and recurrence outputs as labeled diagnostics and test
  that mutations confined to those claims do not change proof-grade tubes,
  clocks, handoffs, or terminal enclosures.
- Test the strict boundaries: \(E=r\) rejects, a zero collision/denominator
  floor rejects, anchors inside an interval cover both time directions, and
  terminal preimage containment is inclusive.
- Test all three canonical LC pairs with unequal positive masses and reject a
  mismatch at any field/lift/projection boundary.
- Test that a standalone LC tube does not imply constrained Newton projection;
  constrainedness must be carried from the actual entry lift.
- Keep frozen-v0.3 status parity, exact-rational proof-grade soundness, and
  cross-language outcome comparison as distinct claims.

## Explicit nonclaims

Even after every gate above is closed, Theorems 7 and 8 do not prove:

- that a certificate producer terminates or finds a chain for every initial
  condition and target time;
- a general closed form for the three-body problem;
- coverage of spatial motion, triple collision, escape endpoints, or any chart
  type not present in the supplied admitted chain;
- uniqueness of a physical continuation at a total collision;
- occurrence of a selected binary collision merely because an LC chart was
  used;
- correctness of an unproved serialized Taylor remainder or claimed
  `tail_bound`;
- frozen-v0.3 binary64 status parity;
- equality of outcomes between arithmetic profiles; or
- foundational verification of the parser, rational arithmetic, interval
  library, square root, exponential, or automatic differentiation code.

The contribution is narrower and concrete: a finite, independently replayable
certificate can prove that one supplied planar chain encloses one exact carried
branch and, when the terminal preimage is covered, rigorously encloses that
branch at the requested physical time.
