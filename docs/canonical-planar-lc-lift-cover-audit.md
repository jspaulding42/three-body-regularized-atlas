# Canonical planar LC square-root lift-cover audit

## Status, scope, and verdict

This note audits the canonical planar Levi--Civita (LC) square-root cover in
[planar_lc_lift.rs](../verifiers/rust-v1/src/planar_lc_lift.rs) against
Section 5.3 of the
[raw-v1 certificate specification](raw-v1-certificate-specification.md) and
Lemma 2 of the
[proof-grade supplied planar-chain theorem](proof-grade-supplied-planar-chain-theorem.md).
It is a source-level mathematical proof, not a formal semantics proof of Rust.

Subject to the arithmetic and source-to-binary trust boundary in Section 8,
the ordered branch construction has no missing collision-free box case. Every
successful patch construction outwardly contains the appropriate pointwise
square-root lift, and the two-patch negative-cut case has the required
parity-one deck transition. This establishes the finite-cover part of Lemma 2
at source-audit level.

Two real implementation limitations remain, both fail closed:

1. Rectangular dependency can make the computed patch **rho_lower_bound** zero
   even when the Cartesian box has a strictly positive distance floor.
2. A positive distance below the fixed dyadic square-root resolution can give
   the radius enclosure a zero lower endpoint. The lift now attributes this
   directly to an unmeasured square-root-precision resource limit before the
   division used for \(h\), but fixed-resolution completeness is unchanged.

Neither limitation permits a false complete cover. They can reject a
mathematically valid cover and therefore are completeness limitations, not
soundness gaps. The second limitation's former diagnostic gap is repaired.
The foundational Rust, big-integer, rational, compiler, and
executable-identity boundary also remains open.

## 1. Reviewed construction and notation

Let the selected relative-position rectangle be

\[
 B=X\times Y=[a,b]\times[c,d],
 \qquad a\le b,\quad c\le d,
\]

and let the relative-velocity rectangle be \(V\). The implementation computes
the exact interval hulls

\[
 X^2=\{x^2:x\in X\}^{\rm hull},\qquad
 Y^2=\{y^2:y\in Y\}^{\rm hull},
\]

then \(D=X^2+Y^2\). Because the variables separate,

\[
 \underline D
 =\min_{x\in X}x^2+\min_{y\in Y}y^2
 =\min_{(x,y)\in B}(x^2+y^2).
\]

Thus the code's strict test **distance_squared.lower() > 0** is equivalent to
\((0,0)\notin B\). This audit assumes that strict premise. If it is false,
**replay_planar_lc_lift_cover_exact_rational** returns no patches,
**complete_cover = false**, and exposes no analytic kernel.

For one point \(q=(x,y)\in B\), define

\[
 r=|q|=\sqrt{x^2+y^2}>0,\qquad
 A={r+x\over2},\qquad B_r={r-x\over2},
\]

\[
 u=\sqrt A\ge0,\qquad v=\sqrt{B_r}\ge0.
\]

The subscript on \(B_r\) distinguishes this scalar radicand from the source
box \(B\). Since \(r\ge |x|\), both radicands are nonnegative.

## 2. Ordered-case exhaustiveness, including equality boundaries

**canonical_case** applies the following tests in order:

| order | code condition | selected cover |
|---:|---|---|
| 1 | \(c\ge0\) | one closed-upper patch |
| 2 | \(d\le0\) | one closed-lower patch |
| 3 | \(a>0\) | one right-half patch |
| 4 | \(c<0<d\) and \(b<0\) | upper and lower patches |
| 5 | otherwise | unresolved |

The order matters. If branch 1 fails, \(c<0\). If branch 2 then fails,
\(d>0\). Therefore every box reaching branch 3 strictly straddles the
\(x\)-axis: \(c<0<d\). If branch 3 fails as well, \(a\le0\). If branch 4
fails after those prior facts, its only new strict test can fail only as
\(b\ge0\). Hence an unresolved box satisfies

\[
 0\in[a,b]\quad\hbox{and}\quad0\in[c,d].
\]

It contains \(q=0\), so \(\underline D=0\). Contrapositively, every
axis-aligned box with \(\underline D>0\) reaches exactly one of branches 1--4.

The weak and strict comparisons handle every axis boundary deliberately:

- \(c=0\) goes to the closed-upper branch, including \(Y=\{0\}\).
- If \(c<0\) and \(d=0\), the box goes to the closed-lower branch.
- If \(Y=\{0\}\), both half-plane descriptions are mathematically possible,
  but the ordered first branch chooses upper deterministically.
- Once \(c<0<d\), the right branch requires the strict \(a>0\), while the
  negative-cut branch requires the strict \(b<0\).
- \(a=0\) or \(b=0\) in the straddling-\(Y\) regime is correctly unresolved,
  because then both coordinate intervals contain zero and
  \(\underline D=0\).

There is consequently no omitted equality case under the strict positive
distance-floor premise.

## 3. Pointwise square-root identities

The definitions give

\[
 u^2-v^2=A-B_r=x.
\]

Moreover,

\[
 4u^2v^2=(r+x)(r-x)=r^2-x^2=y^2.
\]

Since \(u,v\ge0\), \(2uv=|y|\). These identities prove the three
pointwise branch formulas.

### Closed upper

For \(y\ge0\), choose \(z=(u,v)\). Then

\[
 Q(z)=(u^2-v^2,2uv)=(x,y).
\]

This includes \(y=0\). If \(x>0\), then \(v=0\); if \(x<0\), then \(u=0\)
and the ordered upper convention chooses \(z_y=+\sqrt{-x}\).

### Closed lower

For \(y\le0\), choose \(z=(u,-v)\). Then

\[
 Q(z)=(u^2-v^2,-2uv)=(x,y).
\]

At a negative-axis point the lower convention chooses
\(z_y=-\sqrt{-x}\).

### Right half

Here \(x>0\) throughout the box and \(y\) may have either sign. Choose

\[
 z=(u,\operatorname{sgn}(y)v),
\]

with \(z_y=0\) when \(y=0\). The same identities give \(Q(z)=q\). The Rust
right patch deliberately stores one rectangular symmetric enclosure for
both signs of \(z_y\); it does not choose one sign for the whole box.

## 4. Outward interval containment

The successful-operation interval contract used here is proved more broadly
in the
[Rust LC formula and interval-inclusion audit](rust-lc-formula-and-interval-inclusion-audit.md).
The lift-specific chain is as follows.

1. **interval_square** returns the exact hull of a coordinate square. Addition
   therefore makes **distance_squared** contain \(x^2+y^2\) for every
   \(q\in B\).
2. **sqrt_interval** combines the lower endpoint of a verified dyadic
   enclosure of \(\sqrt{\underline D}\) with the upper endpoint of one for
   \(\sqrt{\overline D}\). Monotonicity makes the resulting **radius**
   interval contain every \(r=|q|\).
3. Exact interval addition, subtraction, and scaling make
   \((\mathtt{radius}+X)/2\) contain \(A\), and
   \((\mathtt{radius}-X)/2\) contain \(B_r\). Dependency can give either
   interval a spurious negative lower endpoint.
   **sqrt_nonnegative_forced** clamps that lower endpoint to zero. This is
   sound because the pointwise identities already prove \(A,B_r\ge0\); the
   clamp discards only impossible dependency overhang.
4. Applying the verified dyadic square-root enclosure to the clamped
   intervals yields **zx** containing \(u\) and **zy_magnitude** containing
   \(v\).
5. The upper patch stores that magnitude directly, so it contains \(+v\).
   The lower patch scales by \(-1\), reversing endpoints and containing
   \(-v\). The right patch stores
   \([-\overline z_y,\overline z_y]\), where
   \(\overline z_y=\mathtt{zy\_magnitude.upper()}\); this contains both signed
   choices and zero.

Thus every pointwise \(z\) selected in Section 3 lies in the corresponding
rectangular \(z\)-patch. This is an inclusion statement: dependency widening
means the rectangle also contains \(z\)-values that are not square roots of
any \(q\) in the source box.

For an actual relative velocity
\(v_{\rm rel}=(v_x,v_y)\in V\), the code forms

\[
 w_x={z_xv_x+z_yv_y\over2},\qquad
 w_y={z_xv_y-z_yv_x\over2},
\]

which is

\[
 w={1\over2}\Lambda(z)^Tv_{\rm rel}
   ={1\over4}L(z)^Tv_{\rm rel}.
\]

The product, sum, difference, and scale interval operations contain these
exact values even though \(q,z,v_{\rm rel}\) may be correlated in the actual
state. Similarly, successful division by the positive **radius** interval
makes

\[
 h={1\over2}|v_{\rm rel}|^2-{m_i+m_j\over r}
\]

lie in the stored \(h\)-interval. The affine formulas contain

\[
 R=\beta q_i+\alpha q_j,\quad U=\beta v_i+\alpha v_j,
 \quad y=q_k-R,\quad V=v_k-U.
\]

Consequently, for every exact Cartesian state in the source box and every
successful branch construction, at least one patch contains all 13
components of its coherent exact lift.

## 5. The negative cut, deck antipodality, and the F2 graph

In the fourth case, \(b<0\) and \(c<0<d\). Rust splits the source rectangle
into

\[
 B_-=X\times[c,0],\qquad B_+=X\times[0,d].
\]

Their union is the original box. The lower patch covers \(B_-\), the upper
patch covers \(B_+\), and both include the seam \(X\times\{0\}\). Because
\(b<0\), that seam is a punctured negative-axis interval, never the origin.

At a seam point write \(x=-s\) with \(s>0\). Then \(r=s\), \(u=0\), and
\(v=\sqrt{s}\). The upper and lower exact lifts are

\[
 z_+=(0,\sqrt{s}),\qquad z_-=(0,-\sqrt{s})=-z_+.
\]

For the same relative velocity,

\[
 w_+={1\over2}(\sqrt{s}\,v_y,-\sqrt{s}\,v_x),
 \qquad w_-=-w_+.
\]

All other exact lift components agree. The transition on the overlap is
therefore exactly the deck involution

\[
 (z,w,h,R,U,y,V)\longmapsto(-z,-w,h,R,U,y,V).
\]

This proves parity \(1\) for the sole edge from patch 0 (upper) to patch 1
(lower), matching **PlanarLcLiftParityEdge { source_patch: 0,
target_patch: 1, parity: 1 }**. The equation

\[
 g_0\mathbin{\mathrm{XOR}}g_1=1
\]

has exactly the complementary assignments

\[
 (g_0,g_1)=(0,1)\quad\hbox{and}\quad(1,0).
\]

For any singleton cover, the connected one-vertex graph has the two
complementary assignments \((0)\) and \((1)\). The consuming entry code in
**canonical_gauge_assignments_from_graph** accepts exactly these singleton and
parity-one shapes. It then checks both complete transformed patch families
against the target anchor. The parity proof is pointwise on the negative-axis
overlap; it does not claim that the two widened full patch rectangles are
literal deck transforms of each other.

## 6. Positive rho and the constrained actual lift

For every pointwise square-root lift,

\[
 \rho=|z|^2=u^2+v^2=A+B_r=r=|q|>0.
\]

The code also computes an interval enclosure of \(z_x^2+z_y^2\) for each
patch and stores its lower endpoint. **complete_cover** is true only if every
stored lower endpoint is strictly positive. Therefore a complete patch gives
the stronger universal conclusion that every \(z\) in its rectangular
\(z\)-projection has \(\rho>0\). The analytic kernel ID is exposed only in
that complete case. The carried-entry replay separately requires the same
strict positivity for every patch before its positive-rho obligation can
pass.

For the coherent exact lift of an actual Cartesian state,

\[
 \Lambda(z)\Lambda(z)^T=\rho I,
\]

so

\[
 2|w|^2={1\over2}\rho|v_{\rm rel}|^2.
\]

Using \(h=|v_{\rm rel}|^2/2-(m_i+m_j)/\rho\) gives

\[
 2|w|^2-(m_i+m_j)-\rho h=0.
\]

This conclusion is existential in exactly the sense required by Lemma 2. The
ordinary endpoint theorem supplies one actual Cartesian state in the source
box; the cover selects a coherent lift of that state; interval inclusion puts
that lift in at least one complete patch; and the displayed algebra proves
the constraint for that lift.

It is not a universal constraint statement about a patch rectangle. The
stored \(z,w,h\) intervals contain independent combinations introduced by
interval dependency, and many such combinations need not project to a common
Cartesian state or satisfy the constraint. Only the positive-\(\rho\) claim is
universal over a patch when its computed rho lower bound is positive.

The deck choices preserve the physical data because

\[
 Q(-z)=Q(z),\qquad L(-z)(-w)=L(z)w,
 \qquad|-z|^2=|z|^2.
\]

Thus either complementary F2 assignment represents the same Cartesian
source state and physical clock rate.

## 7. Fail-closed control flow and resource behavior

The lift function returns no complete evidence when the Cartesian dimension
is wrong, rational admission fails, the mass profile fails, a square-root
domain check fails, interval arithmetic fails, or configured square-root
precision exceeds the hard limit. The square-root kernel caps precision at
16,384 bits before shifting, uses the shared rational-component ceiling, and
rechecks exact lower-square, upper-square, dyadic-grid, and width
postconditions before returning an enclosure.

Every lift and arithmetic failure propagates as **Err**; the entry replay does
not turn it into a satisfied obligation. A noncollision box classified as
unresolved returns a replay with no patches, no parity edges,
**complete_cover = false**, and no analytic kernel ID. A constructed atlas
with any nonpositive patch rho lower bound also has
**complete_cover = false**. In the entry consumer, atlas reconstruction,
positive rho, analytic-kernel presence, graph certification, and one globally
contained complement are separate required Booleans. Failure of any one
prevents certification.

The failure-disposition layer maps an excessive requested square-root
precision and rational-component bit ceilings to bounded-resource evidence.
Domain and internal-postcondition failures also reject. Process-level memory
allocation failure or a Rust panic is outside the typed **Result** model; it
still cannot produce an accepted replay through this path, but graceful
totalization of such a process failure belongs to the executable runner and
is not proved here.

## 8. Trust boundary and genuine residual gaps

This proof assumes:

- the reviewed Rust expressions and branch order have their ordinary
  mathematical meaning;
- **num_bigint::BigInt** and **num_rational::BigRational** correctly implement
  normalized integer/rational arithmetic and comparison;
- **RationalInterval** operations return the endpoint hulls reviewed above;
- **sqrt_enclosure_dyadic** satisfies the exact postconditions it rechecks;
- the compiled verifier corresponds to the reviewed source; and
- every returned error, false obligation, absent kernel ID, or process failure
  is rejection rather than acceptance.

No proof assistant models this Rust, its libraries, compiler, or binary. That
foundational boundary remains a genuine open gate, consistent with the
theorem note and the raw-v1 specification.

There are also two narrower source-level limitations worth preserving in the
record.

### Rectangular rho dependency loses valid covers

Collision freedom does not force the independently computed rectangular
**rho** lower bound to be positive. For example,

\[
 X=[-1,1],\qquad Y=[1,2]
\]

has \(\underline D=1>0\) and selects the upper branch. Yet the natural
interval evaluations of both \((r+x)/2\) and \((r-x)/2\) have lower endpoint
zero: the same lower radius endpoint is combined independently with opposite
extremes of \(X\). Both stored square-root component intervals can therefore
have lower endpoint zero, and the computed rectangular rho lower bound is
zero. Pointwise, every coherent lift still has \(\rho=r\ge1\); the verifier
nevertheless marks the cover incomplete. A dependency-aware rho witness, for
example the already available positive radius lower bound, could recover such
cases, but the current stricter rectangle test is sound and fail closed.

### Very small positive radii remain a fixed-resolution completeness limit

At precision \(p\), a positive distance-squared lower endpoint below
\(2^{-2p}\) can receive a square-root enclosure whose lower grid endpoint is
zero. The pointwise radius is positive, but the subsequent interval division
used to construct \(h\) cannot accept a zero-containing radius interval. With
the default \(p=256\), admitted rationals can express such smaller positive
values. The lift now checks the verified radius enclosure before any radicand
or \(h\)-division work and returns **PositiveRadiusUnresolved** with the
selected precision. The failure-disposition layer normalizes that typed error
as an unmeasured **SquareRootPrecision** resource limit. Generic numeric
division-by-zero and internal-shape failures remain internal invariants.

This repairs the diagnostic attribution without changing acceptance,
rejection, the public verifier profile, or the fixed precision. These inputs
still fail closed, so the fixed-resolution completeness limitation and the
foundational trust boundary above both remain.

Subject to these explicit boundaries, no ordered-case, equality-boundary,
square-root-identity, negative-cut, deck-parity, or existential-containment
gap was found in the canonical lift-cover construction.
