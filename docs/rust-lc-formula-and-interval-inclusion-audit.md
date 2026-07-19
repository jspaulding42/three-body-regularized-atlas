# Rust LC formula-correspondence and interval-inclusion audit

## Status and scope

This is a source-level proof audit of the planar Levi--Civita (LC) arithmetic
used by the proof-grade supplied-chain theorem. Conditional on the arithmetic
trust boundary in Section 7, it discharges two implementation obligations:

1. the Rust LC field, lift, mass-profile, and projection kernels evaluate the
   formulas used in Lemma 5 of
   [the theorem note](proof-grade-supplied-planar-chain-theorem.md); and
2. every successful rational-interval evaluation encloses the corresponding
   exact real evaluation, including the complete positive-\(\rho\) Cartesian
   projection slice.

This is not a machine-checked Rust semantics proof. It audits `interval.rs`,
`dual.rs`, `sqrt.rs`, `outward_mass.rs`, `planar_lc_field.rs`,
`planar_lc_lift.rs`, and `planar_lc_projection.rs` in the source snapshot
committed with this note.

## 1. Coordinates and selected-pair map

The lifted order is exactly

\[
(z_x,z_y,w_x,w_y,h,R_x,R_y,U_x,U_y,y_x,y_y,V_x,V_y,t),
\]

at indices \(0,\ldots,13\), respectively:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name | \(z_x\) | \(z_y\) | \(w_x\) | \(w_y\) | \(h\) | \(R_x\) | \(R_y\) | \(U_x\) | \(U_y\) | \(y_x\) | \(y_y\) | \(V_x\) | \(V_y\) | \(t\) |

The Cartesian lift input is

\[
(q_{0x},q_{0y},q_{1x},q_{1y},q_{2x},q_{2y},
v_{0x},v_{0y},v_{1x},v_{1y},v_{2x},v_{2y}).
\]

`body_vector(state,b,offset)` selects `offset+2*b` and `offset+2*b+1`,
with offset 0 for position and 6 for velocity. Semantic admission requires
\(0\le i<j<3\). `relative_pair` returns second minus first, so
\(q=q_j-q_i\) and \(v=v_j-v_i\). The complete finite map is:

| pair \([i,j]\) | third \(k\) | relative blocks |
|---|---:|---|
| `[0,1]` | 2 | \(q_1-q_0,\ v_1-v_0\) |
| `[0,2]` | 1 | \(q_2-q_0,\ v_2-v_0\) |
| `[1,2]` | 0 | \(q_2-q_1,\ v_2-v_1\) |

The finite complement in `validate_mass_inputs` returns exactly this \(k\).

## 2. Exact mass witnesses for all three pairs

The input masses are positive finite binary64 values decoded as exact dyadic
rationals. For each pair the mass profile recomputes

\[
M=m_i+m_j,\quad \alpha={m_j\over M},\quad \beta={m_i\over M},
\quad \gamma={m_k\over M},
\]

\[
c_i={m_im_k\over M},\quad c_j={m_jm_k\over M},\quad
o_i=m_i\left(1+{m_k\over M}\right),\quad
o_j=m_j\left(1+{m_k\over M}\right).
\]

These are `pair_mass`, `alpha`, `beta`, `third_over_pair`,
`center_first`, `center_second`, `offset_first`, and `offset_second`.
Substitution for every admitted pair is:

| pair | \(M\) | \(\alpha,\beta\) | \(c_i,c_j\) | \(o_i,o_j\) |
|---|---|---|---|---|
| `[0,1]` | \(m_0+m_1\) | \(m_1/M,m_0/M\) | \(m_0m_2/M,m_1m_2/M\) | \(m_0(1+m_2/M),m_1(1+m_2/M)\) |
| `[0,2]` | \(m_0+m_2\) | \(m_2/M,m_0/M\) | \(m_0m_1/M,m_2m_1/M\) | \(m_0(1+m_1/M),m_2(1+m_1/M)\) |
| `[1,2]` | \(m_1+m_2\) | \(m_2/M,m_1/M\) | \(m_1m_0/M,m_2m_0/M\) | \(m_1(1+m_0/M),m_2(1+m_0/M)\) |

`validate_against` checks mass bits, ordered pair, complement, and every exact
formula. The audited field, lift, and projection consume `.exact()`, not the
additional outward-binary64 members; no rounded mass ratio enters them.

## 3. Formula correspondence

Put

\[
\Lambda(z)=\begin{pmatrix}z_x&-z_y\\z_y&z_x\end{pmatrix},\quad
L(z)=2\Lambda(z),\quad \rho=z_x^2+z_y^2,
\]

\[
Q(z)=(z_x^2-z_y^2,2z_xz_y)=q.
\]

### Lift

`planar_lc_lift.rs` computes

\[
R=\beta q_i+\alpha q_j,\quad U=\beta v_i+\alpha v_j,
\quad y=q_k-R,\quad V=v_k-U.
\]

Its cover constructs interval branches for \(Q(z)=q\). It then computes

\[
w_x={z_xv_x+z_yv_y\over2},\qquad
w_y={z_xv_y-z_yv_x\over2},
\]

so \(w=\tfrac12\Lambda^Tv=\tfrac14L^Tv\). Since
\(\Lambda^T\Lambda=\rho I\), on \(\rho>0\),

\[
v={2\Lambda w\over\rho}={Lw\over\rho}.
\]

The remaining scalar is \(h=\tfrac12|v|^2-M/|q|\). These are exactly the
entry-lift conventions used by the theorem.

### All 14 field components

Let

\[
d_i=y+\alpha q,\quad d_j=y-\beta q,\quad
f_i={d_i\over|d_i|^3},\quad f_j={d_j\over|d_j|^3},
\]

\[
P=m_k(f_j-f_i),\quad B=c_if_i+c_jf_j,
\quad Y=-o_if_i-o_jf_j.
\]

Reading `rhs_dual` in the coordinate order above gives exactly

\[
\begin{aligned}
z'&=w,\\
w'&={h\over2}z+{\rho\over2}\Lambda^TP
    ={h\over2}z+{\rho\over4}L^TP,\\
h'&=(2\Lambda w)\cdot P=(Lw)\cdot P,\\
R'&=\rho U,&U'&=\rho B,\\
y'&=\rho V,&V'&=\rho Y,\\
t'&=\rho.
\end{aligned}
\]

The code's `ltp` is \(2\Lambda^TP\), subsequently multiplied by `rho/4`;
`qprime` is \(2\Lambda w\). The Section 2 substitutions make `rdd` equal to
\(B\) and `ydd` equal to \(Y\). This accounts for all 14 slots without a
permutation or unused component.

The inverse-cube helper forms \(|d|^2\), requires a strictly positive interval
lower bound, encloses its square root, forms \(|d|^3\), and takes a reciprocal.
Thus every successful field box excludes both relevant third-body collisions.

### Punctured Cartesian projection

`project_planar_lc_full_state_exact_rational` recomputes and validates the
same mass profile and requires a strictly positive lower bound for \(\rho\).
It computes

\[
q=Q(z),\qquad
v={2\over\rho}(z_xw_x-z_yw_y,z_yw_x+z_xw_y)={Lw\over\rho},
\]

and then, on both axes,

\[
q_i=R-\alpha q,\ q_j=R+\beta q,\ q_k=R+y,
\]

\[
v_i=U-\alpha v,\ v_j=U+\beta v,\ v_k=U+V.
\]

These invert the lift blocks. They imply
\(q_k-q_i=y+\alpha q=d_i\) and
\(q_k-q_j=y-\beta q=d_j\), matching the theorem's force convention.
They are invariant under \((z,w)\mapsto(-z,-w)\). Formula correspondence is
therefore established for every admitted canonical pair.

## 4. Rational-interval inclusion

For an interval \(X=[\underline X,\overline X]\), the successful-operation
contract is: if each exact input lies in its interval, the exact result lies in
the returned interval. The proof is structural:

- addition and subtraction use their monotone endpoint formulas;
- multiplication takes the min/max of all four corner products, which are the
  extrema of a bilinear function on a rectangle;
- scaling orders endpoint products according to the exact scalar's sign;
- reciprocal rejects zero-containing intervals and returns \([1/b,1/a]\),
  sound because reciprocal is decreasing on either remaining sign component;
- division is multiplication by the reciprocal enclosure;
- the dependency-aware square returns the exact interval hull of \(x^2\);
- Horner starts with the point interval for the highest coefficient and
  repeatedly multiplies and adds. Induction over coefficients proves
  componentwise polynomial inclusion.

Resource and domain failures propagate as `Err`; these paths do not return an
alleged enclosure.

### Square root

For nonnegative rational \(r\), `sqrt_enclosure_dyadic` constructs nonnegative
dyadic \(l,u\) and rechecks exactly that \(l^2\le r\le u^2\). Hence
\(l\le\sqrt r\le u\). For \([a,b]\subseteq[0,\infty)\), the helpers combine
the lower endpoint for \(a\) with the upper endpoint for \(b\); monotonicity
encloses \(\sqrt x\) throughout the interval. Dual square root requires
\(a>0\) and a root interval excluding zero before forming
\(1/(2\sqrt X)\), so every successful derivative enclosure is finite.

## 5. Dual inclusion, Jacobians, and row sums

Each coordinate interval \(X_r\) is seeded with derivative vector \(e_r\).
Assuming two dual expressions enclose their values and all first partials, the
implemented rules are the exact identities

\[
(A\mathbin{\pm}B)'=A'\mathbin{\pm}B',\quad
(AB)'=A'B+AB',\quad (A^{-1})'=-A'A^{-2},
\]

\[
(A^2)'=2AA',\qquad (\sqrt A)'={A'\over2\sqrt A},
\]

evaluated with the inclusion operations proved above. Domain checks make
reciprocal and square-root derivatives finite. Structural induction over the
field expression tree therefore proves that `rhs` contains every field value
and `jacobian[row][column]` contains the corresponding exact partial
throughout the box.

For derivative interval \([a,b]\), \(\max(|a|,|b|)\) bounds every contained
absolute value. `maximum_row_sum` sums these bounds by row and takes the
maximum, so it bounds

\[
\sup_{x\in X}\|DF_{LC}(x)\|_\infty
=\sup_{x\in X}\max_r\sum_c|\partial_cF_r(x)|.
\]

Dependency overestimation can enlarge this result but cannot make it unsound.

## 6. Complete positive-\(\rho\) slice projection

Let \(X\) be a complete 14-component lifted slice and let the exact state
\(x\) satisfy \(x_r\in X_r\) for every component. Suppose projection succeeds,
so its computed \(\rho\) interval has positive lower endpoint. Then:

1. square arithmetic contains \(\rho(x)\) and \(Q(z(x))\);
2. product/sum arithmetic contains both components of \(L(z(x))w(x)\);
3. positive-\(\rho\) division contains \(L(z(x))w(x)/\rho(x)\);
4. exact mass point scalars and affine interval operations contain all six
   projected position and velocity vectors; and
5. \(i,j,k\) are a permutation of \(0,1,2\), so the loop writes every body.

Thus the returned Cartesian box contains the exact punctured projection of
**every** exact point in the complete positive-\(\rho\) slice, not only a
polynomial center or samples. Combined with the separate tube theorem that
places the selected exact lifted solution in \(X\), this is precisely the exit
inclusion premise used in Lemma 6.

## 7. Arithmetic trust boundary and residual gap

The two Lemma 5 gates are closed at source-audit level, not at foundational
implementation-verification level. This proof assumes:

- Rust executes the reviewed expressions and control flow correctly;
- `num_bigint::BigInt` and `num_rational::BigRational` implement mathematical
  integer/rational arithmetic and comparison correctly;
- admitted rationals satisfy the checked invariants before arithmetic;
- the reviewed source is the source compiled into the verifier; and
- every `Err` is rejection, never acceptance.

There is no proof-assistant Rust model, independently verified big-integer
kernel, translation validation, or proof-producing compilation here. Closing
that residual gap would require a different artifact, such as a formal model
linked to the executable by verified extraction or independently checkable
proof certificates with a separately justified arithmetic kernel.

Subject to this explicit boundary, the formula correspondence, interval/dual
structural inclusion, dyadic-square-root and Horner inclusion, Jacobian row-sum
bound, and complete positive-\(\rho\) projection inclusion are proved above.
