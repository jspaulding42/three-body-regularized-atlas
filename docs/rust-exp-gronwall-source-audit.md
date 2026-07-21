# Rust exponential and Grönwall source audit

## Status and claim

This note audits the exact-rational exponential kernel in `exp.rs` and its
ordinary and planar Levi--Civita (LC) tube call paths in `ordinary_tube.rs` and
`planar_lc_tube.rs`. It is a source-level mathematical proof, not a formal
machine proof of Rust.

Subject to the trust boundary in Section 4, every successful call to
`exp_enclosure_rational(x,...)` with \(x\ge0\) returns an exact rational upper
bound \(B\ge e^x\). Both tube paths use this bound monotonically to compute an
upper bound for the quantity \(E\) in Theorem 1 and require the strict
comparison \(E<r\).

## 1. Exact rational exponential enclosure

### Range reduction

`range_reduce` starts from an admitted nonnegative rational \(x\) and divides
by two until

\[
y={x\over2^k}\le {1\over2}.
\]

The divisions are exact rational operations. If \(k>0\), the replay also
checks \(2y>1/2\), making \(k\) the deterministic minimal reduction count.
The identity later used is

\[
e^x=(e^y)^{2^k}.
\]

The zero input is included: it gives \(k=0\), \(y=0\).

### Taylor sum and geometric tail

For a supplied cutoff \(N\), `taylor_tail_data` constructs by exact recurrence

\[
t_0=1,\qquad t_n={t_{n-1}y\over n}={y^n\over n!},\qquad
S_N=\sum_{n=0}^N t_n.
\]

If \(y=0\), every positive-order term, the omitted remainder, and the
constructed tail bound \(T\) are exactly zero, so the enclosure is the point
\([1,1]\). Now suppose \(y>0\). Then every term is positive, and for every
term after the first omitted one,

\[
{t_{n+1}\over t_n}={y\over n+1}\le {y\over N+2}=:q
\quad(n\ge N+1).
\]

Because \(0<y\le1/2\), \(0<q<1\). The nonnegative Taylor remainder is
therefore bounded by the geometric series

\[
0\le e^y-S_N\le {t_{N+1}\over1-q}=:T.
\]

The code explicitly checks \(q\ge0\), \(q<1\), \(1-q>0\), \(T\ge0\), and
the exact identity \(T(1-q)=t_{N+1}\). It also requires
\(T\) not to exceed the caller's positive reduced-tail ceiling. Hence

\[
e^y\in[S_N,S_N+T].
\]

### Positive repeated squaring

The reduced interval is nonnegative. At each of the \(k\) transcript steps,
the kernel maps \([a,b]\) to the exact interval \([a^2,b^2]\). Since squaring
is increasing on \([0,\infty)\), induction gives

\[
(e^y)^{2^j}\in[a_j,b_j]
\]

after step \(j\), and after \(k\) steps \(e^x\in[a_k,b_k]\). The returned
`upper_bound()` is exactly \(b_k\).

### Mandatory witness replay

Construction does not merely cache the computed endpoint. Before returning,
`verify_with_budget` replays and compares:

- canonical admitted inputs and all intermediate rational fields;
- deterministic \(k\) and \(y\);
- the complete Taylor recurrence, \(S_N\), \(t_{N+1}\), \(q\), \(1-q\), and
  \(T\);
- the declared tail-ceiling inequality;
- both reduced-interval endpoints; and
- every repeated-squaring interval and the final interval.

The standalone `verify` method performs the same replay with a fresh bounded
work budget. These exact algebraic postconditions validate the stored witness;
the Taylor and geometric-series argument above is the mathematical reason
those postconditions enclose the transcendental exponential.

## 2. Ordinary-tube Grönwall path

The ordinary replay recomputes nonnegative quantities

\[
\widehat\delta,\quad \widehat L,\quad
h=\max(|a_--s_0|,|a_+-s_0|),\quad \epsilon,
\]

where \(\widehat\delta\) is the maximum absolute interval-defect endpoint and
\(\widehat L\) is the analytic complete-tube Lipschitz bound. The semantic
input check requires \(\epsilon\ge0\). The analytic construction initializes
\(\widehat L\) to one and takes maxima with nonnegative body bounds, so this
path has \(\widehat L\ge1\); division by zero cannot occur.

It computes \(x=\widehat Lh\), then `dyadic_upper` constructs the exact ceiling
\(\widehat x\ge x\) on a fixed dyadic grid. Exponential monotonicity and
Section 1 give

\[
B\ge e^{\widehat x}\ge e^{\widehat Lh}.
\]

`gronwall_upper` then evaluates exactly

\[
\widehat E
=B\epsilon+\widehat\delta{B-1\over\widehat L}.
\]

All factors are nonnegative. Replacing \(e^{\widehat Lh}\) by its upper bound
\(B\) can only increase both terms, so

\[
\widehat E\ge
e^{\widehat Lh}\epsilon+\widehat\delta
{e^{\widehat Lh}-1\over\widehat L}.
\]

Theorem 1 may use the recomputed outward values \(\widehat\delta\) and
\(\widehat L\) themselves as its defect and Lipschitz constants. Supplied
maximum fields are only separately checked caps and do not replace either
recomputed value. The replay accepts the Grönwall obligation only when the
exact comparison `gronwall_upper < tube_radius` succeeds. Equality rejects, as
the theorem requires.

The ordinary helper has no explicit \(L=0\) branch, but this is not a missing
case on its call path: the constructed ordinary \(\widehat L\) is at least one.

## 3. Planar-LC-tube Grönwall path

The LC replay similarly obtains:

- \(\widehat\delta\ge0\) from the maximum absolute direct lifted-defect
  endpoint;
- \(\widehat L\ge0\) from the interval-Jacobian maximum row sum over the
  complete radius-inflated lifted state box;
- the exact two-sided horizon \(h\ge0\); and
- the admitted initial error bound \(\epsilon\ge0\).

Only after the third-body denominator boxes are strictly separated does it
form \(\widehat Lh\), round it upward dyadically, and obtain
\(B\ge e^{\widehat Lh}\) through the same exponential kernel. When
\(\widehat L>0\), the LC helper uses the same exact formula

\[
\widehat E=B\epsilon+\widehat\delta{B-1\over\widehat L},
\]

and the same nonnegative monotonicity argument applies.

When \(\widehat L=0\), the helper does not divide: it returns exactly

\[
\widehat E=\epsilon+\widehat\delta h,
\]

which is Theorem 1's zero-Lipschitz branch. The exponential call made before
this branch receives exponent zero and returns the exact point interval
\([1,1]\), but its value is not needed by the zero branch. In the present LC
field, the rows \(z'_x=w_x\) and \(z'_y=w_y\) normally force the interval
Jacobian row-sum maximum to be at least one; nevertheless, the helper's
zero-case formula is independently correct.

Finally, `strictly_inside_radius` is exactly `bound < radius`. Equality is a
false obligation, not acceptance.

## 4. Fail-closed behavior and trust boundary

The exponential kernel rejects negative exponents, nonpositive tail ceilings,
malformed or noncanonical rationals, invalid or excessive configurable limits,
excessive range reductions or Taylor cutoff, insufficient Taylor cutoff,
division by zero, intermediate bit overflow, work-budget exhaustion, witness
storage exhaustion, and any replay mismatch. Hard ceilings cannot be relaxed
by caller configuration. The tube callers propagate exponential and rational
arithmetic errors as replay errors; they do not turn them into satisfied tube
obligations. Domain failures represented as false obligations likewise do not
make `certified()` true.

This proof assumes:

- Rust executes the reviewed expressions and control flow correctly;
- `num_bigint::BigInt` and `num_rational::BigRational` correctly implement
  mathematical integer and normalized rational arithmetic/comparison;
- the standard real exponential has its usual power-series, monotonicity, and
  algebraic laws;
- the source compiled into the verifier is identical to the reviewed source;
  and
- every returned `Err` is treated as rejection by the enclosing verifier.

No proof assistant verifies Rust, the big-integer library, compiler, linked
binary, or source-to-binary identity here. Subject to this explicit boundary,
the exact rational exponential upper enclosure and both complete Grönwall
computations match Theorem 1. The remaining gap is foundational verification
of that boundary, not an omitted range-reduction, tail, squaring, \(L=0\), or
strict-radius argument.
