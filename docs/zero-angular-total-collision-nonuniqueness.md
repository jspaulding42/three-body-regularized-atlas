# Zero-Angular Total-Collision Nonuniqueness

## Claim

Zero-angular-momentum total collision does not have a branch-independent Newtonian continuation determined by the collision event alone. Any theorem that claims a general continued solution through total collision must therefore supply extra collision-manifold data or an explicit branch-selection convention.

This is weaker than a full continuation theorem, but it is a real obstruction: the remaining zero-angular-momentum triple-collision obligation cannot be closed by saying "continue through the collision" without specifying what data selects the outgoing branch.

Equivalently, there is no theorem of the following form:

```text
At a zero-angular total-collision event, the Newtonian equations plus the
collapsed position, masses, total energy, center-of-mass data, linear momentum,
and angular momentum determine one branch-independent outgoing solution.
```

The counterexample below gives two outgoing Newtonian branches with the same collapsed collision event and the same classical integral values.

## Homothetic Total-Collision Solutions

Take equal masses and gravitational constant `G = 1`. Let `c = (c_1, c_2, c_3)` be a centered central configuration:

```text
sum_i c_i = 0,
acc(c)_i = -lambda c_i,   lambda > 0.
```

Because Newtonian acceleration is homogeneous of degree `-2`,

```text
acc(r c)_i = r^(-2) acc(c)_i = -lambda r^(-2) c_i.
```

Choose

```text
r(t) = a |t|^(2/3),       a^3 = (9/2) lambda.
```

For `t != 0`,

```text
r''(t) = -(2/9) a |t|^(-4/3) = -lambda / r(t)^2.
```

So `q_i(t) = r(t)c_i` satisfies Newton's equations for every `t != 0`:

```text
q_i''(t) = r''(t)c_i = -lambda r(t)^(-2)c_i = acc(q(t))_i.
```

The motion has zero centered angular momentum because each velocity is parallel to its position:

```text
q_i(t) x q_i'(t) = r(t)r'(t) c_i x c_i = 0.
```

It has zero total energy in the parabolic case. If

```text
I(c) = sum_i |c_i|^2,       U(c) = sum_{i<j} 1/|c_i-c_j|,
```

then the central-configuration identity gives `lambda = U(c)/I(c)`, and

```text
E = (1/2) I(c) r'(t)^2 - U(c)/r(t)
  = (2/9) I(c)a^2 |t|^(-2/3) - U(c)a^(-1)|t|^(-2/3)
  = 0.
```

## Nonuniqueness

There is more than one equal-mass centered central configuration. Two explicit examples are:

```text
Equilateral:
  (1, 0), (-1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2),       lambda = 1/sqrt(3)

Euler collinear:
  (-1, 0), (0, 0), (1, 0),                              lambda = 5/4
```

Both produce parabolic zero-angular total-collision/ejection solutions. At `t = 0`, all bodies occupy the same point in both solutions. For `t > 0`, the equilateral and Euler ejections are different curves, yet each solves Newton's equations away from `0`, has zero angular momentum, and has zero energy.

Thus the collapsed position, zero angular momentum, and classical integral values do not select a unique outgoing branch. A valid zero-angular total-collision continuation theorem must add a convention such as collision-manifold coordinates plus a branch-selection rule. Without that extra data, the general closed-form target is underdetermined at this branch.

## Consequence for the Closed-Form Target

This does not disprove Sundman-style continued solutions with an explicit convention. It proves that the convention is part of the mathematical data. A claimed general closed-form solution must therefore do one of the following:

1. Exclude total collision from its time interval.
2. Stop at total collision and state that the maximal classical solution ends there.
3. Add collision-manifold branch data and prove a continuation rule using it.
4. Add a deterministic convention that deliberately selects one branch and prove that the selected branch satisfies the stated continuation semantics.

The harness should not certify the zero-angular total-collision branch from invariants alone, because the two homothetic examples have the same relevant invariant data at collision and different outgoing Newtonian curves.

## Resonant Branch Parameters

The obstruction is stronger than the homothetic equilateral-versus-Euler
choice. In resonant central-configuration cases, even the central-configuration
second jet and energy do not select a unique continuation.

For every arbitrary-mass equilateral resonance on the positive surface:

```text
beta = (m_1m_2+m_1m_3+m_2m_3)/(m_1+m_2+m_3)^2 = 8/27,
```

the regularized branch has the form:

```text
q(tau) = tau^2 S(tau),
S_0 = C,
S_1 = bD,
S_2 = E_0 b^2 + alpha C,
```

where `D` is the centered nontranslation kernel vector of `DA(C)`. The sign
choices `b` and `-b` have the same collapsed position, the same second
regularized-time jet, the same energy, and zero angular momentum, but the
outgoing curves differ at order `tau^3`. Masses proportional to `(1,1,5/2)`
are only one point on this resonant surface.

For the symmetric Euler resonances with masses:

```text
(1, 11/12, 1),   (1, 1/4, 1),   (1, 1/24, 1),
```

the resonant shape parameter enters at orders `tau^5`, `tau^6`, and `tau^7`.
The `+b` and `-b` branches again share the same second jet and energy while
producing different outgoing curves.

Thus a continuation rule cannot be a function only of the collapsed event,
classical integrals, central-configuration second jet, and homothetic energy
parameter. Resonant branch coordinates are additional mathematical data unless
the continuation convention deliberately chooses them by some extra rule.
