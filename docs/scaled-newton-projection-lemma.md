# Scaled Newton Projection Lemma

## Claim

The escape-scaled coordinate

```text
q_i(t) = (1 + t) X_i(t)
```

is not only a boundedness trick. It gives an exact lifted Newton equation whose projection recovers the original Newtonian equations on every finite interval where `1 + t != 0`.

This is the constructive algebra needed after the escape-scaled-variable lemma: the lifted space can carry scaled variables, and projection back to unscaled positions remains mechanically verifiable.

## Setup

Let `A_i(q)` denote the Newtonian acceleration field:

```text
A_i(q) = sum_{j != i} m_j (q_j - q_i) / |q_j - q_i|^3.
```

This field is homogeneous of degree `-2`:

```text
A_i(aX) = a^(-2) A_i(X)
```

for every positive scalar `a`.

Set:

```text
a(t) = 1 + t,
q_i(t) = a(t) X_i(t).
```

Then:

```text
q_i''(t) = a(t) X_i''(t) + 2 X_i'(t).
```

## Lifted Equation

Therefore `q(t)` satisfies Newton's equations

```text
q_i'' = A_i(q)
```

if and only if the scaled variables satisfy:

```text
(1 + t) X_i'' + 2 X_i'
  = (1 + t)^(-2) A_i(X).
```

The projection identity is exact:

```text
q_i(t) = (1 + t) X_i(t).
```

So any proof in the scaled variables can be projected back to an unscaled Newtonian solution at every finite target time.

## Compact Physical Time Form

With compact physical time

```text
u = tanh(lambda_t t),
t = atanh(u) / lambda_t,
T(u) = dt/du = 1 / (lambda_t (1 - u^2)),
a(u) = 1 + t(u),
```

the same lifted equation becomes:

```text
a X_{uu} + (2T - a T_u/T) X_u
  = T^2 a^(-2) A(X).
```

The derivation is a direct chain-rule calculation. Since `q = aX` and `a_u = T`,

```text
q_u = T X + a X_u,
q_uu = T_u X + 2T X_u + a X_uu.
```

Also,

```text
q_tt = (q_uu - q_u T_u/T) / T^2.
```

Substituting `q_u` and `q_uu` cancels the `T_u X` term and gives:

```text
q_tt = [a X_uu + (2T - a T_u/T) X_u] / T^2.
```

Using Newton's equation `q_tt = A(q) = a^(-2)A(X)` gives the compact-time scaled equation above.

This is the equation a compact-time global construction should use near hyperbolic escape if it wants bounded state variables. The unscaled positions are then recovered by multiplying by the explicit scale `a(u)`.

## Consequence

The previous escape lemmas identified a failure mode for unscaled global state bounds. This lemma gives the corresponding repair at the equation level:

```text
lift:      solve the scaled equation for X
construct: prove bounded or endpoint-controlled X in the compact domain
project:   q = (1+t)X for finite t
verify:    use homogeneity of A to recover q'' = A(q)
```

This still does not prove the general three-body solution. It supplies the exact lifted equation a future proof must use, or generalize, if it wants to include escape data in a bounded compact-time representation.
