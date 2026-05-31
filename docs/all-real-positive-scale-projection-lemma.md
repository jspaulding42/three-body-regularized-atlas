# All-Real Positive-Scale Projection Lemma

## Claim

The escape-scaled construction does not need the special scale `1 + t`, which has an artificial singularity at `t = -1`. For any positive scalar scale `a(t) > 0`, the lift

```text
q_i(t) = a(t) X_i(t)
```

has an exact projected Newton equation. Choosing, for example,

```text
a(t) = sqrt(1 + t^2)
```

keeps the scale positive for every real physical time and still captures linear escape growth.

## General Lifted Equation

Let `A(q)` be the Newtonian acceleration field. It is homogeneous of degree `-2`:

```text
A(aX) = a^(-2) A(X).
```

For `q = aX`,

```text
q'' = a X'' + 2a' X' + a'' X.
```

Thus `q'' = A(q)` holds if and only if:

```text
a X'' + 2a' X' + a'' X = a^(-2) A(X).
```

Projection back to unscaled coordinates is exact for every finite `t`:

```text
q(t) = a(t) X(t).
```

## Compact Physical-Time Form

Let

```text
u = tanh(lambda_t t),
T = dt/du,
a_u = da/du,
a_uu = d^2a/du^2.
```

The chain rule gives:

```text
q_tt = (q_uu - q_u T_u/T) / T^2.
```

Substituting `q = aX` yields the compact-time scaled equation:

```text
a X_uu
  + (2a_u - a T_u/T) X_u
  + (a_uu - a_u T_u/T) X
  = T^2 a^(-2) A(X).
```

This is the all-real positive-scale version of the compact-time scaled equation. The earlier `a(t)=1+t` formula is the special case where `a_u=T` and `a_uu=T_u`, so the final `X` coefficient cancels.

## Escape-Bounded Scale

For the exact positive-energy homothetic escape branch, `R(t) <= R_0 + V_0 t` for future time. With

```text
a(t) = sqrt(1 + t^2),
X_i(t) = R(t)Q_i / a(t),
```

the scaled coordinate is bounded for `t >= 0`:

```text
|X_i(t)| <= sqrt(R_0^2 + V_0^2) |Q_i|.
```

Also, since `R(t)/t -> sqrt(2E)`,

```text
X_i(t) -> sqrt(2E) Q_i
```

as `t -> +infinity`.

## Consequence

This removes a technical defect from the escape-scaled route. A future global construction can use an everywhere-positive scale such as `sqrt(1+t^2)`, construct bounded scaled variables near escape, project back to unscaled coordinates at finite target times, and verify Newton's equations by the positive-scale equation above.

This still does not prove the general solution. It provides the correct algebraic form for a bounded escape-compatible lift that does not introduce a finite-time scale singularity.
