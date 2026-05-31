# Physical-Time Compactification Escape Growth

## Claim

Physical-time compactification

```text
u = tanh(lambda_t t)
```

correctly maps every finite physical target time to `-1 < u < 1`, but it does not make unscaled three-body positions bounded at the compact endpoint. Positive-energy escape data gives an explicit obstruction to any global proof that requires a finite state-supremum envelope for `q(u)` on the whole endpoint-exhausting physical-time compact domain.

This sharpens the route after the compact-Sundman escape obstruction: switching to compact physical time fixes all-real-time coverage, but a global finite-coordinate recurrence still needs scaled variables, endpoint singular terms, or a theorem that only requires convergence on the open interval.

## Exact Escape Family

Use the same centered equal-mass homothetic central-configuration branch:

```text
q_i(t) = R(t) Q_i,
v_i(t) = R'(t) Q_i,
R''(t) = -mu / R(t)^2.
```

The radial energy

```text
E = (1/2) R'(t)^2 - mu / R(t)
```

is conserved. If `E > 0` and the branch is expanding, then:

```text
R'(t)^2 = 2E + 2mu / R(t) >= 2E.
```

With `c = sqrt(2E)`, this implies:

```text
R(t) >= R(0) + c t.
```

## Endpoint Growth In Compact Physical Time

The inverse compact-time map is:

```text
t(u) = atanh(u) / lambda_t
     = (1 / (2 lambda_t)) log((1 + u) / (1 - u)).
```

Therefore, as `u -> 1^-`,

```text
R(t(u)) >= R(0) + (c / (2 lambda_t)) log((1 + u) / (1 - u)) -> infinity.
```

For any finite bound `B`, choose:

```text
t_B > (B / max_i |Q_i| - R(0)) / c,
u_B = tanh(lambda_t t_B).
```

Then `u_B < 1`, but:

```text
max_i |q_i(t_B)| > B.
```

Thus no finite state-supremum envelope can hold uniformly up to the physical-time compact endpoint for unscaled positions.

## Consequence

The scalar compact-time map is still the correct coverage mechanism for finite target times:

```text
t in R  <->  u in (-1, 1).
```

But the global closed-form route cannot ask for a bounded, finite-valued unscaled position recurrence on the closed compact interval. A viable proof has to choose one of these sharper targets:

1. Prove convergence of the unscaled series only on the open compact interval, allowing boundary divergence for escape.
2. Lift into scaled variables that remain finite at escape endpoints, then project back to unscaled positions for finite `t`.
3. Add explicit endpoint singular terms, such as logarithmic escape growth, before applying a bounded analytic recurrence.

This is the same lesson as the unit-distance-inspired method: the lift must put the real obstruction into the representation where the construction lives. Compactifying time alone is not enough; the escaping state variables also need the right representation.
