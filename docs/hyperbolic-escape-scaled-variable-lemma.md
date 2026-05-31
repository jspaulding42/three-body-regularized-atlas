# Hyperbolic Escape Scaled-Variable Lemma

## Claim

For positive-energy homothetic escape, physical-time compactification makes unscaled positions diverge at `u = 1`, but the lifted variables

```text
X_i(t) = q_i(t) / (1 + t)
```

remain bounded and have a finite endpoint limit. This gives a concrete representation target for the global route: escape endpoints should be handled by scaled variables or explicit logarithmic singular terms before projecting back to unscaled positions at finite times.

## Setup

Use a centered equal-mass central configuration `Q` and the homothetic branch:

```text
q_i(t) = R(t) Q_i,
v_i(t) = R'(t) Q_i,
R''(t) = -mu / R(t)^2.
```

Assume the branch is expanding with:

```text
R(0) = R_0 > 0,
R'(0) = V_0 > 0,
E = (1/2)V_0^2 - mu/R_0 > 0.
```

Let:

```text
c = sqrt(2E).
```

The energy identity gives:

```text
R'(t)^2 = c^2 + 2mu / R(t).
```

Because `R(t) >= R_0`, we also have:

```text
0 < R'(t) <= V_0.
```

## Bounded Scaled Position

Since `R'(t) <= V_0`,

```text
R(t) <= R_0 + V_0 t.
```

Therefore:

```text
|X_i(t)|
  = R(t)|Q_i|/(1+t)
  <= max(R_0, V_0)|Q_i|.
```

So the scaled position remains bounded for all future physical time, including as compact physical time approaches:

```text
u = tanh(lambda_t t) -> 1^-.
```

## Endpoint Limit

The lower bound `R'(t) >= c > 0` implies `R(t) -> infinity`. Hence:

```text
R'(t) = sqrt(c^2 + 2mu/R(t)) -> c.
```

By averaging,

```text
R(t)/(1+t) -> c.
```

Thus:

```text
X_i(t) -> c Q_i.
```

The velocities also have a finite endpoint limit:

```text
v_i(t) = R'(t)Q_i -> c Q_i.
```

## Consequence

This is a constructive repair of the escape obstruction. The unscaled projection `q_i(t)` is still recovered at every finite target time by:

```text
q_i(t) = (1+t) X_i(t).
```

But the lifted construction should propagate `X_i` or a comparable escape-scaled coordinate near `u = 1`, not unscaled `q_i`. In compact physical time, the scale factor is explicit:

```text
1 + t(u) = 1 + atanh(u)/lambda_t.
```

So the lift/construct/project/verify route becomes:

```text
lift:      X_i(u) = q_i(t(u)) / (1 + t(u))
construct: prove bounded endpoint behavior for X_i
project:   q_i(t) = (1+t) X_i(t) for finite t
verify:    substitute back into Newton's equations on every finite interval
```

This does not prove the general three-body solution. It gives a mathematically necessary form for any global proof that wants to include escape data without falsely demanding a finite unscaled state bound at the compact endpoint.
