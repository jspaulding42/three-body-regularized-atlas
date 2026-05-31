# Compact-Sundman Escape Endpoint Obstruction

## Claim

The compact-Sundman parameter

```text
w = tanh(lambda_s * s)
```

cannot by itself be the all-real-physical-time global parameter for the general three-body problem. There are exact noncollision escape solutions for which the Sundman time `s` has a finite future limit as physical time `t -> +infinity`. Therefore `w` also has a finite interior limit `w_* < 1`.

This is an obstruction to a specific global proof route, not to Sundman's global-series idea as a whole. It says that a proof which only exhausts the endpoint `w -> 1` of compactified Sundman time is proving the wrong global coverage statement for escape data. The all-time route must add physical-time compactification, change the independent variable, or explicitly handle interior escape endpoints.

## Exact Escape Family

Let `Q = (Q_1, Q_2, Q_3)` be a centered equal-mass central configuration:

```text
a_i(Q) = -mu Q_i.
```

For an equilateral central configuration, `mu > 0`. A homothetic solution has the form:

```text
q_i(t) = R(t) Q_i,
v_i(t) = R'(t) Q_i,
```

where the scalar scale factor solves:

```text
R''(t) = -mu / R(t)^2.
```

The radial energy

```text
E = (1/2) R'(t)^2 - mu / R(t)
```

is conserved. If `E > 0` and `R'(0) > 0`, then the branch expands forever and:

```text
R'(t)^2 = 2E + 2mu / R(t) >= 2E.
```

Thus, with `c = sqrt(2E)`,

```text
R(t) >= R(0) + c t.
```

## Finite Sundman Future

For the default compact-Sundman lift in this repository, the Sundman factor is:

```text
g(q) = product_{i<j} |q_i - q_j|.
```

Along the homothetic branch:

```text
g(q(t)) = C R(t)^3,
C = product_{i<j} |Q_i - Q_j| > 0.
```

Since `dt/ds = g(q)`, we have:

```text
ds/dt = 1 / (C R(t)^3).
```

Therefore:

```text
s_infty - s(0)
  = integral_0^infty dt / (C R(t)^3)
  <= integral_0^infty dt / (C (R(0) + c t)^3)
  = 1 / (2 C c R(0)^2)
  < infinity.
```

Consequently:

```text
w_infty = tanh(lambda_s * s_infty) < 1.
```

The physical solution exists for every finite future time, but `t = +infinity` accumulates at an interior compact-Sundman value `w_*`.

## Consequence

The compact-Sundman finite-atlas and shell-recurrence work remains useful for finite target-time enclosures and for local analytic continuation in `w`. But a theorem claiming all real physical target times cannot be completed by only proving shell exhaustion toward `w = +/-1`.

For the general closed-form target, this changes the remaining proof obligation:

```text
wrong target:  prove compact-Sundman shells exhaust w -> +/-1 and infer all physical times;
right target:  prove physical-time coverage separately, or handle finite interior escape endpoints.
```

This explains why the current theorem harness correctly separates compact-time real-line coverage from Sundman-time targeting. The compact-Sundman recurrence may still be part of the local construction and verification pipeline, but it is not alone an all-real-time global parameter.

## Domain-Exhaustion Obstruction

The obstruction is stronger than a missing endpoint argument. Suppose a compact-Sundman shell induction uses centered domains

```text
[-1 + delta_n, 1 - delta_n],
delta_n -> 0.
```

For the escape branch above, `w_infty < 1`. Therefore there is an index `N` with:

```text
1 - delta_N > w_infty.
```

Any proof that the unscaled position curve `q(w)` is finite and analytic on that whole compact domain would have to cross the escape endpoint. But as `w -> w_infty` from below, physical time tends to `+infinity` and:

```text
|q_i(t)| = R(t) |Q_i| -> infinity
```

for every nonzero body vector `Q_i`. Thus there is no finite-valued continuation of the unscaled physical positions through `w_infty`.

Equivalently, for every finite state bound `B`, choose

```text
t > (B / max_i |Q_i| - R(0)) / sqrt(2E).
```

Then `w(t) < w_infty`, but:

```text
max_i |q_i(t)| > B.
```

So an all-future recurrence in the compact-Sundman variable cannot have a global finite state-supremum envelope on the unscaled coordinates for arbitrary noncollision data. It must either stop at and classify the finite interior escape endpoint, switch to physical-time compactification, or lift into scaled variables that keep escape finite.
