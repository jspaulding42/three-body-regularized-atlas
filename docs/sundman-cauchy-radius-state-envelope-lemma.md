# Sundman Cauchy-Radius State-Envelope Lemma

## Claim

The Sundman-time Cauchy radius used by the current majorant construction has an explicit lower bound in terms of ordinary state-envelope quantities. Combined with the compact `atanh` lower-bound lemma, this turns the compact lower-step premise into a concrete dynamical inequality.

This is a real recurrence subproof, not a new certificate layer: it proves what state control would be sufficient to keep future compact-Sundman steps from collapsing under Cauchy capping.

## Setup

For one noncollision three-body state, define:

```text
d = min_{i<j} |q_i - q_j| > 0,
D = max_{i<j} |q_i - q_j|,
V = max_i |v_i|,
M = m_1 + m_2 + m_3.
```

The point Sundman Cauchy majorant in the implementation chooses the default position radius

```text
rho = d / 10.
```

For every pair with distance `r_ij`, the complex position ball gives:

```text
r_ij^2 - 4 r_ij rho - 4 rho^2
```

as a lower bound for squared pair distance, and

```text
r_ij^2 + 4 r_ij rho + 4 rho^2
```

as an upper bound.

## Pair-Distance Bounds

Because `rho = d/10` and `r_ij >= d`,

```text
r_ij^2 - 4 r_ij rho - 4 rho^2
  >= d^2 - 4d(d/10) - 4(d/10)^2
  = (14/25)d^2.
```

Because `r_ij <= D`,

```text
r_ij^2 + 4 r_ij rho + 4 rho^2
  <= D^2 + (2D d)/5 + d^2/25
  = (D + d/5)^2.
```

For distance-power `p > 0`, the Sundman factor bound therefore satisfies:

```text
G <= (D + d/5)^(3p).
```

## Acceleration Bound

The Cauchy acceleration majorant has terms of the form:

```text
m_j * (r_ij + 2 rho) * (lower squared distance)^(-3/2).
```

Using the pair-distance bounds above and summing all possible attracting masses by `M` gives:

```text
A <= A_bar
```

where

```text
A_bar = M * (D + d/5) / (((14/25)^(3/2)) d^3).
```

The implementation chooses the velocity radius as:

```text
nu = sqrt(A rho).
```

Hence:

```text
nu <= nu_bar = sqrt(A_bar rho).
```

## Sundman-Radius Lower Bound

Before the final one-ulp inward rounding used for floating-point safety, the implemented certified Sundman radius is the minimum of the position and velocity self-map radii:

```text
R_s = min(
  rho / (G * (V + nu)),
  nu / (G * A)
).
```

Since `G <= G_bar = (D + d/5)^(3p)`, `A <= A_bar`, and `nu <= nu_bar`, the radius is bounded below by:

```text
R_s >= min(
  rho / (G_bar * (V + nu_bar)),
  sqrt(rho / A_bar) / G_bar
).
```

This is the state-envelope lower bound.

The stored floating-point radius may be one representable value below this mathematical radius because the code uses `nextafter` toward zero. That implementation detail only changes the numerical regression tolerance; it does not change the analytic inequality.

## Executable Constructor

The bound is now executable in `three_body_symmetry/compact_sundman.py` as:

```text
construct_sundman_cauchy_radius_state_envelope(q, v, m)
```

The constructor derives `d`, `D`, `V`, `M`, `rho=d/10`, `A_bar`,
`G_bar`, `nu_bar`, and the one-ulp-inward lower bound:

```text
R_s,lower =
min(
  rho / (G_bar (V + nu_bar)),
  sqrt(rho / A_bar) / G_bar
).
```

It also exposes two direct checks:

```text
certifies_compact_cauchy_radius(r, a, lambda)
certifies_cauchy_capped_lower_step(a, delta_next, lambda, H, alpha)
```

The first check proves that the compact disk of radius `r` centered at `a`
maps under `atanh(w)/lambda` into the certified Sundman disk by verifying:

```text
r / (lambda (1 - (|a| + r)^2)) <= R_s,lower.
```

The second check applies the lower-step radius
`r = 2 min(H, alpha delta_next)`. Therefore it proves that Cauchy-radius
capping cannot reduce the accepted compact step below
`min(H, alpha delta_next)`.

There is also a margin-only eventual-shell check. If
`r <= eta (1-|a|)` with `0 < eta < 1`, it is enough to require:

```text
2 alpha / (lambda (1-eta)) <= R_s,lower.
```

Equivalently:

```text
alpha <= (lambda (1-eta) R_s,lower) / 2,
alpha <= eta / 2.
```

The constructor returns this upper bound for `alpha`. This is deliberately a
local state-envelope proof: it closes the Cauchy-radius lower-step premise for
states satisfying the envelope inequality, but it does not by itself prove that
all future compact-Sundman shell states continue to satisfy such an envelope.

## Consequence For The Compact Recurrence

The compact `atanh` lemma needs, for future shell margin `delta_{n+1}`,

```text
R_s(a) >= r_* / (lambda * (1 - eta) * (1 - |a|)),
r_* = 2 * min(H, alpha * delta_{n+1}).
```

It is therefore enough to prove the state-envelope inequality:

```text
min(
  rho / (G_bar * (V + nu_bar)),
  sqrt(rho / A_bar) / G_bar
)
>= r_* / (lambda * (1 - eta) * (1 - |a|)).
```

If this holds for every nonterminal future shell chart, then the Cauchy-capped lower-step lemma applies, and the geometric shell segment-count recurrence is proved.

This still does not prove the full all-future tail recurrence. It identifies a specific state-envelope theorem that remains to be proved: future compact-Sundman charts must keep the right-hand inequality above true while also satisfying the already identified state-supremum and tail-transfer bounds.
