# Compact Atanh Cauchy-Radius Lower Bound

## Claim

The compact-Sundman Cauchy-radius selector has an explicit analytic lower-bound condition. This closes the implementation-level gap between the Cauchy-capped lower-step lemma and the actual compact map:

```text
s = atanh(w) / lambda.
```

## Setup

Fix a compact chart center `a` with `|a| < 1`, Sundman rate `lambda > 0`, and an already certified Sundman-time Cauchy radius `R_s > 0` around

```text
s(a) = atanh(a) / lambda.
```

The compact selector accepts a compact radius `r` when the disk stays away from the compact endpoints and the derivative majorant for `atanh` maps the whole disk into the certified Sundman disk:

```text
0 < r < 1 - |a|,

r / (lambda * (1 - (|a| + r)^2)) <= R_s.
```

Let

```text
b = 1 - |a|
```

be the compact endpoint margin at the chart center. For the lower-step recurrence, the radius needed by the Cauchy cap is

```text
r_* = 2 * min(H, alpha * delta_{n+1}),
```

where `H` is the fixed maximum compact step, `alpha` is the boundary-fraction constant from the shell step rule, and `delta_{n+1}` is the next geometric endpoint margin.

## Bound

The sharp sufficient condition is:

```text
r_* < b,

r_* / (lambda * (1 - (|a| + r_*)^2)) <= R_s.
```

Under these two inequalities, `r_*` is an admissible compact Cauchy radius. Since the implementation chooses the largest admissible compact radius, the selected compact radius `R(a)` satisfies:

```text
R(a) >= r_*.
```

A simpler margin-only sufficient condition follows by choosing any `eta` with `0 < eta < 1` and assuming:

```text
r_* <= eta * b,

R_s >= r_* / (lambda * (1 - eta) * b).
```

Indeed, `r_* <= eta b` gives:

```text
b - r_* >= (1 - eta)b.
```

Also,

```text
1 - (|a| + r_*)^2
  = (1 - |a| - r_*) * (1 + |a| + r_*)
  >= (b - r_*) * 1
  >= (1 - eta)b.
```

Therefore:

```text
r_* / (lambda * (1 - (|a| + r_*)^2))
  <= r_* / (lambda * (1 - eta) * b)
  <= R_s.
```

So the sharp selector condition holds.

## Eventual Geometric-Shell Form

The margin-only condition has a useful all-future shell form. Suppose the
centered geometric exhaustion has next boundary margin `delta_{n+1}`, and every
nonterminal chart center in that shell satisfies:

```text
b = 1 - |a| >= delta_{n+1}.
```

Assume also that the shell is late enough to lie in the boundary-fraction
regime:

```text
alpha * delta_{n+1} <= H.
```

Then the Cauchy radius needed by the lower-step lemma is:

```text
r_* = 2 * alpha * delta_{n+1}.
```

If `0 < alpha < 1/2` and `eta` is chosen with:

```text
2 * alpha <= eta < 1,
```

then every nonterminal chart in every such future shell satisfies:

```text
r_* <= eta * delta_{n+1} <= eta * b.
```

The remaining Sundman-radius hypothesis becomes uniform in the shell margin:

```text
r_* / (lambda * (1 - eta) * b)
  <= 2 * alpha / (lambda * (1 - eta)).
```

Therefore the lower-step/Cauchy-radius requirement is implied on all sufficiently
late shells by the single margin-independent bound:

```text
R_s(a) >= 2 * alpha / (lambda * (1 - eta))
```

for every nonterminal chart center `a` in those shells. This does not prove that
the underlying Sundman Cauchy radius has that lower bound; it isolates exactly
what a future state-envelope theorem must prove.

## Consequence

For every nonterminal future shell chart, the Cauchy-capped lower-step lemma follows from the compact `atanh` map once the underlying Sundman-time Cauchy radius satisfies either the sharp inequality or the margin-only inequality above.

Combining this lemma with the earlier segment-count result gives an all-future segment-count recurrence, conditional only on the future lower bound for the underlying Sundman-time Cauchy radius:

```text
R_s(a) >= r_* / (lambda * (1 - eta) * (1 - |a|)).
```

In the eventual boundary-fraction regime, the previous section simplifies this
to the uniform condition:

```text
R_s(a) >= 2 * alpha / (lambda * (1 - eta)).
```

It does not by itself prove the full all-future tail recurrence. The remaining analytic issue is now narrower: prove this Sundman-time Cauchy-radius lower bound, plus the already identified state-supremum and tail-transfer bounds, uniformly across all future shells.
