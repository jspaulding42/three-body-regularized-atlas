# Cauchy-Capped Lower-Step Lemma

## Claim

The geometric shell segment-count bound remains valid after Cauchy-radius capping if every accepted chart in the shell has a compact Cauchy radius large enough compared with the shell's target step scale.

## Setup

In a centered endpoint shell with next boundary margin `delta_{n+1}`, the uncapped step rule has the form:

```text
h_raw(w) = min(H, beta * b(w), 0.5 * b(w)),
```

where `H > 0` is the fixed maximum step, `0 < beta < 1` is the configured boundary-fraction parameter, and `b(w)` is the remaining distance to the endpoint in the direction of travel.

For all nonterminal steps in that shell, `b(w) >= delta_{n+1}`. Therefore:

```text
h_raw(w) >= min(H, alpha * delta_{n+1}),
alpha = min(beta, 0.5).
```

In Cauchy mode, the accepted step is capped by half the compact Cauchy radius:

```text
h(w) = min(h_raw(w), 0.5 * R(w)).
```

## Bound

If the compact Cauchy radius satisfies

```text
R(w) >= 2 * min(H, alpha * delta_{n+1})
```

for every nonterminal chart in the shell, then:

```text
h(w) >= min(H, alpha * delta_{n+1}).
```

The only shorter steps left are endpoint remainders and previous-boundary crossing remainders. Those are finite bookkeeping terms already covered by the `+1` per side in the segment-count lemma.

## Consequence

The all-future segment-count recurrence can be proved by combining:

1. the geometric shell segment-count lemma,
2. a uniform compact Cauchy radius lower bound `R(w) >= 2 * min(H, alpha * delta_{n+1})`, and
3. the existing step construction.

The remaining analytic work is therefore a Cauchy-radius lower-bound theorem for the compact-Sundman charts on future shells.

## Executable Constructor

This implication is now represented by
`certify_cauchy_capped_geometric_segment_count_from_state_envelope(...)` in
`three_body_symmetry/compact_sundman.py`.

The constructor takes a certified Sundman Cauchy-radius state envelope, the
centered geometric schedule constants:

```text
delta_n = delta_0 c^n,
```

and the lower-step parameters `H`, `alpha`, `eta`, and `lambda`. It checks the
compact `atanh` margin condition:

```text
2 alpha <= eta,
2 alpha / (lambda(1-eta)) <= R_s,
```

where `R_s` is the state-envelope lower bound for the Sundman-time Cauchy
radius. It then identifies the first shell where the boundary-fraction regime
applies:

```text
alpha delta_{n+1} <= H,
```

and proves that every later nonterminal chart center with boundary margin at
least `delta_{n+1}` has accepted Cauchy-capped step:

```text
h_n >= alpha delta_{n+1}.
```

Combining this with the geometric shell segment-count lemma gives:

```text
M_* = 2 * (ceil(max((1-c)delta_0/H, (1-c)/(alpha c))) + 1),
M_n <= M_*        for every eventual shell.
```

Thus the segment-count part of the compact-Sundman all-future recurrence has
growth ratio `1` once the same state envelope is proved uniformly on the future
shell centers. The constructor does not prove that uniform state envelope; it
removes the separate lower-step and segment-count assumptions after that
state-envelope theorem is supplied.
