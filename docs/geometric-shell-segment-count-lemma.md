# Geometric Shell Segment-Count Lemma

## Claim

A centered geometric endpoint-exhaustion schedule has uniformly bounded shell segment counts if each accepted step advances at least a fixed fraction of the remaining boundary margin, up to a fixed maximum step cap.

This addresses one concrete component of the all-future compact-Sundman recurrence: segment-count growth.

## Setup

Let the centered compact domain after shell `n` be:

```text
[-1 + delta_n, 1 - delta_n]
```

with geometric boundary margins:

```text
delta_n = delta_0 c^n,       0 < c < 1.
```

The next shell adds two endpoint intervals, each of width:

```text
delta_n - delta_{n+1} = (1 - c) delta_n.
```

Assume that every accepted step in shell `n` has length at least:

```text
h_n >= min(H, alpha * delta_{n+1})
```

where `H > 0` is the fixed maximum-step regime and `alpha > 0` is the boundary-fraction regime.

## Bound

Each side of the shell needs no more than

```text
ceil((1 - c) delta_n / min(H, alpha delta_{n+1}))
```

interior steps, plus at most one boundary-crossing step when counting segments outside the previous compact domain.

Since `delta_{n+1} = c delta_n`,

```text
(1 - c) delta_n / min(H, alpha delta_{n+1})
 <= max((1 - c) delta_0 / H, (1 - c)/(alpha c)).
```

Therefore the two-sided shell segment count is bounded uniformly by:

```text
M_* = 2 * (ceil(max((1 - c) delta_0 / H, (1 - c)/(alpha c))) + 1).
```

## Recurrence Form

Let `M_n` be the number of accepted segments in shell `n`, counted with the same
one-crossing bookkeeping convention. The bound above is independent of `n`, so:

```text
M_n <= M_*        for every n >= 0.
```

This is an all-future recurrence envelope. In the uniform-radius Cauchy-envelope
proof one may use the envelope sequence:

```text
widehat M_n = M_*.
```

It has growth ratio:

```text
widehat M_{n+1} / widehat M_n = 1.
```

Thus the segment-count contribution to the structural recurrence is closed once
the lower-step hypothesis is proved. The raw observed ratios `M_{n+1}/M_n` may
fluctuate in a finite prefix, but every future shell is dominated by the same
constant envelope, so the recurrence factor contributed by segment count is
`B_M = 1`.

## Consequence

This proves a possible all-future segment-count recurrence with growth ratio `1`, provided the step-construction proof supplies the lower bound `h_n >= min(H, alpha delta_{n+1})` for all future shells. The current compact-Sundman code already enforces an upper step-to-Cauchy-radius ratio; the missing analytic work is the matching lower-step guarantee after any Cauchy-radius cap.

The matching Cauchy-capped lower-step guarantee is now executable for the
state-envelope subcase. `certify_cauchy_capped_geometric_segment_count_from_state_envelope(...)`
combines the compact `atanh` radius bound with a
`SundmanCauchyRadiusStateEnvelopeCertificate` and returns this `M_*` directly.
The remaining recurrence task is therefore not a segment-count theorem; it is
the dynamical theorem that the required state envelope persists on the future
shell centers, together with the separate state-supremum and tail-transfer
envelopes.
