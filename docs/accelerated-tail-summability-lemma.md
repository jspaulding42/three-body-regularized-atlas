# Accelerated Tail-Summability Lemma

## Claim

The accelerated compact-Sundman route needs two distinct proof parts:

1. A recurrence proof that every future shell obeys explicit structural and transfer bounds.
2. A summability proof showing that those bounds imply a convergent infinite tail.

This note proves the second part. It does not prove the future-shell recurrence hypotheses themselves.

## Setup

Let `E_N` be the certified incremental tail bound on the last checked shell. Suppose every future shell satisfies the following transition estimate:

```text
E_{n+1} <= E_n * A * sigma^(d0 + n q) * tau0 * gamma^n,     n = 0, 1, 2, ...
```

where:

- `A >= 0` bounds the product of structural growth factors, such as segment count, state supremum, and Cauchy-denominator growth.
- `0 <= sigma < 1` bounds the step-to-Cauchy-radius ratio.
- `d0 >= 1` is the first future retained-order increment.
- `q >= 0` is the per-shell growth in retained-order increment.
- `tau0 >= 0` bounds the first future tail-transfer multiplier.
- `gamma >= 0` bounds geometric growth of the tail-transfer multiplier.

Define:

```text
R0 = A * sigma^d0 * tau0
G  = gamma * sigma^q
```

Then the future transition ratios are bounded by:

```text
R_n <= R0 * G^n.
```

## Proof

By the transition estimate,

```text
E_{n+1} / E_n <= A * sigma^(d0 + n q) * tau0 * gamma^n
              = (A * sigma^d0 * tau0) * (gamma * sigma^q)^n
              = R0 * G^n.
```

If `R0 < 1` and `G < 1`, then `R_n <= R0 < 1` for every `n >= 0`, and the ratios strictly improve unless `G = 1`.

The future tail terms satisfy:

```text
E_1 <= E_N R0
E_2 <= E_N R0^2 G
E_3 <= E_N R0^3 G^3
...
E_k <= E_N R0^k G^(k(k-1)/2).
```

Since `0 <= G < 1`, each term is bounded by the ordinary geometric majorant:

```text
E_k <= E_N R0^k.
```

Therefore:

```text
sum_{k>=1} E_k <= E_N * sum_{k>=1} R0^k
                = E_N * R0 / (1 - R0).
```

This proves convergence of the future-shell tail whenever:

```text
A * sigma^d0 * tau0 < 1
gamma * sigma^q < 1.
```

## Consequence

The accelerated retained-order schedule is not just numerically helpful; it is the analytic mechanism that can absorb geometric tail-transfer growth. If `gamma >= 1`, a constant retained-order increment (`q = 0`) generally cannot absorb it because `G = gamma`. Increasing the retained-order increment by `q` per shell multiplies the transfer-growth factor by `sigma^q`, and because `sigma < 1`, a large enough `q` forces `G < 1`.

For the full general closed-form target, this lemma closes only the summability algebra after all future-shell envelopes have already been proved. The remaining hard proof is to establish those envelopes for every later compact-Sundman shell: the segment-count, state-supremum, Cauchy-denominator, step-radius, tail-transfer, and retained-order recurrence bounds.
