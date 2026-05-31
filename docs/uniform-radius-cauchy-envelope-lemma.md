# Uniform-Radius Cauchy Envelope Lemma

## Claim

The radius-drift term in adjacent-shell comparisons can be avoided by proving a uniform step-to-Cauchy-radius bound and comparing every shell to a common envelope instead of comparing raw adjacent Cauchy ratios.

This gives a cleaner route for the all-future compact-Sundman recurrence proof.

## Setup

For shell `n`, suppose a Cauchy cover has:

- at most `M_n` segments,
- state supremum bound at most `S_n`,
- retained order at least `p_n`,
- segment step-to-Cauchy-radius ratios all bounded by a fixed `sigma`, with `0 <= sigma < 1`.

The standard segment tail estimate is:

```text
tail <= S_n * rho^(p_n + 1) / (1 - rho).
```

Since `rho <= sigma`, every segment obeys:

```text
tail <= S_n * sigma^(p_n + 1) / (1 - sigma).
```

Summing over at most `M_n` segments gives the shell envelope:

```text
E_n <= M_n S_n sigma^(p_n + 1) / (1 - sigma).
```

## Transition Bound

Assume future shells satisfy:

```text
M_{n+1}/M_n <= B_M
S_{n+1}/S_n <= B_S
p_{n+1} - p_n >= d
```

with the same `sigma` for all future shells. Then the uniform envelopes satisfy:

```text
E_{n+1}/E_n <= B_M B_S sigma^d.
```

There is no `(rho_{n+1}/rho_n)^(p_n+1)` radius-drift factor because the comparison is between common-`sigma` envelopes, not between adjacent raw Cauchy ratios.

## All-Future Summability Bound

The same uniform-envelope comparison gives a genuine all-future recurrence bound
once the future inequalities are proved for every shell. More generally, suppose
the retained-order increments are accelerated:

```text
p_{n+1} - p_n >= d + n q,      d >= 1, q >= 0,
```

and the structural growth recurrence is uniform:

```text
(M_{n+1} S_{n+1}) / (M_n S_n) <= B.
```

Then:

```text
E_{n+1}/E_n <= B sigma^(d+nq)
             = R0 G^n,

R0 = B sigma^d,
G  = sigma^q.
```

If `R0 < 1`, the future shell tails are summable. Indeed:

```text
E_{N+k} <= E_N R0^k G^(k(k-1)/2) <= E_N R0^k,
```

because `0 <= G <= 1`. Therefore:

```text
sum_{k>=1} E_{N+k} <= E_N R0 / (1 - R0).
```

If an additional tail-transfer multiplier is required and satisfies
`T_{n+1}/T_n <= Gamma`, replace `G` by `Gamma sigma^q`. The same proof gives a
summable bound when:

```text
B sigma^d T_0 < 1,
Gamma sigma^q < 1.
```

This is now the precise all-future induction target: prove the uniform `sigma`,
structural growth `B`, retained-order acceleration `(d,q)`, and any transfer
growth bound for every later shell. Once those recurrence hypotheses are true,
the infinite compact-Sundman tail has the explicit remainder above.

## Consequence

A viable all-future recurrence strategy is:

1. Prove a global cover-construction rule with `rho <= sigma < 1` for every future compact-Sundman shell.
2. Prove segment-count and state-supremum recurrence bounds.
3. Choose retained-order growth so `B_M B_S sigma^d < 1`, or combine this envelope with the accelerated tail-transfer summability lemma when additional transfer multipliers remain.

This does not yet prove those recurrence bounds. It gives a sharper theorem target that avoids the radius-drift obstruction by changing the comparison object.
