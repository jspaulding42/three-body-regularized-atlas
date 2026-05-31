# Shell Cauchy-Transition Lemma

## Claim

For a future-shell recurrence proof, the Cauchy tail transition has one more factor than the simplified `rho^d` intuition suggests. If the step-to-Cauchy-radius ratio changes from shell to shell, the old retained order contributes a radius-drift factor that must either be bounded separately or absorbed into the tail-transfer recurrence.

This is a proof obligation, not a new certificate layer.

## Setup

Suppose shell `n` is covered by at most `M_n` Cauchy segments. Each segment has state supremum at most `S_n`, step-to-Cauchy-radius ratio `0 <= rho_n < 1`, and retained order `p_n`. The standard Cauchy tail estimate gives an incremental shell tail of the form

```text
E_n <= M_n S_n rho_n^(p_n + 1) / (1 - rho_n).
```

Write

```text
B^M_n = M_{n+1}/M_n
B^S_n = S_{n+1}/S_n
B^D_n = (1 - rho_n)/(1 - rho_{n+1})
d_n   = p_{n+1} - p_n
```

with `d_n >= 0`.

## Transition Bound

Using the Cauchy estimates for adjacent shells,

```text
E_{n+1}/E_n
 <= (M_{n+1}/M_n)
    (S_{n+1}/S_n)
    ((1 - rho_n)/(1 - rho_{n+1}))
    (rho_{n+1}^(p_{n+1}+1) / rho_n^(p_n+1)).
```

Since `p_{n+1} = p_n + d_n`,

```text
rho_{n+1}^(p_{n+1}+1) / rho_n^(p_n+1)
 = rho_{n+1}^d_n * (rho_{n+1}/rho_n)^(p_n + 1).
```

Therefore:

```text
E_{n+1}/E_n
 <= B^M_n B^S_n B^D_n
    rho_{n+1}^d_n
    (rho_{n+1}/rho_n)^(p_n + 1).
```

The final factor is the radius-drift term.

## Consequence

The simplified structural transition

```text
B^M_n B^S_n B^D_n rho_{n+1}^d_n
```

is valid only under an additional condition such as `rho_{n+1} <= rho_n`, or after proving that the radius-drift factor is included in another multiplier, for example the tail-transfer factor.

The current accelerated compact-Sundman diagnostics use a tail-transfer multiplier computed from the actual observed tail ratio divided by the simplified structural factor. That multiplier is therefore allowed to contain radius drift, coefficient-coupling effects, and other slack. A future all-shell recurrence proof cannot ignore those effects: it must either prove a monotone/nonincreasing radius schedule, prove a separate bound on

```text
(rho_{n+1}/rho_n)^(p_n + 1),
```

or prove that the tail-transfer recurrence bounds this factor together with the remaining unmodeled transition effects.

This narrows the missing all-future recurrence theorem. The summability lemma proves that accelerated retained order absorbs a bounded geometric transfer multiplier; this transition lemma identifies one concrete analytic source that such a multiplier has to dominate.
