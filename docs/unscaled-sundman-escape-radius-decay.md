# Unscaled Sundman Escape Radius Decay

## Claim

The eventual-shell compact `atanh` lower-step condition cannot be proved for
all arbitrary data in unscaled compact-Sundman coordinates. Along a
positive-energy homothetic escape branch, the Sundman-time Cauchy radius used by
the current majorant construction decays to zero.

This is an obstruction to the unscaled compact-Sundman all-future recurrence
route. It is not an obstruction to a global solution in a richer lifted space:
the escape branch must be handled by physical-time compactification, scaled
variables, logarithmic endpoint terms, or an explicit escape classification.

## Setup

Let `Q = (Q_1,Q_2,Q_3)` be a centered central configuration with at least two
distinct bodies, and let:

```text
q_i(t) = R(t) Q_i,
v_i(t) = R'(t) Q_i.
```

Assume the branch has positive radial energy:

```text
E = (1/2) R'(t)^2 - mu/R(t) > 0.
```

Then for all sufficiently large future time:

```text
R(t) -> infinity,
R'(t) >= c > 0.
```

For the current default Sundman lift with distance power `1`,

```text
dt/ds = g(q) = product_{i<j} |q_i-q_j|.
```

The Cauchy majorant used by the implementation chooses:

```text
rho = d(q)/10,
R_s <= rho / (G(q) (V(q)+nu)),
```

where `d(q)` is the minimum pair distance, `G(q)` is the majorant bound for the
Sundman factor, `V(q)=max_i |v_i|`, and `nu > 0` is the velocity ball radius.

## Decay

Define constants from the fixed shape:

```text
d_Q = min_{i<j} |Q_i-Q_j| > 0,
D_Q = max_{i<j} |Q_i-Q_j| > 0,
Q_max = max_i |Q_i| > 0.
```

Along the homothetic branch:

```text
d(q) = R d_Q,
rho = R d_Q / 10,
V(q) >= c Q_max.
```

The Sundman-factor majorant is at least the true largest-pair scale cubed:

```text
G(q) >= (R D_Q)^3.
```

Therefore:

```text
R_s
  <= rho / (G(q) V(q))
  <= (d_Q / 10) / (D_Q^3 c Q_max) * R^-2.
```

Hence:

```text
R_s -> 0      as      R -> infinity.
```

## Consequence

The eventual-shell lower-step corollary in the compact `atanh` lemma needs a
positive, margin-independent floor:

```text
R_s(a) >= 2 alpha / (lambda (1-eta)).
```

The decay above proves that no such positive floor can hold uniformly over
arbitrary unscaled escape data. A proof of the full general closed-form theorem
therefore cannot close the all-future recurrence by unscaled compact-Sundman
shells alone. It must either stop and classify the finite Sundman escape
endpoint or lift escape branches into bounded scaled/log-subtracted variables
before attempting a global recurrence.
