# Hyperbolic Escape Log-Term Lemma

## Claim

For positive-energy homothetic escape, scaling the position makes the endpoint finite, but a plain inverse-time power series is still not the right endpoint representation. The scaled variable has a forced logarithmic correction:

```text
X(t) = q(t)/t
     = c Q + (mu/c^2)(log t)/t Q + O(1/t),
```

or, with `tau = 1/t`,

```text
X(tau) = c Q - (mu/c^2) tau log(tau) Q + O(tau).
```

Thus a global closed-form series route that includes hyperbolic escape must allow logarithmic endpoint terms, or an equivalent variable that absorbs them.

## Setup

For a centered equal-mass central configuration:

```text
A(Q) = -mu Q.
```

A positive-energy homothetic escape has:

```text
q_i(t) = R(t) Q_i,
R'' = -mu/R^2,
E = (1/2)R'^2 - mu/R > 0.
```

Let:

```text
c = sqrt(2E).
```

The energy identity gives:

```text
R' = sqrt(c^2 + 2mu/R).
```

As `R -> infinity`,

```text
R' = c + mu/(cR) + O(R^-2).
```

Since `R(t) ~ ct`, integration gives:

```text
R(t) = ct + (mu/c^2) log t + O(1).
```

Therefore:

```text
R(t)/t = c + (mu/c^2)(log t)/t + O(1/t).
```

## Inverse-Time Endpoint Equation

Set:

```text
tau = 1/t,
q = X(tau)/tau.
```

Then:

```text
q_t = X - tau X_tau,
q_tt = tau^3 X_{tau tau}.
```

By homogeneity:

```text
A(q) = A(X/tau) = tau^2 A(X).
```

Newton's equation becomes:

```text
tau X_{tau tau} = A(X).
```

At the escape endpoint `X -> cQ`, the right side tends to:

```text
A(cQ) = c^-2 A(Q) = -(mu/c^2) Q.
```

A pure analytic endpoint expansion `X = cQ + O(tau)` would have finite `X_{tau tau}` and hence `tau X_{tau tau} -> 0`, which cannot equal `A(cQ)` unless the limiting configuration has zero acceleration. The logarithmic term supplies the missing balance:

```text
X(tau) = cQ - (mu/c^2) tau log(tau) Q
```

gives:

```text
tau X_{tau tau} = -(mu/c^2) Q = A(cQ).
```

## Consequence

The scaled escape lift is still the right direction, but the endpoint function class must be richer than ordinary power series in compact time or inverse time. For hyperbolic escape, the lifted representation must include `tau log tau` terms, or choose a further transformed variable that makes those terms analytic.

This narrows the global closed-form route: construct in a space that includes collision regularization, escape scaling, and logarithmic escape endpoint terms; project back to unscaled coordinates only for finite target times; verify Newton's equations through the lifted projection identities.
