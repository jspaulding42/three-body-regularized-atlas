# Log-Subtracted Escape Endpoint Lemma

## Claim

The forced hyperbolic escape logarithm can be absorbed into the lifted representation. For the homothetic escape branch, define:

```text
tau = 1/t,
c = sqrt(2E),
alpha = mu/c^2,
X(tau) = q(t)/t,
Y(tau) = X(tau) + alpha tau log(tau) Q.
```

Then the leading nonzero endpoint forcing in the inverse-time equation cancels. This gives a constructive endpoint variable rather than only an obstruction.

## Starting Equation

The scaled inverse-time variable satisfies:

```text
tau X_{tau tau} = A(X).
```

For the central configuration endpoint:

```text
X -> cQ,
A(cQ) = c^-2 A(Q) = -alpha Q.
```

This is why a pure analytic expansion for `X` fails: the left side would tend to zero while the right side tends to `-alpha Q`.

## Subtract The Forced Log Term

Write:

```text
X = Y - alpha tau log(tau) Q.
```

Since:

```text
(tau log tau)_{tau tau} = 1/tau,
```

we have:

```text
tau X_{tau tau} = tau Y_{tau tau} - alpha Q.
```

Substituting into the scaled endpoint equation gives:

```text
tau Y_{tau tau}
  = A(Y - alpha tau log(tau) Q) + alpha Q.
```

As `tau -> 0` and `Y -> cQ`, the right side tends to:

```text
A(cQ) + alpha Q = 0.
```

The leading endpoint obstruction is therefore removed.

## Consequence

The right endpoint function class can be stated more constructively:

```text
X(tau) = Y(tau) - alpha tau log(tau) Q,
```

where `Y` is the variable that should admit a regular endpoint construction for the homothetic model. The general three-body escape problem will need the corresponding asymptotic velocity configuration and logarithmic coefficient, but the mechanism is now explicit:

1. lift to an escape-scaled coordinate,
2. subtract the forced logarithmic term,
3. construct the regular remainder,
4. project back to unscaled positions for finite target times.

This still does not prove the general theorem. It narrows the required lifted function class from "some logarithmic terms" to a concrete log-subtraction mechanism that cancels the leading endpoint force.
