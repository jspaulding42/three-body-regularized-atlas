# Total-Collision Stop Chart Existence Lemma

## Statement

For positive masses and exact noncollision initial data, suppose the
binary-regularized maximal branch reaches a first unselected total collision
before a finite target time.  Under the finite generalized Fuchsian/Puiseux-log
entry data lemma, there exists a finite maximal-classical total-stop chart
certificate.

The stop chart covers a punctured incoming interval ending at the collision
time, verifies the projected Newton equations on the punctured side, proves
that all three pair distances vanish at the endpoint, and records that the
default policy stops rather than choosing a continuation.

## Hypotheses

- the branch reaches a first total collision at `T`,
- the total-collision policy is `maximal_classical_stop`,
- finite generalized Fuchsian/Puiseux-log entry data are available on the
  incoming side,
- the entry chart has a punctured isolation interval with positive pair
  distances away from `tau=0`,
- the generalized Fuchsian analytic remainder tail and coefficient residual
  bounds are certified.

## Proof Sketch

Use cubic collision time `t = T + tau^3`.  The entry-data lemma supplies a
central leading shape and finitely many stable selector rows, so the lifted
configuration has the form

```text
q(tau) = tau^2 S(tau, log(tau))
```

on the incoming punctured side, with generalized powers allowed inside `S`.
The finite generalized Fuchsian/Puiseux-log recurrence constructs the retained
coefficients of `S`; the Cauchy majorant bounds the analytic remainder beyond
those rows.  Substituting `q(tau)` into the transformed Newton equations gives
a residual that is zero up to the certified coefficient equations plus the
explicit tail bound.

The projection certificate maps the lifted branch back to inertial body
coordinates and checks the center-of-mass, momentum, angular-momentum, and
energy ledgers on `0 < |tau| <= rho`.  Punctured isolation gives positive pair
distances throughout that punctured interval.  The factor `tau^2` gives
`q_i(T)=q_j(T)` for all pairs at `tau=0`, so the endpoint is total collision.

Because the policy is maximal-classical stop, the certificate records no
outgoing branch.  If a selector continuation is desired, it must be supplied as
a separate theorem with its own selector semantics.

The executable generalized supplied-entry bridge now proves this implication
when its local inputs are already constructor-certified: supplied generalized
entry rows, finite-row truncation budget, and Banach-majorized analytic
remainder.  The bridge checks cubic-time interval binding, residual tail within
tolerance, zero angular momentum, endpoint collapse, and the
`maximal_classical_stop` policy.  It is not yet the independent serialized
checker for arbitrary generalized rows.

## Certificate Fields Consumed By Code

- `stop_time`
- `total_collision_policy_id`
- `entry_data_certificate`
- `punctured_isolation_certificate`
- `generalized_fuchsian_tail_budget`
- `analytic_remainder_majorant_certificate`
- `newton_residual_ledger`
- `projection_ledger`
- `invariant_ledger`
- `endpoint_pair_distance_collapse`

## Proof-Grade Status

This is currently a theorem scaffold in the repository.  It becomes proof-grade
only when the generalized Fuchsian entry-data lemma, analytic remainder
Cauchy-majorant theorem, and chart residual/tail checker are externally audited
or machine-checkable.  Declared prose is not sufficient for top-level
closed-form proof certification.
