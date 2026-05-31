# Nonzero-Angular Event-Shell Invariance Lemma

This note records the scoped theorem implemented by
`certify_nonzero_angular_event_shell_invariance_from_handoff(...)`.

It is not an arbitrary-data classifier. It proves that, once explicit uniform
event-envelope hypotheses have been supplied and checked, the finite handoff
and the geometric event recurrence compose into an all-future event-shell
invariance certificate.

## Setup

Use compact physical time

```text
u = tanh(rate * t)
```

and future shells

```text
I_n^+ = [1 - delta * theta^n, 1 - delta * theta^(n+1)]
```

with the time-reversed-past shells defined by reflection. The event-envelope
specification supplies `delta`, `theta`, event-isolation and boundary-clearance
scales, and one ordinary-gap envelope

```text
pair_distance >= d0,
pair_diameter <= D0,
speed <= V0.
```

The event-regime assembly supplies the same shell geometry plus ordinary-gap
and separated-binary Levi-Civita chart-family Cauchy inputs. The all-future
budget gives a geometric value-tail bound for every shell and a tail from any
prefix.

## Claim

Assume:

1. The finite middle reaches the first shell boundary and its endpoint lies
   inside the ordinary-gap envelope.
2. A finite validated atlas covers the first full event shell, from
   `u = 1 - delta` to `u = 1 - delta * theta` and similarly in reversed past
   time.
3. The event recurrence shell geometry is exactly the geometry in the envelope
   specification.
4. The ordinary-gap Cauchy source certificate uses the same `(d0, D0, V0)`
   envelope.
5. The value-majorant rows are uniform across shells.
6. The remaining value tail after shell zero fits inside the first-shell
   endpoint metric margins.

Then the ordinary-gap envelope remains valid for every later shell covered by
the event recurrence, and the projected compact-time atlas has a summable
value tail from the first event-shell endpoint.

## Metric Perturbation Bound

Let every position and velocity coordinate be perturbed by at most `epsilon` in
a `d`-dimensional physical space.

For a pair distance,

```text
|| (q_i + e_i) - (q_j + e_j) || >= ||q_i - q_j|| - ||e_i - e_j||
                                  >= ||q_i - q_j|| - 2 sqrt(d) epsilon.
```

The same `2 sqrt(d) epsilon` bound controls pair-diameter increase. For each
body speed,

```text
||v_i + f_i|| <= ||v_i|| + sqrt(d) epsilon.
```

Therefore the remaining projected value-tail bound `epsilon` preserves the
ordinary-gap envelope whenever the first-shell endpoint margins satisfy

```text
pair_distance_margin > 2 sqrt(d) epsilon,
pair_diameter_margin > 2 sqrt(d) epsilon,
speed_margin > sqrt(d) epsilon.
```

## Why This Is Still Scoped

The lemma proves a composition step:

```text
finite middle
  -> first event-shell prefix
  -> matching geometric recurrence
  -> all-future shell invariance under supplied envelopes
```

It does not prove that arbitrary nonzero-angular initial data enter those
uniform event envelopes. That remains a separate global-regime classification
problem, and the top-level theorem still requires `global_regime_exhaustion`
before it can be advertised as an unrestricted general solution.
