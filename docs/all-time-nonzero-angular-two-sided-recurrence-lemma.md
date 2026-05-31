# All-Time Nonzero-Angular Two-Sided Recurrence Lemma

An all-future event recurrence certifies only one compact-time endpoint.  With
`u=tanh(rate*t)`, the future endpoint is `u -> 1`; the past endpoint is
`u -> -1`.  Therefore an all-real nonzero-angular atlas needs two one-sided
recurrences:

1. the future recurrence for the supplied initial data;
2. the future recurrence for the time-reversed initial data `(q0,-v0)`.

The second recurrence is the past recurrence for the original flow.  If
`q_+(s)` solves Newton's equations with initial velocity `-v0`, then

```text
q_-(t) = q_+(-t)
```

solves Newton's equations with initial velocity `v0`.  Pair distances and the
potential are unchanged, velocities change sign, energy is unchanged, and
centered angular momentum changes sign.  Thus a nonzero angular-momentum lower
bound excludes total collision in both time directions.

The event families are also preserved.  Collision-free ordinary charts remain
ordinary charts after reversing time.  A separated planar binary collision
regularized by Levi-Civita remains a separated planar binary collision; the
regularized local variables may reverse orientation, but the projected
Newtonian chart, Cauchy tail bounds, and separation hypotheses are unchanged
because they use absolute state, radius, and majorant bounds.

Consequently, if the future and time-reversed-future event assemblies both
derive:

- geometric shell isolation,
- ordinary-gap chart-family Cauchy inputs,
- separated-binary Levi-Civita chart-family Cauchy inputs,
- no total-collision event family, and
- summable value, first-jet, lifted-residual, and physical-residual tails,

then their two geometric sums certify the two compact-time endpoints.  Finite
compact prefixes cover the middle.  The resulting all-time nonzero-angular
claim is constructor-backed only when both one-sided recurrences are present.

The executable hook is
`certify_two_sided_nonzero_angular_event_budget(...)`.  The
`construct_nonzero_angular_global_atlas(...)` theorem path now reports
`past_all_future_event_budget` as a missing obligation when the past/time-
reversed recurrence is absent, instead of treating a single all-future budget as
an all-real proof.

For uniform all-pair nonzero-angular event envelopes, the constructor
`construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(...)`
takes separate `NonzeroAngularUniformPairEventEnvelopeSpec` objects for the
future endpoint and the time-reversed past endpoint.  This allows the proof
pipeline to certify asymmetric endpoint hypotheses, for example when the past
shell isolation constants are weaker than the future constants.  The older
single-spec wrapper remains available only for explicitly time-reversal-
invariant absolute envelopes; internally it still derives two distinct endpoint
recurrences before the theorem assembler sees them.

Each endpoint spec carries a `time_direction` field.  The future argument must
be marked `future` or `time_reversal_invariant`; the past argument must be
marked `time_reversed_past` or `time_reversal_invariant`.  This prevents a
future-only envelope table from being accidentally reused as past-endpoint
evidence without saying that the bounds are invariant under time reversal.

This is still a recurrence theorem, not a global regime classifier.  The
two-sided event budget proves a summable tail once the branch is already known
to enter the ordinary/separated-binary shell hypotheses in both time
directions.  The theorem assembler therefore reports
`nonzero_angular_event_regime_membership_from_initial_data` until a constructor
derives those future and past shell hypotheses from the supplied initial data.
