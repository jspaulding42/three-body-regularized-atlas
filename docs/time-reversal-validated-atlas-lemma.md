# Time-Reversal Validated-Atlas Lemma

Newton's equation is invariant under the involution

```text
R(t,q,v)=(-t,q,-v).
```

If `q(t)` solves

```text
q''(t)=A(q(t)),
```

then `Q(s)=q(-s)` satisfies

```text
Q'(s)=-q'(-s),        Q''(s)=q''(-s)=A(q(-s))=A(Q(s)).
```

Thus every certified atlas for initial data `(q0,-v0)` on `[0,T]` gives a
certified atlas for `(q0,v0)` on `[-T,0]` by negating physical-time intervals
and velocity coordinates while leaving positions, masses, chart equations,
tail bounds, and transition containment unchanged.

For interval state sets, the transform is componentwise:

```text
[q_i^-,q_i^+]       -> [q_i^-,q_i^+],
[v_i^-,v_i^+]       -> [-v_i^+,-v_i^-].
```

For a chart whose certified physical-time interval is `[a,b]`, the reversed
chart interval is `[-b,-a]`. The final chart still contains the requested
target time because

```text
T in [a,b]  iff  -T in [-b,-a].
```

All local proof ledgers are preserved:

- Newton residuals are preserved by the calculation above.
- Projection witnesses are preserved because the coordinate projection is
  unchanged and only the velocity sign is reversed.
- Center-of-mass position intervals are unchanged, while center-of-mass
  velocity and linear momentum change sign, so zero or interval-contained
  ledgers remain certified.
- Angular momentum `sum m_i q_i wedge v_i` changes sign, so conservation and
  zero-containment ledgers remain certified.
- Energy is invariant because the kinetic term uses `|v|^2` and the potential
  depends only on positions.
- Tail and Cauchy budgets are unchanged because they depend on absolute
  coefficient and step-size bounds.

The executable constructor is
`time_reverse_validated_atlas_solution(...)`. The public finite-time evaluator
uses it for negative planar `method="validated_atlas"` calls: it constructs the
positive-time planar hybrid atlas for reversed velocities, then maps the result
back to the requested negative target time. This extends the planar
ordinary/Levi-Civita validated-atlas route to both signs of finite physical
time without adding a new dynamical assumption.
