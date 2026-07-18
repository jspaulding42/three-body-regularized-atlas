# Rust verifier v1

This standalone crate is the implementation-independent verifier's Rust
foundation. The current checkpoint contains:

- a manual, resource-bounded RFC 8259 JSON parser with duplicate-key, UTF-8,
  scalar-class, and resource-limit enforcement;
- canonical raw-byte validation against the v0.3 JSON profile;
- an explicit `V03Compatible` typed decoder for the complete finite v0.3
  raw-v1 record grammar, with exact field sets, segment tags, and the legacy
  parser-versus-`UNRESOLVED` shape boundary documented in the
  [Rust schema map](../../docs/raw-v1-rust-schema-map.md);
- a non-certifying semantic conversion from decoded ordinary chart/tube wires
  to bounded exact-rational inputs, using every real's proved binary64 dyadic,
  exact body-major coefficient shapes, and strictly increasing parameter and
  physical-time intervals;
- exact decoding of finite IEEE-754 binary64 bit patterns;
- manual, resource-bounded RFC 8259 number parsing and exact nearest-even
  decimal-to-binary64 proof, retaining the exact decimal lexeme value
  separately from the theorem-facing exact binary64 dyadic;
- bounded canonical exact-rational admission, including fail-closed rejection
  of malformed or unreduced raw ratios;
- checked exact rational intervals and resource-capped Horner evaluation;
- bounded nonempty rectangular exact-rational vector polynomials in ascending
  degree-major order, with resource-capped interval Horner evaluation and exact
  integer formal differentiation;
- a bounded first-order rational interval-dual primitive with at most 64
  gradient coordinates, checked value/gradient algebra, and reciprocal and
  square-root chain rules;
- an independently evaluated planar 12-state Newton RHS and 12-by-12
  rational-interval Jacobian, with an exact infinity-row-sum Lipschitz upper
  bound, fixed-pair separation, exact positive-mass preflight, and an
  unequal-mass conservation regression;
- a direct ordinary polynomial-defect kernel enclosing all 12 residuals in
  `q' - v`, `v' - a(q)` order and returning the exact maximum of the absolute
  values of their rational interval endpoints, with regressions for unequal
  masses, an acceleration-block maximum, and early mass/precision preflight;
- integer-arithmetic dyadic square-root enclosures with exact postconditions
  and a hard pre-allocation precision cap;
- an exact rational exponential enclosure using range reduction, a Taylor
  partial sum with an explicit geometric-tail majorant, exact squaring, full
  witness replay, and hard component/work/storage budgets;
- exact outward binary64 mass coefficients for all eight raw-v1 Section 3.3
  mass formulas, with exact adjacency and nearest-even endpoint checks tested
  against hand-derived boundaries and deterministic lattice stress cases;
- partial replay of the 13 ordered ordinary-chart ledger entries under
  `exact_rational_ordinary_chart_claimed_tail_v04`, consuming the semantic
  adapter's exact binary64 dyadics, using bounded adaptive 256-, 512-, 1024-,
  and 2048-bit square-root interval precision, and carrying exact rational
  interval formal-series recurrence, residual, and Horner arithmetic. All six
  ordinary charts in each canonical baseline chain, twelve chart replays total,
  satisfy the ordered ledger under this claimed-tail profile. Its inherited
  `ordinary_taylor_exact_rational_residual_polynomials` ID denotes exact
  rational interval arithmetic, not a proved remainder; and
- conditional replay of the six ordered ordinary-tube obligations under the
  named `exact_rational_ordinary_tube_v04` profile. It uses the exact analytic
  v0.3 pair-distance and Lipschitz formula family, 256-bit square-root
  enclosures, a 32-bit upward dyadic enclosure of the exponential argument,
  Taylor cutoff 32, and maximum reduced Taylor tail `2^-128`. All six ordinary
  tubes in each baseline chain, twelve replays total, certify conditionally;
- exact replay of the eight direct Python initial-value-binding obligations
  under `exact_rational_initial_value_binding_v04`, after the ordinary semantic
  adapter has admitted a planar chart. The profile evaluates the chart and its
  affine physical-time map at the binding parameter with exact rational
  arithmetic. It deliberately does not treat `source` as part of direct
  binding identity. Both canonical roots satisfy all eight obligations; and
- an eight-obligation proof-oriented root composition under
  `exact_rational_validated_ordinary_root_v04`. This strengthens the historical
  six-obligation wrapper with exact chart unit speed and an exact physical-time
  anchor, composes the exact binding and exact ordinary-tube profiles, and does
  not consume the claimed-tail chart ledger. It computes the actual initial
  error whenever the binding position and velocity gaps are available, and it
  retains the exact root clock origin only when both the unit-speed and
  physical-time-anchor obligations hold. Both canonical roots satisfy all
  eight obligations; and
- local replay of the nine historical ordinary-bridge obligation IDs under
  `exact_rational_carried_ordinary_bridge_v04`. It freshly replays both source
  and target tubes under the exact ordinary-tube profile, evaluates the source
  endpoint in `q`-then-`v` order, inflates it by the source replay's rational
  Gronwall upper bound, checks inclusive containment in the target initial ball,
  and derives the exact clock update `B'=B+e-a`. The replay pins
  `ordinary_autonomous_uniqueness_bridge_kernel_v1`, while deliberately
  ignoring both charts' claimed-tail ledgers. Physical-time metadata is used
  only for semantic/schema admission and never updates or overrides the bridge
  clock. Both canonical first bridges pass and advance `[0,0]` to
  `[2^-40,2^-40]`.

At this checkpoint, 158 Rust unit tests and five integration tests pass;
`cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are clean.

The checked-in [raw-v1 seed corpus](../../conformance/raw-v1/README.md) supplies
two `ACCEPT` baselines and sixteen isolated `REJECT` mutations. It is
`seed_incomplete`, not the complete release corpus.

The corpus's strict
[`ordinary-chart profile observation`](../../conformance/raw-v1/ordinary-chart-profile-expectations.json)
also records that both baseline inputs have equal all-true ordered Boolean
ledgers under the frozen Python direct-object profile and this Rust
claimed-tail profile. The comparison deliberately sets mathematical
comparability false, release status `BLOCKED`, and the OPEN-V1-06
profile-semantics blocker. It is not evidence of cross-profile mathematical
agreement, frozen arithmetic parity, or independent chain replay.

The crate still stops before theorem-facing certificate replay: there is no
raw SHA-256 layer. The ordinary-chart profile's serialized `tail_bound` remains
an unproved claimed allowance. It provides no interval-wide collision-free
witness, no formal-series convergence proof, and no rigorous remainder
witness; it is not frozen-v0.3 primitive parity and does not close
`OPEN-V1-06`. The ordinary-tube result alone proves only a conditional
a-posteriori estimate; the separately named binding and validated-root
profiles now establish the corresponding exact local root obligations only
for admitted planar semantic inputs. They are not parity with the Python
direct-object API on three-dimensional inputs, and they are not raw-chain root
admission. A raw-chain root must additionally require a nonempty binding
`source`, exact-zero time/position/velocity tolerances, and equality of the
binding parameter, chart left endpoint, and tube anchor. Raw-chain root
admission remains unimplemented. The exact ordinary-bridge profile is a local
component under a separate arithmetic profile: it neither reproduces v0.3
binary64 status parity nor folds or commits a raw chain. Fixed-time semantics,
every LC field, full chain replay, and clock/gauge induction beyond the local
components remain unimplemented. `OPEN-V1-08` remains open for historical
binary64 arithmetic status parity. The crate does not call Python and makes no
complete-certificate, end-to-end independent-verifier, or independent-chain-
replay claim. `OPEN-V1-01` also remains open: the current
canonical float renderer is tested against the frozen fixtures but is not yet
a language-neutral normative algorithm, and its Rust toolchain is not pinned.
