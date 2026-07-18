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
- fixed-limit opaque raw-v1 admission that owns the exact canonical bytes and
  their strict `V03Compatible` typed decode, preserves wire-versus-schema
  failure stages, and exposes immutable borrows. This is not SHA-256, global
  namespace checking, or outer-obligation replay;
- non-certifying exact-dyadic semantic inputs for planar LC charts and tubes,
  including common bounded coefficient shapes for six planar vector series
  and two scalar series, positive masses, ascending canonical pairs, strict
  intervals, tube signs/anchors, and finite untrusted chart claims without
  adding sign hypotheses. Public LC-entry semantic construction additionally
  requires the opaque admitted chain and a segment index, then checks the
  carried IDs, constants, masses, and exact source-right/target-left bindings;
- exact arithmetic for the explicitly ordered lifted LC state
  `(zx,zy,wx,wy,h,Rx,Ry,Ux,Uy,yx,yy,Vx,Vy,t)`: one bounded 14-component
  rational polynomial assembled degree-by-degree, interval and derivative
  evaluation, 13-plus-time anchor points, uniform nonnegative inflation,
  dependency-aware squares for `rho`, the LC square and pair-energy
  constraint, and the deck involution. Admitted charts retain the three
  original positive binary64 mass encodings alongside their exact rational
  masses solely as input identity for future mass replay;
- an independent 14-variable exact-rational interval-dual LC vector field,
  including mass reconstruction and profile revalidation, guarded third-body
  denominators, the full ordered RHS and Jacobian, and an exact maximum
  infinity row-sum bound; plus a direct uninflated chart-polynomial defect and
  exact maximum absolute residual-endpoint bound. The defect does not consume
  the serialized tail claim;
- bounded exact-rational interval formal-series replay of the full ordered LC
  field coefficients, derivative-minus-field residual coefficients, and pair
  energy constraint coefficients. Its inverse-cube recurrence uses guarded
  denominator-square series, a recursively constructed positive square-root
  series initialized by a dyadic enclosure, reciprocal convolution, exact
  endpoint maxima, and an explicit quadratic work cap. Recurrence divisions
  are enclosed on a fixed dyadic grid (64 bits by default, with an explicit
  bounded precision API). It consumes coefficient and mass data only, not
  chart intervals, tolerances, tail claims, projection claims, or samples;
- fixed-dimensional exact-rational interval projection from an explicit
  lifted LC state to three Cartesian positions and accelerations. It enforces
  a nonnegative claimed rho floor and strict whole-interval separation above
  that floor, projects the regularized acceleration through the frozen LC
  formula, independently recomputes guarded Newton accelerations from exact
  masses, and returns six residuals plus their exact endpoint maximum. It
  consumes no tail, tolerance, sample, time interval, or chart coefficient;
- the ordered 15-obligation Section 4.2 LC-chart ledger under the separately
  named `exact_rational_planar_lc_chart_claimed_tail_v04` profile. It composes
  a 64-piece exact-rational physical-time hull, the coefficient recurrence and
  constraint replay, interval-coefficient residual Horner bounds, and the
  projection kernel with the historical dependency gates. Domain failures are
  false obligations; numeric/resource/kernel failures remain typed errors;
- the ordered nine-obligation Section 4.5 conditional LC-tube ledger under
  `exact_rational_planar_lc_tube_v04`. It directly replays the uninflated
  14-state polynomial defect, inflates the full lifted state, checks strict
  third-body separation and an interval-Jacobian row-sum cap, optionally
  checks the anchor pair-energy constraint exactly, and applies a strict
  rational Gronwall self-consistency test. It independently reconstructs the
  outward mass profile and reports exact diagnostic bounds. Certification is
  conditional tube evidence only: it proves no IVP containment, lift/entry,
  exit, chain step, claimed-tail theorem, or frozen binary64 parity;
- the separately named `exact_rational_planar_lc_lift_cover_v04` arithmetic
  profile. From a complete 12-component Cartesian interval box and an admitted
  LC mass/pair identity, it applies the exact Section 5.3 canonical branch
  order, constructs one or two 13-component lifted patches, derives the
  parity-one negative-cut edge, and checks strict selected-pair and patch-rho
  lower bounds. Its component formula
  `w=(1/2)(zx vx+zy vy, zx vy-zy vx)` is exactly the frozen
  `w=(1/4)L(z)^T v` convention because that specification defines `L` with a
  leading factor two. A complete result invokes only the pinned existential
  constrained-lift/deck/gauge implication: rectangular patches are not wholly
  constrained, and no entry IDs, target containment, gauge graph, IVP carry,
  exit, or chain step is replayed;
- the exact ordered 21-row Section 5.3 carried-entry composer under
  `exact_rational_carried_planar_lc_entry_v04`. It derives the entry record
  from opaque canonical raw admission, binds the caller's source chart/tube
  exactly to the preceding raw frontier, checks the bounded global defining-ID
  namespace including all reserved entry IDs, freshly replays both chart/tube
  pairs, reconstructs the complete source endpoint box, invokes the canonical
  lift cover, derives the F2 assignments and exact `D_in`, and tests both
  global complements against the target 14D anchor ball. The claimed-tail
  chart ledgers are compatibility gates only; the source tube, direct endpoint,
  lift, and containment are decisive. Its all-true conclusion is conditional
  on the parent having carried its actual branch inside the source tube and
  proves no LC evolution, exit, chain commit, or fixed-time theorem;
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
  `[2^-40,2^-40]`; and
- bounded orchestration under `exact_rational_raw_ordinary_only_chain_v04`,
  beginning at raw obligations 6--13. It applies the strict raw root gate and
  consumes the claimed-tail chart replay only as a fail-closed compatibility
  gate, not theorem evidence. It transactionally commits ordinary bridges,
  records an exact clock ledger, freshly replays the current tube for
  absolute-parameter interval Horner evaluation and radius inflation, and
  checks the inclusive maximum width across all 12 components. A width-only
  failure retains the fixed-time enclosure; otherwise a fresh ordinary right
  frontier is retained when available. Malformed later ordinary records are
  structured failures, segment count is capped at 256 before replay, and LC is
  an explicit unsupported stop. Both canonical chains commit one bridge and
  stop at their first LC segment.

At this checkpoint, 236 Rust unit tests and 15 integration cases across five
integration test targets pass;
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
profiles establish the corresponding exact local root obligations only for
admitted planar semantic inputs. They are not parity with the Python
direct-object API on three-dimensional inputs. The bounded ordinary-only
checkpoint enforces the additional raw root conditions and
ordinary-prefix/fixed-time semantics described above. The conditional LC tube
ledger now exists; LC entry, exit, and full-chain replay remain unimplemented.
`OPEN-V1-08` remains
open for historical
binary64 arithmetic status parity. The crate does not call Python and makes no
complete-certificate, end-to-end independent-verifier, or independent-chain-
replay claim. `OPEN-V1-01` also remains open: the current
canonical float renderer is tested against the frozen fixtures but is not yet
a language-neutral normative algorithm, and its Rust toolchain is not pinned.

The opaque admission and LC semantic adapters are non-certifying. The
arithmetic field reconstructs and revalidates its mass profile, but these
layers alone prove no LC chart, tube, or entry obligation and perform no gauge,
lift, projection, exit, or chain-commit replay. They are not
frozen parity, cross-profile agreement, verifier independence, or release-gate
evidence.

The lifted-state, field, and direct-defect layers are likewise arithmetic
only. Enclosing a residual does not establish an accepted defect threshold or
remainder theorem, and these layers do not prove the LC constraint,
chart/tube/entry validity, gauge, projection, exit, or any continuation step.
The formal-series layer also makes no chart-acceptance or frozen binary64
status-parity claim.
The projection residual is likewise arithmetic evidence only: it supplies no
accepted tail/remainder bound, chart acceptance, or continuation result.
Even an all-true LC-chart profile is conditional on the unproved serialized
tail allowance. It is not convergence, frozen binary64 parity, or theorem-
facing chart acceptance. The separate all-true LC-tube profile is conditional
a-posteriori evidence and does not supply root/IVP containment or a transition.

The ordinary-only checkpoint starts at obligations 6--13 and makes no claim about outer
obligations 1--5, namespace uniqueness, canonicalization or SHA-256 results,
LC replay, complete-chain replay, frozen parity, cross-profile agreement,
independence, or release-gate satisfaction.
