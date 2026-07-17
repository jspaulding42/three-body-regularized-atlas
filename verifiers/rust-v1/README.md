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
  against hand-derived boundaries and deterministic lattice stress cases; and
- conditional replay of the six ordered ordinary-tube obligations under the
  named `exact_rational_ordinary_tube_v04` profile. It uses the exact analytic
  v0.3 pair-distance and Lipschitz formula family, 256-bit square-root
  enclosures, a 32-bit upward dyadic enclosure of the exponential argument,
  Taylor cutoff 32, and maximum reduced Taylor tail `2^-128`. All six ordinary
  tubes in each baseline chain, twelve replays total, certify conditionally.

At this checkpoint, 137 Rust unit tests and two raw-v1 corpus tests pass;
`cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are clean.

The checked-in [raw-v1 seed corpus](../../conformance/raw-v1/README.md) supplies
two `ACCEPT` baselines and sixteen isolated `REJECT` mutations. It is
`seed_incomplete`, not the complete release corpus.

The crate still stops before theorem-facing certificate replay: there is no
raw SHA-256 layer. The semantic ordinary-chart input is not primitive chart
certification—it does not replay the coefficient recurrence or Taylor-model
residual/tail obligations—and the ordinary-tube result proves only a
conditional a-posteriori estimate, not root/IVP containment. Root/IVP binding,
handoff/ordinary-bridge/fixed-time semantics, every LC field, full chain
replay, clock/gauge logic, and the semantic result serializer remain
unimplemented. `OPEN-V1-08` remains open for status parity between this exact
rational profile and the historical binary64 outward profile. The crate does
not call Python and makes no certificate-level or independent-replay claim.
`OPEN-V1-01` also remains open: the current canonical float renderer is tested
against the frozen fixtures but is not yet a language-neutral normative
algorithm, and its Rust toolchain is not pinned.
