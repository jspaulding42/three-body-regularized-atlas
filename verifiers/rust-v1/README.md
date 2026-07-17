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

- exact decoding of finite IEEE-754 binary64 bit patterns;
- manual, resource-bounded RFC 8259 number parsing and exact nearest-even
  decimal-to-binary64 proof, retaining the exact decimal lexeme value
  separately from the theorem-facing exact binary64 dyadic;
- arbitrary-size exact rational intervals and Horner evaluation; and
- integer-arithmetic dyadic square-root enclosures with exact postconditions.

The checked-in [raw-v1 seed corpus](../../conformance/raw-v1/README.md) supplies
two `ACCEPT` baselines and sixteen isolated `REJECT` mutations. It is
`seed_incomplete`, not the complete release corpus.

The crate validates bytes and decodes the typed schema only. It stops before
the theorem-facing replay: there is no raw SHA-256 layer, exponential or mass-
coefficient kernel, interval automatic differentiation, ordinary/LC equation
or chart/tube/transition check, chain fold, clock/gauge/fixed-time logic,
obligation ledger, or semantic result serializer. It does not call Python and
makes no certificate-level claim. `OPEN-V1-01` also remains open: the current
canonical float renderer is tested against the frozen fixtures but is not yet
a language-neutral normative algorithm, and its Rust toolchain is not pinned.
