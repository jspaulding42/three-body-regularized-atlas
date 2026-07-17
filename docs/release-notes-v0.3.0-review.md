# v0.3.0-review release notes

This review snapshot exposes a proof-carrying finite planar continuation-chain
surface for one supplied exact binary64-dyadic point IVP. This is a review
artifact, not a claim that the general three-body problem is solved.

## Implemented theorem surface

- Strict raw-v1 replay starts from an exact point-IVP binding and freshly
  checks an initial ordinary chart and tube.
- A finite chain may contain ordinary bridges and repeated
  `N -> LC_ij -> N` passages for all three canonical pairs, including a
  same-pair revisit after return to the ordinary chart.
- The checker derives conditional clock, cocycle, and passage-local gauge
  ledgers; it accepts no producer-supplied propagated clock or gauge result.
- The theorem-facing LC tube and carried entry/exit paths derive coefficients
  exactly from binary64 mass values and then enclose them outward.
- Replay returns either `CERTIFIED_TO_T` with an exact-rational fixed-time
  ordinary enclosure or `UNRESOLVED` with only the accepted prefix and an
  appropriately typed retained frontier.

The exact scope and proof obligations are stated in
[`raw-repeated-planar-continuation-chain-theorem.md`](raw-repeated-planar-continuation-chain-theorem.md).

## Canonical review bundle

[`../artifacts/v0.3.0-review/planar-chain/`](../artifacts/v0.3.0-review/planar-chain/)
contains:

- `success.raw.json` and `success.replay.json`: fresh replay returns
  `CERTIFIED_TO_T` after five segments. The raw theorem-evidence SHA-256 is
  `ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11`.
- `failed-revisit.raw.json` and `failed-revisit.replay.json`: fresh replay
  returns `UNRESOLVED`, accepts a four-segment prefix, rejects segment index 4
  at complete LC-exit containment, and retains a typed lifted LC frontier.
  The raw theorem-evidence SHA-256 is
  `c6830919726c22fbca6e45d5781b2465e54876bfa89c25bc26e97284ff918727`.
- `manifest.json`: schema versions, pinned checker/kernel identifiers,
  informational build-environment versions, and SHA-256 transport hashes for
  every raw and replay payload.

Verify the checked-in bundle:

```bash
python scripts/certify_repeated_planar_chain.py verify-bundle
```

Run the focused regressions:

```bash
python -m pytest -q \
  tests/test_planar_lc_mass_coefficients.py \
  tests/test_proof_carrying_planar_chain.py \
  tests/test_planar_chain_review_artifact.py
```

## Review paper

The visually inspected 13-page A4 review PDF is
[`proof-carrying-finite-planar-three-body-continuation-v0.3.0-review.pdf`](../artifacts/v0.3.0-review/paper/proof-carrying-finite-planar-three-body-continuation-v0.3.0-review.pdf),
with SHA-256
`2fd7dbae92e0e81b2be46ee9b7c36728345b0d88b3071405864708ef2eb2259e`.
The digest identifies the attached review asset; the project does not claim
bit-reproducible PDF generation across TeX environments.

## Trust boundary and nonclaims

Acceptance remains conditional on the documented Python exact-arithmetic,
outward binary64 interval, ODE enclosure, Newtonian/Levi-Civita formula, and
analytic uniqueness/projection kernels. The included command performs exact
self-replay with the same implementation; it is not an independent verifier
or proof-assistant proof.

The snapshot does not claim automatic certificate production, arbitrary-data
or all-time completeness, total-collision continuation, spatial coverage, a
general closed-form solution, or a physical collision in any LC segment.
