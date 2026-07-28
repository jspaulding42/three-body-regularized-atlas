# Proof-grade raw-v1 v0.4 foundation corpus

This is a deliberately bounded conformance foundation for the separately
named proof-grade raw-v1 execution profile. Its status is
`foundation_incomplete`: it contains the two archived admitted chains and the
sixteen archived strict-admission rejections only. It is not a release corpus,
does not establish cross-verifier agreement, and does not claim family
coverage or independent replay.

`cases.json` records the expected total execution classification and, for the
two admitted chains, a hand-auditable semantic signature. The signature is a
selected proof-grade execution projection, not a raw certificate producer.
Each input is referenced by repository-relative path and SHA-256. The frozen
raw-v1 specification is also bound by SHA-256.

Run the standalone, standard-library-only checker from the repository root:

```text
python3 scripts/proof_grade_v04_conformance.py validate
python3 scripts/proof_grade_v04_conformance.py run-rust --rust-executable verifiers/rust-v1/target/release/raw_v1_proof_grade_verify
```

The manifest hashes this README, `cases.schema.json`, `cases.json`, and the
out-of-directory standalone runner script. `manifest.json` alone excludes its
own hash by design; each referenced input is bound in the cases and manifest
input tables instead.
