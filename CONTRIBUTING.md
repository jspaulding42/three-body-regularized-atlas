# Contributing

Thank you for helping improve this research artifact. The most valuable
contributions are independent mathematical review, trusted-kernel audits,
reproduction reports, and narrowly scoped corrections.

## Before opening a change

Open an issue describing the mathematical or software question first when the
change affects a theorem statement, certificate semantics, or the trusted
computational kernel. For ordinary bug fixes and documentation corrections, a
pull request may be opened directly.

Please keep claims within the repository's demonstrated scope: the current
artifact is a proof-of-concept validation of one local planar Levi-Civita
binary-collision passage, conditional on the stated trusted kernel. It is not a
general closed-form solution of the three-body problem.

## Development workflow

Use a supported Python version (3.10 or newer), install the project with its
test dependencies, and run the test suite:

```console
python -m pip install -e '.[test]'
python -m pytest
```

To exercise the principal reproducible certificate and the symbolic projection
checks, run:

```console
python scripts/certify_planar_binary_collision_passage.py
python scripts/verify_lc_projection_identities.py
```

Pull requests should explain:

- what changed and why;
- which mathematical claims, trusted assumptions, or certificate fields are
  affected;
- how the change was tested; and
- whether generative AI materially contributed to the change.

Do not commit generated caches, local environments, private data, credentials,
or the historical `three-body-problem.zip` archive.

## Mathematical and computational review

For corrections to a proof or certificate, include a minimal counterexample or
a derivation that can be reviewed independently. Changes to outward rounding,
transcendental bounds, interval operations, parsers, or certificate acceptance
logic require focused tests at relevant boundary cases. A passing test suite is
necessary but does not by itself establish a mathematical theorem.

## AI-assisted contributions

AI-assisted work is welcome when disclosed. Human contributors must inspect
the proposed change, verify sources and generated calculations to the extent
claimed, and accept responsibility for submitting it. AI systems must not be
listed as human authors or sign off on reviews.

By contributing, you agree that your contribution is licensed under the
BSD-3-Clause license in this repository.
