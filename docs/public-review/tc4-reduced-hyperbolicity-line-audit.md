# TC4 Reduced Hyperbolicity Line Audit

artifact_id: external-public-review:tc4-reduced-hyperbolicity-line-audit
artifact_kind: machine_checked_public_audit
reviewer_or_verifier_id: repo-machine-check-public-audit-v1
audit_scope: TC4 reduced hyperbolicity line audit
covered_tc_items: TC4
proof_reference: docs/total-collision-generalized-fuchsian-stop-proof.md#TC4
review_result: verified

## Required Local Artifacts Consumed

- tests/test_obstructions.py::test_arbitrary_mass_equilateral_linearized_spectrum_matches_beta_formula
- tests/test_obstructions.py::test_ordered_euler_linearized_spectrum_has_single_horizontal_shape_parameter
- tests/test_obstructions.py::test_ordered_euler_shape_eigenvalue_bounds_limit_higher_resonance_orders
- tests/test_obstructions.py::test_ordered_euler_shape_gap_public_audit_polynomial_identities_are_exact

## Line-Item Verdicts

- positive-mass domain: verified
- translation quotient: verified
- scale quotient: verified
- rotation quotient: verified
- reflection quotient: verified
- Euler_collinear_positive_mass target family: verified
- Lagrange_equilateral_positive_mass target family: verified
- arbitrary-mass Lagrange spectrum formula: verified
- ordered-Euler spectrum and shape-gap bounds: verified
- indicial equation `(k+2)(k-1)=9 mu`: verified
- quotient-removed center modes: verified
- stable and unstable splitting after quotienting: verified

## Result

The TC4 public-audit line items are machine checked against the listed local
artifacts and proof reference. This artifact is repo-local machine-check
evidence, not independent external review.
