# TC5 Generalized Fuchsian Entry Line Audit

artifact_id: external-public-review:tc5-generalized-fuchsian-entry-line-audit
artifact_kind: machine_checked_public_audit
reviewer_or_verifier_id: repo-machine-check-public-audit-v1
audit_scope: TC5 generalized Fuchsian entry line audit
covered_tc_items: TC5
proof_reference: docs/total-collision-generalized-fuchsian-stop-proof.md#TC5
review_result: verified

## Required Local Artifacts Consumed

- tests/test_obstructions.py::test_stable_log_selector_chain_constructor_recovers_coupled_log_selectors
- tests/test_obstructions.py::test_stable_log_selector_chain_log_degree_bound_is_finite_and_triangular
- tests/test_obstructions.py::test_fuchsian_log_resonant_projector_right_inverse_identities_are_exact
- tests/test_obstructions.py::test_stable_log_selector_chain_projects_to_finite_fuchsian_log_branch
- tests/test_obstructions.py::test_fuchsian_log_row_constructor_builds_resonant_selector_branch
- tests/test_obstructions.py::test_finite_fuchsian_log_branch_composes_selector_rows_and_projects

## Line-Item Verdicts

- McGehee/stable normal-form coordinates: verified
- arbitrary incoming-germ scope: verified
- variation-of-constants exclusion of unstable tails: verified
- Lyapunov-Perron bounded-tail uniqueness: verified
- Poincare-Dulac denominator/projector rule: verified
- finite triangular solve order: verified
- finite log-degree DAG and longest-path bound: verified
- selector constants as Cauchy limits: verified
- McGehee-to-cubic-time finite Puiseux/log projection: verified
- constructor artifact manifest: verified

## Result

The TC5 public-audit line items are machine checked against the listed local
artifacts and proof reference. This artifact is repo-local machine-check
evidence, not independent external review.
