# TC6 Cauchy Majorant Backend Line Audit

artifact_id: external-public-review:tc6-cauchy-majorant-backend-line-audit
artifact_kind: machine_checked_public_audit
reviewer_or_verifier_id: repo-machine-check-public-audit-v1
audit_scope: TC6 Cauchy majorant backend line audit
covered_tc_items: TC6
proof_reference: docs/total-collision-generalized-fuchsian-stop-proof.md#TC6
review_result: verified

## Required Local Artifacts Consumed

- tests/test_certificate_checker.py::test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture
- tests/test_certificate_checker.py::test_generalized_fuchsian_remainder_majorant_picard_tail_formula_is_explicit
- tests/test_certificate_checker.py::test_independent_checker_recomputes_generalized_fuchsian_primitive_tail_bounds
- tests/test_certificate_checker.py::test_generalized_fuchsian_projected_residual_direct_interval_is_certification_gated
- tests/test_certificate_checker.py::test_generalized_fuchsian_projected_budget_does_not_weaken_lifted_gate
- tests/test_certificate_checker.py::test_generalized_fuchsian_projected_residual_weight_shift_is_cubic_time_exact
- tests/test_certificate_checker.py::test_generalized_fuchsian_stop_checker_certification_is_interval_not_sample_gated
- tests/test_certificate_checker.py::test_independent_checker_rejects_corrupted_generalized_fuchsian_majorant
- tests/test_certificate_checker.py::test_independent_checker_rejects_generalized_fuchsian_without_projected_tail
- TotalCollisionGeneralizedFuchsianStopChartCertificate
- check_total_collision_generalized_fuchsian_stop_chart

## Line-Item Verdicts

- tc6-polydisc:P(r): verified
- tc6-norm:sup-polydisc: verified
- primitive constants C0, Lambda, sigma, p0, d: verified
- certify_rational_interval_arithmetic_backend_soundness: verified
- Banach inequality `B L_N < 1`: verified
- Banach inequality `B D + B L_N R_0 <= R_0`: verified
- Picard tail formula `q^N B D / (1-q)`: verified
- projected residual weight shift under `t_c - t = tau^3`: verified
- endpoint collapse envelope: verified
- center-of-mass ledger: verified
- linear momentum ledger: verified
- angular momentum ledger: verified
- energy ledger: verified
- independent_total_collision_generalized_fuchsian_stop_checker_interval_cauchy_projected_v3: verified
- generalized_fuchsian_remainder_majorant_certifies: verified
- generalized_fuchsian_remainder_component_inputs_certify: verified
- generalized_fuchsian_remainder_required_residual_components: verified
- cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells: verified
- generalized_fuchsian_endpoint_collapse_envelope: verified
- first_jet: verified
- lifted_residual: verified
- physical_residual: verified
- regularized_position_value: verified
- value: verified

## Result

The TC6 public-audit line items are machine checked against the listed local
artifacts and proof reference. This artifact is repo-local machine-check
evidence, not independent external review.
