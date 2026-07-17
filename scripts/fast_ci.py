"""Run the fast theorem/checker CI target from LAST_PRO_INSTRUCTIONS."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from three_body_symmetry.public_proof_audit import (  # noqa: E402
    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
    PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
)


def _pytest_k_expression(artifact_ids: tuple[str, ...]) -> str:
    return " or ".join(
        artifact_id.split("::", maxsplit=1)[1]
        for artifact_id in artifact_ids
    )


TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K = _pytest_k_expression(
    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
    + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
)

PUBLIC_AUDIT_FAST_K = (
    "public_total_collision_audit_keeps_tc4_tc6_open_by_default or "
    "public_regularized_atlas_audit_keeps_external_review_open_after_local_tc4_tc6_package or "
    "public_general_closed_form_target_keeps_external_review_open_after_local_public_audit_package or "
    "public_audit_manifest_resolver_certifies_local_artifact_manifest or "
    "public_audit_required_artifacts_are_covered_by_ci_targets"
)

PUBLIC_REVIEW_ARTIFACT_HARDENING_K = (
    "public_review_artifact_resolution_certifies_required_tc4_tc6_artifacts or "
    "public_review_artifact_resolution_rejects_missing_artifact_object or "
    "public_review_artifact_resolution_rejects_duplicate_artifact_reuse or "
    "public_review_artifact_resolution_rejects_missing_proof_reference or "
    "public_review_artifact_resolution_rejects_stale_proof_note_reference or "
    "public_review_artifact_resolution_rejects_repo_artifacts_labeled_external_review or "
    "public_review_artifact_resolution_rejects_missing_checked_artifacts or "
    "public_review_resolver_rejects_stale_local_audit_evidence or "
    "public_review_resolver_rejects_stale_top_level_proof_reference_manifest or "
    "review_ready_total_collision_package_accepts_neutral_public_review_artifact_ids or "
    "review_ready_total_collision_package_rejects_conflicting_public_review_ids or "
    "public_total_collision_audit_accepts_neutral_public_review_aliases or "
    "public_total_collision_audit_rejects_conflicting_public_review_alias_ids or "
    "public_total_collision_audit_rejects_conflicting_public_review_resolver_aliases"
)

EVENT_REGIME_ASSEMBLY_HARDENING_K = (
    "event_regime_assembly_rejects_spoofed_obligation_ledgers or "
    "event_regime_local_chart_family_rejects_truthy_source_certificate"
)

GENERAL_SOLUTION_THEOREM_HARDENING_K = (
    "theorem_pipeline_obligation_rejects_truthy_nonboolean_certification or "
    "general_solution_theorem_assembly_rejects_spoofed_obligation_ledgers"
)

FINITE_TIME_ATLAS_HARDENING_K = (
    "validated_atlas_proof_ledger_rejects_truthy_fake_and_optional_only_entries or "
    "validated_atlas_rejects_truthy_component_certification_flags or "
    "validated_atlas_rejects_attribute_compatible_nested_proof_objects or "
    "finite_time_regime_classifier_rejects_spoofed_obligation_ledgers or "
    "finite_time_regime_classifier_ignores_fake_proof_ledger_entries"
)

ZERO_ANGULAR_SELECTOR_HARDENING_K = (
    "zero_angular_compact_finite_atlas_requires_constructor_selector_entry or "
    "positive_energy_homothetic_escape_rejects_fake_majorant_and_selector_entry"
)

CONSTRUCTOR_OBLIGATION_HARDENING_K = (
    "required_constructor_obligations_reject_truthy_attribute_fields or "
    "compact_finite_atlas_rejects_attribute_compatible_validated_atlas"
)

OPEN_TIME_CHECKED_PREFIX_K = (
    "open_time_accepts_supplied_branch_union_checked_prefix or "
    "open_time_accepts_stratified_branch_union_checked_prefix or "
    "independent_checker_bridge_rejects_attribute_compatible_fake_verifier or "
    "independent_checker_bridge_rejects_stale_real_verifier_for_different_atlas or "
    "open_time_arithmetic_gate_rejects_attribute_compatible_fake_verifier or "
    "independent_finite_target_checker_rejects_truthy_atlas_proof_flag or "
    "pointwise_open_time_theorem_rejects_attribute_compatible_nested_components or "
    "pointwise_open_time_theorem_rejects_stale_finite_target_theorem_source or "
    "open_time_theorem_surfaces_reject_spoofed_obligation_ledgers or "
    "open_time_child_certificates_reject_attribute_compatible_nested_components or "
    "finite_target_completeness_reduction_rejects_spoofed_nested_theorems or "
    "total_collision_policy_certificate_requires_exact_policy_flags or "
    "proof_certified_properties_do_not_raise_or_promote_scaffolds or "
    "compact_interval_rejects_attribute_compatible_finite_target_certificates"
)

FINITE_TARGET_FUCHSIAN_STOP_LEDGER_HARDENING_K = (
    "supplied_generalized_fuchsian_certificates_reject_spoofed_obligation_ledgers or "
    "supplied_finite_fuchsian_log_stop_chart_rejects_spoofed_obligation_ledger or "
    "supplied_finite_fuchsian_log_stop_chart_rejects_fake_constructor_inputs"
)

FINAL_PACKAGE_FAST_K = (
    "final_regularized_atlas_package_closes_internal_and_machine_checked_public_routes or "
    "final_regularized_atlas_package_keeps_default_public_route_open or "
    "final_regularized_atlas_package_keeps_interval_box_theorem_separate or "
    "final_regularized_atlas_package_reuses_tc6_evidence_without_rechecking or "
    "final_regularized_atlas_package_does_not_import_test_helpers or "
    "final_regularized_atlas_package_rejects_spoofed_evidence_bundle"
)

COMMANDS = (
    (
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "three_body_symmetry",
        "tests",
        "scripts",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_lc_gauge_gluing.py",
        "tests/test_lc_exact_overlap.py",
        "tests/test_lc_exact_gauge_atlas.py",
        "tests/test_lc_gauge_aware_transition.py",
        "tests/test_lc_projection_identities.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_last_pro_instructions.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_final_theorem_package.py",
        "-k",
        FINAL_PACKAGE_FAST_K,
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_closed_form.py",
        "-k",
        "regularized_locally_finite_atlas or pointwise or "
        "certificate_language or enumeration or "
        "internal_route_rejects_truthy_replaced_gate_flags or "
        "sundman_route_rejects_truthy_global_series_and_scope_flags or "
        "scope_and_sundman_witness_constructors_reject_truthy_flags or "
        "maximal_classical_policy_constructor_rejects_truthy_flags or "
        "general_closed_form_solution_rejects_spoofed_requirement_statuses or "
        "set_valued_constructor_regularized_atlas_route_rejects_fake_verifier",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_general_solution.py",
        "-k",
        f"{GENERAL_SOLUTION_THEOREM_HARDENING_K} or {FINITE_TIME_ATLAS_HARDENING_K}",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_general_solution_theorem.py",
        "-k",
        f"{ZERO_ANGULAR_SELECTOR_HARDENING_K} or "
        f"{CONSTRUCTOR_OBLIGATION_HARDENING_K}",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_public_proof_audit.py",
        "-k",
        f"{PUBLIC_AUDIT_FAST_K} or {PUBLIC_REVIEW_ARTIFACT_HARDENING_K}",
        "--maxfail=1",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_finite_target_completeness.py",
        "-k",
        "supported_event_function_generation or "
        "finite_target_search_completeness_rejects_spoofed_nested_theorem or "
        "validated_set_valued_constructor_rejects_truthy_nested_proof_flags or "
        "terminal_policy_stratification_rejects_truthy_terminal_policy_evidence or "
        "stratified_branch_tree_rejects_truthy_supplied_leaf_and_tree_flags or "
        "branch_event_tree_rejects_attribute_compatible_obligation_spoof or "
        "recursive_stratified_consumption_rejects_truthy_child_consumption or "
        "recursive_stratified_consumption_rejects_attribute_compatible_obligation_spoof or "
        f"{FINITE_TARGET_FUCHSIAN_STOP_LEDGER_HARDENING_K}",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_open_time_atlas.py",
        "-k",
        OPEN_TIME_CHECKED_PREFIX_K,
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_zero_angular_entry.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_obstructions.py",
        "-k",
        f"{TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K} or {EVENT_REGIME_ASSEMBLY_HARDENING_K}",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_certificate_checker.py",
        "-m",
        "not slow",
        "--maxfail=1",
    ),
)


def main() -> int:
    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
