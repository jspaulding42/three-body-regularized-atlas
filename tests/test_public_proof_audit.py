from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import three_body_symmetry as three_body_api
import three_body_symmetry.certificate_checker as certificate_checker_module

from three_body_symmetry.certificate_checker import (
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
)
from three_body_symmetry.closed_form import (
    certify_certificate_language_soundness,
    certify_computable_atlas_certificate_enumeration,
    certify_maximal_classical_total_collision_policy,
    certify_pointwise_regularized_atlas_closed_form_theorem,
    derive_certificate_language_soundness_from_checker_kernel,
    derive_computable_atlas_certificate_enumeration_from_pointwise_theorem,
)
from three_body_symmetry.general_solution_theorem import TheoremPipelineObligation
from three_body_symmetry.open_time_atlas import (
    certify_pointwise_open_time_locally_finite_atlas_theorem,
)
from three_body_symmetry.public_proof_audit import (
    PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES,
    PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
    PUBLIC_TC4_REQUIRED_QUOTIENT_MODES,
    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
    PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
    PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES,
    PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
    PUBLIC_TC5_REQUIRED_RESONANCE_RULES,
    PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS,
    PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS,
    PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID,
    PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
    PUBLIC_TC6_REQUIRED_CHECKER_ID,
    PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS,
    PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS,
    PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID,
    PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS,
    PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES,
    PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
    PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID,
    PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS,
    PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
    PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
    PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL,
    PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE,
    TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,
    PublicCauchyMajorantAuditEvidence,
    PublicAuditManifestResolutionCertificate,
    PublicGeneralClosedFormSolutionCertificate,
    PublicGeneralizedFuchsianEntryAuditEvidence,
    PublicReducedHyperbolicityAuditEvidence,
    PublicRegularizedAtlasClosedFormProofCertificate,
    PublicReviewArtifactEvidence,
    PublicReviewArtifactResolutionCertificate,
    PublicTotalCollisionProofAuditCertificate,
    build_public_tc4_tc6_audit_manifest,
    certify_public_audit_manifest_resolution,
    certify_public_cauchy_majorant_audit_evidence,
    certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart,
    certify_public_general_closed_form_solution_target,
    certify_public_generalized_fuchsian_entry_audit_evidence,
    certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest,
    certify_public_regularized_atlas_closed_form_proof,
    certify_public_review_artifact_evidence,
    certify_public_review_artifact_resolution,
    certify_public_reduced_hyperbolicity_audit_evidence,
    certify_public_reduced_hyperbolicity_audit_evidence_from_manifest,
    certify_review_ready_total_collision_audit_package,
    certify_public_total_collision_proof_audit,
)
from tests.test_certificate_checker import (
    _fast_total_collision_generalized_fuchsian_stop_chart_certificate,
)


ROOT = Path(__file__).resolve().parents[1]


def _assert_pytest_artifact_ids_exist(
    artifact_ids: tuple[str, ...],
    *,
    expected_file: str,
) -> None:
    source = (ROOT / expected_file).read_text()
    for artifact_id in artifact_ids:
        file_name, test_name = artifact_id.split("::", maxsplit=1)
        assert file_name == expected_file
        assert f"def {test_name}(" in source


def _assert_proof_references_resolve(references: tuple[str, ...]) -> None:
    proof_note = (ROOT / "docs" / "total-collision-generalized-fuchsian-stop-proof.md").read_text()
    for reference in references:
        path, anchor = reference.split("#", maxsplit=1)
        assert path == "docs/total-collision-generalized-fuchsian-stop-proof.md"
        assert f"## {anchor}." in proof_note


def _assert_checker_artifact_ids_resolve(artifact_ids: tuple[str, ...]) -> None:
    for artifact_id in artifact_ids:
        artifact = getattr(certificate_checker_module, artifact_id, None)
        assert artifact is not None
        assert getattr(artifact, "__name__", "") == artifact_id


def _derived_soundness():
    return derive_certificate_language_soundness_from_checker_kernel(
        certify_certificate_checker_kernel_support(
            proof_grade_arithmetic_backend_certificate=(
                certify_rational_interval_arithmetic_backend_soundness()
            ),
        )
    )


def _raw_soundness():
    return certify_certificate_language_soundness(
        ordinary_taylor_sound=True,
        levi_civita_sound=True,
        spatial_ks_sound=True,
        fuchsian_stop_sound=True,
        generalized_fuchsian_stop_sound=True,
        transition_sound=True,
        branch_union_sound=True,
        chart_chain_sound=True,
        verifier_kernel_sound=True,
        proof_grade_arithmetic_backend_sound=True,
    )


def _raw_enumeration():
    return certify_computable_atlas_certificate_enumeration(
        chart_family_words_enumerated=True,
        pair_labels_enumerated=True,
        rational_domains_enumerated=True,
        truncation_orders_enumerated=True,
        rational_or_interval_coefficients_enumerated=True,
        rational_tail_budgets_enumerated=True,
        generalized_fuchsian_exponent_data_enumerated=True,
        fuchsian_selector_constants_enumerated=True,
        cauchy_majorants_enumerated=True,
        transition_witnesses_enumerated=True,
        collision_policy_data_enumerated=True,
        independent_checker_dovetailed=True,
        dovetailing_fairness_certified=True,
        finite_target_query_terminates_certified=True,
    )


def _internal_pointwise_closed_form_theorem():
    pointwise = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )
    return certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=pointwise,
        certificate_language_soundness=_derived_soundness(),
        computable_certificate_enumeration=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
        maximal_classical_total_collision_policy=(
            certify_maximal_classical_total_collision_policy(
                pointwise_open_time_theorem=pointwise,
            )
        ),
    )


def _pointwise_open_time_theorem():
    return certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=3,
        compact_time_rate=1.3,
        total_collision_policy_id="maximal_classical_stop",
    )


def _complete_tc4_evidence():
    return certify_public_reduced_hyperbolicity_audit_evidence(
        positive_mass_domain_declared=True,
        quotient_coordinates_declared=True,
        translations_removed=True,
        scale_rotation_reflection_quotiented=True,
        central_target_family_covered=True,
        mass_metric_hessian_spectrum_audited=True,
        zero_modes_removed=True,
        stable_unstable_splitting_certified=True,
        central_target_families=PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES,
        quotient_modes_removed=PUBLIC_TC4_REQUIRED_QUOTIENT_MODES,
        spectrum_audit_reference_ids=PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
        proof_references=PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
    )


def _complete_tc5_evidence():
    return certify_public_generalized_fuchsian_entry_audit_evidence(
        normal_form_coordinates_declared=True,
        exponent_conventions_declared=True,
        resonance_lattice_declared=True,
        finite_log_degree_rule_declared=True,
        denominator_projector_rule_declared=True,
        triangular_solve_order_declared=True,
        finite_selector_data_extraction_audited=True,
        arbitrary_incoming_germ_scope_declared=True,
        normal_form_coordinate_ids=PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES,
        resonance_rule_ids=PUBLIC_TC5_REQUIRED_RESONANCE_RULES,
        triangular_order_keys=PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS,
        selector_data_fields=PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS,
        constructor_artifact_ids=PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
        proof_references=PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
    )


def _complete_tc6_evidence():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    return certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
        certificate,
    )


def _complete_total_collision_audit():
    return certify_review_ready_total_collision_audit_package(
        replace(
            _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
            residual_tolerance=1.0e-5,
            projected_residual_tolerance=2.0e4,
        ),
    )


def _complete_total_collision_audit_with_machine_public_artifacts():
    return certify_review_ready_total_collision_audit_package(
        replace(
            _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
            residual_tolerance=1.0e-5,
            projected_residual_tolerance=2.0e4,
        ),
        public_review_resolution_certificate=(
            certify_public_review_artifact_resolution(project_root=ROOT)
        ),
    )


def test_public_total_collision_audit_keeps_tc4_tc6_open_by_default():
    audit = certify_public_total_collision_proof_audit()

    assert isinstance(audit, PublicTotalCollisionProofAuditCertificate)
    assert not audit.public_proof_certified
    assert audit.tc1_zero_angular_audited
    assert audit.tc2_binary_degenerate_exclusion_audited
    assert audit.tc3_central_shape_limit_audited
    assert audit.tc7_total_stop_chart_soundness_audited
    assert audit.public_audit_blockers == (
        "tc4_reduced_hyperbolicity_audited",
        "tc5_generalized_fuchsian_entry_audited",
        "tc6_cauchy_majorant_constants_audited",
        "public_audit_proof_references_supplied",
        "public_audit_proof_reference_manifest",
        "public_audit_local_manifest_resolution",
    )


def test_public_regularized_atlas_audit_consumes_internal_theorem_without_overclaiming():
    internal = _internal_pointwise_closed_form_theorem()
    public = certify_public_regularized_atlas_closed_form_proof(internal)

    assert isinstance(public, PublicRegularizedAtlasClosedFormProofCertificate)
    assert not internal.proof_certified
    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "public_total_collision_proof_audit_certified" in (
        public.public_audit_blockers
    )
    assert "tc4_reduced_hyperbolicity_audited" in public.public_audit_blockers
    assert "tc5_generalized_fuchsian_entry_audited" in public.public_audit_blockers
    assert "tc6_cauchy_majorant_constants_audited" in public.public_audit_blockers
    assert public.route_summary == (
        "public proof audit is missing the internal pointwise theorem certificate"
    )


def test_public_regularized_atlas_audit_keeps_external_review_open_after_local_tc4_tc6_package():
    internal = _internal_pointwise_closed_form_theorem()
    total_collision_audit = _complete_total_collision_audit()
    public = certify_public_regularized_atlas_closed_form_proof(
        internal,
        total_collision_audit=total_collision_audit,
    )

    assert total_collision_audit.local_audit_package_certified
    assert not total_collision_audit.public_proof_certified
    assert not public.public_proof_certified
    assert "public_total_collision_proof_audit_certified" in (
        public.public_audit_blockers
    )
    assert "public_review_artifact_manifest" in (
        public.public_audit_blockers
    )


def test_public_total_collision_audit_requires_top_level_references_and_checker_artifacts():
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=(),
        checker_artifacts=("TotalCollisionGeneralizedFuchsianStopChartCertificate",),
    )

    assert not audit.public_proof_certified
    assert not audit.checker_artifact_manifest_certified
    assert not audit.proof_reference_manifest_certified
    assert "public_audit_proof_references_supplied" in audit.public_audit_blockers
    assert "public_audit_proof_reference_manifest" in audit.public_audit_blockers
    assert "public_audit_checker_artifact_manifest" in audit.public_audit_blockers

    placeholder = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=("public-review:tc4-tc6",),
        checker_artifacts=PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
    )

    assert not placeholder.public_proof_certified
    assert not placeholder.proof_reference_manifest_certified
    assert "public_audit_proof_references_supplied" not in (
        placeholder.public_audit_blockers
    )
    assert "public_audit_proof_reference_manifest" in (
        placeholder.public_audit_blockers
    )
    assert "public_audit_local_manifest_resolution" in (
        placeholder.public_audit_blockers
    )

    complete = _complete_total_collision_audit()
    assert complete.local_audit_package_certified
    assert not complete.public_proof_certified
    assert complete.checker_artifact_manifest_certified
    assert complete.proof_reference_manifest_certified
    assert complete.local_manifest_resolution_certified
    assert complete.local_manifest_resolution_certificate.proof_certified
    assert not complete.external_public_review_artifact_manifest_supplied
    assert not complete.external_public_review_artifact_manifest_certified
    assert not complete.public_review_artifact_manifest_supplied
    assert not complete.public_review_artifact_manifest_certified
    assert complete.public_review_artifact_kind is None
    assert not complete.public_review_resolution_is_external
    assert not complete.machine_checked_public_audit_resolved
    assert "public_review_artifact_manifest" in (
        complete.public_audit_blockers
    )
    assert set(complete.checker_artifacts).issuperset(
        PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
    )
    assert set(complete.proof_references).issuperset(
        PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES
    )
    assert three_body_api.PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS == (
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )


def test_external_public_review_strings_do_not_close_without_verified_artifact_resolver():
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
    )

    assert audit.local_audit_package_certified
    assert audit.external_public_review_artifact_manifest_supplied
    assert audit.public_review_artifact_manifest_supplied
    assert not audit.external_public_review_artifact_manifest_certified
    assert not audit.public_review_artifact_manifest_certified
    assert audit.public_review_artifact_kind is None
    assert not audit.public_review_resolution_is_external
    assert not audit.machine_checked_public_audit_resolved
    assert not audit.public_proof_certified
    assert "public_review_artifact_manifest" not in (
        audit.public_audit_blockers
    )
    assert "public_review_artifact_verification" in (
        audit.public_audit_blockers
    )


def test_public_review_artifact_resolution_certifies_required_tc4_tc6_artifacts():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    manifest = build_public_tc4_tc6_audit_manifest(
        project_root=ROOT,
        public_review_resolution_certificate=resolution,
    )

    assert isinstance(resolution, PublicReviewArtifactResolutionCertificate)
    assert resolution.proof_certified
    assert resolution.machine_checked_public_audit_resolved
    assert not resolution.external_public_review_resolved
    assert resolution.artifact_ids == (
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )
    assert resolution.missing_obligations == ()
    assert isinstance(resolution.tc4_artifact, PublicReviewArtifactEvidence)
    assert isinstance(resolution.tc5_artifact, PublicReviewArtifactEvidence)
    assert isinstance(resolution.tc6_artifact, PublicReviewArtifactEvidence)
    assert manifest["public_closure_status"] == (
        "machine_checked_public_audit_verified"
    )
    assert manifest["public_review_artifact_ids"] == list(
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )
    assert manifest["public_review_artifact_kind"] == (
        PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE
    )
    assert manifest["public_review_resolved_by_artifact_resolution"] is True
    assert manifest["public_review_resolution_status"] == (
        "machine_checked_public_audit_verified"
    )
    assert manifest["public_review_resolution_is_external"] is False
    assert manifest["external_public_review_resolved_by_local_manifest"] is False
    assert manifest["machine_checked_public_audit_resolved"] is True
    assert manifest["public_review_artifact_resolution"]["proof_certified"] is True
    assert three_body_api.PublicReviewArtifactResolutionCertificate is (
        PublicReviewArtifactResolutionCertificate
    )
    assert three_body_api.certify_public_review_artifact_resolution is (
        certify_public_review_artifact_resolution
    )


def test_review_ready_total_collision_package_constructor_builds_local_package():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    tc4 = certify_public_reduced_hyperbolicity_audit_evidence_from_manifest()
    tc5 = certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest()
    audit = certify_review_ready_total_collision_audit_package(certificate)

    assert tc4.proof_certified
    assert tc5.proof_certified
    assert audit.local_audit_package_certified
    assert not audit.public_proof_certified
    assert audit.public_audit_blockers == (
        "public_review_artifact_manifest",
    )
    assert three_body_api.certify_review_ready_total_collision_audit_package is (
        certify_review_ready_total_collision_audit_package
    )
    assert (
        three_body_api.certify_public_reduced_hyperbolicity_audit_evidence_from_manifest
        is certify_public_reduced_hyperbolicity_audit_evidence_from_manifest
    )
    assert (
        three_body_api.certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest
        is certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest
    )


def test_review_ready_total_collision_package_accepts_neutral_public_review_artifact_ids():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    audit = certify_review_ready_total_collision_audit_package(
        certificate,
        public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        public_review_resolution_certificate=(
            certify_public_review_artifact_resolution(project_root=ROOT)
        ),
    )

    assert audit.local_audit_package_certified
    assert audit.public_review_artifact_manifest_supplied
    assert audit.public_review_artifact_manifest_certified
    assert audit.machine_checked_public_audit_resolved
    assert audit.public_proof_certified
    assert audit.public_audit_blockers == ()


def test_review_ready_total_collision_package_rejects_conflicting_public_review_ids():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    audit = certify_review_ready_total_collision_audit_package(
        certificate,
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        public_review_artifact_ids=("public-review:stale-alias",),
        public_review_resolution_certificate=(
            certify_public_review_artifact_resolution(project_root=ROOT)
        ),
    )

    assert audit.local_audit_package_certified
    assert not audit.public_review_artifact_manifest_supplied
    assert not audit.public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "public_review_artifact_manifest" in audit.public_audit_blockers


def test_public_total_collision_audit_closes_with_verified_public_review_artifacts():
    audit = _complete_total_collision_audit_with_machine_public_artifacts()

    assert audit.local_audit_package_certified
    assert audit.external_public_review_artifact_manifest_supplied
    assert audit.external_public_review_artifact_manifest_certified
    assert audit.public_review_artifact_manifest_supplied
    assert audit.public_review_artifact_manifest_certified
    assert audit.public_review_artifact_kind == "machine_checked_public_audit"
    assert not audit.public_review_resolution_is_external
    assert audit.machine_checked_public_audit_resolved
    assert audit.public_proof_certified
    assert audit.public_audit_blockers == ()


def test_public_total_collision_audit_accepts_neutral_public_review_aliases():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        public_review_resolution_certificate=resolution,
    )

    assert audit.local_audit_package_certified
    assert audit.public_review_artifact_manifest_supplied
    assert audit.public_review_artifact_manifest_certified
    assert audit.machine_checked_public_audit_resolved
    assert not audit.public_review_resolution_is_external
    assert audit.public_proof_certified
    assert audit.public_audit_blockers == ()


def test_public_total_collision_audit_rejects_conflicting_public_review_alias_ids():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        public_review_artifact_ids=("public-review:stale-alias",),
        public_review_resolution_certificate=resolution,
    )

    assert audit.local_audit_package_certified
    assert not audit.public_review_artifact_manifest_supplied
    assert not audit.public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "public_review_artifact_manifest" in audit.public_audit_blockers


def test_public_total_collision_audit_rejects_conflicting_public_review_resolver_aliases():
    legacy_resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    neutral_resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        external_public_review_resolution_certificate=legacy_resolution,
        public_review_resolution_certificate=neutral_resolution,
    )

    assert audit.local_audit_package_certified
    assert audit.public_review_artifact_manifest_supplied
    assert not audit.public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "public_review_artifact_verification" in audit.public_audit_blockers


def test_public_total_collision_audit_rejects_attribute_compatible_fake_review_resolver():
    fake_resolver = SimpleNamespace(
        proof_certified=True,
        artifact_kind="machine_checked_public_audit",
        resolves=lambda *_args: True,
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        external_public_review_resolution_certificate=fake_resolver,
    )

    assert audit.local_audit_package_certified
    assert not audit.external_public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "public_review_artifact_verification" in (
        audit.public_audit_blockers
    )


def test_public_review_resolver_rejects_stale_local_audit_evidence():
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=replace(
            _complete_tc4_evidence(),
            proof_references=("public-review:tc4-stale",),
        ),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        external_public_review_resolution_certificate=(
            certify_public_review_artifact_resolution(project_root=ROOT)
        ),
    )

    assert not audit.local_audit_package_certified
    assert not audit.external_public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "tc4:tc4_public_proof_reference_manifest" in (
        audit.public_audit_blockers
    )


def test_public_review_resolver_rejects_stale_top_level_proof_reference_manifest():
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=(
            *PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
            "docs/total-collision-generalized-fuchsian-stop-proof.md#stale",
        ),
        external_public_review_artifact_ids=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        external_public_review_resolution_certificate=(
            certify_public_review_artifact_resolution(project_root=ROOT)
        ),
    )

    assert not audit.local_audit_package_certified
    assert not audit.external_public_review_artifact_manifest_certified
    assert not audit.public_proof_certified
    assert "public_audit_local_manifest_resolution" in audit.public_audit_blockers


def test_public_review_artifact_resolution_rejects_wrong_artifact_id():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=replace(
            resolution.tc4_artifact,
            artifact_id="external-public-review:wrong-tc4-artifact",
        ),
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not stale.proof_certified
    assert "public_review_artifact_id_manifest" in stale.missing_obligations
    assert "public_review_tc4_artifact_scope" in stale.missing_obligations


def test_public_review_artifact_resolution_rejects_digest_mismatch():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=replace(resolution.tc4_artifact, sha256="0" * 64),
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not stale.proof_certified
    assert "public_review_tc4_artifact_scope" in stale.missing_obligations
    assert "tc4:public_review_artifact_sha256_matches" in (
        stale.missing_obligations
    )


def test_public_review_artifact_resolution_rejects_proof_note_self_reference():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    self_reference = certify_public_review_artifact_evidence(
        artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[0],
        artifact_path=TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,
        audit_scope="TC4 reduced hyperbolicity line audit",
        proof_references=PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
        covered_tc_items=("TC4",),
        checked_local_artifact_ids=PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
        project_root=ROOT,
    )
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=self_reference,
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not self_reference.proof_certified
    assert "public_review_artifact_not_proof_note_self_reference" in (
        self_reference.missing_obligations
    )
    assert not stale.proof_certified
    assert "tc4:public_review_artifact_not_proof_note_self_reference" in (
        stale.missing_obligations
    )


def test_public_review_artifact_resolution_rejects_placeholder_text(tmp_path):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    artifact_path = (
        project_root / "docs/public-review/tc4-reduced-hyperbolicity-line-audit.md"
    )
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8") + "\nTODO: incomplete audit\n",
        encoding="utf-8",
    )
    resolution = certify_public_review_artifact_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_review_tc4_artifact_scope" in resolution.missing_obligations
    assert "tc4:public_review_artifact_no_placeholder_text" in (
        resolution.missing_obligations
    )


def test_public_review_artifact_resolution_rejects_missing_artifact_object():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=resolution.tc4_artifact,
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=SimpleNamespace(
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[2],
            proof_certified=True,
        ),
        project_root=ROOT,
    )

    assert not stale.proof_certified
    assert "public_review_artifact_types" in stale.missing_obligations
    assert "tc6:artifact_type" in stale.missing_obligations


def test_public_review_artifact_resolution_rejects_duplicate_artifact_reuse():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    duplicate = certify_public_review_artifact_resolution(
        tc4_artifact=resolution.tc4_artifact,
        tc5_artifact=resolution.tc4_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not duplicate.proof_certified
    assert "public_review_artifact_id_manifest" in duplicate.missing_obligations
    assert "public_review_tc5_artifact_scope" in duplicate.missing_obligations


def test_public_review_artifact_resolution_rejects_missing_proof_reference():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=replace(resolution.tc4_artifact, proof_references=()),
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not stale.proof_certified
    assert "public_review_tc4_artifact_scope" in stale.missing_obligations


def test_public_review_artifact_resolution_rejects_stale_proof_note_reference():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale_reference = f"{PUBLIC_TC4_REQUIRED_PROOF_REFERENCES[0]}-stale"
    stale = certify_public_review_artifact_resolution(
        tc4_artifact=replace(
            resolution.tc4_artifact,
            proof_references=(stale_reference,),
        ),
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )

    assert not stale.proof_certified
    assert "public_review_tc4_artifact_scope" in stale.missing_obligations
    assert "tc4:public_review_artifact_metadata_present" in (
        stale.missing_obligations
    )


def test_public_review_artifact_resolution_rejects_repo_artifacts_labeled_external_review():
    resolution = certify_public_review_artifact_resolution(
        project_root=ROOT,
        artifact_kind=PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL,
    )
    manifest = build_public_tc4_tc6_audit_manifest(
        project_root=ROOT,
        public_review_resolution_certificate=resolution,
    )

    assert not resolution.proof_certified
    assert not resolution.external_public_review_resolved
    assert not resolution.machine_checked_public_audit_resolved
    assert manifest["public_closure_status"] == "external_review_open"
    assert manifest["public_review_artifact_kind"] is None
    assert manifest["public_review_resolved_by_artifact_resolution"] is False
    assert manifest["public_review_resolution_status"] == "external_review_open"
    assert manifest["public_review_resolution_is_external"] is False
    assert manifest["external_public_review_resolved_by_local_manifest"] is False
    assert "public_review_tc4_artifact_scope" in resolution.missing_obligations
    assert "tc4:public_review_artifact_provenance_truthful" in (
        resolution.missing_obligations
    )


def test_public_review_artifact_resolution_rejects_missing_checked_artifacts():
    resolution = certify_public_review_artifact_resolution(project_root=ROOT)
    stale_tc5 = certify_public_review_artifact_resolution(
        tc4_artifact=resolution.tc4_artifact,
        tc5_artifact=replace(
            resolution.tc5_artifact,
            checked_local_artifact_ids=PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS[:-1],
        ),
        tc6_artifact=resolution.tc6_artifact,
        project_root=ROOT,
    )
    stale_tc6 = certify_public_review_artifact_resolution(
        tc4_artifact=resolution.tc4_artifact,
        tc5_artifact=resolution.tc5_artifact,
        tc6_artifact=replace(
            resolution.tc6_artifact,
            checked_local_artifact_ids=PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS[:-1],
        ),
        project_root=ROOT,
    )

    assert not stale_tc5.proof_certified
    assert "public_review_tc5_artifact_scope" in stale_tc5.missing_obligations
    assert not stale_tc6.proof_certified
    assert "public_review_tc6_artifact_scope" in stale_tc6.missing_obligations


def test_public_regularized_atlas_proof_closes_with_verified_total_collision_public_audit():
    public = certify_public_regularized_atlas_closed_form_proof(
        _internal_pointwise_closed_form_theorem(),
        total_collision_audit=(
            _complete_total_collision_audit_with_machine_public_artifacts()
        ),
    )

    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "internal_pointwise_closed_form_theorem_certified" in public.public_audit_blockers


def test_public_general_closed_form_solution_target_closes_with_verified_public_review_artifacts():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
        total_collision_audit=(
            _complete_total_collision_audit_with_machine_public_artifacts()
        ),
    )

    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "internal_general_closed_form_solution_certified" in public.public_audit_blockers


def test_public_total_collision_audit_requires_local_manifest_resolution():
    stale_resolution = certify_public_audit_manifest_resolution(
        tc6_checker_artifact_ids=(
            "tests/test_certificate_checker.py::test_missing_tc6_checker_artifact",
        ),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        local_manifest_resolution_certificate=stale_resolution,
    )

    assert not stale_resolution.proof_certified
    assert not audit.local_audit_package_certified
    assert not audit.public_proof_certified
    assert "public_audit_local_manifest_resolution" in audit.public_audit_blockers


def test_public_general_closed_form_target_keeps_external_review_open_after_local_public_audit_package():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
        total_collision_audit=_complete_total_collision_audit(),
    )

    assert isinstance(public, PublicGeneralClosedFormSolutionCertificate)
    assert not public.internal_proof_certified
    total_collision_audit = (
        public.public_regularized_atlas_proof.total_collision_audit
    )
    assert total_collision_audit.local_audit_package_certified
    assert not public.public_proof_certified
    assert not public.certified
    assert "public_regularized_atlas_proof_certified" in (
        public.public_audit_blockers
    )
    assert "public_review_artifact_manifest" in (
        public.public_audit_blockers
    )
    assert public.route_summary == (
        "public general closed-form target is missing internal proof closure"
    )


def test_public_general_closed_form_target_rejects_raw_scaffold_gates():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_raw_soundness(),
        computable_atlas_enumeration_certificate=_raw_enumeration(),
        total_collision_audit=_complete_total_collision_audit(),
    )

    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "internal_general_closed_form_solution_certified" in (
        public.public_audit_blockers
    )
    assert "public_certificate_language_soundness_derived" in (
        public.public_audit_blockers
    )
    assert "public_computable_enumeration_derived" in public.public_audit_blockers
    assert "raw_boolean_gates_not_public_evidence" in public.public_audit_blockers


def test_public_regularized_facade_rejects_subclassed_audit_and_forged_ledger():
    class SpoofedTotalCollisionAudit(PublicTotalCollisionProofAuditCertificate):
        @property
        def public_proof_certified(self):
            return True

        @property
        def public_audit_blockers(self):
            return ()

    base = certify_public_regularized_atlas_closed_form_proof(
        _internal_pointwise_closed_form_theorem(),
    )
    forged = replace(
        base,
        total_collision_audit=SpoofedTotalCollisionAudit(),
        obligations=(
            TheoremPipelineObligation(
                obligation="fake_public_regularized_obligation",
                certified=True,
                source="spoof",
                detail="forged all-true ledger",
            ),
        ),
    )

    assert not forged.public_proof_certified
    assert "public_regularized_total_collision_audit_field" in (
        forged.public_audit_blockers
    )
    assert "public_regularized_obligation_manifest" in (
        forged.public_audit_blockers
    )


def test_public_general_facade_rejects_subclassed_route_and_forged_ledger():
    class SpoofedRegularizedRoute(PublicRegularizedAtlasClosedFormProofCertificate):
        @property
        def public_proof_certified(self):
            return True

        @property
        def public_audit_blockers(self):
            return ()

    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
    )
    route = public.public_regularized_atlas_proof
    spoofed_route = SpoofedRegularizedRoute(
        internal_closed_form_theorem=route.internal_closed_form_theorem,
        total_collision_audit=route.total_collision_audit,
        obligations=route.obligations,
    )
    forged = replace(
        public,
        public_regularized_atlas_proof=spoofed_route,
        obligations=(
            TheoremPipelineObligation(
                obligation="fake_public_general_obligation",
                certified=True,
                source="spoof",
                detail="forged all-true ledger",
            ),
        ),
    )

    assert not forged.public_proof_certified
    assert "public_regularized_atlas_proof_type" in forged.public_audit_blockers
    assert "public_regularized_atlas_proof_source_matches_pointwise" in (
        forged.public_audit_blockers
    )
    assert "public_general_obligation_manifest" in forged.public_audit_blockers


def test_public_general_closed_form_target_keeps_total_collision_audit_separate():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
    )

    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "public_regularized_atlas_proof_certified" in (
        public.public_audit_blockers
    )
    assert "tc4_reduced_hyperbolicity_audited" in public.public_audit_blockers
    assert "tc5_generalized_fuchsian_entry_audited" in public.public_audit_blockers
    assert "tc6_cauchy_majorant_constants_audited" in public.public_audit_blockers


def test_tc4_tc6_booleans_without_standalone_evidence_do_not_public_certify():
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
    )

    assert not audit.tc4_evidence_certified
    assert not audit.tc5_evidence_certified
    assert not audit.tc6_evidence_certified
    assert not audit.public_proof_certified
    assert audit.public_audit_blockers == (
        "tc4_reduced_hyperbolicity_audited",
        "tc5_generalized_fuchsian_entry_audited",
        "tc6_cauchy_majorant_constants_audited",
        "public_audit_proof_references_supplied",
        "public_audit_proof_reference_manifest",
        "public_audit_local_manifest_resolution",
        "tc4:evidence_supplied",
        "tc5:evidence_supplied",
        "tc6:evidence_supplied",
    )


def test_public_audit_rejects_bad_tc6_cauchy_majorant_constants():
    bad_tc6 = certify_public_cauchy_majorant_audit_evidence(
        cauchy_polydisc_declared=True,
        primitive_constants_declared=True,
        defect_bound=0.03,
        right_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.6,
        majorant_radius=0.2,
        proof_grade_arithmetic_backend_audited=True,
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=bad_tc6,
    )

    assert isinstance(bad_tc6, PublicCauchyMajorantAuditEvidence)
    assert bad_tc6.constants_finite
    assert bad_tc6.contraction_factor >= 1.0
    assert "tc6_banach_contraction_inequality" in bad_tc6.missing_obligations
    assert not bad_tc6.proof_certified
    assert not audit.public_proof_certified
    assert "tc6_cauchy_majorant_constants_audited" in audit.public_audit_blockers
    assert "tc6:tc6_banach_contraction_inequality" in audit.public_audit_blockers


def test_public_audit_evidence_constructors_are_typed_and_checkable():
    tc4 = _complete_tc4_evidence()
    tc5 = _complete_tc5_evidence()
    tc6 = _complete_tc6_evidence()

    assert isinstance(tc4, PublicReducedHyperbolicityAuditEvidence)
    assert isinstance(tc5, PublicGeneralizedFuchsianEntryAuditEvidence)
    assert isinstance(tc6, PublicCauchyMajorantAuditEvidence)
    assert tc4.proof_certified
    assert tc5.proof_certified
    assert tc6.proof_certified
    assert tc4.central_target_manifest_certified
    assert tc4.quotient_mode_manifest_certified
    assert tc4.spectrum_audit_reference_manifest_certified
    assert tc5.normal_form_coordinate_manifest_certified
    assert tc5.resonance_rule_manifest_certified
    assert tc5.triangular_order_manifest_certified
    assert tc5.selector_data_manifest_certified
    assert tc5.constructor_artifact_manifest_certified
    assert tc6.primitive_constant_manifest_certified
    assert tc6.cauchy_polydisc_manifest_certified
    assert tc6.majorant_norm_manifest_certified
    assert tc6.proof_grade_backend_manifest_certified
    assert tc6.checker_artifact_manifest_certified
    assert tc6.checker_id_manifest_certified
    assert tc6.checker_certificate_manifest_certified
    assert tc6.checker_obligation_manifest_certified
    assert tc6.majorant_component_manifest_certified
    assert tc6.contraction_factor < 1.0
    assert tc6.self_map_inequality_certified


def test_public_audit_evidence_rejects_forged_theorem_ids():
    tc4 = replace(_complete_tc4_evidence(), theorem_id="spoofed_tc4")
    tc5 = replace(_complete_tc5_evidence(), theorem_id="spoofed_tc5")
    tc6 = replace(_complete_tc6_evidence(), theorem_id="spoofed_tc6")

    assert not tc4.proof_certified
    assert not tc5.proof_certified
    assert not tc6.proof_certified
    assert tc4.missing_obligations == (
        "public_tc4_reduced_hyperbolicity_audit_theorem_id",
    )
    assert tc5.missing_obligations == (
        "public_tc5_generalized_fuchsian_entry_audit_theorem_id",
    )
    assert tc6.missing_obligations == (
        "public_tc6_cauchy_majorant_audit_theorem_id",
    )


def test_public_tc6_evidence_is_derived_from_checked_stop_chart():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
        certificate,
    )
    majorant = certificate.remainder_majorant

    assert majorant is not None
    assert tc6.proof_certified
    assert tc6.checker_id == PUBLIC_TC6_REQUIRED_CHECKER_ID
    assert tc6.checker_certificate_id == certificate.certificate_id
    assert tc6.defect_bound == majorant.defect_bound
    assert tc6.right_inverse_bound == majorant.linear_inverse_bound
    assert tc6.nonlinear_lipschitz_bound == majorant.nonlinear_lipschitz_bound
    assert tc6.majorant_radius == majorant.remainder_ball_radius
    assert set(PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS).issubset(
        tc6.checker_obligation_ids,
    )
    assert tc6.majorant_component_ids == PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS


def test_public_tc6_checked_evidence_rejects_corrupted_majorant_constants():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    assert certificate.remainder_majorant is not None
    corrupted = replace(
        certificate,
        remainder_majorant=replace(
            certificate.remainder_majorant,
            nonlinear_lipschitz_bound=0.8,
        ),
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
        corrupted,
    )

    assert not tc6.proof_certified
    assert "tc6_banach_contraction_inequality" in tc6.missing_obligations
    assert "tc6_checker_obligation_manifest" in tc6.missing_obligations


def test_public_tc6_checked_evidence_requires_projected_residual_component():
    certificate = replace(
        _fast_total_collision_generalized_fuchsian_stop_chart_certificate(),
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )
    assert certificate.remainder_majorant is not None
    missing_physical = replace(
        certificate,
        remainder_majorant=replace(
            certificate.remainder_majorant,
            component_effective_exponents=tuple(
                item
                for item in certificate.remainder_majorant.component_effective_exponents
                if item[0] != "physical_residual"
            ),
            component_inputs=tuple(
                item
                for item in certificate.remainder_majorant.component_inputs
                if item[0] != "physical_residual"
            ),
        ),
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
        missing_physical,
    )

    assert not tc6.proof_certified
    assert not tc6.majorant_component_manifest_certified
    assert "tc6_majorant_component_manifest" in tc6.missing_obligations
    assert "tc6_checker_obligation_manifest" in tc6.missing_obligations


def test_public_audit_requires_named_tc4_tc6_structured_evidence():
    tc4 = certify_public_reduced_hyperbolicity_audit_evidence(
        positive_mass_domain_declared=True,
        quotient_coordinates_declared=True,
        translations_removed=True,
        scale_rotation_reflection_quotiented=True,
        central_target_family_covered=True,
        mass_metric_hessian_spectrum_audited=True,
        zero_modes_removed=True,
        stable_unstable_splitting_certified=True,
        proof_references=("public-review:tc4",),
    )
    tc5 = certify_public_generalized_fuchsian_entry_audit_evidence(
        normal_form_coordinates_declared=True,
        exponent_conventions_declared=True,
        resonance_lattice_declared=True,
        finite_log_degree_rule_declared=True,
        denominator_projector_rule_declared=True,
        triangular_solve_order_declared=True,
        finite_selector_data_extraction_audited=True,
        arbitrary_incoming_germ_scope_declared=True,
        proof_references=("public-review:tc5",),
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence(
        cauchy_polydisc_declared=True,
        primitive_constants_declared=True,
        defect_bound=0.03,
        right_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.2,
        majorant_radius=0.2,
        proof_grade_arithmetic_backend_audited=True,
        proof_references=("public-review:tc6",),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=tc4,
        tc5_generalized_fuchsian_entry_evidence=tc5,
        tc6_cauchy_majorant_constants_evidence=tc6,
    )

    assert not tc4.proof_certified
    assert not tc5.proof_certified
    assert not tc6.proof_certified
    assert "tc4_central_target_family_manifest" in tc4.missing_obligations
    assert "tc4_quotient_mode_manifest" in tc4.missing_obligations
    assert "tc4_spectrum_audit_reference_ids_supplied" in (
        tc4.missing_obligations
    )
    assert "tc4_spectrum_audit_reference_manifest" in tc4.missing_obligations
    assert "tc4_public_proof_reference_manifest" in tc4.missing_obligations
    assert "tc5_normal_form_coordinate_manifest" in tc5.missing_obligations
    assert "tc5_resonance_rule_manifest" in tc5.missing_obligations
    assert "tc5_triangular_order_manifest" in tc5.missing_obligations
    assert "tc5_selector_data_manifest" in tc5.missing_obligations
    assert "tc5_constructor_artifact_manifest" in tc5.missing_obligations
    assert "tc5_public_proof_reference_manifest" in tc5.missing_obligations
    assert "tc6_cauchy_polydisc_id_supplied" in tc6.missing_obligations
    assert "tc6_primitive_constant_manifest" in tc6.missing_obligations
    assert "tc6_majorant_norm_id_supplied" in tc6.missing_obligations
    assert "tc6_proof_grade_backend_id_supplied" in tc6.missing_obligations
    assert "tc6_checker_artifact_manifest" in tc6.missing_obligations
    assert "tc6_checker_id_manifest" in tc6.missing_obligations
    assert "tc6_checker_certificate_manifest" in tc6.missing_obligations
    assert "tc6_checker_obligation_manifest" in tc6.missing_obligations
    assert "tc6_majorant_component_manifest" in tc6.missing_obligations
    assert "tc6_public_proof_reference_manifest" in tc6.missing_obligations
    assert not audit.public_proof_certified
    assert "tc4:tc4_central_target_family_manifest" in audit.public_audit_blockers
    assert "tc5:tc5_resonance_rule_manifest" in audit.public_audit_blockers
    assert "tc6:tc6_primitive_constant_manifest" in audit.public_audit_blockers
    assert "tc6:tc6_checker_artifact_manifest" in audit.public_audit_blockers
    assert "tc6:tc6_checker_obligation_manifest" in audit.public_audit_blockers


def test_public_total_collision_audit_rejects_truthy_fake_nested_evidence():
    fake_tc4 = SimpleNamespace(
        proof_certified="yes",
        missing_obligations=("spoofed_tc4_missing_detail",),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=fake_tc4,
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not audit.local_audit_package_certified
    assert not audit.public_proof_certified
    assert "tc4_reduced_hyperbolicity_audited" in audit.public_audit_blockers
    assert "tc4:evidence_type" in audit.public_audit_blockers
    assert "tc4:spoofed_tc4_missing_detail" not in audit.public_audit_blockers


def test_public_total_collision_audit_rejects_subclassed_nested_evidence():
    class SpoofedTC4(PublicReducedHyperbolicityAuditEvidence):
        @property
        def proof_certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    class SpoofedTC5(PublicGeneralizedFuchsianEntryAuditEvidence):
        @property
        def proof_certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    class SpoofedTC6(PublicCauchyMajorantAuditEvidence):
        @property
        def proof_certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=SpoofedTC4(),
        tc5_generalized_fuchsian_entry_evidence=SpoofedTC5(),
        tc6_cauchy_majorant_constants_evidence=SpoofedTC6(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not audit.local_audit_package_certified
    assert not audit.public_proof_certified
    assert "tc4:evidence_type" in audit.public_audit_blockers
    assert "tc5:evidence_type" in audit.public_audit_blockers
    assert "tc6:evidence_type" in audit.public_audit_blockers


def test_public_audit_evidence_requires_public_proof_references():
    tc4 = certify_public_reduced_hyperbolicity_audit_evidence(
        positive_mass_domain_declared=True,
        quotient_coordinates_declared=True,
        translations_removed=True,
        scale_rotation_reflection_quotiented=True,
        central_target_family_covered=True,
        mass_metric_hessian_spectrum_audited=True,
        zero_modes_removed=True,
        stable_unstable_splitting_certified=True,
    )
    tc5 = certify_public_generalized_fuchsian_entry_audit_evidence(
        normal_form_coordinates_declared=True,
        exponent_conventions_declared=True,
        resonance_lattice_declared=True,
        finite_log_degree_rule_declared=True,
        denominator_projector_rule_declared=True,
        triangular_solve_order_declared=True,
        finite_selector_data_extraction_audited=True,
        arbitrary_incoming_germ_scope_declared=True,
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence(
        cauchy_polydisc_declared=True,
        primitive_constants_declared=True,
        defect_bound=0.03,
        right_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.2,
        majorant_radius=0.2,
        proof_grade_arithmetic_backend_audited=True,
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=tc4,
        tc5_generalized_fuchsian_entry_evidence=tc5,
        tc6_cauchy_majorant_constants_evidence=tc6,
    )

    assert not tc4.proof_certified
    assert not tc5.proof_certified
    assert not tc6.proof_certified
    assert "tc4_public_proof_references_supplied" in tc4.missing_obligations
    assert "tc5_public_proof_references_supplied" in tc5.missing_obligations
    assert "tc6_public_proof_references_supplied" in tc6.missing_obligations
    assert not audit.public_proof_certified
    assert "tc4:tc4_public_proof_references_supplied" in (
        audit.public_audit_blockers
    )
    assert "tc5:tc5_public_proof_references_supplied" in (
        audit.public_audit_blockers
    )
    assert "tc6:tc6_public_proof_references_supplied" in (
        audit.public_audit_blockers
    )


def test_public_tc4_audit_requires_machine_checked_spectrum_artifact_manifest():
    tc4 = replace(
        _complete_tc4_evidence(),
        spectrum_audit_reference_ids=("tc4-spectrum:euler-lagrange",),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=tc4,
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not tc4.proof_certified
    assert not tc4.spectrum_audit_reference_manifest_certified
    assert "tc4_spectrum_audit_reference_ids_supplied" not in (
        tc4.missing_obligations
    )
    assert "tc4_spectrum_audit_reference_manifest" in tc4.missing_obligations
    assert not audit.public_proof_certified
    assert "tc4:tc4_spectrum_audit_reference_manifest" in (
        audit.public_audit_blockers
    )


def test_public_tc4_tc5_required_artifacts_exist_in_obstruction_suite():
    _assert_pytest_artifact_ids_exist(
        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
        expected_file="tests/test_obstructions.py",
    )
    _assert_pytest_artifact_ids_exist(
        PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
        expected_file="tests/test_obstructions.py",
    )


def test_public_tc4_tc6_required_proof_references_resolve_to_proof_note_sections():
    _assert_proof_references_resolve(PUBLIC_TC4_REQUIRED_PROOF_REFERENCES)
    _assert_proof_references_resolve(PUBLIC_TC5_REQUIRED_PROOF_REFERENCES)
    _assert_proof_references_resolve(PUBLIC_TC6_REQUIRED_PROOF_REFERENCES)
    _assert_proof_references_resolve(
        PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES
    )


def test_public_tc4_tc6_proof_note_contains_line_item_audit_map():
    proof_note = (
        ROOT / "docs" / "total-collision-generalized-fuchsian-stop-proof.md"
    ).read_text()
    required_markers = (
        "## Public TC4-TC6 Audit Line-Item Map",
        "TC4-Audit-01, quotient and target coverage",
        "TC4-Audit-02, indicial equation",
        "TC5-Audit-01, generalized entry language",
        "TC5-Audit-02, resonance/projector solve",
        "TC6-Audit-01, Banach majorant",
        "TC6-Audit-02, projected residual transfer",
        "external-public-review:tc4-reduced-hyperbolicity-line-audit",
        "external-public-review:tc5-generalized-fuchsian-entry-line-audit",
        "external-public-review:tc6-cauchy-majorant-backend-line-audit",
        "q = B L_N < 1",
        "B D + q R_0 <= R_0",
        "projected_residual",
        "omega-4",
    )
    required_artifact_ids = (
        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
        + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
        + PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
        + PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
        + PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )

    for marker in required_markers:
        assert marker in proof_note
    for artifact_id in required_artifact_ids:
        assert artifact_id in proof_note


def test_public_audit_manifest_resolver_certifies_local_artifact_manifest():
    resolution = certify_public_audit_manifest_resolution()

    assert isinstance(resolution, PublicAuditManifestResolutionCertificate)
    assert resolution.proof_certified
    assert resolution.missing_obligations == ()
    assert any(
        artifact_id.endswith(
            "test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture"
        )
        for artifact_id in resolution.fast_ci_artifact_ids
    )
    assert len(resolution.slow_ci_artifact_ids) == 3
    assert three_body_api.PublicAuditManifestResolutionCertificate is (
        PublicAuditManifestResolutionCertificate
    )
    assert three_body_api.certify_public_audit_manifest_resolution is (
        certify_public_audit_manifest_resolution
    )
    assert three_body_api.build_public_tc4_tc6_audit_manifest is (
        build_public_tc4_tc6_audit_manifest
    )
    assert (
        three_body_api.certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart
        is certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart
    )


def test_public_audit_manifest_resolver_requires_real_pytest_function_defs(tmp_path):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    artifact_id = PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS[0]
    file_name, test_name = artifact_id.split("::", maxsplit=1)
    artifact_path = project_root / file_name
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8").replace(
            f"def {test_name}(",
            f"# def {test_name}(",
            1,
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "tc4_spectrum_pytest_artifacts_resolve" in resolution.missing_obligations
    assert artifact_id in _audit_obligation_detail(
        resolution,
        "tc4_spectrum_pytest_artifacts_resolve",
    )


def test_public_audit_manifest_resolver_requires_real_markdown_heading_anchors(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    proof_note = project_root / TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT
    proof_note.write_text(
        proof_note.read_text(encoding="utf-8").replace(
            "## TC4.",
            "Paragraph spoof retaining anchor text: ## TC4.",
            1,
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_proof_references_resolve" in (
        resolution.missing_obligations
    )
    assert PUBLIC_TC4_REQUIRED_PROOF_REFERENCES[0] in _audit_obligation_detail(
        resolution,
        "public_audit_proof_references_resolve",
    )


def test_public_tc4_tc6_audit_manifest_export_is_constant_backed_and_resolved():
    resolution = certify_public_audit_manifest_resolution(project_root=ROOT)
    manifest = build_public_tc4_tc6_audit_manifest(project_root=ROOT)

    assert resolution.proof_certified
    assert manifest["manifest_id"] == "public_tc4_tc6_audit_manifest"
    assert manifest["proof_document"] == TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT
    assert manifest["proof_references"] == list(
        PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES
    )
    assert manifest["public_closure_status"] == "external_review_open"
    assert manifest["external_public_review_resolved_by_local_manifest"] is False
    assert manifest["tc4"]["central_target_families"] == list(
        PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES
    )
    assert manifest["tc4"]["quotient_modes_removed"] == list(
        PUBLIC_TC4_REQUIRED_QUOTIENT_MODES
    )
    assert manifest["tc4"]["spectrum_audit_reference_ids"] == list(
        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
    )
    assert manifest["tc5"]["normal_form_coordinate_ids"] == list(
        PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES
    )
    assert manifest["tc5"]["resonance_rule_ids"] == list(
        PUBLIC_TC5_REQUIRED_RESONANCE_RULES
    )
    assert manifest["tc5"]["triangular_order_keys"] == list(
        PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS
    )
    assert manifest["tc5"]["selector_data_fields"] == list(
        PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS
    )
    assert manifest["tc5"]["constructor_artifact_ids"] == list(
        PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
    )
    assert manifest["tc6"]["cauchy_polydisc_id"] == (
        PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID
    )
    assert manifest["tc6"]["primitive_constant_names"] == list(
        PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES
    )
    assert manifest["tc6"]["majorant_norm_id"] == (
        PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID
    )
    assert manifest["tc6"]["proof_grade_backend_id"] == (
        PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID
    )
    assert manifest["tc6"]["checker_artifact_ids"] == list(
        PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
    )
    assert manifest["tc6"]["checker_id"] == PUBLIC_TC6_REQUIRED_CHECKER_ID
    assert manifest["tc6"]["checker_obligation_ids"] == list(
        PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS
    )
    assert manifest["tc6"]["majorant_component_ids"] == list(
        PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS
    )
    assert manifest["tc6"]["fast_checker_artifact_ids"] == list(
        PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS
    )
    assert manifest["tc6"]["slow_checker_artifact_ids"] == list(
        PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS
    )
    assert manifest["total_collision_checker_artifacts"] == list(
        PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
    )
    assert manifest["external_public_review_artifact_ids"] == list(
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )
    assert manifest["public_review_artifact_ids"] == list(
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    )
    assert manifest["public_review_artifact_kind"] is None
    assert manifest["public_review_resolved_by_artifact_resolution"] is False
    assert manifest["public_review_resolution_status"] == "external_review_open"
    assert manifest["public_review_resolution_is_external"] is False
    assert manifest["ci_artifacts"]["fast"] == list(resolution.fast_ci_artifact_ids)
    assert manifest["ci_artifacts"]["slow"] == list(resolution.slow_ci_artifact_ids)
    assert manifest["local_manifest_resolution"] == {
        "proof_certified": True,
        "missing_obligations": [],
        "theorem_id": "public_audit_manifest_resolution",
    }


def test_public_tc4_tc6_audit_manifest_is_deterministic_json_payload():
    manifest = build_public_tc4_tc6_audit_manifest(project_root=ROOT)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    regenerated = json.dumps(
        build_public_tc4_tc6_audit_manifest(project_root=ROOT),
        indent=2,
        sort_keys=True,
    )

    assert json.loads(payload) == manifest
    assert payload == regenerated
    assert '"public_closure_status": "external_review_open"' in payload
    assert "external TC4-TC6 line-audit artifacts are still required" in payload


def test_export_public_audit_manifest_script_prints_manifest_json():
    completed = subprocess.run(
        [sys.executable, "scripts/export_public_audit_manifest.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == build_public_tc4_tc6_audit_manifest(
        project_root=ROOT
    )
    assert completed.stderr == ""


def test_public_audit_manifest_resolution_rejects_spoofed_obligation_ledgers():
    resolution = certify_public_audit_manifest_resolution()
    spoofed = replace(
        resolution,
        obligations=(
            SimpleNamespace(
                obligation="fake_public_audit_manifest_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    empty = replace(resolution, obligations=())
    optional_only = replace(
        resolution,
        obligations=tuple(
            replace(obligation, required=False)
            for obligation in resolution.obligations
        ),
    )

    assert resolution.proof_certified
    assert not spoofed.proof_certified
    assert "public_audit_manifest_resolution_obligation_type" in (
        spoofed.missing_obligations
    )
    assert not empty.proof_certified
    assert "public_audit_manifest_resolution_obligations_present" in (
        empty.missing_obligations
    )
    assert not optional_only.proof_certified
    assert "public_audit_manifest_resolution_required_obligation_present" in (
        optional_only.missing_obligations
    )


def test_public_audit_manifest_resolution_rejects_forged_theorem_id():
    resolution = certify_public_audit_manifest_resolution()
    forged = replace(resolution, theorem_id="spoofed_manifest_resolution")

    assert resolution.proof_certified
    assert not forged.proof_certified
    assert forged.missing_obligations == (
        "public_audit_manifest_resolution_theorem_id",
    )


def test_public_audit_manifest_resolution_rejects_stale_replaced_manifest_fields():
    resolution = certify_public_audit_manifest_resolution()
    stale_proof_reference = replace(
        resolution,
        proof_references=("docs/total-collision-generalized-fuchsian-stop-proof.md#TC4",),
    )
    stale_tc4_artifact = replace(
        resolution,
        tc4_spectrum_audit_reference_ids=(
            PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS[0],
        ),
    )
    stale_tc5_artifact = replace(
        resolution,
        tc5_constructor_artifact_ids=(
            PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS[0],
        ),
    )
    stale_tc6_artifact = replace(
        resolution,
        tc6_checker_artifact_ids=(
            PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS[0],
        ),
    )
    stale_total_checker = replace(
        resolution,
        total_collision_checker_artifacts=(
            PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS[0],
        ),
    )
    stale_fast_ci = replace(
        resolution,
        fast_ci_artifact_ids=PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
    )
    stale_slow_ci = replace(
        resolution,
        slow_ci_artifact_ids=(),
    )

    assert resolution.proof_certified
    assert not stale_proof_reference.proof_certified
    assert "public_audit_manifest_proof_reference_field" in (
        stale_proof_reference.missing_obligations
    )
    assert not stale_tc4_artifact.proof_certified
    assert "public_audit_manifest_tc4_spectrum_field" in (
        stale_tc4_artifact.missing_obligations
    )
    assert not stale_tc5_artifact.proof_certified
    assert "public_audit_manifest_tc5_constructor_field" in (
        stale_tc5_artifact.missing_obligations
    )
    assert not stale_tc6_artifact.proof_certified
    assert "public_audit_manifest_tc6_checker_field" in (
        stale_tc6_artifact.missing_obligations
    )
    assert not stale_total_checker.proof_certified
    assert "public_audit_manifest_total_checker_field" in (
        stale_total_checker.missing_obligations
    )
    assert not stale_fast_ci.proof_certified
    assert "public_audit_manifest_fast_ci_artifact_field" in (
        stale_fast_ci.missing_obligations
    )
    assert not stale_slow_ci.proof_certified
    assert "public_audit_manifest_slow_ci_artifact_field" in (
        stale_slow_ci.missing_obligations
    )


def test_public_audit_manifest_resolution_rejects_duplicate_manifest_fields():
    duplicate_proof_reference = certify_public_audit_manifest_resolution(
        proof_references=(
            *PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
            PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES[0],
        ),
    )
    duplicate_checker_artifact = certify_public_audit_manifest_resolution(
        total_collision_checker_artifacts=(
            *PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
            PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS[0],
        ),
    )

    assert not duplicate_proof_reference.proof_certified
    assert "public_audit_manifest_proof_reference_field" in (
        duplicate_proof_reference.missing_obligations
    )
    assert not duplicate_checker_artifact.proof_certified
    assert "public_audit_manifest_total_checker_field" in (
        duplicate_checker_artifact.missing_obligations
    )


def test_public_total_collision_audit_rejects_stale_local_manifest_replay():
    audit = _complete_total_collision_audit()
    stale_proof_references = replace(
        audit,
        proof_references=(
            *audit.proof_references,
            audit.proof_references[0],
        ),
    )
    stale_tc4_artifacts = replace(
        audit,
        tc4_reduced_hyperbolicity_evidence=replace(
            audit.tc4_reduced_hyperbolicity_evidence,
            spectrum_audit_reference_ids=(
                *(
                    audit.tc4_reduced_hyperbolicity_evidence
                    .spectrum_audit_reference_ids
                ),
                (
                    audit.tc4_reduced_hyperbolicity_evidence
                    .spectrum_audit_reference_ids[0]
                ),
            ),
        ),
    )
    stale_tc5_artifacts = replace(
        audit,
        tc5_generalized_fuchsian_entry_evidence=replace(
            audit.tc5_generalized_fuchsian_entry_evidence,
            constructor_artifact_ids=(
                *(
                    audit.tc5_generalized_fuchsian_entry_evidence
                    .constructor_artifact_ids
                ),
                (
                    audit.tc5_generalized_fuchsian_entry_evidence
                    .constructor_artifact_ids[0]
                ),
            ),
        ),
    )
    stale_tc6_artifacts = replace(
        audit,
        tc6_cauchy_majorant_constants_evidence=replace(
            audit.tc6_cauchy_majorant_constants_evidence,
            checker_artifact_ids=(
                *(
                    audit.tc6_cauchy_majorant_constants_evidence
                    .checker_artifact_ids
                ),
                (
                    audit.tc6_cauchy_majorant_constants_evidence
                    .checker_artifact_ids[0]
                ),
            ),
        ),
    )
    stale_checker_artifacts = replace(
        audit,
        checker_artifacts=(
            *audit.checker_artifacts,
            audit.checker_artifacts[0],
        ),
    )

    assert audit.local_audit_package_certified
    for stale in (
        stale_proof_references,
        stale_tc4_artifacts,
        stale_tc5_artifacts,
        stale_tc6_artifacts,
        stale_checker_artifacts,
    ):
        assert not stale.local_audit_package_certified
        assert "public_audit_local_manifest_matches_current_fields" in (
            stale.public_audit_blockers
        )


def test_public_total_collision_audit_rejects_forged_theorem_id_locally():
    audit = _complete_total_collision_audit()
    forged = replace(audit, theorem_id="spoofed_total_collision_audit")

    assert audit.local_audit_package_certified
    assert not forged.local_audit_package_certified
    assert not forged.public_proof_certified
    assert "public_total_collision_proof_audit_theorem_id" in (
        forged.public_audit_blockers
    )


def test_public_total_collision_audit_rejects_subclassed_manifest_resolution():
    class SpoofedManifestResolution(PublicAuditManifestResolutionCertificate):
        @property
        def proof_certified(self):
            return True

        @property
        def missing_obligations(self):
            return ()

    audit = _complete_total_collision_audit()
    spoofed_manifest = SpoofedManifestResolution(
        proof_references=audit.proof_references,
        tc4_spectrum_audit_reference_ids=audit.tc4_manifest_artifact_ids,
        tc5_constructor_artifact_ids=audit.tc5_manifest_artifact_ids,
        tc6_checker_artifact_ids=audit.tc6_manifest_artifact_ids,
        total_collision_checker_artifacts=audit.checker_artifacts,
        fast_ci_artifact_ids=(
            audit.local_manifest_resolution_certificate.fast_ci_artifact_ids
        ),
        slow_ci_artifact_ids=(
            audit.local_manifest_resolution_certificate.slow_ci_artifact_ids
        ),
        obligations=(),
    )
    forged = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=(
            audit.tc4_reduced_hyperbolicity_evidence
        ),
        tc5_generalized_fuchsian_entry_evidence=(
            audit.tc5_generalized_fuchsian_entry_evidence
        ),
        tc6_cauchy_majorant_constants_evidence=(
            audit.tc6_cauchy_majorant_constants_evidence
        ),
        proof_references=audit.proof_references,
        local_manifest_resolution_certificate=spoofed_manifest,
    )

    assert audit.local_audit_package_certified
    assert not forged.local_audit_package_certified
    assert "public_audit_local_manifest_resolution" in forged.public_audit_blockers
    assert "public_audit_local_manifest_matches_current_fields" in (
        forged.public_audit_blockers
    )


def test_public_audit_manifest_resolver_rejects_stale_local_artifacts():
    stale_proof = certify_public_audit_manifest_resolution(
        proof_references=(
            "docs/total-collision-generalized-fuchsian-stop-proof.md#TC999",
        ),
    )
    stale_test = certify_public_audit_manifest_resolution(
        tc4_spectrum_audit_reference_ids=(
            "tests/test_obstructions.py::test_missing_tc4_artifact",
        ),
    )
    stale_checker = certify_public_audit_manifest_resolution(
        total_collision_checker_artifacts=("MissingCheckerArtifact",),
    )
    malformed_fast_split = certify_public_audit_manifest_resolution(
        tc6_checker_artifact_ids=PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS,
    )
    malformed_slow_split = certify_public_audit_manifest_resolution(
        tc6_checker_artifact_ids=PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS,
    )

    assert not stale_proof.proof_certified
    assert "public_audit_proof_references_resolve" in (
        stale_proof.missing_obligations
    )
    assert not stale_test.proof_certified
    assert "tc4_spectrum_pytest_artifacts_resolve" in (
        stale_test.missing_obligations
    )
    assert not stale_checker.proof_certified
    assert "total_collision_checker_module_artifacts_resolve" in (
        stale_checker.missing_obligations
    )
    assert not malformed_fast_split.proof_certified
    assert "public_audit_manifest_tc6_checker_field" in (
        malformed_fast_split.missing_obligations
    )
    assert "public_audit_manifest_fast_ci_artifact_field" in (
        malformed_fast_split.missing_obligations
    )
    assert not malformed_slow_split.proof_certified
    assert "public_audit_manifest_tc6_checker_field" in (
        malformed_slow_split.missing_obligations
    )
    assert "public_audit_manifest_slow_ci_artifact_field" in (
        malformed_slow_split.missing_obligations
    )


def test_public_audit_manifest_resolver_rejects_name_spoofed_checker_artifacts(
    monkeypatch,
):
    class SpoofedStopChartCertificate:
        pass

    def spoofed_stop_chart_checker(certificate):
        return certificate

    SpoofedStopChartCertificate.__name__ = (
        "TotalCollisionGeneralizedFuchsianStopChartCertificate"
    )
    spoofed_stop_chart_checker.__name__ = (
        "check_total_collision_generalized_fuchsian_stop_chart"
    )
    monkeypatch.setattr(
        certificate_checker_module,
        "TotalCollisionGeneralizedFuchsianStopChartCertificate",
        SpoofedStopChartCertificate,
    )
    monkeypatch.setattr(
        certificate_checker_module,
        "check_total_collision_generalized_fuchsian_stop_chart",
        spoofed_stop_chart_checker,
    )

    resolution = certify_public_audit_manifest_resolution()

    assert not resolution.proof_certified
    assert "total_collision_checker_module_artifacts_resolve" in (
        resolution.missing_obligations
    )
    detail = _audit_obligation_detail(
        resolution,
        "total_collision_checker_module_artifacts_resolve",
    )
    assert "TotalCollisionGeneralizedFuchsianStopChartCertificate" in detail
    assert "check_total_collision_generalized_fuchsian_stop_chart" in detail


def _copy_public_audit_resolution_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "public_audit_resolution"
    for file_name in (
        TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,
        "tests/test_obstructions.py",
        "tests/test_certificate_checker.py",
        "scripts/fast_ci.py",
        "scripts/slow_certificate_checker.py",
        "docs/public-review/tc4-reduced-hyperbolicity-line-audit.md",
        "docs/public-review/tc5-generalized-fuchsian-entry-line-audit.md",
        "docs/public-review/tc6-cauchy-majorant-backend-line-audit.md",
    ):
        source = ROOT / file_name
        target = project_root / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return project_root


def _audit_obligation_detail(
    resolution: PublicAuditManifestResolutionCertificate,
    obligation_id: str,
) -> str:
    for obligation in resolution.obligations:
        if obligation.obligation == obligation_id:
            return obligation.detail
    raise AssertionError(f"missing obligation {obligation_id}")


def test_public_audit_manifest_resolver_rejects_stale_fast_ci_command_selection(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    fast_ci_path.write_text(
        fast_ci_source.replace(
            'f"{TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K} or '
            '{EVENT_REGIME_ASSEMBLY_HARDENING_K}"',
            '"event_regime_assembly_rejects_spoofed_obligation_ledgers"',
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    assert "fast_ci:obstruction_command_selection" in _audit_obligation_detail(
        resolution,
        "public_audit_fast_ci_artifact_coverage",
    )


def test_public_audit_manifest_resolver_rejects_substring_fast_ci_selection(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    spoofed_selection = " or ".join(
        f"{artifact_id.split('::', maxsplit=1)[1]}_not_the_required_test"
        for artifact_id in (
            PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
            + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
        )
    )
    fast_ci_path.write_text(
        fast_ci_source.replace(
            'f"{TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K} or '
            '{EVENT_REGIME_ASSEMBLY_HARDENING_K}"',
            repr(spoofed_selection),
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    assert "fast_ci:obstruction_command_selection" in _audit_obligation_detail(
        resolution,
        "public_audit_fast_ci_artifact_coverage",
    )


def test_public_audit_manifest_resolver_rejects_comment_only_fast_manifest_imports(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    literal_selection = " or ".join(
        artifact_id.split("::", maxsplit=1)[1]
        for artifact_id in (
            PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
            + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
        )
    )
    fast_ci_source = fast_ci_source.replace(
        "from three_body_symmetry.public_proof_audit import (  # noqa: E402\n"
        "    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,\n"
        "    PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,\n"
        ")\n",
        "from three_body_symmetry.public_proof_audit import build_public_tc4_tc6_audit_manifest  # noqa: E402\n"
        "# PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS\n"
        "# PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS\n",
    )
    fast_ci_path.write_text(
        fast_ci_source.replace(
            "TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K = _pytest_k_expression(\n"
            "    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS\n"
            "    + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,\n"
            ")\n",
            f"TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K = {literal_selection!r}\n",
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    detail = _audit_obligation_detail(
        resolution,
        "public_audit_fast_ci_artifact_coverage",
    )
    assert "fast_ci:tc4_manifest_constant" in detail
    assert "fast_ci:tc5_manifest_constant" in detail


def test_public_audit_manifest_resolver_rejects_stale_fast_checker_command_selection(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    fast_ci_path.write_text(
        fast_ci_source.replace(
            '"not slow",\n        "--maxfail=1",',
            '"not slow",\n        "-k",\n        "unrelated_checker_test",\n        "--maxfail=1",',
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    assert "fast_ci:certificate_checker_command_selection" in (
        _audit_obligation_detail(
            resolution,
            "public_audit_fast_ci_artifact_coverage",
        )
    )


def test_public_audit_manifest_resolver_rejects_stale_slow_checker_command(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    slow_path = project_root / "scripts/slow_certificate_checker.py"
    slow_path.write_text(
        slow_path.read_text(encoding="utf-8").replace(
            "COMMAND = (",
            "LOCAL_COMMAND = (",
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_slow_ci_artifact_coverage" in resolution.missing_obligations
    assert "slow_ci:checker_command_selection" in _audit_obligation_detail(
        resolution,
        "public_audit_slow_ci_artifact_coverage",
    )


def test_public_audit_manifest_resolver_rejects_comment_only_slow_manifest_import(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    slow_path = project_root / "scripts/slow_certificate_checker.py"
    slow_source = slow_path.read_text(encoding="utf-8")
    literal_selection = (
        "independent_checker_accepts_serialized_generalized_fuchsian_stop_chart or "
        + " or ".join(
            artifact_id.split("::", maxsplit=1)[1]
            for artifact_id in PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
        )
    )
    slow_source = slow_source.replace(
        "from three_body_symmetry.public_proof_audit import (  # noqa: E402\n"
        "    PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,\n"
        ")\n",
        "from three_body_symmetry.public_proof_audit import build_public_tc4_tc6_audit_manifest  # noqa: E402\n"
        "# PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS\n",
    )
    slow_path.write_text(
        slow_source.replace(
            "SLOW_GENERALIZED_FUCHSIAN_CHECKER_K = (\n"
            "    \"independent_checker_accepts_serialized_generalized_fuchsian_stop_chart or \"\n"
            "    + _pytest_k_expression(PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS)\n"
            ")\n",
            f"SLOW_GENERALIZED_FUCHSIAN_CHECKER_K = {literal_selection!r}\n",
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_slow_ci_artifact_coverage" in resolution.missing_obligations
    assert "slow_ci:tc6_manifest_constant" in _audit_obligation_detail(
        resolution,
        "public_audit_slow_ci_artifact_coverage",
    )


def test_public_route_wrappers_report_malformed_obligation_ledgers():
    internal = _internal_pointwise_closed_form_theorem()
    total_collision_audit = _complete_total_collision_audit()
    route = certify_public_regularized_atlas_closed_form_proof(
        internal,
        total_collision_audit=total_collision_audit,
    )
    spoofed_route = replace(
        route,
        obligations=(
            SimpleNamespace(
                obligation="fake_public_route_obligation",
                certified=True,
                required=True,
            ),
        ),
    )
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=internal,
        total_collision_audit=total_collision_audit,
    )
    empty_public = replace(public, obligations=())

    assert "public_regularized_atlas_closed_form_proof_obligation_type" in (
        spoofed_route.public_audit_blockers
    )
    assert not spoofed_route.public_proof_certified
    assert "public_general_closed_form_solution_target_obligations_present" in (
        empty_public.public_audit_blockers
    )
    assert not empty_public.public_proof_certified


def test_public_general_closed_form_target_rejects_fake_public_route_wrapper():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
        total_collision_audit=_complete_total_collision_audit(),
    )
    spoofed = replace(
        public,
        public_regularized_atlas_proof=SimpleNamespace(
            public_proof_certified="yes",
            public_audit_blockers=(),
        ),
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in public.obligations
        ),
    )

    assert not public.internal_proof_certified
    assert not spoofed.public_proof_certified
    assert "public_regularized_atlas_proof_type" in (
        spoofed.public_audit_blockers
    )


def test_public_route_wrappers_recompute_stale_replaced_field_blockers():
    internal = _internal_pointwise_closed_form_theorem()
    alternate_internal = _internal_pointwise_closed_form_theorem()
    total_collision_audit = _complete_total_collision_audit()
    route = certify_public_regularized_atlas_closed_form_proof(
        internal,
        total_collision_audit=total_collision_audit,
    )
    alternate_route = certify_public_regularized_atlas_closed_form_proof(
        alternate_internal,
        total_collision_audit=total_collision_audit,
    )
    stale_route_internal = replace(
        route,
        internal_closed_form_theorem=SimpleNamespace(
            proof_certified=True,
            route_summary="fake internal pointwise theorem",
        ),
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in route.obligations
        ),
    )
    stale_route_audit = replace(
        route,
        total_collision_audit=SimpleNamespace(
            public_proof_certified=True,
            public_audit_blockers=(),
        ),
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in route.obligations
        ),
    )
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=internal,
        total_collision_audit=total_collision_audit,
    )
    stale_requested_class = replace(
        public,
        requested_class="finite algebraic first integrals",
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in public.obligations
        ),
    )
    stale_derived_gate = replace(
        public,
        require_derived_gates=False,
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in public.obligations
        ),
    )
    stale_derived_gate_type = replace(
        public,
        require_derived_gates="yes",
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in public.obligations
        ),
    )
    stale_public_route_source = replace(
        public,
        public_regularized_atlas_proof=alternate_route,
        obligations=tuple(
            replace(obligation, certified=True)
            for obligation in public.obligations
        ),
    )

    assert not stale_route_internal.public_proof_certified
    assert "public_regularized_internal_theorem_field" in (
        stale_route_internal.public_audit_blockers
    )
    assert not stale_route_audit.public_proof_certified
    assert "public_regularized_total_collision_audit_field" in (
        stale_route_audit.public_audit_blockers
    )
    assert not stale_requested_class.public_proof_certified
    assert "public_requested_class_field" in (
        stale_requested_class.public_audit_blockers
    )
    assert not stale_derived_gate.public_proof_certified
    assert "public_require_derived_gates_field" in (
        stale_derived_gate.public_audit_blockers
    )
    assert not stale_derived_gate_type.public_proof_certified
    assert "public_require_derived_gates_bool_field" in (
        stale_derived_gate_type.public_audit_blockers
    )
    assert not public.public_route_source_matches_pointwise_theorem
    assert not stale_public_route_source.public_route_source_matches_pointwise_theorem
    assert not stale_public_route_source.public_proof_certified
    assert "public_regularized_atlas_proof_source_matches_pointwise" in (
        stale_public_route_source.public_audit_blockers
    )


def test_public_regularized_atlas_audit_rejects_truthy_fake_policy_gate():
    internal = _internal_pointwise_closed_form_theorem()
    tampered_internal = replace(
        internal,
        maximal_classical_total_collision_policy=SimpleNamespace(
            proof_certified="yes",
            pointwise_theorem_derived="yes",
            policy_id="maximal_classical_stop",
        ),
    )
    public = certify_public_regularized_atlas_closed_form_proof(
        tampered_internal,
        total_collision_audit=_complete_total_collision_audit(),
    )

    assert not internal.proof_certified
    assert not tampered_internal.proof_certified
    assert not public.public_proof_certified
    assert "internal_pointwise_closed_form_theorem_certified" in (
        public.public_audit_blockers
    )
    assert "maximal_classical_policy_derived_from_pointwise_theorem" in (
        public.public_audit_blockers
    )


def test_public_tc6_audit_requires_named_polydisc_norm_and_backend_manifests():
    tc6 = replace(
        _complete_tc6_evidence(),
        cauchy_polydisc_id="some-polydisc",
        majorant_norm_id="some-norm",
        proof_grade_backend_id="rational_interval_backend",
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=tc6,
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not tc6.proof_certified
    assert not tc6.cauchy_polydisc_manifest_certified
    assert not tc6.majorant_norm_manifest_certified
    assert not tc6.proof_grade_backend_manifest_certified
    assert "tc6_cauchy_polydisc_id_supplied" in tc6.missing_obligations
    assert "tc6_majorant_norm_id_supplied" in tc6.missing_obligations
    assert "tc6_proof_grade_backend_id_supplied" in tc6.missing_obligations
    assert not audit.public_proof_certified
    assert "tc6:tc6_cauchy_polydisc_id_supplied" in audit.public_audit_blockers
    assert "tc6:tc6_majorant_norm_id_supplied" in audit.public_audit_blockers
    assert "tc6:tc6_proof_grade_backend_id_supplied" in (
        audit.public_audit_blockers
    )


def test_public_tc6_audit_requires_checker_artifact_manifest():
    tc6 = replace(
        _complete_tc6_evidence(),
        checker_artifact_ids=(
            "tests/test_certificate_checker.py::"
            "test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture",
        ),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=_complete_tc5_evidence(),
        tc6_cauchy_majorant_constants_evidence=tc6,
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not tc6.proof_certified
    assert not tc6.checker_artifact_manifest_certified
    assert "tc6_checker_artifact_manifest" in tc6.missing_obligations
    assert not audit.public_proof_certified
    assert "tc6:tc6_checker_artifact_manifest" in audit.public_audit_blockers


def test_public_tc6_required_checker_artifacts_exist_in_checker_suite():
    _assert_pytest_artifact_ids_exist(
        PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
        expected_file="tests/test_certificate_checker.py",
    )


def test_public_total_collision_required_checker_artifacts_resolve_to_checker_module():
    _assert_checker_artifact_ids_resolve(
        PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
    )


def test_public_audit_required_artifacts_are_covered_by_ci_targets():
    resolution = certify_public_audit_manifest_resolution()

    assert resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" not in (
        resolution.missing_obligations
    )
    assert "public_audit_slow_ci_artifact_coverage" not in (
        resolution.missing_obligations
    )
    assert set(resolution.fast_ci_artifact_ids).issuperset(
        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
        + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
        + PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS
    )
    assert resolution.slow_ci_artifact_ids == PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS


def test_public_audit_manifest_resolver_rejects_unfocused_fast_public_audit_command(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    fast_ci_path.write_text(
        fast_ci_source.replace(
            '        "tests/test_public_proof_audit.py",\n'
            '        "-k",\n'
            "        f\"{PUBLIC_AUDIT_FAST_K} or {PUBLIC_REVIEW_ARTIFACT_HARDENING_K}\",\n"
            '        "--maxfail=1",\n',
            '        "tests/test_public_proof_audit.py",\n',
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    detail = _audit_obligation_detail(
        resolution,
        "public_audit_fast_ci_artifact_coverage",
    )
    assert "fast_ci:public_audit_command_selection" in detail
    assert "fast_ci:public_audit_full_suite" in detail


def test_public_audit_manifest_resolver_rejects_stale_fast_public_audit_selection(
    tmp_path,
):
    project_root = _copy_public_audit_resolution_root(tmp_path)
    fast_ci_path = project_root / "scripts/fast_ci.py"
    fast_ci_source = fast_ci_path.read_text(encoding="utf-8")
    required_name = "public_audit_required_artifacts_are_covered_by_ci_targets"
    fast_ci_path.write_text(
        fast_ci_source.replace(
            required_name,
            f"{required_name}_stale",
        ),
        encoding="utf-8",
    )

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)

    assert not resolution.proof_certified
    assert "public_audit_fast_ci_artifact_coverage" in resolution.missing_obligations
    detail = _audit_obligation_detail(
        resolution,
        "public_audit_fast_ci_artifact_coverage",
    )
    assert f"fast_ci:test_{required_name}" in detail
    assert "fast_ci:public_audit_command_selection" in detail


def test_public_tc5_audit_requires_constructor_artifact_manifest():
    tc5 = replace(
        _complete_tc5_evidence(),
        constructor_artifact_ids=(
            "tests/test_obstructions.py::test_poincare_dulac_stable_normal_form_has_complete_log_selectors",
        ),
    )
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=_complete_tc4_evidence(),
        tc5_generalized_fuchsian_entry_evidence=tc5,
        tc6_cauchy_majorant_constants_evidence=_complete_tc6_evidence(),
        proof_references=PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    )

    assert not tc5.proof_certified
    assert not tc5.constructor_artifact_manifest_certified
    assert "tc5_constructor_artifact_manifest" in tc5.missing_obligations
    assert not audit.public_proof_certified
    assert "tc5:tc5_constructor_artifact_manifest" in audit.public_audit_blockers


def test_public_audit_rejects_placeholder_proof_references():
    tc4 = replace(_complete_tc4_evidence(), proof_references=("public-review:tc4",))
    tc5 = replace(_complete_tc5_evidence(), proof_references=("public-review:tc5",))
    tc6 = replace(_complete_tc6_evidence(), proof_references=("public-review:tc6",))
    audit = certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=tc4,
        tc5_generalized_fuchsian_entry_evidence=tc5,
        tc6_cauchy_majorant_constants_evidence=tc6,
        proof_references=("public-review:tc4-tc6",),
    )

    assert not tc4.proof_certified
    assert not tc5.proof_certified
    assert not tc6.proof_certified
    assert "tc4_public_proof_reference_manifest" in tc4.missing_obligations
    assert "tc5_public_proof_reference_manifest" in tc5.missing_obligations
    assert "tc6_public_proof_reference_manifest" in tc6.missing_obligations
    assert not audit.public_proof_certified
    assert "public_audit_proof_reference_manifest" in audit.public_audit_blockers
    assert "tc4:tc4_public_proof_reference_manifest" in (
        audit.public_audit_blockers
    )
    assert "tc5:tc5_public_proof_reference_manifest" in (
        audit.public_audit_blockers
    )
    assert "tc6:tc6_public_proof_reference_manifest" in (
        audit.public_audit_blockers
    )


def test_public_regularized_atlas_audit_rejects_raw_scaffold_gates():
    internal = _internal_pointwise_closed_form_theorem()
    raw_internal = replace(
        internal,
        certificate_language_soundness=_raw_soundness(),
        computable_certificate_enumeration=_raw_enumeration(),
    )
    total_collision_audit = _complete_total_collision_audit()
    public = certify_public_regularized_atlas_closed_form_proof(
        raw_internal,
        total_collision_audit=total_collision_audit,
    )

    assert not raw_internal.proof_certified
    assert not public.public_proof_certified
    assert "internal_pointwise_closed_form_theorem_certified" in (
        public.public_audit_blockers
    )
    assert "certificate_language_soundness_derived_from_checker_kernel" in (
        public.public_audit_blockers
    )
    assert "computable_enumeration_derived_from_same_pointwise_theorem" in (
        public.public_audit_blockers
    )


def test_public_regularized_atlas_audit_rejects_raw_total_collision_audit_gate():
    internal = _internal_pointwise_closed_form_theorem()
    public = certify_public_regularized_atlas_closed_form_proof(
        internal,
        total_collision_audit=True,
    )

    assert not internal.proof_certified
    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "public_total_collision_proof_audit_type" in (
        public.public_audit_blockers
    )
    assert "public_total_collision_proof_audit_certified" in (
        public.public_audit_blockers
    )


def test_public_audit_constructors_reject_truthy_nonboolean_flags():
    try:
        certify_public_reduced_hyperbolicity_audit_evidence(
            positive_mass_domain_declared="yes",
        )
    except TypeError as error:
        assert "positive_mass_domain_declared must be a bool" in str(error)
    else:
        raise AssertionError("truthy TC4 audit flag was accepted")

    try:
        certify_public_generalized_fuchsian_entry_audit_evidence(
            normal_form_coordinates_declared=1,
        )
    except TypeError as error:
        assert "normal_form_coordinates_declared must be a bool" in str(error)
    else:
        raise AssertionError("integer TC5 audit flag was accepted")

    try:
        certify_public_cauchy_majorant_audit_evidence(
            cauchy_polydisc_declared="true",
        )
    except TypeError as error:
        assert "cauchy_polydisc_declared must be a bool" in str(error)
    else:
        raise AssertionError("truthy TC6 audit flag was accepted")

    try:
        certify_public_total_collision_proof_audit(
            tc4_reduced_hyperbolicity_audited=1,
        )
    except TypeError as error:
        assert "tc4_reduced_hyperbolicity_audited must be a bool" in str(error)
    else:
        raise AssertionError("integer total-collision audit flag was accepted")


def test_public_general_closed_form_target_rejects_truthy_require_derived_gate():
    try:
        certify_public_general_closed_form_solution_target(
            require_derived_gates="yes",
        )
    except TypeError as error:
        assert "require_derived_gates must be a bool" in str(error)
    else:
        raise AssertionError("truthy require_derived_gates value was accepted")


def test_public_general_closed_form_target_rejects_raw_total_collision_audit_gate():
    pointwise = _pointwise_open_time_theorem()
    public = certify_public_general_closed_form_solution_target(
        "regularized locally finite atlas",
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=_derived_soundness(),
        computable_atlas_enumeration_certificate=(
            derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
                pointwise,
            )
        ),
        total_collision_audit=True,
    )

    assert not public.internal_proof_certified
    assert not public.public_proof_certified
    assert "public_regularized_atlas_proof_certified" in (
        public.public_audit_blockers
    )
    assert "public_total_collision_proof_audit_type" in (
        public.public_audit_blockers
    )
