import ast
from functools import lru_cache
from pathlib import Path

import pytest

from three_body_symmetry.certificate_checker import (
    build_fast_total_collision_generalized_fuchsian_stop_chart_certificate,
)
from three_body_symmetry.final_theorem_package import (
    INTERVAL_BOX_SEARCH_BLOCKERS,
    FinalRegularizedAtlasProofPackage,
    certify_final_regularized_atlas_proof_package,
)
from three_body_symmetry.public_proof_audit import (
    PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
    ReviewReadyTotalCollisionAuditEvidenceBundle,
    build_review_ready_total_collision_audit_evidence_bundle,
    certify_public_review_artifact_resolution,
    certify_review_ready_total_collision_audit_package_from_evidence_bundle,
)
import three_body_symmetry.public_proof_audit as public_proof_audit


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _final_package() -> FinalRegularizedAtlasProofPackage:
    return certify_final_regularized_atlas_proof_package(project_root=ROOT)


def test_final_regularized_atlas_package_preserves_local_checks_without_promoting_universal_routes():
    package = _final_package()

    assert package.finite_target_theorem.scaffold_certified
    assert not package.finite_target_theorem.proof_certified
    assert not package.pointwise_open_time_theorem.proof_certified
    assert package.certificate_language_soundness.proof_certified
    assert not package.computable_atlas_enumeration.proof_certified
    assert not package.internal_general_closed_form.proof_certified
    assert package.internal_general_closed_form.blocking_obligations
    assert not package.pointwise_regularized_atlas_closed_form.proof_certified
    assert package.total_collision_evidence_bundle.local_audit_evidence_certified
    assert package.local_total_collision_audit.local_audit_package_certified
    assert not package.local_total_collision_audit.public_proof_certified
    assert package.public_review_resolution.proof_certified
    assert package.public_review_resolution.artifact_kind == (
        "machine_checked_public_audit"
    )
    assert package.machine_checked_total_collision_audit.public_proof_certified
    assert not package.public_regularized_atlas_proof.public_proof_certified
    assert not package.public_general_closed_form.public_proof_certified
    assert package.public_general_closed_form.public_audit_blockers
    assert not package.internal_proof_certified
    assert not package.machine_checked_public_proof_certified
    assert not package.ready_to_publish


def test_final_regularized_atlas_package_keeps_default_public_route_open():
    package = _final_package()

    assert not package.default_public_general_closed_form.internal_proof_certified
    assert not package.default_public_general_closed_form.public_proof_certified
    assert "tc4_reduced_hyperbolicity_audited" in (
        package.default_public_general_closed_form.public_audit_blockers
    )
    assert package.public_manifest["public_closure_status"] == "external_review_open"
    assert package.machine_checked_public_manifest["public_closure_status"] == (
        "machine_checked_public_audit_verified"
    )
    assert (
        package.machine_checked_public_manifest["public_review_resolution_is_external"]
        is False
    )
    assert not package.default_public_route_remains_open
    assert not package.default_public_general_closed_form.public_proof_certified


def test_final_regularized_atlas_package_keeps_interval_box_theorem_separate():
    package = _final_package()

    assert not package.raw_certificate_search.certified
    assert package.raw_certificate_search.missing_obligations == (
        *INTERVAL_BOX_SEARCH_BLOCKERS,
        "pointwise_finite_target_theorem_proof_certified",
    )
    # The legacy convenience property also required the universal pointwise
    # route to be proved. The explicit blocker assertion above is the
    # authoritative separation check while that route remains open.
    assert not package.interval_box_theorem_separate


def test_final_regularized_atlas_package_reuses_tc6_evidence_without_rechecking(
    monkeypatch,
):
    checked_stop_chart = (
        build_fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    )
    real_check = public_proof_audit.check_total_collision_generalized_fuchsian_stop_chart
    calls = {"count": 0}

    def counted_check(certificate):
        calls["count"] += 1
        return real_check(certificate)

    monkeypatch.setattr(
        public_proof_audit,
        "check_total_collision_generalized_fuchsian_stop_chart",
        counted_check,
    )

    bundle = build_review_ready_total_collision_audit_evidence_bundle(
        checked_stop_chart,
    )
    resolver = certify_public_review_artifact_resolution(project_root=ROOT)
    local_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
        )
    )
    public_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
            public_review_resolution_certificate=resolver,
        )
    )
    raw_package = (
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            bundle,
            public_review_artifact_ids=(
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
            ),
        )
    )

    assert calls["count"] == 1
    assert (
        local_package.tc6_cauchy_majorant_constants_evidence
        is bundle.tc6_cauchy_majorant_constants_evidence
    )
    assert (
        public_package.tc6_cauchy_majorant_constants_evidence
        is bundle.tc6_cauchy_majorant_constants_evidence
    )
    assert (
        raw_package.tc6_cauchy_majorant_constants_evidence
        is bundle.tc6_cauchy_majorant_constants_evidence
    )
    assert local_package.local_audit_package_certified
    assert public_package.public_proof_certified
    assert not raw_package.public_proof_certified


def test_final_regularized_atlas_package_does_not_import_test_helpers():
    source = (ROOT / "three_body_symmetry" / "final_theorem_package.py").read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("tests")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("tests")
    assert "tests." not in source


def test_final_regularized_atlas_package_rejects_spoofed_evidence_bundle():
    checked_stop_chart = (
        build_fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    )
    bundle = build_review_ready_total_collision_audit_evidence_bundle(
        checked_stop_chart,
    )

    with pytest.raises(TypeError):
        certify_review_ready_total_collision_audit_package_from_evidence_bundle(
            object(),
        )

    spoofed = ReviewReadyTotalCollisionAuditEvidenceBundle(
        checked_stop_chart_certificate=bundle.checked_stop_chart_certificate,
        tc4_reduced_hyperbolicity_evidence=bundle.tc4_reduced_hyperbolicity_evidence,
        tc5_generalized_fuchsian_entry_evidence=bundle.tc5_generalized_fuchsian_entry_evidence,
        tc6_cauchy_majorant_constants_evidence=object(),
        source_documents=bundle.source_documents,
        proof_references=bundle.proof_references,
    )
    package = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
        spoofed,
    )

    assert not spoofed.local_audit_evidence_certified
    assert not package.local_audit_package_certified
    assert "tc6:evidence_type" in package.public_audit_blockers
