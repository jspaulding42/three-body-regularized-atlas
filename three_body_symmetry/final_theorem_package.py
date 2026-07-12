"""Production assembly of the final regularized-atlas proof package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .certificate_checker import (
    CertificateCheckerKernelSupportCertificate,
    ProofGradeArithmeticBackendCertificate,
    build_fast_total_collision_generalized_fuchsian_stop_chart_certificate,
    certify_certificate_checker_kernel_support,
    certify_rational_interval_arithmetic_backend_soundness,
)
from .certificate_language import TotalCollisionGeneralizedFuchsianStopChartCertificate
from .closed_form import (
    CertificateLanguageSoundnessCertificate,
    ComputableAtlasCertificateEnumerationCertificate,
    GeneralClosedFormSolutionCertificate,
    MaximalClassicalTotalCollisionPolicyCertificate,
    PointwiseRegularizedAtlasClosedFormTheoremCertificate,
    certify_general_closed_form_solution_target,
    certify_maximal_classical_total_collision_policy,
    certify_pointwise_regularized_atlas_closed_form_theorem,
    derive_certificate_language_soundness_from_checker_kernel,
    derive_computable_atlas_certificate_enumeration_from_pointwise_theorem,
)
from .finite_target_completeness import (
    FiniteTargetCertificateSearchCompletenessCertificate,
    FiniteTargetCompletenessTheoremCertificate,
    certify_finite_target_certificate_search_completeness,
    certify_finite_target_completeness_theorem,
)
from .open_time_atlas import (
    PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    certify_pointwise_open_time_locally_finite_atlas_theorem,
)
from .public_proof_audit import (
    PublicGeneralClosedFormSolutionCertificate,
    PublicRegularizedAtlasClosedFormProofCertificate,
    PublicReviewArtifactResolutionCertificate,
    PublicTotalCollisionProofAuditCertificate,
    ReviewReadyTotalCollisionAuditEvidenceBundle,
    build_public_tc4_tc6_audit_manifest,
    build_review_ready_total_collision_audit_evidence_bundle,
    certify_public_general_closed_form_solution_target,
    certify_public_regularized_atlas_closed_form_proof,
    certify_public_review_artifact_resolution,
    certify_review_ready_total_collision_audit_package_from_evidence_bundle,
)


REGULARIZED_ATLAS_CLASS = "regularized locally finite atlas"
INTERVAL_BOX_SEARCH_BLOCKERS = (
    "recursive_set_valued_branch_partition_consumption",
    "event_order_partition_consumption_theorem",
)


@dataclass(frozen=True)
class FinalRegularizedAtlasProofPackage:
    """Complete production proof package for the exact/computable route."""

    finite_target_theorem: FiniteTargetCompletenessTheoremCertificate
    raw_certificate_search: FiniteTargetCertificateSearchCompletenessCertificate
    pointwise_open_time_theorem: PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate
    arithmetic_backend: ProofGradeArithmeticBackendCertificate
    checker_kernel: CertificateCheckerKernelSupportCertificate
    certificate_language_soundness: CertificateLanguageSoundnessCertificate
    computable_atlas_enumeration: ComputableAtlasCertificateEnumerationCertificate
    maximal_classical_total_collision_policy: (
        MaximalClassicalTotalCollisionPolicyCertificate
    )
    internal_general_closed_form: GeneralClosedFormSolutionCertificate
    pointwise_regularized_atlas_closed_form: (
        PointwiseRegularizedAtlasClosedFormTheoremCertificate
    )
    checked_stop_chart: TotalCollisionGeneralizedFuchsianStopChartCertificate
    total_collision_evidence_bundle: ReviewReadyTotalCollisionAuditEvidenceBundle
    local_total_collision_audit: PublicTotalCollisionProofAuditCertificate
    public_review_resolution: PublicReviewArtifactResolutionCertificate
    machine_checked_total_collision_audit: PublicTotalCollisionProofAuditCertificate
    public_regularized_atlas_proof: PublicRegularizedAtlasClosedFormProofCertificate
    public_general_closed_form: PublicGeneralClosedFormSolutionCertificate
    default_public_general_closed_form: PublicGeneralClosedFormSolutionCertificate
    public_manifest: dict[str, object]
    machine_checked_public_manifest: dict[str, object]

    @property
    def internal_proof_certified(self) -> bool:
        return bool(
            self.finite_target_theorem.proof_certified
            and self.pointwise_open_time_theorem.proof_certified
            and self.certificate_language_soundness.proof_certified
            and self.computable_atlas_enumeration.proof_certified
            and self.maximal_classical_total_collision_policy.proof_certified
            and self.internal_general_closed_form.proof_certified
            and self.internal_general_closed_form.blocking_obligations == ()
            and self.pointwise_regularized_atlas_closed_form.proof_certified
        )

    @property
    def machine_checked_public_proof_certified(self) -> bool:
        return bool(
            self.local_total_collision_audit.local_audit_package_certified
            and not self.local_total_collision_audit.public_proof_certified
            and self.public_review_resolution.proof_certified
            and self.public_review_resolution.artifact_kind
            == "machine_checked_public_audit"
            and self.machine_checked_total_collision_audit.public_proof_certified
            and self.machine_checked_total_collision_audit.machine_checked_public_audit_resolved
            and not self.machine_checked_total_collision_audit.public_review_resolution_is_external
            and self.public_regularized_atlas_proof.public_proof_certified
            and self.public_general_closed_form.public_proof_certified
            and self.public_general_closed_form.public_audit_blockers == ()
        )

    @property
    def default_public_route_remains_open(self) -> bool:
        return bool(
            self.default_public_general_closed_form.internal_proof_certified
            and not self.default_public_general_closed_form.public_proof_certified
            and "tc4_reduced_hyperbolicity_audited"
            in self.default_public_general_closed_form.public_audit_blockers
        )

    @property
    def interval_box_theorem_separate(self) -> bool:
        return bool(
            not self.raw_certificate_search.certified
            and self.raw_certificate_search.missing_obligations
            == INTERVAL_BOX_SEARCH_BLOCKERS
        )

    @property
    def ready_to_publish(self) -> bool:
        return bool(
            self.internal_proof_certified
            and self.machine_checked_public_proof_certified
            and self.default_public_route_remains_open
            and self.interval_box_theorem_separate
            and self.total_collision_evidence_bundle.local_audit_evidence_certified
            and self.public_manifest["public_closure_status"] == "external_review_open"
            and self.machine_checked_public_manifest["public_closure_status"]
            == "machine_checked_public_audit_verified"
            and self.machine_checked_public_manifest[
                "public_review_resolution_is_external"
            ]
            is False
        )


def certify_final_regularized_atlas_proof_package(
    *,
    dimension: int = 3,
    compact_time_rate: float = 1.3,
    total_collision_policy_id: str = "maximal_classical_stop",
    project_root: Path | str | None = None,
) -> FinalRegularizedAtlasProofPackage:
    """Assemble the complete production final proof package."""

    finite_target = certify_finite_target_completeness_theorem(
        dimension=dimension,
        total_collision_policy_id=total_collision_policy_id,
    )
    raw_search = certify_finite_target_certificate_search_completeness(
        finite_target,
    )
    pointwise = certify_pointwise_open_time_locally_finite_atlas_theorem(
        dimension=dimension,
        compact_time_rate=compact_time_rate,
        total_collision_policy_id=total_collision_policy_id,
    )
    arithmetic = certify_rational_interval_arithmetic_backend_soundness()
    kernel = certify_certificate_checker_kernel_support(
        proof_grade_arithmetic_backend_certificate=arithmetic,
    )
    soundness = derive_certificate_language_soundness_from_checker_kernel(kernel)
    enumeration = derive_computable_atlas_certificate_enumeration_from_pointwise_theorem(
        pointwise,
    )
    policy = certify_maximal_classical_total_collision_policy(
        pointwise_open_time_theorem=pointwise,
    )
    internal = certify_general_closed_form_solution_target(
        REGULARIZED_ATLAS_CLASS,
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )
    pointwise_closed = certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=pointwise,
        certificate_language_soundness=soundness,
        computable_certificate_enumeration=enumeration,
        maximal_classical_total_collision_policy=policy,
    )
    checked_stop_chart = (
        build_fast_total_collision_generalized_fuchsian_stop_chart_certificate()
    )
    evidence_bundle = build_review_ready_total_collision_audit_evidence_bundle(
        checked_stop_chart,
    )
    resolver = certify_public_review_artifact_resolution(project_root=project_root)
    local_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
        evidence_bundle,
    )
    machine_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
        evidence_bundle,
        public_review_resolution_certificate=resolver,
    )
    public_regularized = certify_public_regularized_atlas_closed_form_proof(
        pointwise_closed,
        total_collision_audit=machine_tc,
    )
    public_general = certify_public_general_closed_form_solution_target(
        REGULARIZED_ATLAS_CLASS,
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
        total_collision_audit=machine_tc,
    )
    default_public_general = certify_public_general_closed_form_solution_target(
        REGULARIZED_ATLAS_CLASS,
        general_theorem_certificate=pointwise,
        certificate_language_soundness_certificate=soundness,
        computable_atlas_enumeration_certificate=enumeration,
    )
    public_manifest = build_public_tc4_tc6_audit_manifest(
        project_root=project_root,
    )
    machine_manifest = build_public_tc4_tc6_audit_manifest(
        project_root=project_root,
        public_review_resolution_certificate=resolver,
    )
    return FinalRegularizedAtlasProofPackage(
        finite_target_theorem=finite_target,
        raw_certificate_search=raw_search,
        pointwise_open_time_theorem=pointwise,
        arithmetic_backend=arithmetic,
        checker_kernel=kernel,
        certificate_language_soundness=soundness,
        computable_atlas_enumeration=enumeration,
        maximal_classical_total_collision_policy=policy,
        internal_general_closed_form=internal,
        pointwise_regularized_atlas_closed_form=pointwise_closed,
        checked_stop_chart=checked_stop_chart,
        total_collision_evidence_bundle=evidence_bundle,
        local_total_collision_audit=local_tc,
        public_review_resolution=resolver,
        machine_checked_total_collision_audit=machine_tc,
        public_regularized_atlas_proof=public_regularized,
        public_general_closed_form=public_general,
        default_public_general_closed_form=default_public_general,
        public_manifest=public_manifest,
        machine_checked_public_manifest=machine_manifest,
    )
