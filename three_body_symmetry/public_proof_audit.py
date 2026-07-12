"""Public proof-audit layer for the pointwise closed-form theorem.

The internal pointwise theorem certificates say what the checker/kernel
currently accepts as proof-certified.  This module keeps a separate public
review surface for the parts of that proof that still need a line-by-line
mathematical audit before the project should present the result as a public
proof package.  A complete local audit package is review-ready evidence, not
public proof closure until external public-review artifacts, or truthfully
named machine-checked public-audit artifacts, are attached and verified.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import re
from typing import Any

from .closed_form import (
    CertificateLanguageSoundnessCertificate,
    ComputableAtlasCertificateEnumerationCertificate,
    GeneralClosedFormSolutionCertificate,
    MaximalClassicalTotalCollisionPolicyCertificate,
    PointwiseRegularizedAtlasClosedFormTheoremCertificate,
    certify_general_closed_form_solution_target,
    certify_maximal_classical_total_collision_policy,
    certify_pointwise_regularized_atlas_closed_form_theorem,
)
from .certificate_checker import check_total_collision_generalized_fuchsian_stop_chart
from .certificate_language import TotalCollisionGeneralizedFuchsianStopChartCertificate
from .general_solution_theorem import TheoremPipelineObligation
from .open_time_atlas import PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate


TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT = (
    "docs/total-collision-generalized-fuchsian-stop-proof.md"
)

PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES = (
    "Euler_collinear_positive_mass",
    "Lagrange_equilateral_positive_mass",
)

PUBLIC_TC4_REQUIRED_QUOTIENT_MODES = (
    "translation",
    "scale",
    "rotation",
    "reflection",
)

PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS = (
    (
        "tests/test_obstructions.py::"
        "test_arbitrary_mass_equilateral_linearized_spectrum_matches_beta_formula"
    ),
    (
        "tests/test_obstructions.py::"
        "test_ordered_euler_linearized_spectrum_has_single_horizontal_shape_parameter"
    ),
    (
        "tests/test_obstructions.py::"
        "test_ordered_euler_shape_eigenvalue_bounds_limit_higher_resonance_orders"
    ),
    (
        "tests/test_obstructions.py::"
        "test_ordered_euler_shape_gap_public_audit_polynomial_identities_are_exact"
    ),
)

PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES = (
    "mcgehee_time",
    "scale_energy",
    "stable_shape_modes",
)

PUBLIC_TC5_REQUIRED_RESONANCE_RULES = (
    "poincare_dulac_denominator_projector",
    "finite_log_degree_bound",
)

PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS = (
    "weight",
    "row",
    "log_degree",
)

PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS = (
    "exponents",
    "resonant_rows",
    "cauchy_tail_majorants",
)

PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS = (
    (
        "tests/test_obstructions.py::"
        "test_stable_log_selector_chain_constructor_recovers_coupled_log_selectors"
    ),
    (
        "tests/test_obstructions.py::"
        "test_stable_log_selector_chain_log_degree_bound_is_finite_and_triangular"
    ),
    (
        "tests/test_obstructions.py::"
        "test_fuchsian_log_resonant_projector_right_inverse_identities_are_exact"
    ),
    (
        "tests/test_obstructions.py::"
        "test_stable_log_selector_chain_projects_to_finite_fuchsian_log_branch"
    ),
    (
        "tests/test_obstructions.py::"
        "test_fuchsian_log_row_constructor_builds_resonant_selector_branch"
    ),
    (
        "tests/test_obstructions.py::"
        "test_finite_fuchsian_log_branch_composes_selector_rows_and_projects"
    ),
)

PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES = (
    "C0",
    "Lambda",
    "sigma",
    "p0",
    "d",
)

PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID = "tc6-polydisc:P(r)"
PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID = "tc6-norm:sup-polydisc"
PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID = (
    "certify_rational_interval_arithmetic_backend_soundness"
)

PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS = (
    (
        "tests/test_certificate_checker.py::"
        "test_fast_reduced_order_generalized_fuchsian_stop_checker_ci_fixture"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_generalized_fuchsian_remainder_majorant_picard_tail_formula_is_explicit"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_independent_checker_recomputes_generalized_fuchsian_primitive_tail_bounds"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_generalized_fuchsian_projected_residual_direct_interval_is_certification_gated"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_generalized_fuchsian_projected_budget_does_not_weaken_lifted_gate"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_generalized_fuchsian_projected_residual_weight_shift_is_cubic_time_exact"
    ),
)

PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS = (
    (
        "tests/test_certificate_checker.py::"
        "test_generalized_fuchsian_stop_checker_certification_is_interval_not_sample_gated"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_independent_checker_rejects_corrupted_generalized_fuchsian_majorant"
    ),
    (
        "tests/test_certificate_checker.py::"
        "test_independent_checker_rejects_generalized_fuchsian_without_projected_tail"
    ),
)

PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS = (
    PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS
    + PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS
)

PUBLIC_TC6_REQUIRED_CHECKER_ID = (
    "independent_total_collision_generalized_fuchsian_stop_checker_"
    "interval_cauchy_projected_v3"
)

PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS = (
    "generalized_fuchsian_remainder_majorant_certifies",
    "generalized_fuchsian_remainder_component_inputs_certify",
    "generalized_fuchsian_remainder_required_residual_components",
    "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells",
    "generalized_fuchsian_endpoint_collapse_envelope",
)

PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS = (
    "first_jet",
    "lifted_residual",
    "physical_residual",
    "regularized_position_value",
    "value",
)

PUBLIC_TC4_REQUIRED_PROOF_REFERENCES = (
    f"{TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT}#TC4",
)

PUBLIC_TC5_REQUIRED_PROOF_REFERENCES = (
    f"{TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT}#TC5",
)

PUBLIC_TC6_REQUIRED_PROOF_REFERENCES = (
    f"{TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT}#TC6",
)

PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES = (
    PUBLIC_TC4_REQUIRED_PROOF_REFERENCES
    + PUBLIC_TC5_REQUIRED_PROOF_REFERENCES
    + PUBLIC_TC6_REQUIRED_PROOF_REFERENCES
)

PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS = (
    "TotalCollisionGeneralizedFuchsianStopChartCertificate",
    "check_total_collision_generalized_fuchsian_stop_chart",
)

_PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_OBJECTS = {
    "TotalCollisionGeneralizedFuchsianStopChartCertificate": (
        TotalCollisionGeneralizedFuchsianStopChartCertificate
    ),
    "check_total_collision_generalized_fuchsian_stop_chart": (
        check_total_collision_generalized_fuchsian_stop_chart
    ),
}

PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS = (
    "external-public-review:tc4-reduced-hyperbolicity-line-audit",
    "external-public-review:tc5-generalized-fuchsian-entry-line-audit",
    "external-public-review:tc6-cauchy-majorant-backend-line-audit",
)

PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL = "external_review"
PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE = "machine_checked_public_audit"
PUBLIC_REVIEW_ARTIFACT_KINDS = (
    PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL,
    PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE,
)

PUBLIC_REVIEW_VERIFIER_ID = "repo-machine-check-public-audit-v1"

PUBLIC_REVIEW_ARTIFACT_PATHS = {
    "external-public-review:tc4-reduced-hyperbolicity-line-audit": (
        "docs/public-review/tc4-reduced-hyperbolicity-line-audit.md"
    ),
    "external-public-review:tc5-generalized-fuchsian-entry-line-audit": (
        "docs/public-review/tc5-generalized-fuchsian-entry-line-audit.md"
    ),
    "external-public-review:tc6-cauchy-majorant-backend-line-audit": (
        "docs/public-review/tc6-cauchy-majorant-backend-line-audit.md"
    ),
}

PUBLIC_REVIEW_PLACEHOLDER_MARKERS = (
    "todo",
    "fixme",
    "tbd",
    "placeholder",
    "assumed",
    "to be filled",
    "external-review pending",
)


def build_public_tc4_tc6_audit_manifest(
    *,
    project_root: str | Path | None = None,
    public_review_resolution_certificate: Any | None = None,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable TC4-TC6 review manifest.

    This is an export surface for reviewers, not a public-proof certificate.
    Local proof-note and checker/test artifacts are resolved below; the
    external public-review artifacts remain explicit open obligations.
    """

    resolution = certify_public_audit_manifest_resolution(project_root=project_root)
    public_review_certified = bool(
        type(public_review_resolution_certificate)
        is PublicReviewArtifactResolutionCertificate
        and public_review_resolution_certificate.proof_certified is True
    )
    artifact_kind = (
        public_review_resolution_certificate.artifact_kind
        if public_review_certified
        else None
    )
    machine_checked = artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE
    external_review = artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL
    public_closure_status = "external_review_open"
    public_closure_note = (
        "Local proof-note, checker, and regression artifacts resolve; "
        "external TC4-TC6 line-audit artifacts are still required before "
        "public proof closure."
    )
    if machine_checked:
        public_closure_status = "machine_checked_public_audit_verified"
        public_closure_note = (
            "Local proof-note, checker, regression, and machine-check public "
            "audit artifacts resolve. This is not claimed as external peer review."
        )
    elif external_review:
        public_closure_status = "public_review_verified"
        public_closure_note = (
            "External public-review artifacts have been resolved and checked "
            "against the TC4-TC6 manifest."
        )
    return {
        "manifest_id": "public_tc4_tc6_audit_manifest",
        "proof_document": TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,
        "proof_references": list(PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES),
        "scope": "TC4-TC6 public total-collision proof review support",
        "public_closure_status": public_closure_status,
        "public_closure_note": public_closure_note,
        "tc4": {
            "topic": "reduced hyperbolicity",
            "proof_references": list(PUBLIC_TC4_REQUIRED_PROOF_REFERENCES),
            "central_target_families": list(
                PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES
            ),
            "quotient_modes_removed": list(PUBLIC_TC4_REQUIRED_QUOTIENT_MODES),
            "spectrum_audit_reference_ids": list(
                PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
            ),
        },
        "tc5": {
            "topic": "generalized Fuchsian entry",
            "proof_references": list(PUBLIC_TC5_REQUIRED_PROOF_REFERENCES),
            "normal_form_coordinate_ids": list(
                PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES
            ),
            "resonance_rule_ids": list(PUBLIC_TC5_REQUIRED_RESONANCE_RULES),
            "triangular_order_keys": list(PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS),
            "selector_data_fields": list(PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS),
            "constructor_artifact_ids": list(
                PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
            ),
        },
        "tc6": {
            "topic": "Cauchy majorant backend",
            "proof_references": list(PUBLIC_TC6_REQUIRED_PROOF_REFERENCES),
            "cauchy_polydisc_id": PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID,
            "majorant_norm_id": PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID,
            "primitive_constant_names": list(
                PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES
            ),
            "proof_grade_backend_id": PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID,
            "fast_checker_artifact_ids": list(
                PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS
            ),
            "slow_checker_artifact_ids": list(
                PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS
            ),
            "checker_artifact_ids": list(PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS),
            "checker_id": PUBLIC_TC6_REQUIRED_CHECKER_ID,
            "checker_obligation_ids": list(
                PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS
            ),
            "majorant_component_ids": list(
                PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS
            ),
        },
        "total_collision_checker_artifacts": list(
            PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
        ),
        "ci_artifacts": {
            "fast": list(resolution.fast_ci_artifact_ids),
            "slow": list(resolution.slow_ci_artifact_ids),
        },
        "external_public_review_artifact_ids": list(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        "public_review_artifact_ids": list(
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        ),
        "public_review_artifact_kind": artifact_kind,
        "public_review_resolved_by_artifact_resolution": public_review_certified,
        "public_review_resolution_status": public_closure_status,
        "public_review_resolution_is_external": external_review,
        "external_public_review_resolved_by_local_manifest": external_review,
        "machine_checked_public_audit_resolved": machine_checked,
        "public_review_artifact_resolution": (
            {
                "proof_certified": True,
                "artifact_kind": artifact_kind,
                "artifact_ids": list(
                    public_review_resolution_certificate.required_artifact_ids
                ),
                "artifact_sha256": {
                    public_review_resolution_certificate.tc4_artifact.artifact_id: (
                        public_review_resolution_certificate.tc4_artifact.sha256
                    ),
                    public_review_resolution_certificate.tc5_artifact.artifact_id: (
                        public_review_resolution_certificate.tc5_artifact.sha256
                    ),
                    public_review_resolution_certificate.tc6_artifact.artifact_id: (
                        public_review_resolution_certificate.tc6_artifact.sha256
                    ),
                },
            }
            if public_review_certified
            else {
                "proof_certified": False,
                "artifact_kind": None,
                "artifact_ids": [],
                "artifact_sha256": {},
            }
        ),
        "local_manifest_resolution": {
            "proof_certified": resolution.proof_certified,
            "missing_obligations": list(resolution.missing_obligations),
            "theorem_id": resolution.theorem_id,
        },
    }


_PUBLIC_REGULARIZED_ATLAS_CLASS_ALIASES = {
    "regularized_atlas": "regularized_locally_finite_atlas",
    "regularized_locally_finite_atlas": "regularized_locally_finite_atlas",
    "piecewise_analytic_regularized_atlas": "regularized_locally_finite_atlas",
}

PUBLIC_REGULARIZED_ATLAS_PROOF_REQUIRED_OBLIGATIONS = (
    "internal_pointwise_closed_form_theorem_certified",
    "public_total_collision_proof_audit_certified",
    "public_total_collision_proof_audit_type",
    "certificate_language_soundness_derived_from_checker_kernel",
    "computable_enumeration_derived_from_same_pointwise_theorem",
    "maximal_classical_policy_derived_from_pointwise_theorem",
)

PUBLIC_GENERAL_CLOSED_FORM_REQUIRED_OBLIGATIONS = (
    "internal_general_closed_form_solution_certified",
    "public_regularized_atlas_proof_certified",
    "public_route_requires_derived_gates",
    "public_certificate_language_soundness_derived",
    "public_computable_enumeration_derived",
    "raw_boolean_gates_not_public_evidence",
)


def _contains_all(items: tuple[str, ...], required: tuple[str, ...]) -> bool:
    return set(items).issuperset(required)


def _matches_exact_manifest(items: tuple[str, ...], required: tuple[str, ...]) -> bool:
    return items == required


def _normalize_public_closed_form_class(closed_form_class: Any) -> str:
    key = str(closed_form_class).strip().lower().replace(" ", "_").replace("-", "_")
    return _PUBLIC_REGULARIZED_ATLAS_CLASS_ALIASES.get(key, key)


def _public_requested_class_is_regularized_atlas(closed_form_class: Any) -> bool:
    return (
        _normalize_public_closed_form_class(closed_form_class)
        == "regularized_locally_finite_atlas"
    )


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    return value


def _pipeline_obligation_ledger_certified(
    obligations: tuple[Any, ...],
) -> bool:
    return bool(
        obligations
        and any(
            isinstance(obligation, TheoremPipelineObligation)
            and obligation.required is True
            for obligation in obligations
        )
        and all(
            isinstance(obligation, TheoremPipelineObligation)
            for obligation in obligations
        )
        and all(
            obligation.certified is True
            for obligation in obligations
            if obligation.required
        )
    )


def _pipeline_obligation_ledger_missing(
    obligations: tuple[Any, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not obligations:
        missing.append(f"{ledger_name}_obligations_present")
    if obligations and not any(
        isinstance(obligation, TheoremPipelineObligation)
        and obligation.required is True
        for obligation in obligations
    ):
        missing.append(f"{ledger_name}_required_obligation_present")
    for obligation in obligations:
        if not isinstance(obligation, TheoremPipelineObligation):
            missing.append(f"{ledger_name}_obligation_type")
            continue
        if obligation.required and obligation.certified is not True:
            missing.append(obligation.obligation)
    return tuple(dict.fromkeys(missing))


def _pipeline_obligation_manifest_exact(
    obligations: tuple[Any, ...],
    required_obligation_ids: tuple[str, ...],
) -> bool:
    return bool(
        tuple(
            obligation.obligation
            for obligation in obligations
            if isinstance(obligation, TheoremPipelineObligation)
        )
        == tuple(required_obligation_ids)
        and len(obligations) == len(required_obligation_ids)
    )


@dataclass(frozen=True)
class PublicAuditManifestResolutionCertificate:
    """Local resolver for the public-audit proof/reference manifests."""

    proof_references: tuple[str, ...]
    tc4_spectrum_audit_reference_ids: tuple[str, ...]
    tc5_constructor_artifact_ids: tuple[str, ...]
    tc6_checker_artifact_ids: tuple[str, ...]
    total_collision_checker_artifacts: tuple[str, ...]
    fast_ci_artifact_ids: tuple[str, ...]
    slow_ci_artifact_ids: tuple[str, ...]
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "public_audit_manifest_resolution"

    @property
    def manifest_field_obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_audit_manifest_resolution_theorem_id",
                certified=self.theorem_id == "public_audit_manifest_resolution",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_proof_reference_field",
                certified=_matches_exact_manifest(
                    self.proof_references,
                    PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
                ),
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_tc4_spectrum_field",
                certified=_matches_exact_manifest(
                    self.tc4_spectrum_audit_reference_ids,
                    PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
                ),
                source=self.theorem_id,
                detail=",".join(self.tc4_spectrum_audit_reference_ids),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_tc5_constructor_field",
                certified=_matches_exact_manifest(
                    self.tc5_constructor_artifact_ids,
                    PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
                ),
                source=self.theorem_id,
                detail=",".join(self.tc5_constructor_artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_tc6_checker_field",
                certified=_matches_exact_manifest(
                    self.tc6_checker_artifact_ids,
                    PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
                ),
                source=self.theorem_id,
                detail=",".join(self.tc6_checker_artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_total_checker_field",
                certified=_matches_exact_manifest(
                    self.total_collision_checker_artifacts,
                    PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
                ),
                source=self.theorem_id,
                detail=",".join(self.total_collision_checker_artifacts),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_fast_ci_artifact_field",
                certified=_matches_exact_manifest(
                    self.fast_ci_artifact_ids,
                    (
                        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
                        + PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
                        + PUBLIC_TC6_REQUIRED_FAST_CHECKER_ARTIFACT_IDS
                    ),
                ),
                source=self.theorem_id,
                detail=",".join(self.fast_ci_artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_manifest_slow_ci_artifact_field",
                certified=_matches_exact_manifest(
                    self.slow_ci_artifact_ids,
                    PUBLIC_TC6_REQUIRED_SLOW_CHECKER_ARTIFACT_IDS,
                ),
                source=self.theorem_id,
                detail=",".join(self.slow_ci_artifact_ids),
            ),
        )

    @property
    def manifest_fields_certified(self) -> bool:
        return _pipeline_obligation_ledger_certified(self.manifest_field_obligations)

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_audit_manifest_resolution"
            and self.manifest_fields_certified
            and _pipeline_obligation_ledger_certified(self.obligations)
        )

    @property
    def certified(self) -> bool:
        return self.proof_certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *_pipeline_obligation_ledger_missing(
                        self.manifest_field_obligations,
                        ledger_name="public_audit_manifest_resolution_fields",
                    ),
                    *_pipeline_obligation_ledger_missing(
                        self.obligations,
                        ledger_name="public_audit_manifest_resolution",
                    ),
                )
            )
        )


def certify_public_audit_manifest_resolution(
    *,
    proof_references: tuple[str, ...] = PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
    tc4_spectrum_audit_reference_ids: tuple[str, ...] = (
        PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
    ),
    tc5_constructor_artifact_ids: tuple[str, ...] = (
        PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
    ),
    tc6_checker_artifact_ids: tuple[str, ...] = (
        PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
    ),
    total_collision_checker_artifacts: tuple[str, ...] = (
        PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
    ),
    project_root: str | Path | None = None,
) -> PublicAuditManifestResolutionCertificate:
    """Resolve local public-audit references to actual repo artifacts.

    This verifies the local manifest only.  It does not resolve or certify the
    external public-review artifacts required for public proof closure.
    """

    root = Path(project_root) if project_root is not None else _project_root()
    proof_references = tuple(str(item) for item in proof_references)
    tc4_spectrum_audit_reference_ids = tuple(
        str(item) for item in tc4_spectrum_audit_reference_ids
    )
    tc5_constructor_artifact_ids = tuple(
        str(item) for item in tc5_constructor_artifact_ids
    )
    tc6_checker_artifact_ids = tuple(
        str(item) for item in tc6_checker_artifact_ids
    )
    total_collision_checker_artifacts = tuple(
        str(item) for item in total_collision_checker_artifacts
    )
    unresolved_proof_references = _unresolved_proof_references(
        proof_references,
        root,
    )
    unresolved_tc4 = _unresolved_pytest_artifact_ids(
        tc4_spectrum_audit_reference_ids,
        root,
    )
    unresolved_tc5 = _unresolved_pytest_artifact_ids(
        tc5_constructor_artifact_ids,
        root,
    )
    unresolved_tc6 = _unresolved_pytest_artifact_ids(
        tc6_checker_artifact_ids,
        root,
    )
    unresolved_total_checker = _unresolved_checker_module_artifact_ids(
        total_collision_checker_artifacts,
    )
    fast_ci_artifact_ids, slow_ci_artifact_ids = _public_audit_ci_artifact_split(
        tc4_spectrum_audit_reference_ids,
        tc5_constructor_artifact_ids,
        tc6_checker_artifact_ids,
        root,
    )
    unresolved_fast_ci = _unresolved_fast_ci_public_audit_coverage(
        tc4_spectrum_audit_reference_ids,
        tc5_constructor_artifact_ids,
        tc6_checker_artifact_ids,
        fast_ci_artifact_ids,
        root,
    )
    unresolved_slow_ci = _unresolved_slow_ci_public_audit_coverage(
        slow_ci_artifact_ids,
        root,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="public_audit_proof_references_resolve",
            certified=not unresolved_proof_references and bool(proof_references),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_proof_references),
        ),
        TheoremPipelineObligation(
            obligation="tc4_spectrum_pytest_artifacts_resolve",
            certified=not unresolved_tc4 and bool(tc4_spectrum_audit_reference_ids),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_tc4),
        ),
        TheoremPipelineObligation(
            obligation="tc5_constructor_pytest_artifacts_resolve",
            certified=not unresolved_tc5 and bool(tc5_constructor_artifact_ids),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_tc5),
        ),
        TheoremPipelineObligation(
            obligation="tc6_checker_pytest_artifacts_resolve",
            certified=not unresolved_tc6 and bool(tc6_checker_artifact_ids),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_tc6),
        ),
        TheoremPipelineObligation(
            obligation="total_collision_checker_module_artifacts_resolve",
            certified=not unresolved_total_checker
            and bool(total_collision_checker_artifacts),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_total_checker),
        ),
        TheoremPipelineObligation(
            obligation="public_audit_fast_ci_artifact_coverage",
            certified=not unresolved_fast_ci and bool(fast_ci_artifact_ids),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_fast_ci),
        ),
        TheoremPipelineObligation(
            obligation="public_audit_slow_ci_artifact_coverage",
            certified=not unresolved_slow_ci and bool(slow_ci_artifact_ids),
            source="public_audit_manifest_resolution",
            detail="unresolved=" + ",".join(unresolved_slow_ci),
        ),
        TheoremPipelineObligation(
            obligation="external_public_review_not_resolved_by_local_manifest",
            certified=True,
            source="public_audit_manifest_resolution",
            required=False,
            detail=(
                "local proof-note and checker/test artifacts resolve; external "
                "TC4-TC6 public-review artifacts remain separate obligations"
            ),
        ),
    )
    return PublicAuditManifestResolutionCertificate(
        proof_references=proof_references,
        tc4_spectrum_audit_reference_ids=tc4_spectrum_audit_reference_ids,
        tc5_constructor_artifact_ids=tc5_constructor_artifact_ids,
        tc6_checker_artifact_ids=tc6_checker_artifact_ids,
        total_collision_checker_artifacts=total_collision_checker_artifacts,
        fast_ci_artifact_ids=fast_ci_artifact_ids,
        slow_ci_artifact_ids=slow_ci_artifact_ids,
        obligations=obligations,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unresolved_proof_references(
    references: tuple[str, ...],
    root: Path,
) -> tuple[str, ...]:
    unresolved: list[str] = []
    for reference in references:
        if "#" not in reference:
            unresolved.append(reference)
            continue
        path, anchor = reference.split("#", maxsplit=1)
        document_path = root / path
        if not document_path.is_file():
            unresolved.append(reference)
            continue
        if anchor not in _markdown_level2_heading_anchors(document_path):
            unresolved.append(reference)
    return tuple(unresolved)


def _markdown_level2_heading_anchors(path: Path) -> frozenset[str]:
    if not path.is_file():
        return frozenset()
    anchors: set[str] = set()
    in_fenced_block = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        if not stripped.startswith("## ") or stripped.startswith("### "):
            continue
        heading_text = stripped[3:].strip()
        heading_anchor = heading_text.split(".", maxsplit=1)[0].strip()
        if heading_anchor:
            anchors.add(heading_anchor)
    return frozenset(anchors)


def _unresolved_pytest_artifact_ids(
    artifact_ids: tuple[str, ...],
    root: Path,
) -> tuple[str, ...]:
    unresolved: list[str] = []
    for artifact_id in artifact_ids:
        if "::" not in artifact_id:
            unresolved.append(artifact_id)
            continue
        file_name, test_name = artifact_id.split("::", maxsplit=1)
        test_path = root / file_name
        if not test_path.is_file():
            unresolved.append(artifact_id)
            continue
        if test_name not in _python_function_defs(test_path):
            unresolved.append(artifact_id)
    return tuple(unresolved)


def _python_function_defs(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    if not path.is_file():
        return {}
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return {}
    return {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _unresolved_checker_module_artifact_ids(
    artifact_ids: tuple[str, ...],
) -> tuple[str, ...]:
    from . import certificate_checker as certificate_checker_module

    unresolved: list[str] = []
    for artifact_id in artifact_ids:
        expected_artifact = _PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_OBJECTS.get(
            artifact_id,
        )
        artifact = getattr(certificate_checker_module, artifact_id, None)
        if (
            artifact is None
            or getattr(artifact, "__name__", "") != artifact_id
            or artifact is not expected_artifact
        ):
            unresolved.append(artifact_id)
    return tuple(unresolved)


def _public_audit_ci_artifact_split(
    tc4_spectrum_audit_reference_ids: tuple[str, ...],
    tc5_constructor_artifact_ids: tuple[str, ...],
    tc6_checker_artifact_ids: tuple[str, ...],
    root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tc4_tc5_fast = tc4_spectrum_audit_reference_ids + tc5_constructor_artifact_ids
    tc6_fast: list[str] = []
    tc6_slow: list[str] = []
    for artifact_id in tc6_checker_artifact_ids:
        if "::" not in artifact_id:
            tc6_slow.append(artifact_id)
            continue
        file_name, test_name = artifact_id.split("::", maxsplit=1)
        if _pytest_test_has_slow_marker(test_name, expected_file=file_name, root=root):
            tc6_slow.append(artifact_id)
        else:
            tc6_fast.append(artifact_id)
    return tc4_tc5_fast + tuple(tc6_fast), tuple(tc6_slow)


_PUBLIC_AUDIT_FAST_TEST_NAMES = (
    "test_public_total_collision_audit_keeps_tc4_tc6_open_by_default",
    "test_public_regularized_atlas_audit_keeps_external_review_open_after_local_tc4_tc6_package",
    "test_public_general_closed_form_target_keeps_external_review_open_after_local_public_audit_package",
    "test_public_audit_manifest_resolver_certifies_local_artifact_manifest",
    "test_public_audit_required_artifacts_are_covered_by_ci_targets",
)


def _unresolved_fast_ci_public_audit_coverage(
    tc4_spectrum_audit_reference_ids: tuple[str, ...],
    tc5_constructor_artifact_ids: tuple[str, ...],
    tc6_checker_artifact_ids: tuple[str, ...],
    expected_fast_artifact_ids: tuple[str, ...],
    root: Path,
) -> tuple[str, ...]:
    fast_ci = _load_script_module("fast_ci.py", root)
    fast_ci_path = root / "scripts" / "fast_ci.py"
    unresolved: list[str] = []
    tc4_tc5_names = _pytest_artifact_test_names(
        tc4_spectrum_audit_reference_ids + tc5_constructor_artifact_ids
    )
    tc4_tc5_k = str(getattr(fast_ci, "TC4_TC5_PUBLIC_AUDIT_ARTIFACT_K", ""))
    fast_imports = _public_audit_imported_names(fast_ci_path)
    if "PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS" not in fast_imports:
        unresolved.append("fast_ci:tc4_manifest_constant")
    if "PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS" not in fast_imports:
        unresolved.append("fast_ci:tc5_manifest_constant")
    for test_name in tc4_tc5_names:
        if not _ci_selection_mentions(test_name, tc4_tc5_k):
            unresolved.append(f"fast_ci:{test_name}")
    fast_commands = _script_command_tuples(fast_ci)
    public_audit_k = str(getattr(fast_ci, "PUBLIC_AUDIT_FAST_K", ""))
    for test_name in _PUBLIC_AUDIT_FAST_TEST_NAMES:
        if not _ci_selection_mentions(test_name, public_audit_k):
            unresolved.append(f"fast_ci:{test_name}")
    public_audit_commands = tuple(
        command
        for command in fast_commands
        if _command_references(command, "tests/test_public_proof_audit.py")
    )
    if not public_audit_commands:
        unresolved.append("fast_ci:public_audit_suite")
    if not any(
        _command_k_selection_covers(command, _PUBLIC_AUDIT_FAST_TEST_NAMES)
        for command in public_audit_commands
    ):
        unresolved.append("fast_ci:public_audit_command_selection")
    if any(
        not _command_option_value(command, "-k")
        for command in public_audit_commands
    ):
        unresolved.append("fast_ci:public_audit_full_suite")
    obstruction_commands = tuple(
        command
        for command in fast_commands
        if _command_references(command, "tests/test_obstructions.py")
    )
    if not obstruction_commands:
        unresolved.append("fast_ci:obstruction_suite")
    if not any(
        _command_k_selection_covers(command, tc4_tc5_names)
        for command in obstruction_commands
    ):
        unresolved.append("fast_ci:obstruction_command_selection")
    tc6_names = _pytest_artifact_test_names(tc6_checker_artifact_ids)
    fast_tc6_names = _pytest_artifact_test_names(
        tuple(
            artifact_id
            for artifact_id in expected_fast_artifact_ids
            if artifact_id in tc6_checker_artifact_ids
        )
    )
    slow_tc6_names = tuple(name for name in tc6_names if name not in fast_tc6_names)
    certificate_checker_commands = tuple(
        command
        for command in fast_commands
        if _command_references(command, "tests/test_certificate_checker.py")
    )
    if not certificate_checker_commands:
        unresolved.append("fast_ci:certificate_checker_suite")
    if not any(
        _command_option_value(command, "-m") == "not slow"
        for command in certificate_checker_commands
    ):
        unresolved.append("fast_ci:non_slow_marker")
    if not any(
        _fast_certificate_checker_command_covers(command, fast_tc6_names)
        for command in certificate_checker_commands
    ):
        unresolved.append("fast_ci:certificate_checker_command_selection")
    if set(fast_tc6_names).intersection(slow_tc6_names):
        unresolved.append("fast_ci:tc6_slow_fast_overlap")
    if not fast_tc6_names:
        unresolved.append("fast_ci:tc6_fast_artifact")
    return tuple(unresolved)


def _unresolved_slow_ci_public_audit_coverage(
    slow_artifact_ids: tuple[str, ...],
    root: Path,
) -> tuple[str, ...]:
    slow_checker = _load_script_module("slow_certificate_checker.py", root)
    slow_path = root / "scripts" / "slow_certificate_checker.py"
    slow_k = str(getattr(slow_checker, "SLOW_GENERALIZED_FUCHSIAN_CHECKER_K", ""))
    unresolved: list[str] = []
    slow_imports = _public_audit_imported_names(slow_path)
    if "PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS" not in slow_imports:
        unresolved.append("slow_ci:tc6_manifest_constant")
    for test_name in _pytest_artifact_test_names(slow_artifact_ids):
        if not _ci_selection_mentions(test_name, slow_k):
            unresolved.append(f"slow_ci:{test_name}")
    slow_names = _pytest_artifact_test_names(slow_artifact_ids)
    primary_slow_command = _coerce_command_tuple(getattr(slow_checker, "COMMAND", None))
    slow_commands = _script_command_tuples(slow_checker)
    checker_commands = tuple(
        command
        for command in slow_commands
        if _command_references(command, "tests/test_certificate_checker.py")
    )
    if not checker_commands:
        unresolved.append("slow_ci:certificate_checker_suite")
    if not any(
        _command_option_value(command, "-m") == "slow"
        for command in checker_commands
    ):
        unresolved.append("slow_ci:slow_marker")
    if not _slow_certificate_checker_command_covers(
        primary_slow_command,
        slow_names,
    ):
        unresolved.append("slow_ci:checker_command_selection")
    return tuple(unresolved)


def _public_audit_imported_names(script_path: Path) -> frozenset[str]:
    if not script_path.is_file():
        return frozenset()
    try:
        module = ast.parse(
            script_path.read_text(encoding="utf-8"),
            filename=str(script_path),
        )
    except SyntaxError:
        return frozenset()
    imported: set[str] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "three_body_symmetry.public_proof_audit"
        ):
            imported.update(alias.name for alias in node.names)
    return frozenset(imported)


def _pytest_artifact_test_names(artifact_ids: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for artifact_id in artifact_ids:
        if "::" in artifact_id:
            names.append(artifact_id.split("::", maxsplit=1)[1])
    return tuple(names)


def _pytest_test_has_slow_marker(
    test_name: str,
    *,
    expected_file: str,
    root: Path,
) -> bool:
    test_path = root / expected_file
    test_node = _python_function_defs(test_path).get(test_name)
    if test_node is None:
        return False
    return any(
        _decorator_qualified_name(decorator) == "pytest.mark.slow"
        for decorator in test_node.decorator_list
    )


def _decorator_qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_qualified_name(node.func)
    if isinstance(node, ast.Attribute):
        parent = _decorator_qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _ci_selection_mentions(test_name: str, script_selection: str) -> bool:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", script_selection))
    return test_name in tokens or test_name.removeprefix("test_") in tokens


def _script_command_tuples(script_module: Any) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    command = _coerce_command_tuple(getattr(script_module, "COMMAND", None))
    if command:
        commands.append(command)
    script_commands = getattr(script_module, "COMMANDS", ())
    if isinstance(script_commands, (tuple, list)):
        for script_command in script_commands:
            command = _coerce_command_tuple(script_command)
            if command:
                commands.append(command)
    return tuple(commands)


def _coerce_command_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        return ()
    if not all(isinstance(item, str) for item in value):
        return ()
    return tuple(value)


def _command_references(command: tuple[str, ...], token: str) -> bool:
    return token in command


def _command_option_value(command: tuple[str, ...], option: str) -> str:
    for index in range(len(command) - 2, -1, -1):
        if command[index] == option:
            return command[index + 1]
    return ""


def _command_k_selection_covers(
    command: tuple[str, ...],
    test_names: tuple[str, ...],
) -> bool:
    selection = _command_option_value(command, "-k")
    return bool(selection) and all(
        _ci_selection_mentions(test_name, selection)
        for test_name in test_names
    )


def _fast_certificate_checker_command_covers(
    command: tuple[str, ...],
    fast_tc6_names: tuple[str, ...],
) -> bool:
    if not _command_references(command, "tests/test_certificate_checker.py"):
        return False
    if _command_option_value(command, "-m") != "not slow":
        return False
    selection = _command_option_value(command, "-k")
    return bool(
        not selection
        or all(
            _ci_selection_mentions(test_name, selection)
            for test_name in fast_tc6_names
        )
    )


def _slow_certificate_checker_command_covers(
    command: tuple[str, ...],
    slow_tc6_names: tuple[str, ...],
) -> bool:
    if not _command_references(command, "tests/test_certificate_checker.py"):
        return False
    if _command_option_value(command, "-m") != "slow":
        return False
    return _command_k_selection_covers(command, slow_tc6_names)


def _load_script_module(script_name: str, root: Path) -> Any:
    module_path = root / "scripts" / script_name
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"_public_audit_{script_name.removesuffix('.py')}",
        module_path,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_repo_path(path: str | Path, root: Path | None = None) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return (root or _project_root()) / raw_path


def _sha256_for_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _path_is_repo_local_public_review_artifact(path: Path) -> bool:
    parts = path.parts
    return any(
        parts[index : index + 2] == ("docs", "public-review")
        for index in range(max(len(parts) - 1, 0))
    )


def _text_has_no_public_review_placeholders(text: str) -> bool:
    lower_text = text.lower()
    return not any(marker in lower_text for marker in PUBLIC_REVIEW_PLACEHOLDER_MARKERS)


@dataclass(frozen=True)
class PublicReviewArtifactEvidence:
    """One resolved TC4-TC6 public-review or machine-check audit artifact."""

    artifact_id: str
    artifact_path: str
    sha256: str
    audit_scope: str
    proof_references: tuple[str, ...]
    covered_tc_items: tuple[str, ...]
    checked_local_artifact_ids: tuple[str, ...]
    reviewer_or_verifier_id: str
    review_result: str
    no_placeholder_text_certified: bool
    artifact_kind: str
    theorem_id: str = "public_review_artifact_evidence"

    @property
    def resolved_path(self) -> Path:
        return _resolve_repo_path(self.artifact_path)

    @property
    def text(self) -> str:
        return _read_text_file(self.resolved_path)

    @property
    def artifact_exists(self) -> bool:
        return self.resolved_path.is_file()

    @property
    def artifact_is_not_proof_note(self) -> bool:
        proof_note = _resolve_repo_path(TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT)
        try:
            return self.resolved_path.resolve() != proof_note.resolve()
        except OSError:
            return False

    @property
    def sha256_matches_file(self) -> bool:
        return bool(self.sha256 and _sha256_for_file(self.resolved_path) == self.sha256)

    @property
    def no_placeholder_text(self) -> bool:
        return bool(
            self.no_placeholder_text_certified is True
            and self.artifact_exists
            and _text_has_no_public_review_placeholders(self.text)
        )

    @property
    def artifact_kind_certified(self) -> bool:
        return self.artifact_kind in PUBLIC_REVIEW_ARTIFACT_KINDS

    @property
    def artifact_provenance_truthful(self) -> bool:
        if self.artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE:
            return self.reviewer_or_verifier_id == PUBLIC_REVIEW_VERIFIER_ID
        if self.artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL:
            return bool(
                self.reviewer_or_verifier_id != PUBLIC_REVIEW_VERIFIER_ID
                and not _path_is_repo_local_public_review_artifact(self.resolved_path)
            )
        return False

    @property
    def metadata_present_in_text(self) -> bool:
        text = self.text
        required = (
            self.artifact_id,
            self.artifact_kind,
            self.reviewer_or_verifier_id,
            self.review_result,
            self.audit_scope,
            *self.proof_references,
            *self.covered_tc_items,
            *self.checked_local_artifact_ids,
        )
        return bool(text and all(item in text for item in required if item))

    @property
    def verified_result_declared(self) -> bool:
        return self.review_result in {"accepted", "verified"}

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_review_artifact_theorem_id",
                certified=self.theorem_id == "public_review_artifact_evidence",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_id_supplied",
                certified=bool(self.artifact_id),
                source=self.theorem_id,
                detail=self.artifact_id,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_path_exists",
                certified=self.artifact_exists,
                source=self.theorem_id,
                detail=self.artifact_path,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_not_proof_note_self_reference",
                certified=self.artifact_is_not_proof_note,
                source=self.theorem_id,
                detail=self.artifact_path,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_sha256_matches",
                certified=self.sha256_matches_file,
                source=self.theorem_id,
                detail=self.sha256,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_kind",
                certified=self.artifact_kind_certified,
                source=self.theorem_id,
                detail=self.artifact_kind,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_provenance_truthful",
                certified=self.artifact_provenance_truthful,
                source=self.theorem_id,
                detail=(
                    f"kind={self.artifact_kind}; "
                    f"reviewer_or_verifier_id={self.reviewer_or_verifier_id}; "
                    f"path={self.artifact_path}"
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_verified_result",
                certified=self.verified_result_declared,
                source=self.theorem_id,
                detail=self.review_result,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_metadata_present",
                certified=self.metadata_present_in_text,
                source=self.theorem_id,
                detail=self.artifact_path,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_no_placeholder_text",
                certified=self.no_placeholder_text,
                source=self.theorem_id,
                detail=self.artifact_path,
            ),
        )

    @property
    def proof_certified(self) -> bool:
        return _pipeline_obligation_ledger_certified(self.obligations)

    @property
    def certified(self) -> bool:
        return self.proof_certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_review_artifact",
        )


@dataclass(frozen=True)
class PublicReviewArtifactResolutionCertificate:
    """Resolver for the TC4-TC6 public-review/machine-check artifacts."""

    tc4_artifact: PublicReviewArtifactEvidence
    tc5_artifact: PublicReviewArtifactEvidence
    tc6_artifact: PublicReviewArtifactEvidence
    required_artifact_ids: tuple[str, ...]
    manifest_resolution_certificate: PublicAuditManifestResolutionCertificate
    theorem_id: str = "public_review_artifact_resolution"

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return (
            self.tc4_artifact.artifact_id
            if type(self.tc4_artifact) is PublicReviewArtifactEvidence
            else "",
            self.tc5_artifact.artifact_id
            if type(self.tc5_artifact) is PublicReviewArtifactEvidence
            else "",
            self.tc6_artifact.artifact_id
            if type(self.tc6_artifact) is PublicReviewArtifactEvidence
            else "",
        )

    @property
    def artifact_kind(self) -> str:
        kinds = {
            artifact.artifact_kind
            for artifact in (self.tc4_artifact, self.tc5_artifact, self.tc6_artifact)
            if type(artifact) is PublicReviewArtifactEvidence
        }
        if len(kinds) == 1:
            return next(iter(kinds))
        return "mixed"

    @property
    def machine_checked_public_audit_resolved(self) -> bool:
        return bool(
            self.proof_certified
            and self.artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE
        )

    @property
    def external_public_review_resolved(self) -> bool:
        return bool(
            self.proof_certified
            and self.artifact_kind == PUBLIC_REVIEW_ARTIFACT_KIND_EXTERNAL
        )

    @property
    def artifact_types_certified(self) -> bool:
        return bool(
            type(self.tc4_artifact) is PublicReviewArtifactEvidence
            and type(self.tc5_artifact) is PublicReviewArtifactEvidence
            and type(self.tc6_artifact) is PublicReviewArtifactEvidence
        )

    @property
    def artifact_id_manifest_certified(self) -> bool:
        return bool(
            _matches_exact_manifest(
                self.required_artifact_ids,
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
            )
            and self.artifact_ids == self.required_artifact_ids
            and len(set(self.artifact_ids)) == 3
        )

    @property
    def manifest_resolution_certified(self) -> bool:
        return bool(
            type(
                self.manifest_resolution_certificate,
            )
            is PublicAuditManifestResolutionCertificate
            and self.manifest_resolution_certificate.proof_certified is True
        )

    def _artifact_scope_certified(
        self,
        artifact: PublicReviewArtifactEvidence,
        *,
        artifact_id: str,
        tc_item: str,
        proof_references: tuple[str, ...],
        checked_local_artifact_ids: tuple[str, ...],
    ) -> bool:
        return bool(
            type(artifact) is PublicReviewArtifactEvidence
            and artifact.artifact_id == artifact_id
            and artifact.proof_certified is True
            and _contains_all(artifact.covered_tc_items, (tc_item,))
            and _contains_all(artifact.proof_references, proof_references)
            and _contains_all(
                artifact.checked_local_artifact_ids,
                checked_local_artifact_ids,
            )
        )

    @property
    def tc4_artifact_scope_certified(self) -> bool:
        return self._artifact_scope_certified(
            self.tc4_artifact,
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[0],
            tc_item="TC4",
            proof_references=PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
            checked_local_artifact_ids=(
                PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
            ),
        )

    @property
    def tc5_artifact_scope_certified(self) -> bool:
        return self._artifact_scope_certified(
            self.tc5_artifact,
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[1],
            tc_item="TC5",
            proof_references=PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
            checked_local_artifact_ids=PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
        )

    @property
    def tc6_artifact_scope_certified(self) -> bool:
        return self._artifact_scope_certified(
            self.tc6_artifact,
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[2],
            tc_item="TC6",
            proof_references=PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
            checked_local_artifact_ids=(
                PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
                + PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
            ),
        )

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_review_artifact_resolution_theorem_id",
                certified=self.theorem_id == "public_review_artifact_resolution",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_types",
                certified=self.artifact_types_certified,
                source=self.theorem_id,
                detail=",".join(type(item).__name__ for item in (
                    self.tc4_artifact,
                    self.tc5_artifact,
                    self.tc6_artifact,
                )),
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_id_manifest",
                certified=self.artifact_id_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="public_review_manifest_resolution",
                certified=self.manifest_resolution_certified,
                source=type(self.manifest_resolution_certificate).__name__,
                detail=",".join(
                    getattr(
                        self.manifest_resolution_certificate,
                        "missing_obligations",
                        (),
                    )
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_review_tc4_artifact_scope",
                certified=self.tc4_artifact_scope_certified,
                source=type(self.tc4_artifact).__name__,
                detail=",".join(
                    getattr(self.tc4_artifact, "missing_obligations", ())
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_review_tc5_artifact_scope",
                certified=self.tc5_artifact_scope_certified,
                source=type(self.tc5_artifact).__name__,
                detail=",".join(
                    getattr(self.tc5_artifact, "missing_obligations", ())
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_review_tc6_artifact_scope",
                certified=self.tc6_artifact_scope_certified,
                source=type(self.tc6_artifact).__name__,
                detail=",".join(
                    getattr(self.tc6_artifact, "missing_obligations", ())
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_review_artifact_kind_truthful",
                certified=self.artifact_kind in PUBLIC_REVIEW_ARTIFACT_KINDS,
                source=self.theorem_id,
                detail=self.artifact_kind,
            ),
        )

    @property
    def proof_certified(self) -> bool:
        return _pipeline_obligation_ledger_certified(self.obligations)

    @property
    def certified(self) -> bool:
        return self.proof_certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        nested: list[str] = []
        for prefix, artifact in (
            ("tc4", self.tc4_artifact),
            ("tc5", self.tc5_artifact),
            ("tc6", self.tc6_artifact),
        ):
            if type(artifact) is not PublicReviewArtifactEvidence:
                nested.append(f"{prefix}:artifact_type")
                continue
            nested.extend(
                f"{prefix}:{obligation}"
                for obligation in artifact.missing_obligations
            )
        return tuple(
            dict.fromkeys(
                (
                    *_pipeline_obligation_ledger_missing(
                        self.obligations,
                        ledger_name="public_review_artifact_resolution",
                    ),
                    *nested,
                )
            )
        )

    def resolves(
        self,
        external_public_review_artifact_ids: tuple[str, ...],
        proof_references: tuple[str, ...],
        tc4_evidence: Any,
        tc5_evidence: Any,
        tc6_evidence: Any,
    ) -> bool:
        return bool(
            self.proof_certified
            and tuple(external_public_review_artifact_ids)
            == self.required_artifact_ids
            and _matches_exact_manifest(
                proof_references,
                PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
            )
            and type(tc4_evidence) is PublicReducedHyperbolicityAuditEvidence
            and type(tc5_evidence) is PublicGeneralizedFuchsianEntryAuditEvidence
            and type(tc6_evidence) is PublicCauchyMajorantAuditEvidence
            and tc4_evidence.proof_certified is True
            and tc5_evidence.proof_certified is True
            and tc6_evidence.proof_certified is True
            and tuple(tc4_evidence.spectrum_audit_reference_ids)
            == PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS
            and tuple(tc5_evidence.constructor_artifact_ids)
            == PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS
            and tuple(tc6_evidence.checker_artifact_ids)
            == PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
        )


def certify_public_review_artifact_evidence(
    *,
    artifact_id: str,
    artifact_path: str | Path,
    audit_scope: str,
    proof_references: tuple[str, ...],
    covered_tc_items: tuple[str, ...],
    checked_local_artifact_ids: tuple[str, ...],
    reviewer_or_verifier_id: str = PUBLIC_REVIEW_VERIFIER_ID,
    review_result: str = "verified",
    artifact_kind: str = PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE,
    project_root: str | Path | None = None,
) -> PublicReviewArtifactEvidence:
    root = Path(project_root) if project_root is not None else _project_root()
    resolved_path = _resolve_repo_path(artifact_path, root)
    return PublicReviewArtifactEvidence(
        artifact_id=str(artifact_id),
        artifact_path=str(resolved_path),
        sha256=_sha256_for_file(resolved_path),
        audit_scope=str(audit_scope),
        proof_references=tuple(str(item) for item in proof_references),
        covered_tc_items=tuple(str(item) for item in covered_tc_items),
        checked_local_artifact_ids=tuple(
            str(item) for item in checked_local_artifact_ids
        ),
        reviewer_or_verifier_id=str(reviewer_or_verifier_id),
        review_result=str(review_result),
        no_placeholder_text_certified=True,
        artifact_kind=str(artifact_kind),
    )


def certify_public_review_artifact_resolution(
    *,
    tc4_artifact: PublicReviewArtifactEvidence | None = None,
    tc5_artifact: PublicReviewArtifactEvidence | None = None,
    tc6_artifact: PublicReviewArtifactEvidence | None = None,
    required_artifact_ids: tuple[str, ...] = (
        PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
    ),
    manifest_resolution_certificate: (
        PublicAuditManifestResolutionCertificate | None
    ) = None,
    project_root: str | Path | None = None,
    artifact_kind: str = PUBLIC_REVIEW_ARTIFACT_KIND_MACHINE,
) -> PublicReviewArtifactResolutionCertificate:
    root = Path(project_root) if project_root is not None else _project_root()
    if manifest_resolution_certificate is None:
        manifest_resolution_certificate = certify_public_audit_manifest_resolution(
            project_root=root,
        )
    if tc4_artifact is None:
        tc4_artifact = certify_public_review_artifact_evidence(
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[0],
            artifact_path=PUBLIC_REVIEW_ARTIFACT_PATHS[
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[0]
            ],
            audit_scope="TC4 reduced hyperbolicity line audit",
            proof_references=PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
            covered_tc_items=("TC4",),
            checked_local_artifact_ids=PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
            artifact_kind=artifact_kind,
            project_root=root,
        )
    if tc5_artifact is None:
        tc5_artifact = certify_public_review_artifact_evidence(
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[1],
            artifact_path=PUBLIC_REVIEW_ARTIFACT_PATHS[
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[1]
            ],
            audit_scope="TC5 generalized Fuchsian entry line audit",
            proof_references=PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
            covered_tc_items=("TC5",),
            checked_local_artifact_ids=PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
            artifact_kind=artifact_kind,
            project_root=root,
        )
    if tc6_artifact is None:
        tc6_artifact = certify_public_review_artifact_evidence(
            artifact_id=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[2],
            artifact_path=PUBLIC_REVIEW_ARTIFACT_PATHS[
                PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS[2]
            ],
            audit_scope="TC6 Cauchy majorant backend line audit",
            proof_references=PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
            covered_tc_items=("TC6",),
            checked_local_artifact_ids=(
                PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS
                + PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
            ),
            artifact_kind=artifact_kind,
            project_root=root,
        )
    return PublicReviewArtifactResolutionCertificate(
        tc4_artifact=tc4_artifact,
        tc5_artifact=tc5_artifact,
        tc6_artifact=tc6_artifact,
        required_artifact_ids=tuple(str(item) for item in required_artifact_ids),
        manifest_resolution_certificate=manifest_resolution_certificate,
    )


@dataclass(frozen=True)
class PublicReducedHyperbolicityAuditEvidence:
    """Standalone public audit evidence for TC4."""

    positive_mass_domain_declared: bool = False
    quotient_coordinates_declared: bool = False
    translations_removed: bool = False
    scale_rotation_reflection_quotiented: bool = False
    central_target_family_covered: bool = False
    mass_metric_hessian_spectrum_audited: bool = False
    zero_modes_removed: bool = False
    stable_unstable_splitting_certified: bool = False
    central_target_families: tuple[str, ...] = ()
    quotient_modes_removed: tuple[str, ...] = ()
    spectrum_audit_reference_ids: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,)
    proof_references: tuple[str, ...] = ()
    theorem_id: str = "public_tc4_reduced_hyperbolicity_audit"

    @property
    def central_target_manifest_certified(self) -> bool:
        return _contains_all(
            self.central_target_families,
            PUBLIC_TC4_REQUIRED_CENTRAL_TARGET_FAMILIES,
        )

    @property
    def quotient_mode_manifest_certified(self) -> bool:
        return _contains_all(
            self.quotient_modes_removed,
            PUBLIC_TC4_REQUIRED_QUOTIENT_MODES,
        )

    @property
    def spectrum_audit_reference_manifest_certified(self) -> bool:
        return _contains_all(
            self.spectrum_audit_reference_ids,
            PUBLIC_TC4_REQUIRED_SPECTRUM_AUDIT_REFERENCE_IDS,
        )

    @property
    def proof_reference_manifest_certified(self) -> bool:
        return _contains_all(
            self.proof_references,
            PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
        )

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_tc4_reduced_hyperbolicity_audit_theorem_id",
                certified=self.theorem_id == "public_tc4_reduced_hyperbolicity_audit",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="tc4_positive_mass_domain_declared",
                certified=self.positive_mass_domain_declared,
                source=self.theorem_id,
                detail="positive-mass central targets are the stated domain",
            ),
            TheoremPipelineObligation(
                obligation="tc4_quotient_coordinates_declared",
                certified=self.quotient_coordinates_declared,
                source=self.theorem_id,
                detail="normal-form coordinates on reduced shape space are declared",
            ),
            TheoremPipelineObligation(
                obligation="tc4_translations_removed",
                certified=self.translations_removed,
                source=self.theorem_id,
                detail="center-of-mass translations are removed",
            ),
            TheoremPipelineObligation(
                obligation="tc4_scale_rotation_reflection_quotiented",
                certified=self.scale_rotation_reflection_quotiented,
                source=self.theorem_id,
                detail="scale, rotation, and reflection modes are quotiented",
            ),
            TheoremPipelineObligation(
                obligation="tc4_central_target_family_covered",
                certified=self.central_target_family_covered,
                source=self.theorem_id,
                detail="Euler and Lagrange positive-mass central targets are covered",
            ),
            TheoremPipelineObligation(
                obligation="tc4_central_target_family_manifest",
                certified=self.central_target_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.central_target_families),
            ),
            TheoremPipelineObligation(
                obligation="tc4_quotient_mode_manifest",
                certified=self.quotient_mode_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.quotient_modes_removed),
            ),
            TheoremPipelineObligation(
                obligation="tc4_mass_metric_hessian_spectrum_audited",
                certified=self.mass_metric_hessian_spectrum_audited,
                source=self.theorem_id,
                detail="mass-metric Hessian spectrum is audited on the quotient",
            ),
            TheoremPipelineObligation(
                obligation="tc4_spectrum_audit_reference_ids_supplied",
                certified=bool(self.spectrum_audit_reference_ids),
                source=self.theorem_id,
                detail=",".join(self.spectrum_audit_reference_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc4_spectrum_audit_reference_manifest",
                certified=self.spectrum_audit_reference_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.spectrum_audit_reference_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc4_zero_modes_removed",
                certified=self.zero_modes_removed,
                source=self.theorem_id,
                detail="only symmetry zero modes are removed",
            ),
            TheoremPipelineObligation(
                obligation="tc4_stable_unstable_splitting_certified",
                certified=self.stable_unstable_splitting_certified,
                source=self.theorem_id,
                detail="reduced stable/unstable hyperbolic splitting is certified",
            ),
            TheoremPipelineObligation(
                obligation="tc4_source_document",
                certified=TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT in self.source_documents,
                source=self.theorem_id,
                detail=",".join(self.source_documents),
            ),
            TheoremPipelineObligation(
                obligation="tc4_public_proof_references_supplied",
                certified=bool(self.proof_references),
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="tc4_public_proof_reference_manifest",
                certified=self.proof_reference_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_tc4_reduced_hyperbolicity_audit"
            and _pipeline_obligation_ledger_certified(self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_tc4_reduced_hyperbolicity_audit",
        )


@dataclass(frozen=True)
class PublicGeneralizedFuchsianEntryAuditEvidence:
    """Standalone public audit evidence for TC5."""

    normal_form_coordinates_declared: bool = False
    exponent_conventions_declared: bool = False
    resonance_lattice_declared: bool = False
    finite_log_degree_rule_declared: bool = False
    denominator_projector_rule_declared: bool = False
    triangular_solve_order_declared: bool = False
    finite_selector_data_extraction_audited: bool = False
    arbitrary_incoming_germ_scope_declared: bool = False
    normal_form_coordinate_ids: tuple[str, ...] = ()
    resonance_rule_ids: tuple[str, ...] = ()
    triangular_order_keys: tuple[str, ...] = ()
    selector_data_fields: tuple[str, ...] = ()
    constructor_artifact_ids: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,)
    proof_references: tuple[str, ...] = ()
    theorem_id: str = "public_tc5_generalized_fuchsian_entry_audit"

    @property
    def normal_form_coordinate_manifest_certified(self) -> bool:
        return _contains_all(
            self.normal_form_coordinate_ids,
            PUBLIC_TC5_REQUIRED_NORMAL_FORM_COORDINATES,
        )

    @property
    def resonance_rule_manifest_certified(self) -> bool:
        return _contains_all(
            self.resonance_rule_ids,
            PUBLIC_TC5_REQUIRED_RESONANCE_RULES,
        )

    @property
    def triangular_order_manifest_certified(self) -> bool:
        return _contains_all(
            self.triangular_order_keys,
            PUBLIC_TC5_REQUIRED_TRIANGULAR_ORDER_KEYS,
        )

    @property
    def selector_data_manifest_certified(self) -> bool:
        return _contains_all(
            self.selector_data_fields,
            PUBLIC_TC5_REQUIRED_SELECTOR_DATA_FIELDS,
        )

    @property
    def constructor_artifact_manifest_certified(self) -> bool:
        return _contains_all(
            self.constructor_artifact_ids,
            PUBLIC_TC5_REQUIRED_CONSTRUCTOR_ARTIFACT_IDS,
        )

    @property
    def proof_reference_manifest_certified(self) -> bool:
        return _contains_all(
            self.proof_references,
            PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
        )

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_tc5_generalized_fuchsian_entry_audit_theorem_id",
                certified=self.theorem_id == (
                    "public_tc5_generalized_fuchsian_entry_audit"
                ),
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="tc5_normal_form_coordinates_declared",
                certified=self.normal_form_coordinates_declared,
                source=self.theorem_id,
                detail="stable normal-form coordinates are declared",
            ),
            TheoremPipelineObligation(
                obligation="tc5_normal_form_coordinate_manifest",
                certified=self.normal_form_coordinate_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.normal_form_coordinate_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc5_exponent_conventions_declared",
                certified=self.exponent_conventions_declared,
                source=self.theorem_id,
                detail="fractional/irrational exponent conventions are explicit",
            ),
            TheoremPipelineObligation(
                obligation="tc5_resonance_lattice_declared",
                certified=self.resonance_lattice_declared,
                source=self.theorem_id,
                detail="resonance lattice and admissible multiindices are declared",
            ),
            TheoremPipelineObligation(
                obligation="tc5_resonance_rule_manifest",
                certified=self.resonance_rule_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.resonance_rule_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc5_finite_log_degree_rule_declared",
                certified=self.finite_log_degree_rule_declared,
                source=self.theorem_id,
                detail="finite log-degree bound is stated for resonant rows",
            ),
            TheoremPipelineObligation(
                obligation="tc5_denominator_projector_rule_declared",
                certified=self.denominator_projector_rule_declared,
                source=self.theorem_id,
                detail="small-denominator/projector rule for resonances is explicit",
            ),
            TheoremPipelineObligation(
                obligation="tc5_triangular_solve_order_declared",
                certified=self.triangular_solve_order_declared,
                source=self.theorem_id,
                detail="solve order over weight, row, and log degree is declared",
            ),
            TheoremPipelineObligation(
                obligation="tc5_triangular_order_manifest",
                certified=self.triangular_order_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.triangular_order_keys),
            ),
            TheoremPipelineObligation(
                obligation="tc5_finite_selector_data_extraction_audited",
                certified=self.finite_selector_data_extraction_audited,
                source=self.theorem_id,
                detail="arbitrary incoming germs produce finite selector data",
            ),
            TheoremPipelineObligation(
                obligation="tc5_selector_data_manifest",
                certified=self.selector_data_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.selector_data_fields),
            ),
            TheoremPipelineObligation(
                obligation="tc5_constructor_artifact_manifest",
                certified=self.constructor_artifact_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.constructor_artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc5_arbitrary_incoming_germ_scope_declared",
                certified=self.arbitrary_incoming_germ_scope_declared,
                source=self.theorem_id,
                detail="exact incoming-germ scope is stated separately from interval boxes",
            ),
            TheoremPipelineObligation(
                obligation="tc5_source_document",
                certified=TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT in self.source_documents,
                source=self.theorem_id,
                detail=",".join(self.source_documents),
            ),
            TheoremPipelineObligation(
                obligation="tc5_public_proof_references_supplied",
                certified=bool(self.proof_references),
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="tc5_public_proof_reference_manifest",
                certified=self.proof_reference_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_tc5_generalized_fuchsian_entry_audit"
            and _pipeline_obligation_ledger_certified(self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_tc5_generalized_fuchsian_entry_audit",
        )


@dataclass(frozen=True)
class PublicCauchyMajorantAuditEvidence:
    """Standalone public audit evidence for TC6."""

    cauchy_polydisc_declared: bool = False
    primitive_constants_declared: bool = False
    defect_bound: float = math.inf
    right_inverse_bound: float = math.inf
    nonlinear_lipschitz_bound: float = math.inf
    majorant_radius: float = 0.0
    proof_grade_arithmetic_backend_audited: bool = False
    cauchy_polydisc_id: str = ""
    primitive_constant_names: tuple[str, ...] = ()
    majorant_norm_id: str = ""
    proof_grade_backend_id: str = ""
    checker_artifact_ids: tuple[str, ...] = ()
    checker_id: str = ""
    checker_certificate_id: str = ""
    checker_obligation_ids: tuple[str, ...] = ()
    majorant_component_ids: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,)
    proof_references: tuple[str, ...] = ()
    theorem_id: str = "public_tc6_cauchy_majorant_audit"

    @property
    def constants_finite(self) -> bool:
        values = (
            self.defect_bound,
            self.right_inverse_bound,
            self.nonlinear_lipschitz_bound,
            self.majorant_radius,
        )
        return bool(
            all(math.isfinite(float(value)) for value in values)
            and self.defect_bound >= 0.0
            and self.right_inverse_bound > 0.0
            and self.nonlinear_lipschitz_bound >= 0.0
            and self.majorant_radius > 0.0
        )

    @property
    def contraction_factor(self) -> float:
        return float(self.right_inverse_bound * self.nonlinear_lipschitz_bound)

    @property
    def self_map_left_side(self) -> float:
        return float(
            self.right_inverse_bound * self.defect_bound
            + self.contraction_factor * self.majorant_radius
        )

    @property
    def contraction_inequality_certified(self) -> bool:
        return bool(self.constants_finite and self.contraction_factor < 1.0)

    @property
    def self_map_inequality_certified(self) -> bool:
        return bool(
            self.constants_finite
            and self.self_map_left_side
            <= self.majorant_radius * (1.0 + 1.0e-12) + 1.0e-30
        )

    @property
    def primitive_constant_manifest_certified(self) -> bool:
        return _contains_all(
            self.primitive_constant_names,
            PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES,
        )

    @property
    def cauchy_polydisc_manifest_certified(self) -> bool:
        return self.cauchy_polydisc_id == PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID

    @property
    def majorant_norm_manifest_certified(self) -> bool:
        return self.majorant_norm_id == PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID

    @property
    def proof_grade_backend_manifest_certified(self) -> bool:
        return (
            self.proof_grade_backend_id
            == PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID
        )

    @property
    def checker_artifact_manifest_certified(self) -> bool:
        return _contains_all(
            self.checker_artifact_ids,
            PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
        )

    @property
    def checker_id_manifest_certified(self) -> bool:
        return self.checker_id == PUBLIC_TC6_REQUIRED_CHECKER_ID

    @property
    def checker_certificate_manifest_certified(self) -> bool:
        return bool(self.checker_certificate_id)

    @property
    def checker_obligation_manifest_certified(self) -> bool:
        return _contains_all(
            self.checker_obligation_ids,
            PUBLIC_TC6_REQUIRED_CHECKER_OBLIGATION_IDS,
        )

    @property
    def majorant_component_manifest_certified(self) -> bool:
        return _contains_all(
            self.majorant_component_ids,
            PUBLIC_TC6_REQUIRED_MAJORANT_COMPONENT_IDS,
        )

    @property
    def proof_reference_manifest_certified(self) -> bool:
        return _contains_all(
            self.proof_references,
            PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
        )

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_tc6_cauchy_majorant_audit_theorem_id",
                certified=self.theorem_id == "public_tc6_cauchy_majorant_audit",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_cauchy_polydisc_declared",
                certified=self.cauchy_polydisc_declared,
                source=self.theorem_id,
                detail="closed Cauchy polydisc and norm are declared",
            ),
            TheoremPipelineObligation(
                obligation="tc6_cauchy_polydisc_id_supplied",
                certified=self.cauchy_polydisc_manifest_certified,
                source=self.theorem_id,
                detail=self.cauchy_polydisc_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_primitive_constants_declared",
                certified=self.primitive_constants_declared,
                source=self.theorem_id,
                detail="primitive constants (C0,Lambda,sigma,p0,d) are declared",
            ),
            TheoremPipelineObligation(
                obligation="tc6_primitive_constant_manifest",
                certified=self.primitive_constant_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.primitive_constant_names),
            ),
            TheoremPipelineObligation(
                obligation="tc6_majorant_constants_finite",
                certified=self.constants_finite,
                source=self.theorem_id,
                detail=(
                    f"D={self.defect_bound:g}; B={self.right_inverse_bound:g}; "
                    f"L={self.nonlinear_lipschitz_bound:g}; "
                    f"R={self.majorant_radius:g}"
                ),
            ),
            TheoremPipelineObligation(
                obligation="tc6_banach_contraction_inequality",
                certified=self.contraction_inequality_certified,
                source=self.theorem_id,
                detail=f"B*L={self.contraction_factor:g}",
            ),
            TheoremPipelineObligation(
                obligation="tc6_banach_self_map_inequality",
                certified=self.self_map_inequality_certified,
                source=self.theorem_id,
                detail=f"B*D+B*L*R={self.self_map_left_side:g}",
            ),
            TheoremPipelineObligation(
                obligation="tc6_proof_grade_arithmetic_backend_audited",
                certified=self.proof_grade_arithmetic_backend_audited,
                source=self.theorem_id,
                detail="Cauchy constants are checked in proof-grade arithmetic",
            ),
            TheoremPipelineObligation(
                obligation="tc6_majorant_norm_id_supplied",
                certified=self.majorant_norm_manifest_certified,
                source=self.theorem_id,
                detail=self.majorant_norm_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_proof_grade_backend_id_supplied",
                certified=self.proof_grade_backend_manifest_certified,
                source=self.theorem_id,
                detail=self.proof_grade_backend_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_checker_artifact_manifest",
                certified=self.checker_artifact_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.checker_artifact_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc6_checker_id_manifest",
                certified=self.checker_id_manifest_certified,
                source=self.theorem_id,
                detail=self.checker_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_checker_certificate_manifest",
                certified=self.checker_certificate_manifest_certified,
                source=self.theorem_id,
                detail=self.checker_certificate_id,
            ),
            TheoremPipelineObligation(
                obligation="tc6_checker_obligation_manifest",
                certified=self.checker_obligation_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.checker_obligation_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc6_majorant_component_manifest",
                certified=self.majorant_component_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.majorant_component_ids),
            ),
            TheoremPipelineObligation(
                obligation="tc6_source_document",
                certified=TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT in self.source_documents,
                source=self.theorem_id,
                detail=",".join(self.source_documents),
            ),
            TheoremPipelineObligation(
                obligation="tc6_public_proof_references_supplied",
                certified=bool(self.proof_references),
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="tc6_public_proof_reference_manifest",
                certified=self.proof_reference_manifest_certified,
                source=self.theorem_id,
                detail=",".join(self.proof_references),
            ),
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_tc6_cauchy_majorant_audit"
            and _pipeline_obligation_ledger_certified(self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_tc6_cauchy_majorant_audit",
        )


@dataclass(frozen=True)
class PublicTotalCollisionProofAuditCertificate:
    """Public audit status for the TC1-TC7 total-collision proof chain."""

    tc1_zero_angular_audited: bool = False
    tc2_binary_degenerate_exclusion_audited: bool = False
    tc3_central_shape_limit_audited: bool = False
    tc4_reduced_hyperbolicity_audited: bool = False
    tc5_generalized_fuchsian_entry_audited: bool = False
    tc6_cauchy_majorant_constants_audited: bool = False
    tc7_total_stop_chart_soundness_audited: bool = False
    tc4_reduced_hyperbolicity_evidence: (
        PublicReducedHyperbolicityAuditEvidence | None
    ) = None
    tc5_generalized_fuchsian_entry_evidence: (
        PublicGeneralizedFuchsianEntryAuditEvidence | None
    ) = None
    tc6_cauchy_majorant_constants_evidence: (
        PublicCauchyMajorantAuditEvidence | None
    ) = None
    source_documents: tuple[str, ...] = ()
    proof_references: tuple[str, ...] = ()
    checker_artifacts: tuple[str, ...] = ()
    external_public_review_artifact_ids: tuple[str, ...] = ()
    local_manifest_resolution_certificate: (
        PublicAuditManifestResolutionCertificate | None
    ) = None
    external_public_review_resolution_certificate: (
        PublicReviewArtifactResolutionCertificate | None
    ) = None
    theorem_id: str = "public_total_collision_proof_audit"

    @property
    def proof_reference_manifest_certified(self) -> bool:
        return _contains_all(
            self.proof_references,
            PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
        )

    @property
    def checker_artifact_manifest_certified(self) -> bool:
        return _contains_all(
            self.checker_artifacts,
            PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS,
        )

    @property
    def external_public_review_artifact_manifest_supplied(self) -> bool:
        return _contains_all(
            self.external_public_review_artifact_ids,
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
        )

    @property
    def external_public_review_artifact_manifest_certified(self) -> bool:
        resolver = self.external_public_review_resolution_certificate
        return bool(
            type(resolver) is PublicReviewArtifactResolutionCertificate
            and resolver.resolves(
                self.external_public_review_artifact_ids,
                self.proof_references,
                self.tc4_reduced_hyperbolicity_evidence,
                self.tc5_generalized_fuchsian_entry_evidence,
                self.tc6_cauchy_majorant_constants_evidence,
            )
        )

    @property
    def public_review_artifact_manifest_supplied(self) -> bool:
        """Neutral alias for the legacy external-review artifact manifest gate."""

        return self.external_public_review_artifact_manifest_supplied

    @property
    def public_review_artifact_manifest_certified(self) -> bool:
        """Neutral alias for typed public-review or machine-check resolution."""

        return self.external_public_review_artifact_manifest_certified

    @property
    def public_review_artifact_kind(self) -> str | None:
        if (
            type(self.external_public_review_resolution_certificate)
            is PublicReviewArtifactResolutionCertificate
        ):
            return self.external_public_review_resolution_certificate.artifact_kind
        return None

    @property
    def public_review_resolution_is_external(self) -> bool:
        resolver = self.external_public_review_resolution_certificate
        return bool(
            type(resolver) is PublicReviewArtifactResolutionCertificate
            and resolver.external_public_review_resolved
        )

    @property
    def machine_checked_public_audit_resolved(self) -> bool:
        resolver = self.external_public_review_resolution_certificate
        return bool(
            type(resolver) is PublicReviewArtifactResolutionCertificate
            and resolver.machine_checked_public_audit_resolved
        )

    @property
    def local_manifest_resolution_certified(self) -> bool:
        return bool(
            type(
                self.local_manifest_resolution_certificate,
            )
            is PublicAuditManifestResolutionCertificate
            and self.local_manifest_resolution_certificate.proof_certified is True
        )

    @property
    def tc4_evidence_certified(self) -> bool:
        return bool(
            type(
                self.tc4_reduced_hyperbolicity_evidence,
            )
            is PublicReducedHyperbolicityAuditEvidence
            and self.tc4_reduced_hyperbolicity_evidence.proof_certified is True
        )

    @property
    def tc5_evidence_certified(self) -> bool:
        return bool(
            type(
                self.tc5_generalized_fuchsian_entry_evidence,
            )
            is PublicGeneralizedFuchsianEntryAuditEvidence
            and self.tc5_generalized_fuchsian_entry_evidence.proof_certified is True
        )

    @property
    def tc6_evidence_certified(self) -> bool:
        return bool(
            type(
                self.tc6_cauchy_majorant_constants_evidence,
            )
            is PublicCauchyMajorantAuditEvidence
            and self.tc6_cauchy_majorant_constants_evidence.proof_certified is True
        )

    @property
    def tc4_manifest_artifact_ids(self) -> tuple[str, ...]:
        if type(
            self.tc4_reduced_hyperbolicity_evidence,
        ) is PublicReducedHyperbolicityAuditEvidence:
            return tuple(
                self.tc4_reduced_hyperbolicity_evidence.spectrum_audit_reference_ids
            )
        return ()

    @property
    def tc5_manifest_artifact_ids(self) -> tuple[str, ...]:
        if type(
            self.tc5_generalized_fuchsian_entry_evidence,
        ) is PublicGeneralizedFuchsianEntryAuditEvidence:
            return tuple(
                self.tc5_generalized_fuchsian_entry_evidence.constructor_artifact_ids
            )
        return ()

    @property
    def tc6_manifest_artifact_ids(self) -> tuple[str, ...]:
        if type(
            self.tc6_cauchy_majorant_constants_evidence,
        ) is PublicCauchyMajorantAuditEvidence:
            return tuple(
                self.tc6_cauchy_majorant_constants_evidence.checker_artifact_ids
            )
        return ()

    @property
    def local_manifest_matches_current_fields(self) -> bool:
        manifest = self.local_manifest_resolution_certificate
        return bool(
            type(manifest) is PublicAuditManifestResolutionCertificate
            and tuple(manifest.proof_references) == tuple(self.proof_references)
            and tuple(manifest.tc4_spectrum_audit_reference_ids)
            == self.tc4_manifest_artifact_ids
            and tuple(manifest.tc5_constructor_artifact_ids)
            == self.tc5_manifest_artifact_ids
            and tuple(manifest.tc6_checker_artifact_ids)
            == self.tc6_manifest_artifact_ids
            and tuple(manifest.total_collision_checker_artifacts)
            == tuple(self.checker_artifacts)
        )

    @property
    def obligations(self) -> tuple[TheoremPipelineObligation, ...]:
        return (
            TheoremPipelineObligation(
                obligation="public_total_collision_proof_audit_theorem_id",
                certified=self.theorem_id == "public_total_collision_proof_audit",
                source=self.theorem_id,
                detail=self.theorem_id,
            ),
            TheoremPipelineObligation(
                obligation="tc1_zero_angular_audited",
                certified=self.tc1_zero_angular_audited,
                source="public_total_collision_proof_audit",
                detail="total collision implies zero centered angular momentum",
            ),
            TheoremPipelineObligation(
                obligation="tc2_binary_degenerate_exclusion_audited",
                certified=self.tc2_binary_degenerate_exclusion_audited,
                source="public_total_collision_proof_audit",
                detail="binary-degenerate normalized-shape approach is impossible",
            ),
            TheoremPipelineObligation(
                obligation="tc3_central_shape_limit_audited",
                certified=self.tc3_central_shape_limit_audited,
                source="public_total_collision_proof_audit",
                detail="total-collision branches have central-configuration limits",
            ),
            TheoremPipelineObligation(
                obligation="tc4_reduced_hyperbolicity_audited",
                certified=(
                    self.tc4_reduced_hyperbolicity_audited is True
                    and self.tc4_evidence_certified
                ),
                source="public_total_collision_proof_audit",
                detail=(
                    "reduced central targets are hyperbolic after quotienting "
                    "translations, scale, rotations, and reflection; "
                    f"evidence={type(self.tc4_reduced_hyperbolicity_evidence).__name__}"
                ),
            ),
            TheoremPipelineObligation(
                obligation="tc5_generalized_fuchsian_entry_audited",
                certified=(
                    self.tc5_generalized_fuchsian_entry_audited is True
                    and self.tc5_evidence_certified
                ),
                source="public_total_collision_proof_audit",
                detail=(
                    "stable branch supplies finite generalized Fuchsian/Puiseux-log "
                    "selector data with explicit resonance rules; "
                    f"evidence={type(self.tc5_generalized_fuchsian_entry_evidence).__name__}"
                ),
            ),
            TheoremPipelineObligation(
                obligation="tc6_cauchy_majorant_constants_audited",
                certified=(
                    self.tc6_cauchy_majorant_constants_audited is True
                    and self.tc6_evidence_certified
                ),
                source="public_total_collision_proof_audit",
                detail=(
                    "Cauchy-majorant constants are extracted with proof-grade "
                    "arithmetic and explicit polydisc hypotheses; "
                    f"evidence={type(self.tc6_cauchy_majorant_constants_evidence).__name__}"
                ),
            ),
            TheoremPipelineObligation(
                obligation="tc7_total_stop_chart_soundness_audited",
                certified=self.tc7_total_stop_chart_soundness_audited,
                source="public_total_collision_proof_audit",
                detail="supplied generalized entry data produce a checked total-stop chart",
            ),
            TheoremPipelineObligation(
                obligation="public_audit_source_document",
                certified=TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT in self.source_documents,
                source="public_total_collision_proof_audit",
                detail=",".join(self.source_documents),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_proof_references_supplied",
                certified=bool(self.proof_references),
                source="public_total_collision_proof_audit",
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_proof_reference_manifest",
                certified=self.proof_reference_manifest_certified,
                source="public_total_collision_proof_audit",
                detail=",".join(self.proof_references),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_checker_artifact_manifest",
                certified=self.checker_artifact_manifest_certified,
                source="public_total_collision_proof_audit",
                detail=",".join(self.checker_artifacts),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_local_manifest_resolution",
                certified=self.local_manifest_resolution_certified,
                source=type(self.local_manifest_resolution_certificate).__name__,
                detail=(
                    "missing="
                    + ",".join(
                        getattr(
                            self.local_manifest_resolution_certificate,
                            "missing_obligations",
                            (),
                        )
                    )
                ),
            ),
            TheoremPipelineObligation(
                obligation="public_audit_local_manifest_matches_current_fields",
                certified=self.local_manifest_matches_current_fields,
                source=type(self.local_manifest_resolution_certificate).__name__,
                detail=(
                    "local manifest resolution must replay the audit's current "
                    "proof/checker/evidence fields"
                ),
            ),
        )

    @property
    def local_audit_package_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_total_collision_proof_audit"
            and _pipeline_obligation_ledger_certified(self.obligations)
        )

    @property
    def public_proof_certified(self) -> bool:
        return bool(
            self.local_audit_package_certified
            and self.public_review_artifact_manifest_certified
        )

    @property
    def certified(self) -> bool:
        return self.public_proof_certified

    @property
    def public_audit_blockers(self) -> tuple[str, ...]:
        high_level = _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_total_collision_proof_audit",
        )
        nested = (
            self._nested_evidence_blockers(
                "tc4",
                self.tc4_reduced_hyperbolicity_audited,
                self.tc4_reduced_hyperbolicity_evidence,
            )
            + self._nested_evidence_blockers(
                "tc5",
                self.tc5_generalized_fuchsian_entry_audited,
                self.tc5_generalized_fuchsian_entry_evidence,
            )
            + self._nested_evidence_blockers(
                "tc6",
                self.tc6_cauchy_majorant_constants_audited,
                self.tc6_cauchy_majorant_constants_evidence,
            )
        )
        external_review = ()
        if self.local_audit_package_certified:
            if not self.public_review_artifact_manifest_supplied:
                external_review = ("public_review_artifact_manifest",)
            elif not self.public_review_artifact_manifest_certified:
                external_review = (
                    "public_review_artifact_verification",
                )
        return tuple(dict.fromkeys((*high_level, *nested, *external_review)))

    @staticmethod
    def _nested_evidence_blockers(
        prefix: str,
        audit_flag: bool,
        evidence: Any,
    ) -> tuple[str, ...]:
        if audit_flag is not True:
            return ()
        expected_types = {
            "tc4": PublicReducedHyperbolicityAuditEvidence,
            "tc5": PublicGeneralizedFuchsianEntryAuditEvidence,
            "tc6": PublicCauchyMajorantAuditEvidence,
        }
        expected_type = expected_types.get(prefix)
        if evidence is None:
            return (f"{prefix}:evidence_supplied",)
        if expected_type is not None and type(evidence) is not expected_type:
            return (f"{prefix}:evidence_type",)
        if getattr(evidence, "proof_certified", False) is True:
            return ()
        missing = getattr(evidence, "missing_obligations", ())
        if not missing:
            return (f"{prefix}:evidence_proof_certified",)
        return tuple(f"{prefix}:{obligation}" for obligation in missing)


def certify_public_reduced_hyperbolicity_audit_evidence(
    *,
    positive_mass_domain_declared: bool = False,
    quotient_coordinates_declared: bool = False,
    translations_removed: bool = False,
    scale_rotation_reflection_quotiented: bool = False,
    central_target_family_covered: bool = False,
    mass_metric_hessian_spectrum_audited: bool = False,
    zero_modes_removed: bool = False,
    stable_unstable_splitting_certified: bool = False,
    central_target_families: tuple[str, ...] = (),
    quotient_modes_removed: tuple[str, ...] = (),
    spectrum_audit_reference_ids: tuple[str, ...] = (),
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (),
) -> PublicReducedHyperbolicityAuditEvidence:
    """Build the standalone TC4 public-audit evidence object."""

    return PublicReducedHyperbolicityAuditEvidence(
        positive_mass_domain_declared=_strict_bool(
            positive_mass_domain_declared,
            "positive_mass_domain_declared",
        ),
        quotient_coordinates_declared=_strict_bool(
            quotient_coordinates_declared,
            "quotient_coordinates_declared",
        ),
        translations_removed=_strict_bool(
            translations_removed,
            "translations_removed",
        ),
        scale_rotation_reflection_quotiented=_strict_bool(
            scale_rotation_reflection_quotiented,
            "scale_rotation_reflection_quotiented",
        ),
        central_target_family_covered=_strict_bool(
            central_target_family_covered,
            "central_target_family_covered",
        ),
        mass_metric_hessian_spectrum_audited=_strict_bool(
            mass_metric_hessian_spectrum_audited,
            "mass_metric_hessian_spectrum_audited",
        ),
        zero_modes_removed=_strict_bool(
            zero_modes_removed,
            "zero_modes_removed",
        ),
        stable_unstable_splitting_certified=_strict_bool(
            stable_unstable_splitting_certified,
            "stable_unstable_splitting_certified",
        ),
        central_target_families=tuple(str(item) for item in central_target_families),
        quotient_modes_removed=tuple(str(item) for item in quotient_modes_removed),
        spectrum_audit_reference_ids=tuple(
            str(item) for item in spectrum_audit_reference_ids
        ),
        source_documents=tuple(str(item) for item in source_documents),
        proof_references=tuple(str(item) for item in proof_references),
    )


def certify_public_generalized_fuchsian_entry_audit_evidence(
    *,
    normal_form_coordinates_declared: bool = False,
    exponent_conventions_declared: bool = False,
    resonance_lattice_declared: bool = False,
    finite_log_degree_rule_declared: bool = False,
    denominator_projector_rule_declared: bool = False,
    triangular_solve_order_declared: bool = False,
    finite_selector_data_extraction_audited: bool = False,
    arbitrary_incoming_germ_scope_declared: bool = False,
    normal_form_coordinate_ids: tuple[str, ...] = (),
    resonance_rule_ids: tuple[str, ...] = (),
    triangular_order_keys: tuple[str, ...] = (),
    selector_data_fields: tuple[str, ...] = (),
    constructor_artifact_ids: tuple[str, ...] = (),
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (),
) -> PublicGeneralizedFuchsianEntryAuditEvidence:
    """Build the standalone TC5 public-audit evidence object."""

    return PublicGeneralizedFuchsianEntryAuditEvidence(
        normal_form_coordinates_declared=_strict_bool(
            normal_form_coordinates_declared,
            "normal_form_coordinates_declared",
        ),
        exponent_conventions_declared=_strict_bool(
            exponent_conventions_declared,
            "exponent_conventions_declared",
        ),
        resonance_lattice_declared=_strict_bool(
            resonance_lattice_declared,
            "resonance_lattice_declared",
        ),
        finite_log_degree_rule_declared=_strict_bool(
            finite_log_degree_rule_declared,
            "finite_log_degree_rule_declared",
        ),
        denominator_projector_rule_declared=_strict_bool(
            denominator_projector_rule_declared,
            "denominator_projector_rule_declared",
        ),
        triangular_solve_order_declared=_strict_bool(
            triangular_solve_order_declared,
            "triangular_solve_order_declared",
        ),
        finite_selector_data_extraction_audited=_strict_bool(
            finite_selector_data_extraction_audited,
            "finite_selector_data_extraction_audited",
        ),
        arbitrary_incoming_germ_scope_declared=_strict_bool(
            arbitrary_incoming_germ_scope_declared,
            "arbitrary_incoming_germ_scope_declared",
        ),
        normal_form_coordinate_ids=tuple(
            str(item) for item in normal_form_coordinate_ids
        ),
        resonance_rule_ids=tuple(str(item) for item in resonance_rule_ids),
        triangular_order_keys=tuple(str(item) for item in triangular_order_keys),
        selector_data_fields=tuple(str(item) for item in selector_data_fields),
        constructor_artifact_ids=tuple(str(item) for item in constructor_artifact_ids),
        source_documents=tuple(str(item) for item in source_documents),
        proof_references=tuple(str(item) for item in proof_references),
    )


def certify_public_cauchy_majorant_audit_evidence(
    *,
    cauchy_polydisc_declared: bool = False,
    primitive_constants_declared: bool = False,
    defect_bound: float = math.inf,
    right_inverse_bound: float = math.inf,
    nonlinear_lipschitz_bound: float = math.inf,
    majorant_radius: float = 0.0,
    proof_grade_arithmetic_backend_audited: bool = False,
    cauchy_polydisc_id: str = "",
    primitive_constant_names: tuple[str, ...] = (),
    majorant_norm_id: str = "",
    proof_grade_backend_id: str = "",
    checker_artifact_ids: tuple[str, ...] = (),
    checker_id: str = "",
    checker_certificate_id: str = "",
    checker_obligation_ids: tuple[str, ...] = (),
    majorant_component_ids: tuple[str, ...] = (),
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (),
) -> PublicCauchyMajorantAuditEvidence:
    """Build the standalone TC6 audit evidence and check its Banach bounds."""

    return PublicCauchyMajorantAuditEvidence(
        cauchy_polydisc_declared=_strict_bool(
            cauchy_polydisc_declared,
            "cauchy_polydisc_declared",
        ),
        primitive_constants_declared=_strict_bool(
            primitive_constants_declared,
            "primitive_constants_declared",
        ),
        defect_bound=float(defect_bound),
        right_inverse_bound=float(right_inverse_bound),
        nonlinear_lipschitz_bound=float(nonlinear_lipschitz_bound),
        majorant_radius=float(majorant_radius),
        proof_grade_arithmetic_backend_audited=_strict_bool(
            proof_grade_arithmetic_backend_audited,
            "proof_grade_arithmetic_backend_audited",
        ),
        cauchy_polydisc_id=str(cauchy_polydisc_id),
        primitive_constant_names=tuple(
            str(item) for item in primitive_constant_names
        ),
        majorant_norm_id=str(majorant_norm_id),
        proof_grade_backend_id=str(proof_grade_backend_id),
        checker_artifact_ids=tuple(str(item) for item in checker_artifact_ids),
        checker_id=str(checker_id),
        checker_certificate_id=str(checker_certificate_id),
        checker_obligation_ids=tuple(str(item) for item in checker_obligation_ids),
        majorant_component_ids=tuple(str(item) for item in majorant_component_ids),
        source_documents=tuple(str(item) for item in source_documents),
        proof_references=tuple(str(item) for item in proof_references),
    )


def certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
    certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
    *,
    cauchy_polydisc_id: str = PUBLIC_TC6_REQUIRED_CAUCHY_POLYDISC_ID,
    primitive_constant_names: tuple[str, ...] = (
        PUBLIC_TC6_REQUIRED_PRIMITIVE_CONSTANT_NAMES
    ),
    majorant_norm_id: str = PUBLIC_TC6_REQUIRED_MAJORANT_NORM_ID,
    proof_grade_backend_id: str = PUBLIC_TC6_REQUIRED_PROOF_GRADE_BACKEND_ID,
    checker_artifact_ids: tuple[str, ...] = PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
) -> PublicCauchyMajorantAuditEvidence:
    """Derive public TC6 local audit evidence from the independent checker.

    This constructor ties the Banach constants in the public audit surface to
    the serialized generalized-Fuchsian stop chart accepted by the checker.
    The audit can still remain externally open, but local TC6 evidence no
    longer certifies from free-floating majorant constants.
    """

    if not isinstance(certificate, TotalCollisionGeneralizedFuchsianStopChartCertificate):
        raise TypeError(
            "certificate must be a TotalCollisionGeneralizedFuchsianStopChartCertificate",
        )

    result = check_total_collision_generalized_fuchsian_stop_chart(certificate)
    majorant = certificate.remainder_majorant
    checker_accepts_certificate = result.certified is True
    certified_obligation_ids = tuple(
        obligation.obligation
        for obligation in result.obligations
        if obligation.certified is True
    )
    majorant_component_ids = (
        tuple(component for component, _ in majorant.component_inputs)
        if majorant is not None
        else ()
    )

    return certify_public_cauchy_majorant_audit_evidence(
        cauchy_polydisc_declared=checker_accepts_certificate,
        primitive_constants_declared=checker_accepts_certificate,
        defect_bound=(
            float(majorant.defect_bound) if majorant is not None else math.inf
        ),
        right_inverse_bound=(
            float(majorant.linear_inverse_bound) if majorant is not None else math.inf
        ),
        nonlinear_lipschitz_bound=(
            float(majorant.nonlinear_lipschitz_bound)
            if majorant is not None
            else math.inf
        ),
        majorant_radius=(
            float(majorant.remainder_ball_radius)
            if majorant is not None
            else 0.0
        ),
        proof_grade_arithmetic_backend_audited=checker_accepts_certificate,
        cauchy_polydisc_id=cauchy_polydisc_id,
        primitive_constant_names=primitive_constant_names,
        majorant_norm_id=majorant_norm_id,
        proof_grade_backend_id=proof_grade_backend_id,
        checker_artifact_ids=checker_artifact_ids,
        checker_id=result.checker_id,
        checker_certificate_id=result.certificate_id,
        checker_obligation_ids=certified_obligation_ids,
        majorant_component_ids=majorant_component_ids,
        source_documents=source_documents,
        proof_references=proof_references,
    )


def certify_public_reduced_hyperbolicity_audit_evidence_from_manifest(
    *,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
) -> PublicReducedHyperbolicityAuditEvidence:
    """Build the complete local TC4 evidence from the public-audit manifest."""

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
        source_documents=source_documents,
        proof_references=proof_references,
    )


def certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest(
    *,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
) -> PublicGeneralizedFuchsianEntryAuditEvidence:
    """Build the complete local TC5 evidence from the public-audit manifest."""

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
        source_documents=source_documents,
        proof_references=proof_references,
    )


@dataclass(frozen=True)
class ReviewReadyTotalCollisionAuditEvidenceBundle:
    """Reusable complete local TC4-TC6 evidence for a total-stop chart."""

    checked_stop_chart_certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate
    tc4_reduced_hyperbolicity_evidence: PublicReducedHyperbolicityAuditEvidence
    tc5_generalized_fuchsian_entry_evidence: PublicGeneralizedFuchsianEntryAuditEvidence
    tc6_cauchy_majorant_constants_evidence: PublicCauchyMajorantAuditEvidence
    source_documents: tuple[str, ...]
    proof_references: tuple[str, ...]

    @property
    def local_audit_evidence_certified(self) -> bool:
        return bool(
            type(
                self.checked_stop_chart_certificate,
            )
            is TotalCollisionGeneralizedFuchsianStopChartCertificate
            and type(
                self.tc4_reduced_hyperbolicity_evidence,
            )
            is PublicReducedHyperbolicityAuditEvidence
            and self.tc4_reduced_hyperbolicity_evidence.proof_certified is True
            and type(
                self.tc5_generalized_fuchsian_entry_evidence,
            )
            is PublicGeneralizedFuchsianEntryAuditEvidence
            and self.tc5_generalized_fuchsian_entry_evidence.proof_certified is True
            and type(
                self.tc6_cauchy_majorant_constants_evidence,
            )
            is PublicCauchyMajorantAuditEvidence
            and self.tc6_cauchy_majorant_constants_evidence.proof_certified is True
            and (
                self.tc6_cauchy_majorant_constants_evidence.checker_certificate_id
                == self.checked_stop_chart_certificate.certificate_id
            )
            and TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT in self.source_documents
            and _contains_all(
                self.proof_references,
                PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
            )
        )


def build_review_ready_total_collision_audit_evidence_bundle(
    checked_stop_chart_certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
    *,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (
        PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES
    ),
) -> ReviewReadyTotalCollisionAuditEvidenceBundle:
    """Build complete reusable local TC4-TC6 audit evidence once."""

    if type(
        checked_stop_chart_certificate,
    ) is not TotalCollisionGeneralizedFuchsianStopChartCertificate:
        raise TypeError(
            "checked_stop_chart_certificate must be a "
            "TotalCollisionGeneralizedFuchsianStopChartCertificate",
        )
    source_documents = tuple(str(item) for item in source_documents)
    proof_references = tuple(str(item) for item in proof_references)
    tc4 = certify_public_reduced_hyperbolicity_audit_evidence_from_manifest(
        source_documents=source_documents,
        proof_references=PUBLIC_TC4_REQUIRED_PROOF_REFERENCES,
    )
    tc5 = certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest(
        source_documents=source_documents,
        proof_references=PUBLIC_TC5_REQUIRED_PROOF_REFERENCES,
    )
    tc6 = certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(
        checked_stop_chart_certificate,
        source_documents=source_documents,
        proof_references=PUBLIC_TC6_REQUIRED_PROOF_REFERENCES,
    )
    return ReviewReadyTotalCollisionAuditEvidenceBundle(
        checked_stop_chart_certificate=checked_stop_chart_certificate,
        tc4_reduced_hyperbolicity_evidence=tc4,
        tc5_generalized_fuchsian_entry_evidence=tc5,
        tc6_cauchy_majorant_constants_evidence=tc6,
        source_documents=source_documents,
        proof_references=proof_references,
    )


def certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    evidence_bundle: ReviewReadyTotalCollisionAuditEvidenceBundle,
    *,
    external_public_review_artifact_ids: tuple[str, ...] = (),
    public_review_artifact_ids: tuple[str, ...] = (),
    public_review_resolution_certificate: (
        PublicReviewArtifactResolutionCertificate | None
    ) = None,
) -> PublicTotalCollisionProofAuditCertificate:
    """Construct the review package from already-computed TC4-TC6 evidence."""

    if type(evidence_bundle) is not ReviewReadyTotalCollisionAuditEvidenceBundle:
        raise TypeError(
            "evidence_bundle must be a "
            "ReviewReadyTotalCollisionAuditEvidenceBundle",
        )
    if (
        public_review_resolution_certificate is not None
        and not external_public_review_artifact_ids
        and not public_review_artifact_ids
    ):
        public_review_artifact_ids = (
            PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS
        )
    return certify_public_total_collision_proof_audit(
        tc4_reduced_hyperbolicity_audited=True,
        tc5_generalized_fuchsian_entry_audited=True,
        tc6_cauchy_majorant_constants_audited=True,
        tc4_reduced_hyperbolicity_evidence=(
            evidence_bundle.tc4_reduced_hyperbolicity_evidence
        ),
        tc5_generalized_fuchsian_entry_evidence=(
            evidence_bundle.tc5_generalized_fuchsian_entry_evidence
        ),
        tc6_cauchy_majorant_constants_evidence=(
            evidence_bundle.tc6_cauchy_majorant_constants_evidence
        ),
        source_documents=evidence_bundle.source_documents,
        proof_references=evidence_bundle.proof_references,
        external_public_review_artifact_ids=external_public_review_artifact_ids,
        public_review_artifact_ids=public_review_artifact_ids,
        public_review_resolution_certificate=(
            public_review_resolution_certificate
        ),
    )


def certify_review_ready_total_collision_audit_package(
    checked_stop_chart_certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
    *,
    external_public_review_artifact_ids: tuple[str, ...] = (),
    public_review_artifact_ids: tuple[str, ...] = (),
    public_review_resolution_certificate: (
        PublicReviewArtifactResolutionCertificate | None
    ) = None,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (
        PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES
    ),
) -> PublicTotalCollisionProofAuditCertificate:
    """Construct the complete local TC4-TC6 public-audit package.

    TC4 and TC5 are replayed from the fixed public-audit manifests.  TC6 is
    derived from a checked generalized-Fuchsian stop-chart certificate, so the
    Banach constants and checker provenance are constructor-derived rather than
    copied from test helpers.  Without a public-review or machine-check
    artifact resolver, the returned certificate is deliberately only
    review-ready local evidence.
    """

    evidence_bundle = build_review_ready_total_collision_audit_evidence_bundle(
        checked_stop_chart_certificate,
        source_documents=source_documents,
        proof_references=proof_references,
    )
    return certify_review_ready_total_collision_audit_package_from_evidence_bundle(
        evidence_bundle,
        external_public_review_artifact_ids=external_public_review_artifact_ids,
        public_review_artifact_ids=public_review_artifact_ids,
        public_review_resolution_certificate=(
            public_review_resolution_certificate
        ),
    )


@dataclass(frozen=True)
class PublicRegularizedAtlasClosedFormProofCertificate:
    """Public audit wrapper for the internally closed pointwise route."""

    internal_closed_form_theorem: Any
    total_collision_audit: Any
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "public_regularized_atlas_closed_form_proof"

    @property
    def internal_proof_certified(self) -> bool:
        return bool(
            type(
                self.internal_closed_form_theorem,
            )
            is PointwiseRegularizedAtlasClosedFormTheoremCertificate
            and self.internal_closed_form_theorem.proof_certified is True
        )

    @property
    def obligation_manifest_certified(self) -> bool:
        return _pipeline_obligation_manifest_exact(
            self.obligations,
            PUBLIC_REGULARIZED_ATLAS_PROOF_REQUIRED_OBLIGATIONS,
        )

    @property
    def public_proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_regularized_atlas_closed_form_proof"
            and self.internal_proof_certified
            and type(
                self.total_collision_audit,
            )
            is PublicTotalCollisionProofAuditCertificate
            and self.total_collision_audit.public_proof_certified is True
            and _pipeline_obligation_ledger_certified(self.obligations)
            and self.obligation_manifest_certified
            and not self.field_consistency_blockers
        )

    @property
    def certified(self) -> bool:
        return self.public_proof_certified

    @property
    def public_audit_blockers(self) -> tuple[str, ...]:
        fields = self.field_consistency_blockers
        own = _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_regularized_atlas_closed_form_proof",
        )
        nested = (
            self.total_collision_audit.public_audit_blockers
            if type(
                self.total_collision_audit,
            )
            is PublicTotalCollisionProofAuditCertificate
            else ()
        )
        return tuple(dict.fromkeys((*fields, *own, *nested)))

    @property
    def field_consistency_blockers(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.theorem_id != "public_regularized_atlas_closed_form_proof":
            missing.append("public_regularized_route_theorem_id_field")
        if not self.internal_proof_certified:
            missing.append("public_regularized_internal_theorem_field")
        if type(
            self.total_collision_audit,
        ) is not PublicTotalCollisionProofAuditCertificate:
            missing.append("public_regularized_total_collision_audit_field")
        if not self.obligation_manifest_certified:
            missing.append("public_regularized_obligation_manifest")
        return tuple(dict.fromkeys(missing))

    @property
    def route_summary(self) -> str:
        if self.public_proof_certified:
            return "public pointwise regularized-atlas closed-form proof audit certified"
        if self.internal_proof_certified:
            return (
                "internal pointwise closed-form theorem is proof-certified; "
                "public total-collision audit remains open"
            )
        return "public proof audit is missing the internal pointwise theorem certificate"


@dataclass(frozen=True)
class PublicGeneralClosedFormSolutionCertificate:
    """Top-level public audit wrapper for the regularized-atlas target."""

    requested_class: str
    internal_general_solution_certificate: Any
    public_regularized_atlas_proof: PublicRegularizedAtlasClosedFormProofCertificate
    pointwise_closed_form_theorem: Any
    require_derived_gates: bool
    obligations: tuple[TheoremPipelineObligation, ...]
    theorem_id: str = "public_general_closed_form_solution_target"

    @property
    def internal_proof_certified(self) -> bool:
        return bool(
            type(
                self.internal_general_solution_certificate,
            )
            is GeneralClosedFormSolutionCertificate
            and self.internal_general_solution_certificate.proof_certified is True
        )

    @property
    def obligation_manifest_certified(self) -> bool:
        return _pipeline_obligation_manifest_exact(
            self.obligations,
            PUBLIC_GENERAL_CLOSED_FORM_REQUIRED_OBLIGATIONS,
        )

    @property
    def public_proof_certified(self) -> bool:
        return bool(
            self.theorem_id == "public_general_closed_form_solution_target"
            and self.internal_proof_certified
            and type(
                self.public_regularized_atlas_proof,
            )
            is PublicRegularizedAtlasClosedFormProofCertificate
            and self.public_regularized_atlas_proof.public_proof_certified is True
            and self.public_route_source_matches_pointwise_theorem
            and _pipeline_obligation_ledger_certified(self.obligations)
            and self.obligation_manifest_certified
            and not self.field_consistency_blockers
        )

    @property
    def certified(self) -> bool:
        return self.public_proof_certified

    @property
    def public_audit_blockers(self) -> tuple[str, ...]:
        fields = self.field_consistency_blockers
        own = _pipeline_obligation_ledger_missing(
            self.obligations,
            ledger_name="public_general_closed_form_solution_target",
        )
        if type(
            self.public_regularized_atlas_proof,
        ) is PublicRegularizedAtlasClosedFormProofCertificate:
            nested = self.public_regularized_atlas_proof.public_audit_blockers
        else:
            nested = ("public_regularized_atlas_proof_type",)
        return tuple(dict.fromkeys((*fields, *own, *nested)))

    @property
    def field_consistency_blockers(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.theorem_id != "public_general_closed_form_solution_target":
            missing.append("public_general_theorem_id_field")
        if not _public_requested_class_is_regularized_atlas(self.requested_class):
            missing.append("public_requested_class_field")
        if type(self.require_derived_gates) is not bool:
            missing.append("public_require_derived_gates_bool_field")
        elif self.require_derived_gates is not True:
            missing.append("public_require_derived_gates_field")
        if not self.internal_proof_certified:
            missing.append("public_internal_general_solution_certificate_field")
        if type(
            self.public_regularized_atlas_proof,
        ) is not PublicRegularizedAtlasClosedFormProofCertificate:
            missing.append("public_regularized_atlas_proof_type")
        if not self.public_route_source_matches_pointwise_theorem:
            missing.append("public_regularized_atlas_proof_source_matches_pointwise")
        if not self.obligation_manifest_certified:
            missing.append("public_general_obligation_manifest")
        return tuple(dict.fromkeys(missing))

    @property
    def public_route_source_matches_pointwise_theorem(self) -> bool:
        return bool(
            type(
                self.pointwise_closed_form_theorem,
            )
            is PointwiseRegularizedAtlasClosedFormTheoremCertificate
            and self.pointwise_closed_form_theorem.proof_certified is True
            and type(
                self.public_regularized_atlas_proof,
            )
            is PublicRegularizedAtlasClosedFormProofCertificate
            and (
                self.public_regularized_atlas_proof.internal_closed_form_theorem
                is self.pointwise_closed_form_theorem
            )
        )

    @property
    def route_summary(self) -> str:
        if self.public_proof_certified:
            return "public general closed-form solution target certified"
        if self.internal_proof_certified:
            return (
                "internal general closed-form target is proof-certified; "
                "public proof audit remains open"
            )
        return "public general closed-form target is missing internal proof closure"


def certify_public_total_collision_proof_audit(
    *,
    tc1_zero_angular_audited: bool = True,
    tc2_binary_degenerate_exclusion_audited: bool = True,
    tc3_central_shape_limit_audited: bool = True,
    tc4_reduced_hyperbolicity_audited: bool = False,
    tc5_generalized_fuchsian_entry_audited: bool = False,
    tc6_cauchy_majorant_constants_audited: bool = False,
    tc7_total_stop_chart_soundness_audited: bool = True,
    tc4_reduced_hyperbolicity_evidence: (
        PublicReducedHyperbolicityAuditEvidence | None
    ) = None,
    tc5_generalized_fuchsian_entry_evidence: (
        PublicGeneralizedFuchsianEntryAuditEvidence | None
    ) = None,
    tc6_cauchy_majorant_constants_evidence: (
        PublicCauchyMajorantAuditEvidence | None
    ) = None,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = (),
    checker_artifacts: tuple[str, ...] = (
        PUBLIC_TOTAL_COLLISION_REQUIRED_CHECKER_ARTIFACTS
    ),
    external_public_review_artifact_ids: tuple[str, ...] = (),
    public_review_artifact_ids: tuple[str, ...] = (),
    local_manifest_resolution_certificate: (
        PublicAuditManifestResolutionCertificate | None
    ) = None,
    external_public_review_resolution_certificate: (
        PublicReviewArtifactResolutionCertificate | None
    ) = None,
    public_review_resolution_certificate: (
        PublicReviewArtifactResolutionCertificate | None
    ) = None,
) -> PublicTotalCollisionProofAuditCertificate:
    """Record public-audit status for the TC1-TC7 proof note.

    The defaults encode the current project stance: TC1-TC3 and TC7 have
    documented/checker-backed audit evidence, while TC4-TC6 remain the public
    mathematical audit frontier.  Supplying external review artifact ids only
    records that the named obligations were supplied; public closure still
    needs a future resolver/verifier for those artifacts.
    """

    proof_references = tuple(str(item) for item in proof_references)
    checker_artifacts = tuple(str(item) for item in checker_artifacts)
    legacy_public_review_artifact_ids = tuple(
        str(item) for item in external_public_review_artifact_ids
    )
    neutral_public_review_artifact_ids = tuple(
        str(item) for item in public_review_artifact_ids
    )
    if neutral_public_review_artifact_ids:
        if (
            legacy_public_review_artifact_ids
            and neutral_public_review_artifact_ids
            != legacy_public_review_artifact_ids
        ):
            external_public_review_artifact_ids = (
                "__conflicting_public_review_artifact_ids__",
            )
        else:
            external_public_review_artifact_ids = neutral_public_review_artifact_ids
    else:
        external_public_review_artifact_ids = legacy_public_review_artifact_ids
    if public_review_resolution_certificate is not None:
        if (
            external_public_review_resolution_certificate is not None
            and public_review_resolution_certificate
            is not external_public_review_resolution_certificate
        ):
            external_public_review_resolution_certificate = None
        else:
            external_public_review_resolution_certificate = (
                public_review_resolution_certificate
            )
    tc4_artifacts = (
        tc4_reduced_hyperbolicity_evidence.spectrum_audit_reference_ids
        if type(
            tc4_reduced_hyperbolicity_evidence,
        )
        is PublicReducedHyperbolicityAuditEvidence
        else ()
    )
    tc5_artifacts = (
        tc5_generalized_fuchsian_entry_evidence.constructor_artifact_ids
        if type(
            tc5_generalized_fuchsian_entry_evidence,
        )
        is PublicGeneralizedFuchsianEntryAuditEvidence
        else ()
    )
    tc6_artifacts = (
        tc6_cauchy_majorant_constants_evidence.checker_artifact_ids
        if type(
            tc6_cauchy_majorant_constants_evidence,
        )
        is PublicCauchyMajorantAuditEvidence
        else ()
    )
    if local_manifest_resolution_certificate is None:
        local_manifest_resolution_certificate = (
            certify_public_audit_manifest_resolution(
                proof_references=proof_references,
                tc4_spectrum_audit_reference_ids=tc4_artifacts,
                tc5_constructor_artifact_ids=tc5_artifacts,
                tc6_checker_artifact_ids=tc6_artifacts,
                total_collision_checker_artifacts=checker_artifacts,
            )
        )
    return PublicTotalCollisionProofAuditCertificate(
        tc1_zero_angular_audited=_strict_bool(
            tc1_zero_angular_audited,
            "tc1_zero_angular_audited",
        ),
        tc2_binary_degenerate_exclusion_audited=_strict_bool(
            tc2_binary_degenerate_exclusion_audited,
            "tc2_binary_degenerate_exclusion_audited",
        ),
        tc3_central_shape_limit_audited=_strict_bool(
            tc3_central_shape_limit_audited,
            "tc3_central_shape_limit_audited",
        ),
        tc4_reduced_hyperbolicity_audited=_strict_bool(
            tc4_reduced_hyperbolicity_audited,
            "tc4_reduced_hyperbolicity_audited",
        ),
        tc5_generalized_fuchsian_entry_audited=_strict_bool(
            tc5_generalized_fuchsian_entry_audited,
            "tc5_generalized_fuchsian_entry_audited",
        ),
        tc6_cauchy_majorant_constants_audited=_strict_bool(
            tc6_cauchy_majorant_constants_audited,
            "tc6_cauchy_majorant_constants_audited",
        ),
        tc7_total_stop_chart_soundness_audited=_strict_bool(
            tc7_total_stop_chart_soundness_audited,
            "tc7_total_stop_chart_soundness_audited",
        ),
        tc4_reduced_hyperbolicity_evidence=tc4_reduced_hyperbolicity_evidence,
        tc5_generalized_fuchsian_entry_evidence=(
            tc5_generalized_fuchsian_entry_evidence
        ),
        tc6_cauchy_majorant_constants_evidence=(
            tc6_cauchy_majorant_constants_evidence
        ),
        source_documents=tuple(str(item) for item in source_documents),
        proof_references=proof_references,
        checker_artifacts=checker_artifacts,
        external_public_review_artifact_ids=external_public_review_artifact_ids,
        local_manifest_resolution_certificate=local_manifest_resolution_certificate,
        external_public_review_resolution_certificate=(
            external_public_review_resolution_certificate
        ),
    )


def certify_public_regularized_atlas_closed_form_proof(
    internal_closed_form_theorem: Any,
    *,
    total_collision_audit: Any | None = None,
) -> PublicRegularizedAtlasClosedFormProofCertificate:
    """Audit the public proof route without weakening internal theorem gates."""

    if total_collision_audit is None:
        total_collision_audit = certify_public_total_collision_proof_audit()
    total_collision_audit_typed = isinstance(
        total_collision_audit,
        PublicTotalCollisionProofAuditCertificate,
    )
    total_collision_audit_certified = bool(
        total_collision_audit_typed
        and total_collision_audit.public_proof_certified is True
    )
    total_collision_audit_detail = (
        ",".join(total_collision_audit.public_audit_blockers)
        if total_collision_audit_typed
        else "total_collision_audit must be a PublicTotalCollisionProofAuditCertificate"
    )
    pointwise_theorem = getattr(internal_closed_form_theorem, "pointwise_open_time_theorem", None)
    soundness = getattr(internal_closed_form_theorem, "certificate_language_soundness", None)
    enumeration = getattr(
        internal_closed_form_theorem,
        "computable_certificate_enumeration",
        None,
    )
    policy = getattr(
        internal_closed_form_theorem,
        "maximal_classical_total_collision_policy",
        None,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="internal_pointwise_closed_form_theorem_certified",
            certified=(
                isinstance(
                    internal_closed_form_theorem,
                    PointwiseRegularizedAtlasClosedFormTheoremCertificate,
                )
                and internal_closed_form_theorem.proof_certified is True
            ),
            source=type(internal_closed_form_theorem).__name__,
            detail=getattr(internal_closed_form_theorem, "route_summary", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="public_total_collision_proof_audit_certified",
            certified=total_collision_audit_certified,
            source=type(total_collision_audit).__name__,
            detail=total_collision_audit_detail,
        ),
        TheoremPipelineObligation(
            obligation="public_total_collision_proof_audit_type",
            certified=total_collision_audit_typed,
            source=type(total_collision_audit).__name__,
            detail="typed total-collision public audit certificate required",
        ),
        TheoremPipelineObligation(
            obligation="certificate_language_soundness_derived_from_checker_kernel",
            certified=(
                isinstance(soundness, CertificateLanguageSoundnessCertificate)
                and soundness.proof_certified is True
                and soundness.checker_kernel_derived is True
            ),
            source=type(soundness).__name__,
            detail=getattr(soundness, "derive_source", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="computable_enumeration_derived_from_same_pointwise_theorem",
            certified=_public_enumeration_matches_pointwise_theorem(
                enumeration,
                pointwise_theorem,
            ),
            source=type(enumeration).__name__,
            detail=getattr(enumeration, "witness_source", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="maximal_classical_policy_derived_from_pointwise_theorem",
            certified=(
                isinstance(
                    policy,
                    MaximalClassicalTotalCollisionPolicyCertificate,
                )
                and policy.proof_certified is True
                and policy.pointwise_theorem_derived is True
            ),
            source=type(policy).__name__,
            detail=getattr(policy, "policy_id", "missing"),
        ),
    )
    return PublicRegularizedAtlasClosedFormProofCertificate(
        internal_closed_form_theorem=internal_closed_form_theorem,
        total_collision_audit=total_collision_audit,
        obligations=obligations,
    )


def certify_public_general_closed_form_solution_target(
    closed_form_class: str = "regularized locally finite atlas",
    *,
    general_theorem_certificate: Any | None = None,
    certificate_language_soundness_certificate: (
        CertificateLanguageSoundnessCertificate | None
    ) = None,
    computable_atlas_enumeration_certificate: (
        ComputableAtlasCertificateEnumerationCertificate | None
    ) = None,
    total_collision_audit: Any | None = None,
    require_derived_gates: bool = True,
) -> PublicGeneralClosedFormSolutionCertificate:
    """Certify the public top-level route with derived-gate-only semantics."""

    require_derived_gates = _strict_bool(
        require_derived_gates,
        "require_derived_gates",
    )
    pointwise_closed_form_theorem = _pointwise_closed_form_theorem_for_public_route(
        general_theorem_certificate,
        certificate_language_soundness_certificate,
        computable_atlas_enumeration_certificate,
    )
    theorem_for_internal_route = (
        pointwise_closed_form_theorem
        if pointwise_closed_form_theorem is not None
        else general_theorem_certificate
    )
    internal_general = certify_general_closed_form_solution_target(
        closed_form_class,
        general_theorem_certificate=theorem_for_internal_route,
        certificate_language_soundness_certificate=(
            certificate_language_soundness_certificate
        ),
        computable_atlas_enumeration_certificate=(
            computable_atlas_enumeration_certificate
        ),
    )
    public_regularized = certify_public_regularized_atlas_closed_form_proof(
        pointwise_closed_form_theorem,
        total_collision_audit=total_collision_audit,
    )
    soundness = getattr(
        pointwise_closed_form_theorem,
        "certificate_language_soundness",
        certificate_language_soundness_certificate,
    )
    enumeration = getattr(
        pointwise_closed_form_theorem,
        "computable_certificate_enumeration",
        computable_atlas_enumeration_certificate,
    )
    pointwise_theorem = getattr(
        pointwise_closed_form_theorem,
        "pointwise_open_time_theorem",
        general_theorem_certificate,
    )
    derived_soundness = bool(
        isinstance(soundness, CertificateLanguageSoundnessCertificate)
        and soundness.proof_certified is True
        and soundness.checker_kernel_derived is True
    )
    derived_enumeration = _public_enumeration_matches_pointwise_theorem(
        enumeration,
        pointwise_theorem,
    )
    obligations = (
        TheoremPipelineObligation(
            obligation="internal_general_closed_form_solution_certified",
            certified=bool(
                isinstance(internal_general, GeneralClosedFormSolutionCertificate)
                and internal_general.proof_certified is True
            ),
            source=type(internal_general).__name__,
            detail=",".join(internal_general.blocking_obligations),
        ),
        TheoremPipelineObligation(
            obligation="public_regularized_atlas_proof_certified",
            certified=public_regularized.public_proof_certified is True,
            source=type(public_regularized).__name__,
            detail=",".join(public_regularized.public_audit_blockers),
        ),
        TheoremPipelineObligation(
            obligation="public_route_requires_derived_gates",
            certified=require_derived_gates is True,
            source="certify_public_general_closed_form_solution_target",
            detail=f"require_derived_gates={require_derived_gates}",
        ),
        TheoremPipelineObligation(
            obligation="public_certificate_language_soundness_derived",
            certified=derived_soundness,
            source=type(soundness).__name__,
            detail=getattr(soundness, "derive_source", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="public_computable_enumeration_derived",
            certified=derived_enumeration,
            source=type(enumeration).__name__,
            detail=getattr(enumeration, "witness_source", "missing"),
        ),
        TheoremPipelineObligation(
            obligation="raw_boolean_gates_not_public_evidence",
            certified=bool(derived_soundness and derived_enumeration),
            source="certify_public_general_closed_form_solution_target",
            detail=(
                "public closure requires checker-kernel-derived soundness and "
                "same-pointwise-theorem computable enumeration"
            ),
        ),
    )
    return PublicGeneralClosedFormSolutionCertificate(
        requested_class=str(closed_form_class),
        internal_general_solution_certificate=internal_general,
        public_regularized_atlas_proof=public_regularized,
        pointwise_closed_form_theorem=pointwise_closed_form_theorem,
        require_derived_gates=require_derived_gates,
        obligations=obligations,
    )


def _public_enumeration_matches_pointwise_theorem(
    enumeration: Any,
    pointwise_theorem: Any,
) -> bool:
    if not isinstance(
        enumeration,
        ComputableAtlasCertificateEnumerationCertificate,
    ) or not isinstance(
        pointwise_theorem,
        PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    ):
        return False
    finite_target_theorem = pointwise_theorem.finite_target_theorem
    return bool(
        enumeration.proof_certified is True
        and enumeration.pointwise_theorem_derived is True
        and enumeration.source_theorem_id == pointwise_theorem.theorem_id
        and enumeration.source_theorem_proof_certified is True
        and pointwise_theorem.proof_certified is True
        and enumeration.source_theorem_dimension == pointwise_theorem.dimension
        and enumeration.source_theorem_input_model == pointwise_theorem.input_model
        and enumeration.source_total_collision_policy_id
        == pointwise_theorem.total_collision_policy_id
        and tuple(enumeration.finite_target_chart_families)
        == tuple(finite_target_theorem.chart_families)
        and tuple(enumeration.finite_target_allowed_outcomes)
        == tuple(finite_target_theorem.allowed_outcomes)
    )


def _pointwise_closed_form_theorem_for_public_route(
    general_theorem_certificate: Any,
    soundness: CertificateLanguageSoundnessCertificate | None,
    enumeration: ComputableAtlasCertificateEnumerationCertificate | None,
) -> PointwiseRegularizedAtlasClosedFormTheoremCertificate | None:
    if isinstance(
        general_theorem_certificate,
        PointwiseRegularizedAtlasClosedFormTheoremCertificate,
    ):
        return general_theorem_certificate
    if not isinstance(
        general_theorem_certificate,
        PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate,
    ):
        return None
    return certify_pointwise_regularized_atlas_closed_form_theorem(
        pointwise_open_time_theorem=general_theorem_certificate,
        certificate_language_soundness=soundness,
        computable_certificate_enumeration=enumeration,
        maximal_classical_total_collision_policy=(
            certify_maximal_classical_total_collision_policy(
                pointwise_open_time_theorem=general_theorem_certificate,
            )
        ),
    )
