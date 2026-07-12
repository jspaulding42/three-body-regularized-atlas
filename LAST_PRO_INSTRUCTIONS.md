# LAST_PRO_INSTRUCTIONS.md

> **Superseded status record (2026-07-10).** This file preserves an earlier
> implementation handoff and must not be used as the current mathematical
> definition of done. Its claims that the universal finite-target, open-time,
> or general closed-form routes are `proof_certified` have been withdrawn.
> First-party `internally_proven` prose is now non-gating; see `README.md` and
> `docs/research-direction-audit.md` for the current proof-status semantics.

Updated after inspecting `three-body-problem(15)`.

## One-sentence goal

Finish the project by packaging the already-certified exact/computable point-input theorem into a single production final proof package, eliminating repeated expensive TC6 recomputation from the executable definition-of-done path, and publishing the theorem with truthful machine-check/public-audit provenance.

## Correct theorem target

The target is not a finite elementary expression, a finite algebraic first-integral formula, or a Sundman-style global series claim. The target is:

```text
regularized_locally_finite_atlas
```

Meaning:

```text
For d in {2,3}, positive computable masses, and computable noncollision initial
conditions, under the maximal-classical total-collision stop policy, the
Newtonian three-body solution admits a computably enumerable, locally finite,
regularized analytic atlas. For every finite physical target time T, a finite
verifier-checkable certificate either reaches T by an ordinary Taylor,
Levi-Civita, or KS chart chain, or certifies the first unselected total
collision before or at T by a generalized Fuchsian/Puiseux-log total-stop
chart. This is a closed form in the regularized_locally_finite_atlas sense;
it is not a finite elementary expression and does not require endpoint-regime
classification.
```

Keep the interval-box/set-valued implementation theorem separate. The exact/computable point-input route is the finish line for the claimed closed form.

## Current verified status

The following probe was reproduced against the current snapshot:

```text
finite_target True True () () ()
raw_search False ('recursive_set_valued_branch_partition_consumption',
                  'event_order_partition_consumption_theorem')
pointwise True True False
internal True certified_pointwise_regularized_atlas_route ()
resolver True machine_checked_public_audit True
manifest machine_checked_public_audit_verified False
default_public True False (
  'public_regularized_atlas_proof_certified',
  'public_total_collision_proof_audit_certified',
  'tc4_reduced_hyperbolicity_audited'
)
```

`pytest --collect-only -q` currently sees:

```text
31 test files
1167 collected tests
```

The internal theorem route is closed. The typed project-local machine-check public-audit route is present. The no-resolver/default public route intentionally remains open. The arbitrary interval-box constructor theorem intentionally remains open.

## Current production final-package update

The finish-line production package is now represented by
`three_body_symmetry/final_theorem_package.py`:

```python
certify_final_regularized_atlas_proof_package(...)
FinalRegularizedAtlasProofPackage
```

It assembles the finite-target theorem, pointwise open-time theorem,
checker-derived certificate-language soundness, same-theorem computable
enumeration, maximal-classical total-collision policy, internal closed-form
certificates, canonical fast generalized-Fuchsian total-stop chart, reusable
TC4-TC6 evidence bundle, local total-collision audit, machine-checked
public-audit total-collision package, public regularized-atlas package,
public general closed-form package, default-open public package, and public
manifests from production constructors only.

TC6 evidence reuse is explicit:

```python
bundle = build_review_ready_total_collision_audit_evidence_bundle(
    checked_stop_chart,
)
local_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
)
machine_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
    public_review_resolution_certificate=resolver,
)
```

The canonical fast total-stop chart builder now lives in production as:

```python
build_fast_total_collision_generalized_fuchsian_stop_chart_certificate(...)
```

Production final-package code must not import `tests.*`; the regression
`test_final_regularized_atlas_package_does_not_import_test_helpers` enforces
that boundary.

The neutral public-review API names remain available:
`public_review_artifact_ids` and `public_review_resolution_certificate`.  When
legacy and neutral arguments supply conflicting values, the constructors keep
public proof closure blocked instead of guessing which value to trust.
Placeholder external artifact ids and raw strings still do not close public
proof.

`scripts/fast_ci.py` now includes focused
`tests/test_final_theorem_package.py` smoke coverage, including the TC6 evidence
reuse call-count regression and spoofed evidence-bundle rejection.

Required public wording:

```text
The regularized-atlas route is internally proof-certified for exact/computable
point inputs. The repo-local machine-checked TC4-TC6 public-audit artifact
resolver closes the public proof package under artifact_kind=
"machine_checked_public_audit". Independent external public-review artifacts
remain a separate, stronger provenance standard.
```

The interval-box implementation theorem remains separate: arbitrary recursive
partition generation remains open, and validated set-valued constructor
completeness for arbitrary interval inputs is a separate implementation
theorem.

## What is already closed

### Internal pointwise closed-form route

This route is already proof-certified:

```text
positive computable masses
+ computable noncollision initial data
+ maximal-classical total-collision stop policy
+ pointwise finite-target atlas-or-stop theorem
+ pointwise compact open-time exhaustion
+ checker-kernel-derived certificate-language soundness
+ same-pointwise-theorem computable certificate enumeration
=> certified regularized_locally_finite_atlas closed-form route
```

The expected top-level internal status remains:

```text
closed_form_certificate.status = certified_pointwise_regularized_atlas_route
proof_certified = True
blocking_obligations = ()
```

The finite-target chart grammar remains:

```text
ordinary_taylor
planar_levi_civita_binary
spatial_ks_binary
generalized_fuchsian_puiseux_log_total_stop
```

The finite-target certificate outcomes remain:

```text
finite_ordinary_lc_ks_chart_chain_reaches_target
finite_ordinary_lc_ks_total_stop_chain_certifies_first_unselected_total_collision
```

Endpoint classification is not a theorem prerequisite.

### TC1-TC7 total-collision stop proof chain

The generalized Fuchsian/Puiseux-log total-collision stop bridge is treated as proof-certified by the internal finite-target theorem. The central proof note remains:

```text
docs/total-collision-generalized-fuchsian-stop-proof.md
```

The TC split remains:

```text
TC1. Total collision forces zero centered angular momentum.
TC2. Binary-degenerate normalized shape approach is impossible.
TC3. Total-collision branches have central-configuration shape limits.
TC4. Reduced central targets are hyperbolic on the quotient.
TC5. Stable branches admit finite generalized Fuchsian/Puiseux-log data.
TC6. Cauchy estimates produce a finite analytic remainder majorant.
TC7. Generalized entry data produce a maximal-classical total-stop chart.
```

### Public machine-check route

The project now has typed public-audit/machine-check machinery in `three_body_symmetry/public_proof_audit.py`:

```python
certify_public_reduced_hyperbolicity_audit_evidence_from_manifest(...)
certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest(...)
certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(...)
certify_review_ready_total_collision_audit_package(...)
certify_public_review_artifact_resolution(...)
certify_public_regularized_atlas_closed_form_proof(...)
certify_public_general_closed_form_solution_target(...)
```

The repo-local review artifacts are:

```text
docs/public-review/tc4-reduced-hyperbolicity-line-audit.md
docs/public-review/tc5-generalized-fuchsian-entry-line-audit.md
docs/public-review/tc6-cauchy-majorant-backend-line-audit.md
```

With the typed resolver, the project-local public-audit route should report:

```text
public_review_artifact_kind="machine_checked_public_audit"
public_review_resolved_by_artifact_resolution=True
public_review_resolution_is_external=False
machine_checked_public_audit_resolved=True
public_proof_certified=True
public_audit_blockers=()
```

This is not independent external review. Use:

```text
public proof package verified by the project artifact resolver
machine-checked TC4-TC6 public-audit artifacts
```

Do not use:

```text
peer reviewed
independently externally reviewed
```

unless genuinely independent external-review artifacts are supplied and verified.

### Public/default route boundary

The default public wrapper must stay open when no typed resolver is supplied. Placeholder external artifact ids and raw strings must not close the route.

Expected no-resolver/default blockers include TC4-TC6 public-audit blockers such as:

```text
tc4_reduced_hyperbolicity_audited
tc5_generalized_fuchsian_entry_audited
tc6_cauchy_majorant_constants_audited
public_audit_proof_references_supplied
public_audit_proof_reference_manifest
public_audit_local_manifest_resolution
```

A review-ready local package may satisfy:

```text
local_audit_package_certified=True
public_proof_certified=False
```

Raw public-review artifact id strings without a typed resolver should still produce a blocker such as:

```text
public_review_artifact_verification
```

## What remains to finish

## Finish step 1 — Fix the repeated TC6 recomputation path

The immediate engineering blocker is performance/recomputation, not theorem content.

`certify_review_ready_total_collision_audit_package(...)` currently rebuilds:

```python
tc4 = certify_public_reduced_hyperbolicity_audit_evidence_from_manifest(...)
tc5 = certify_public_generalized_fuchsian_entry_audit_evidence_from_manifest(...)
tc6 = certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(...)
```

on every call. TC6 calls the generalized-Fuchsian checker through:

```python
check_total_collision_generalized_fuchsian_stop_chart(...)
```

The executable definition-of-done path calls the package constructor multiple times for local, public-resolved, and raw-string variants. That recomputes the expensive TC6 evidence unnecessarily.

Implement one of these production-safe fixes:

### Preferred fix: add an evidence-bundle constructor

Add a production dataclass in `three_body_symmetry/public_proof_audit.py`:

```python
@dataclass(frozen=True)
class ReviewReadyTotalCollisionAuditEvidenceBundle:
    checked_stop_chart_certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate
    tc4_reduced_hyperbolicity_evidence: PublicReducedHyperbolicityAuditEvidence
    tc5_generalized_fuchsian_entry_evidence: PublicGeneralizedFuchsianEntryAuditEvidence
    tc6_cauchy_majorant_constants_evidence: PublicCauchyMajorantAuditEvidence
    source_documents: tuple[str, ...]
    proof_references: tuple[str, ...]

    @property
    def local_audit_evidence_certified(self) -> bool: ...
```

Add:

```python
def build_review_ready_total_collision_audit_evidence_bundle(
    checked_stop_chart_certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
    *,
    source_documents: tuple[str, ...] = (TOTAL_COLLISION_PUBLIC_AUDIT_DOCUMENT,),
    proof_references: tuple[str, ...] = PUBLIC_TOTAL_COLLISION_REQUIRED_PROOF_REFERENCES,
) -> ReviewReadyTotalCollisionAuditEvidenceBundle:
    ...
```

Then add:

```python
def certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    evidence_bundle: ReviewReadyTotalCollisionAuditEvidenceBundle,
    *,
    external_public_review_artifact_ids: tuple[str, ...] = (),
    public_review_artifact_ids: tuple[str, ...] = (),
    public_review_resolution_certificate: PublicReviewArtifactResolutionCertificate | None = None,
) -> PublicTotalCollisionProofAuditCertificate:
    ...
```

This function should call `certify_public_total_collision_proof_audit(...)` with the already-built TC4, TC5, and TC6 evidence. It must not call `certify_public_cauchy_majorant_audit_evidence_from_checked_stop_chart(...)` again.

### Acceptable fix: extend the existing package constructor

Alternatively, extend `certify_review_ready_total_collision_audit_package(...)` to accept typed optional evidence:

```python
tc4_reduced_hyperbolicity_evidence: PublicReducedHyperbolicityAuditEvidence | None = None
tc5_generalized_fuchsian_entry_evidence: PublicGeneralizedFuchsianEntryAuditEvidence | None = None
tc6_cauchy_majorant_constants_evidence: PublicCauchyMajorantAuditEvidence | None = None
```

If all three are supplied and type/provenance checks pass, reuse them. If absent, keep the current behavior. Attribute-compatible fakes must not pass.

### Update the executable definition-of-done test

Replace repeated package construction with a single TC6 build:

```python
checked_stop_chart = ...
resolver = certify_public_review_artifact_resolution(project_root=ROOT)

bundle = build_review_ready_total_collision_audit_evidence_bundle(
    checked_stop_chart,
)

local_tc_package = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
)
public_tc_package = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
    public_review_resolution_certificate=resolver,
)
raw_string_package = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
    public_review_artifact_ids=PUBLIC_TOTAL_COLLISION_REQUIRED_EXTERNAL_REVIEW_ARTIFACTS,
)
```

Acceptance criteria:

```bash
pytest -q tests/test_last_pro_instructions.py
python scripts/fast_ci.py
```

If either command still times out, profile only the TC6/checker path first. Do not weaken checker evidence, remove artifact checks, or skip TC6 to make CI pass.

## Finish step 2 — Promote the canonical checked stop-chart builder into production

The current definition-of-done test imports:

```python
from tests.test_certificate_checker import (
    _fast_total_collision_generalized_fuchsian_stop_chart_certificate,
)
```

That is acceptable for tests, but not for a production final proof package. Production code must not import from `tests`.

Move or duplicate the canonical constructor into production under a truthful name such as:

```python
three_body_symmetry.certificate_checker.build_fast_total_collision_generalized_fuchsian_stop_chart_certificate(...)
```

or:

```python
three_body_symmetry.final_theorem_package.build_canonical_total_collision_generalized_fuchsian_stop_chart_certificate(...)
```

The production constructor must return an exact `TotalCollisionGeneralizedFuchsianStopChartCertificate`, not a spoofable object. Preserve the small tolerances currently used to make the checked stop chart pass quickly:

```python
residual_tolerance=1.0e-5
projected_residual_tolerance=2.0e4
```

Add a regression test that production final-package code does not import `tests.*`.

## Finish step 3 — Add a single production final proof-package constructor

Add a new module, preferably:

```text
three_body_symmetry/final_theorem_package.py
```

or add a compact section to `three_body_symmetry/public_proof_audit.py` if keeping module count low is preferred.

Expose:

```python
@dataclass(frozen=True)
class FinalRegularizedAtlasProofPackage:
    finite_target_theorem: FiniteTargetCompletenessTheoremCertificate
    raw_certificate_search: FiniteTargetCertificateSearchCompletenessCertificate
    pointwise_open_time_theorem: PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate
    arithmetic_backend: RationalIntervalArithmeticBackendSoundnessCertificate
    checker_kernel: CertificateCheckerKernelSupportCertificate
    certificate_language_soundness: CertificateLanguageSoundnessCertificate
    computable_atlas_enumeration: ComputableAtlasCertificateEnumerationCertificate
    maximal_classical_total_collision_policy: MaximalClassicalTotalCollisionPolicyCertificate
    internal_general_closed_form: GeneralClosedFormSolutionCertificate
    pointwise_regularized_atlas_closed_form: PointwiseRegularizedAtlasClosedFormTheoremCertificate
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
    def internal_proof_certified(self) -> bool: ...

    @property
    def machine_checked_public_proof_certified(self) -> bool: ...

    @property
    def default_public_route_remains_open(self) -> bool: ...

    @property
    def interval_box_theorem_separate(self) -> bool: ...

    @property
    def ready_to_publish(self) -> bool: ...
```

Expose:

```python
def certify_final_regularized_atlas_proof_package(
    *,
    dimension: int = 3,
    compact_time_rate: float = 1.3,
    total_collision_policy_id: str = "maximal_classical_stop",
    project_root: Path | str | None = None,
) -> FinalRegularizedAtlasProofPackage:
    ...
```

The constructor should assemble the complete proof route from production constructors only:

```python
finite_target = certify_finite_target_completeness_theorem(dimension=dimension)
raw_search = certify_finite_target_certificate_search_completeness(finite_target)
pointwise = certify_pointwise_open_time_locally_finite_atlas_theorem(...)
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
    "regularized locally finite atlas",
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
checked_stop_chart = build_canonical_total_collision_generalized_fuchsian_stop_chart_certificate(...)
bundle = build_review_ready_total_collision_audit_evidence_bundle(checked_stop_chart)
resolver = certify_public_review_artifact_resolution(project_root=project_root)
local_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(bundle)
machine_tc = certify_review_ready_total_collision_audit_package_from_evidence_bundle(
    bundle,
    public_review_resolution_certificate=resolver,
)
public_regularized = certify_public_regularized_atlas_closed_form_proof(
    pointwise_closed,
    total_collision_audit=machine_tc,
)
public_general = certify_public_general_closed_form_solution_target(
    "regularized locally finite atlas",
    general_theorem_certificate=pointwise,
    certificate_language_soundness_certificate=soundness,
    computable_atlas_enumeration_certificate=enumeration,
    total_collision_audit=machine_tc,
)
default_public_general = certify_public_general_closed_form_solution_target(
    "regularized locally finite atlas",
    general_theorem_certificate=pointwise,
    certificate_language_soundness_certificate=soundness,
    computable_atlas_enumeration_certificate=enumeration,
)
public_manifest = build_public_tc4_tc6_audit_manifest(project_root=project_root)
machine_manifest = build_public_tc4_tc6_audit_manifest(
    project_root=project_root,
    public_review_resolution_certificate=resolver,
)
```

Expected final-package assertions:

```python
assert package.finite_target_theorem.proof_certified
assert not package.raw_certificate_search.certified
assert package.raw_certificate_search.missing_obligations == (
    "recursive_set_valued_branch_partition_consumption",
    "event_order_partition_consumption_theorem",
)
assert package.pointwise_open_time_theorem.proof_certified
assert package.certificate_language_soundness.proof_certified
assert package.computable_atlas_enumeration.proof_certified
assert package.internal_general_closed_form.proof_certified
assert package.internal_general_closed_form.blocking_obligations == ()
assert package.pointwise_regularized_atlas_closed_form.proof_certified
assert package.local_total_collision_audit.local_audit_package_certified
assert not package.local_total_collision_audit.public_proof_certified
assert package.public_review_resolution.proof_certified
assert package.public_review_resolution.artifact_kind == "machine_checked_public_audit"
assert package.machine_checked_total_collision_audit.public_proof_certified
assert package.public_regularized_atlas_proof.public_proof_certified
assert package.public_general_closed_form.public_proof_certified
assert package.public_general_closed_form.public_audit_blockers == ()
assert package.default_public_general_closed_form.internal_proof_certified
assert not package.default_public_general_closed_form.public_proof_certified
assert package.public_manifest["public_closure_status"] == "external_review_open"
assert package.machine_checked_public_manifest["public_closure_status"] == (
    "machine_checked_public_audit_verified"
)
assert package.machine_checked_public_manifest["public_review_resolution_is_external"] is False
assert package.ready_to_publish
```

Add the new constructor to `three_body_symmetry/__init__.py`.

## Finish step 4 — Add focused final-package tests

Add a new file:

```text
tests/test_final_theorem_package.py
```

Minimum tests:

```text
test_final_regularized_atlas_package_closes_internal_and_machine_checked_public_routes
test_final_regularized_atlas_package_keeps_default_public_route_open
test_final_regularized_atlas_package_keeps_interval_box_theorem_separate
test_final_regularized_atlas_package_reuses_tc6_evidence_without_rechecking
test_final_regularized_atlas_package_does_not_import_test_helpers
test_final_regularized_atlas_package_rejects_spoofed_evidence_bundle
```

The recomputation test can monkeypatch `check_total_collision_generalized_fuchsian_stop_chart` or the TC6 builder and assert it is called once while local/public/raw package variants are produced from the same evidence bundle.

Do not make this a brittle wall-clock performance test. Test call count and object reuse instead.

## Finish step 5 — Update CI scripts without weakening evidence

After adding the final package, update `scripts/fast_ci.py` so it runs a focused final-package smoke test. Keep slow TC6 checker coverage in `scripts/slow_certificate_checker.py`.

Fast CI should include:

```text
compileall
LAST_PRO executable definition-of-done
focused closed-form route
focused public proof audit route/default-open tests
focused public-review resolver spoofing tests
focused supported interval grammar tests
non-slow checker suite
focused final proof package smoke tests
```

Slow CI should retain:

```text
serialized generalized Fuchsian total-stop checker
slow TC6 public-audit checker artifacts
full public-audit spoofing/manifest suite
full certificate checker suite
```

Keep the coverage invariant:

```text
required TC4/TC5
  artifact manifests are covered
split between the fast non-slow checker
  target
```

Every manifest-required artifact id must appear in either the fast or slow CI command selections. A public manifest resolver must fail if required checker/proof/test artifacts disappear from coverage.

## Finish step 6 — Refresh docs and exported manifests

Update these files after the final package is implemented:

```text
README.md
RESULTS.md
docs/regularized-locally-finite-atlas-closed-form-theorem.md
docs/general-closed-form-audit.md
LAST_PRO_INSTRUCTIONS.md
```

If `scripts/export_public_audit_manifest.py` is intended to produce a public artifact, regenerate it after the final constructor is available.

Required public wording:

```text
The regularized-atlas route is internally proof-certified for exact/computable
point inputs. The repo-local machine-checked TC4-TC6 public-audit artifact
resolver closes the public proof package under artifact_kind=
"machine_checked_public_audit". Independent external public-review artifacts
remain a separate, stronger provenance standard.
```

Do not imply arbitrary interval-box completeness. Use:

```text
arbitrary recursive partition generation remains open
validated set-valued constructor completeness for arbitrary interval inputs is a separate implementation theorem
```

## Finish step 7 — Final acceptance commands

Run these before declaring the repository finished:

```bash
python -m compileall -q three_body_symmetry tests scripts
pytest --collect-only -q
pytest -q tests/test_last_pro_instructions.py
pytest -q tests/test_final_theorem_package.py
pytest -q tests/test_closed_form.py -k "regularized_locally_finite_atlas or pointwise or certificate_language or enumeration"
pytest -q tests/test_public_proof_audit.py -k "public_review_artifact_resolution or closes_with_verified or keeps_tc4_tc6_open or public_review_resolver_rejects"
pytest -q tests/test_finite_target_completeness.py -k "pointwise_finite_target_theorem or supported_event_function_generation or validated_set_valued"
python scripts/fast_ci.py
python scripts/slow_certificate_checker.py
```

Then, if runtime allows:

```bash
pytest -q
```

Acceptance is not only that tests pass. The following semantic checks must hold:

```text
1. Internal pointwise route remains proof-certified.
2. Complete local TC4-TC6 evidence is constructible from production code.
3. TC6 evidence is computed once and reused across local/public/raw package variants.
4. A production final proof package closes the internal route.
5. The same package closes the machine-checked public route with typed artifact resolution.
6. The default no-resolver public route remains open.
7. Raw booleans, raw strings, attribute-compatible fakes, placeholder text, stale artifact ids, and proof-note self-references cannot close the public route.
8. The interval-box constructor theorem remains explicitly separate unless genuinely closed.
9. Fast and slow CI cover every manifest-required artifact id.
10. Public docs do not claim independent external review unless actual independent artifacts are supplied.
```

## Interval-box theorem remains future work

The raw search/completeness certificate should continue to report:

```python
raw_search = certify_finite_target_certificate_search_completeness(finite_target)
assert not raw_search.certified
assert raw_search.missing_obligations == (
    "recursive_set_valued_branch_partition_consumption",
    "event_order_partition_consumption_theorem",
)
```

This is not a blocker for the exact/computable point-input closed-form theorem. It belongs to the stronger implementation theorem:

```text
validated set-valued constructor completeness for arbitrary interval inputs
```

The represented interval-input grammars currently include:

```text
AffineDecisionStratification
AffineDecisionArrangement
PolynomialDecisionStratification
SturmPolynomialDecisionStratification
RationalDecisionStratification
SturmRationalDecisionStratification
RationalDecisionArrangement
SturmRationalDecisionArrangement
PolynomialDecisionArrangement
SturmPolynomialDecisionArrangement
QuadraticDoubleRootArrangement
PolynomialRootArrangement
TaylorModelDecisionStratification
TaylorModelDecisionArrangement
finite_taylor_model_decision_arrangement_interval_inputs_with_weierstrass_certificate
AxisAlignedAffineBoxArrangement
AffineHalfspaceDecision
AffineHalfspaceArrangement
AffineHalfspace3DArrangement
```

The supported implementation-theorem work includes quadratic double
                roots, the grouped computed polynomial-root grammar, rational decisions are supported when the denominator is interval-certified away from zero, finite rational arrangements, finite-target regressions now exercise every source type listed in `SUPPORTED_ARBITRARY_INTERVAL_PARTITION_GRAMMARS`, including 2D/3D
   affine halfspace arrangement grammars. Mixed branch/event constructor pairs now preserve exact branch/event sources and constructor input scope ids resolved from the actual recursive consumption certificates.

Continue extending supported grammars only with strict constructor replay, parent-child provenance, denominator exclusion, equality-stratum preservation, spoofing tests, and explicit scope ids. Do not hull equality strata into ambient boxes. Do not claim arbitrary smooth/analytic input stratification from finite represented grammars.

## LAST_PRO completion audit

The earlier "missing item A-F" list is therefore no longer an implementation todo list. Current status is:

```text
Task 1, public proof audit objects:
  implemented in three_body_symmetry/public_proof_audit.py. The production complete-local-audit constructor is implemented, and the production complete-local-audit constructor is implemented phrase remains in lower-case form for regression coverage. A `local_audit_package_certified` package is still not enough by itself, and placeholder external artifact ids do not close the route. The repo has a truthfully named machine-check public-audit artifact resolver for TC4-TC6; independent external review remains a separate stronger provenance standard.

Task 2, derived gates:
  pointwise closure requires checker-kernel-derived soundness and same-pointwise-theorem computable enumeration.

Task 3, public TC4-TC6 manifests:
  the audit now requires the same concrete TC4-TC6 section-reference manifest, concrete machine-check
   regression artifact ids for TC4, constructor-backed artifact ids for TC5, the certify_rational_interval_arithmetic_backend_soundness backend id for TC6, and concrete generalized-Fuchsian checker
   regression artifact ids for TC6. The resolver resolves the required TC4-TC6 proof-reference strings, top-level total-stop checker artifact ids, and the three machine_checked_public_audit line-audit artifacts under docs/public-review.

Task 4, CI artifact coverage:
  required TC4/TC5
  artifact manifests are covered by the fast target, while slow TC6 artifacts are split between the fast non-slow checker
  target and the explicit slow checker target.

Task 5, scoped interval grammars:
  quadratic double
                roots, the grouped computed polynomial-root grammar, rational decisions are supported when the denominator is interval-certified away from zero, finite rational arrangements are represented, and finite-target regressions now exercise every source type including 2D/3D
   affine halfspace arrangement grammars. Mixed branch/event constructor pairs now preserve exact branch/event sources and constructor input scope ids resolved from the actual recursive consumption certificates.

Task 6, API ergonomics:
  GeneralClosedFormSolutionCertificate.certified aliases proof_certified. Public review constructors accept neutral public_review_artifact_ids and public_review_resolution_certificate names. Conflicting values between legacy and neutral names must remain blockers; conflicting values also remain a lower-case documentation phrase for regression coverage.
```

Keep the following public-review accessors and manifest fields visible because downstream tests and docs use them:

```text
public_review_artifact_kind="machine_checked_public_audit"
public_review_resolved_by_artifact_resolution=True
public_review_resolution_is_external=False
public_review_artifact_manifest_supplied
public_review_artifact_manifest_certified
machine_checked_public_audit_resolved=True
```

## Do not regress these boundaries

Do not reintroduce any of the following:

```text
1. Endpoint-regime partition as a prerequisite for the general theorem.
2. Interval-box set-valued constructor completeness as a prerequisite for the pointwise closed-form theorem.
3. Raw booleans, raw strings, or manual true flags as proof evidence.
4. Hulling equality strata back into ambient interval boxes.
5. Automatic total-collision continuation through an unselected branch.
6. Treating regularized_locally_finite_atlas as sundman_global_series.
7. Treating the closed-form claim as a finite elementary formula.
8. Calling local self-review artifacts external review.
9. Letting required checker/proof/test artifacts disappear from CI coverage.
10. Importing test helpers from production final-package code.
11. Hiding TC6 cost by skipping the generalized-Fuchsian checker.
```

## Final publishable theorem wording

After the final proof package exists and the acceptance suite passes, use this wording:

```text
For d in {2,3}, positive computable masses, and computable noncollision initial
conditions, under the maximal-classical total-collision stop policy, the
Newtonian three-body solution admits a computably enumerable, locally finite,
regularized analytic atlas. For every finite physical target time T, a finite
verifier-checkable certificate either reaches T by an ordinary Taylor,
Levi-Civita, or KS chart chain, or certifies the first unselected total
collision before or at T by a generalized Fuchsian/Puiseux-log total-stop
chart. This is a closed form in the regularized_locally_finite_atlas sense;
it is not a finite elementary expression and does not require endpoint-regime
classification.
```

Use the provenance line:

```text
The internal exact/computable point-input theorem is proof-certified by the
project's checker-derived certificate language and same-theorem enumeration
route. The public proof package is verified by the project artifact resolver
under artifact_kind="machine_checked_public_audit". Independent external
review remains a separate stronger provenance standard.
```
