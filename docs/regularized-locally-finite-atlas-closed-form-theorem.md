# Regularized Locally Finite Atlas Closed-Form Theorem

This note records the exact/computable-input theorem that the closed-form audit
is allowed to certify.  It is deliberately narrower than the interval-box
validated-numerics constructor theorem, and it is deliberately different from a
finite elementary or finite algebraic first-integral formula.

## Theorem Statement

For `d in {2,3}`, positive computable masses, and computable noncollision
initial position and velocity data, under the maximal-classical total-collision
stop policy, the Newtonian three-body solution admits a computably enumerable
locally finite regularized analytic atlas.

The permitted chart primitives are exactly:

- ordinary Taylor charts;
- planar Levi-Civita binary charts;
- spatial KS binary charts;
- generalized Fuchsian/Puiseux-log total-collision stop charts.

For every finite physical target time `T`, a finite verifier-checkable
certificate resolves the query in exactly one of two ways:

1. A finite ordinary/Levi-Civita/KS chart chain reaches `T` and verifies the
   projected Newtonian solution on the chain.
2. A finite ordinary/Levi-Civita/KS/total-stop chart chain certifies the first
   unselected total collision before or at `T`.

Equivalently, the second outcome certifies the first unselected total collision
before or at `T` rather than selecting an outgoing branch.

Endpoint classification into scattering, bounded, homothetic, oscillatory, or
other final-motion regimes is not a theorem prerequisite.  Such regimes may be
used later as optional compression certificates for special all-time behavior.

## Meaning Of Closed Form

Here `closed form` means a verifier-checkable regularized locally finite
analytic atlas:

```text
finite target T
  -> fair certificate enumeration
  -> finite accepted chart chain or maximal-classical total-stop certificate
  -> projected Newtonian state or certified stop before/at T
```

It does not mean a finite elementary expression, a finite algebraic expression,
or a finite list of additional algebraic first integrals.  Those finite
algebraic-integral routes are treated separately by the obstruction branch in
the closed-form audit.

## Proof Chain

The theorem is the composition of five constructor-facing pieces.

1. The pointwise finite-target atlas-or-stop theorem proves that each
   computable finite target has a finite ordinary, binary-regularized, or
   total-stop certificate.
2. The pointwise open-time compact-exhaustion theorem applies the finite-target
   theorem on the nested compact intervals `[-nR,nR]`, producing a countable
   locally finite family of finite target certificates.
3. The certificate-language soundness theorem states that accepted ordinary,
   planar Levi-Civita, spatial KS, total-stop, transition, branch-union, and
   chart-chain certificates project to genuine Newtonian solutions or genuine
   maximal-classical stop proofs.
4. The computable enumeration theorem enumerates chart words, pair labels,
   rational domains, truncation orders, rational or interval coefficients,
   rational tail budgets, generalized Fuchsian exponent data, selector
   constants, Cauchy majorants, transition witnesses, and collision-policy
   data, then dovetails the independent verifier fairly.
5. The maximal-classical total-collision policy theorem fixes the singular
   semantics: unselected total collision ends the classical solution branch;
   no outgoing branch is silently selected.

Since the pointwise finite-target theorem proves that at least one finite
certificate exists for each computable target and fair enumeration eventually
tests every finite certificate, finite target evaluation under the
maximal-classical total-collision stop policy is computably enumerable in this
atlas language.

## Production Proof Package

`certify_final_regularized_atlas_proof_package(...)` assembles the theorem
through production constructors only.  The package builds the pointwise
finite-target theorem, the pointwise open-time locally finite atlas theorem,
the checker-kernel-derived certificate language, the same-theorem computable
enumeration gate, the maximal-classical total-collision stop policy, and the
pointwise regularized-atlas closed-form certificate.

The package also builds the canonical checked generalized Fuchsian/Puiseux-log
total-stop chart through
`build_fast_total_collision_generalized_fuchsian_stop_chart_certificate(...)`.
`build_review_ready_total_collision_audit_evidence_bundle(...)` computes the
TC4-TC6 public-audit evidence once, including the TC6 generalized-Fuchsian
checker call, and
`certify_review_ready_total_collision_audit_package_from_evidence_bundle(...)`
reuses that evidence for the local, machine-checked public-audit, and raw
artifact-string variants.

The regularized-atlas route is internally proof-certified for exact/computable
point inputs. The repo-local machine-checked TC4-TC6 public-audit artifact
resolver closes the public proof package under
`artifact_kind="machine_checked_public_audit"`. Independent external
public-review artifacts remain a separate, stronger provenance standard.

## Separate Interval-Box Theorem

The interval-box set-valued constructor theorem is a stronger implementation
theorem.  It must derive recursive branch and event-order partitions for input
sets, consume equality strata, and aggregate certified leaf responses.  The
pointwise theorem above does not require that stronger result.

Current scoped interval-box certificates may prove positive-margin boxes or
supplied recursive stratified trees, but arbitrary recursive partition
generation remains open until a future constructor derives those finite
stratifications from arbitrary interval inputs.
Validated set-valued constructor completeness for arbitrary interval inputs is
a separate implementation theorem.

## Production Final Package

The production constructor
`certify_final_regularized_atlas_proof_package(...)` assembles this theorem
route from project constructors only. It builds the pointwise finite-target and
open-time theorem certificates, derives checker-kernel soundness and
same-theorem computable enumeration, fixes the maximal-classical
total-collision stop policy, constructs the canonical generalized
Fuchsian/Puiseux-log total-stop chart, computes TC4-TC6 public-audit evidence
once, and reuses that evidence for the local, machine-checked public, and
raw-artifact package variants.

The regularized-atlas route is internally proof-certified for exact/computable
point inputs. The repo-local machine-checked TC4-TC6 public-audit artifact
resolver closes the public proof package under
`artifact_kind="machine_checked_public_audit"`. Independent external
public-review artifacts remain a separate, stronger provenance standard.

This final package does not claim arbitrary interval-box completeness:
arbitrary recursive partition generation remains open, and validated
set-valued constructor completeness for arbitrary interval inputs is a separate
implementation theorem.
