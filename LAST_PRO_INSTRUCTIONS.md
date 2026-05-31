The project has made the right major leap: it now has a proof-certified pointwise theorem surface.

The most important status probe is:

certify_finite_target_completeness_theorem(dimension=3):
  certified=True
  proof_certified=True
  unaudited analytic lemmas=0
  critical unaudited analytic lemmas=0

certify_pointwise_open_time_locally_finite_atlas_theorem(...):
  certified=True
  proof_certified=True

certify_finite_target_certificate_search_completeness(...):
  certified=False
  missing:
    recursive_set_valued_branch_partition_consumption
    event_order_partition_consumption_theorem

That means the project now claims to have closed the exact/computable point-input mathematical theorem, but it has not closed the stronger interval-box / set-valued constructor theorem.

That distinction is now the finish line.

External positioning

The correct closed-form target remains not a finite elementary formula and not a finite algebraic first-integral formula. Bruns-type results obstruct algebraic first-integral routes: Julliard-Tosel’s generalized Bruns theorem says algebraic first integrals are algebraic functions of the classical integrals.

Sundman-style regularized infinite representations are the right historical precedent. Yeomans’ NASA exposition describes Sundman regularization, treatment of double collisions, and the convergent-series route, while also showing why convergence/practical evaluation is the issue.

For finite-target completeness, the project is correctly built on Painlevé’s N=3 reduction: finite-time noncollision singularities do not occur in the three-body problem, so finite-time obstruction reduces to collision. Binary collision regularization by Levi-Civita in 2D and KS in 3D is also the right chart grammar; Waldvogel’s regularization overview describes Levi-Civita regularization and its KS generalization as removing binary-collision singularities in transformed variables.

Total collision remains the singular case that must be treated as a maximal-classical stop unless a selector convention is supplied. The literature also treats total collision differently from binary collision; Mercati and Reichert note Sundman’s zero-angular-momentum condition for total collisions and discuss the special singular nature of total-collision points.

What changed in this snapshot

The strongest new progress is in four places.

First, finite_target_completeness.py now marks the pointwise finite-target theorem as internally proof-certified. The theorem no longer leaves the total-collision bridge in the unaudited bucket. It now claims internal proofs for:

binary_degenerate_total_collision_exclusion
reduced_hyperbolic_total_collision_entry
poincare_dulac_fuchsian_log_selector_completeness
arbitrary_total_collision_germ_finite_generalized_fuchsian_entry_data
arbitrary_total_collision_germ_entry_to_stop_chart
total_collision_stop_chart_existence

Second, the total-collision language has been generalized correctly. The project no longer restricts itself to integer-power Fuchsian-log rows. It now uses:

generalized Fuchsian / Puiseux-log entry data
fractional or irrational exponents
finite resonant log-polynomial rows
Cauchy-majorized analytic remainders

That was necessary.

Third, certificate_language.py and certificate_checker.py now include a serialized:

TotalCollisionGeneralizedFuchsianStopChartCertificate
GeneralizedFuchsianRemainderMajorantCertificate

and the checker now verifies a generalized total-stop chart with interval/Cauchy-style obligations, including lifted residuals, projected residual tails, angular momentum, center-of-mass/linear momentum, and endpoint collapse.

Fourth, stratified_branch_tree.py has advanced from positive-margin leaves to polynomial decision arrangements, including affine boundary roots, simultaneous affine strata, quadratic simple roots, quadratic double-root tangent strata, and coincident quadratic simple-root strata. That is the right direction for the set-valued constructor theorem.

Did this take the project over the finish line?

For the pointwise mathematical theorem, yes, internally.

For the top-level closed-form audit, not yet.

The top-level closed_form.py still routes regularized_locally_finite_atlas through the constructor-facing:

OpenTimeLocallyFiniteAtlasTheoremCertificate

which still blocks on:

set_valued_constructor_branch_event_completeness
recursive_set_valued_branch_partition_consumption
event_order_partition_consumption_theorem
independent_chart_verifier

But the project now also has:

PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate.proof_certified=True

That certificate is the one that should close the mathematical closed-form theorem for exact/computable initial data.

So the steering correction is:

Do not make the interval-box constructor theorem a prerequisite for the
pointwise closed-form theorem.

It should be a separate validated-numerics theorem.

Finish-line theorem to declare

Use this as the final theorem target.

Regularized Locally Finite Atlas Closed-Form Theorem.

For d in {2,3}, positive computable masses, and computable noncollision
initial data, under the maximal-classical total-collision policy, the
Newtonian three-body solution admits a computably enumerable locally finite
regularized analytic atlas.

The allowed chart primitives are:

  ordinary Taylor charts;
  planar Levi-Civita binary charts;
  spatial KS binary charts;
  generalized Fuchsian/Puiseux-log total-collision stop charts.

For every finite physical target time T, a finite certificate resolves the
target in exactly one of two ways:

  1. a finite ordinary/LC/KS chart chain reaches T and verifies the projected
     Newtonian solution; or

  2. a finite ordinary/LC/KS/total-stop chain certifies the first unselected
     total collision before or at T.

Endpoint classification into scattering, bounded, homothetic, oscillatory, or
other final-motion classes is not required. Those regimes are optional
compression certificates, not prerequisites for the general solution.

That is the theorem that should be called the “closed form solution” in this project, with the explicit caveat that “closed form” means:

verifier-checkable regularized locally finite analytic atlas

not:

finite elementary expression
What to change next in the code
1. Add a pointwise closed-form certificate route

Add a new theorem-facing object:

@dataclass(frozen=True)
class PointwiseRegularizedAtlasClosedFormTheoremCertificate:
    closed_form_class_id: str
    pointwise_open_time_theorem: PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate
    certificate_language_soundness: object
    computable_certificate_enumeration: object
    maximal_classical_total_collision_policy: object
    statement: str
    proof_sketch: str
    obligations: tuple[TheoremPipelineObligation, ...]

    @property
    def proof_certified(self) -> bool:
        return (
            self.closed_form_class_id == "regularized_locally_finite_atlas"
            and self.pointwise_open_time_theorem.proof_certified
            and getattr(self.certificate_language_soundness, "proof_certified", False)
            and getattr(self.computable_certificate_enumeration, "proof_certified", False)
            and getattr(self.maximal_classical_total_collision_policy, "proof_certified", False)
            and all(o.certified for o in self.obligations if o.required)
        )

Then let certify_general_closed_form_solution_target(...) accept either:

OpenTimeLocallyFiniteAtlasTheoremCertificate

for the constructor/set-valued route, or:

PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate

for the exact/computable point-input mathematical route.

The pointwise route should not require:

set_valued_constructor_branch_event_completeness

That is the current main misalignment.

2. Split the closed-form audit into two scopes

Use two explicit audit statuses:

certified_pointwise_regularized_atlas_route
certified_set_valued_constructor_regularized_atlas_route

The first is the mathematical general solution for exact/computable inputs.

The second is the stronger validated-numerics implementation theorem for interval boxes.

Right now the project is mixing these scopes. That is why the pointwise theorem is proof-certified while the closed-form audit remains incomplete.

3. Add a universal checker-soundness certificate

The current independent_chart_verifier evidence is attached to sample checked prefixes. That is useful, but it is not the same as a theorem-level proof that the certificate language is sound.

Add:

CertificateLanguageSoundnessCertificate

with obligations:

ordinary_taylor_checker_sound
planar_lc_checker_sound
spatial_ks_checker_sound
fuchsian_stop_checker_sound
generalized_fuchsian_stop_checker_sound
transition_checker_sound
branch_union_checker_sound
chart_chain_checker_sound

At first, this can be a theorem-scaffold object. But the top-level closed-form proof should consume this object rather than demanding that a sample constructor prefix happened to be independently checked.

4. Add a computable enumeration certificate

For a closed-form theorem over computable inputs, the project needs to state how finite certificates are found.

Add:

ComputableAtlasCertificateEnumerationCertificate

with proof sketch:

Enumerate chart words, pair labels, rational time slabs, truncation orders,
rational/interval coefficients, algebraic or interval exponent certificates,
Fuchsian selector constants, Cauchy majorants, and transition witnesses.
Dovetail the independent checker over this enumeration.
The pointwise finite-target theorem proves at least one finite certificate
exists for each finite target, so fair enumeration eventually finds one.

This is the bridge from:

certificate exists

to:

closed-form evaluation is computably enumerable
5. Keep the interval-box branch theorem separate

Do continue developing:

recursive_set_valued_branch_partition_consumption
event_order_partition_consumption_theorem

but do not let them block the pointwise closed-form proof.

They should feed a separate theorem:

ValidatedSetValuedConstructorCompletenessTheorem

That theorem is useful for rigorous numerical enclosures of initial boxes, not for the exact-input closed-form existence theorem.

Mathematical proof audit still needed

The current code now marks the total-collision chain internally proven. That is acceptable as an internal theorem scaffold, but before making a public “closed form proof” claim, write this chain as a standalone proof document.

The risky part is not the ordinary/binary atlas argument. The risky part is:

arbitrary incoming total-collision germ
  -> reduced hyperbolic central target
  -> finite generalized Fuchsian/Puiseux-log entry data
  -> Cauchy-majorized analytic remainder
  -> verifier-checkable total-stop chart

The proof document should be split exactly this way:

TC1. total collision implies zero centered angular momentum.
TC2. binary-degenerate normalized-shape approach is impossible.
TC3. every total-collision branch has a collision-free central-configuration shape limit.
TC4. reduced central targets are hyperbolic after quotienting translations, scale, rotation, and reflection.
TC5. the reduced stable branch admits finite generalized Fuchsian/Puiseux-log selector data.
TC6. Cauchy estimates produce a finite analytic remainder majorant.
TC7. supplied generalized entry data produce a finite maximal-classical total-stop chart.

TC5 and TC6 are the proof-critical parts. Those should be written with enough precision that someone can check the normal-form hypotheses, resonance handling, exponent conventions, and Cauchy-majorant extraction.

Performance issue to fix

The generalized Fuchsian checker is now expensive enough to affect test reproducibility. Cache or factor this path.

Immediate improvements:

cache _supplied_generalized_fuchsian_branch() in tests;
cache serialized generalized stop-chart certificates across tests;
avoid rebuilding the same generalized branch inside verifier round trips;
separate slow generalized checker tests under a marker, e.g. pytest -m slow;
add a fast reduced-order generalized Fuchsian fixture for CI.

The checker is valuable, but the full suite should not repeatedly rebuild the same expensive generalized total-collision certificate.

Recommended next milestone

Call it:

Pointwise closed-form theorem closure

Definition of done:

1. closed_form.py accepts PointwiseOpenTimeLocallyFiniteAtlasTheoremCertificate
   as a valid regularized_locally_finite_atlas theorem certificate.

2. The pointwise closed-form route no longer blocks on
   set_valued_constructor_branch_event_completeness.

3. A CertificateLanguageSoundnessCertificate is required for the pointwise
   closed-form proof.

4. A ComputableAtlasCertificateEnumerationCertificate is required for the
   computable-input closed-form proof.

5. The interval-box constructor theorem remains separate and still reports
   recursive branch/event-order blockers.

6. The top-level audit can return:

   certified_pointwise_regularized_atlas_route

   while still refusing:

   certified_set_valued_constructor_regularized_atlas_route

   until the branch/event-order constructor theorem is finished.
Bottom line

The project is now close enough that the next step is not another local chart or endpoint regime.

The next step is to promote the proof-certified pointwise open-time theorem into the top-level closed-form audit.

The final shape should be:

general closed form solution, exact/computable input:
  regularized locally finite atlas
  pointwise finite-target atlas-or-stop theorem
  pointwise compact open-time exhaustion
  certificate language soundness
  fair computable certificate enumeration
  maximal-classical total-collision stop policy

separate implementation theorem:
  interval-box set-valued constructor completeness
  recursive branch/event-order stratification

That separation is what takes the project over the finish line without requiring the stronger interval-box constructor theorem first.