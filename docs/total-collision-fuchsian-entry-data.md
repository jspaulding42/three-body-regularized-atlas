# Total-Collision Fuchsian Entry Data Lemma

## Statement

For positive masses in the planar or spatial three-body problem, let an exact
finite-energy branch approach a first total collision at physical time `T`.
Assume the branch is considered only on the incoming punctured side and no
continuation through `T` is selected.  Then the branch supplies
finite-dimensional generalized Fuchsian entry data after quotienting
translations, scale, rotations, and reflection ambiguity.

The entry data consist of:

- a collision-free central-configuration limiting shape `C`,
- the cubic collision scale in `t = T + tau^3`,
- a finite list of reduced stable-mode exponents, allowing fractional or
  irrational powers in cubic collision time,
- finitely many selector constants for resonant logarithmic rows,
- a Cauchy-majorized analytic remainder after the retained generalized
  Fuchsian/Puiseux-log rows,
- a punctured isolation radius on which all pair distances are positive for
  `0 < |tau| <= rho`.

This lemma is an input to the maximal-classical total-stop theorem.  It does
not assert a unique Newtonian continuation through total collision.

## Hypotheses

- masses are positive and finite,
- the initial state is noncollision,
- the branch has finite energy up to the collision time,
- `q_1(T)=q_2(T)=q_3(T)` is the first total collision on the selected oriented
  interval,
- the branch is analyzed in mass-centered coordinates,
- binary-degenerate normalized-shape collapse has been excluded.

## Proof Sketch

The Sundman inequality gives zero centered angular momentum for finite-energy
total collision: if `I` is the centered moment of inertia, `K` the kinetic
energy, `U` the positive Newtonian potential, `H=K-U`, and `C_ang` the centered
angular momentum, then `|C_ang|^2 <= 2 I K`.  Along total collision, `I -> 0`
and `I K = I(H+U) -> 0`, so conservation forces `C_ang = 0`.

Write the centered configuration as `q = r s`, where `r = sqrt(I)` and `s` is
mass-normalized shape.  The Lagrange-Jacobi identity and finite energy imply
the reduced omega-limit lies at zero reduced shape velocity and at a critical
point of the normalized potential.  Thus every selected limiting shape is a
central configuration.

The binary-degenerate normalized-shape strata are excluded by the Jacobi
perturbed-Kepler comparison: if a tight binary coordinate `r_b` collapsed much
faster than the outer Jacobi coordinate `rho`, both coordinates would satisfy
Kepler collision balances with lower-order perturbations, forcing the same
`u^(2/3)` collision scale and a nonzero limiting ratio.  That contradicts
binary-degenerate collapse.

After quotienting translation, scale, rotation, and reflection ambiguity, the
remaining positive-mass three-body central targets are isolated reduced
hyperbolic equilibria.  On the stable manifold, analytic Poincare-Dulac
coordinates triangularize the reduced equations.  Each stable row has a
positive decay exponent; resonant forcing from lower rows contributes only a
finite polynomial in the logarithmic collision variable.  Substituting cubic
time `t = T + tau^3` converts those rows into finite powers of `tau`, possibly
with fractional or irrational exponents, and in resonant cases finite powers
of `log(tau)`.  The proof-grade entry theorem must also produce a Cauchy
majorant for the analytic remainder beyond the retained rows.

The branch therefore has finite generalized Fuchsian/Puiseux-log entry data on
its incoming side.  In the pointwise exact-germ theorem this is now treated as
an internal proof-grade bridge: reduced hyperbolicity supplies the stable
chart, Poincare-Dulac supplies the finite selector rows, and Cauchy estimates
on a smaller analytic polydisc supply a finite remainder majorant.  This is
not a set-valued constructor theorem.  It does not produce uniform constants
from arbitrary interval boxes, nor does it build the recursive branch/event
partition needed by validated numerics.

The executable supplied-entry path checks one local quantitative version of
the same remainder step: given a constructor-derived generalized branch,
finite-row tail budget, defect bound `D`, right-inverse bound `B`, nonlinear
Lipschitz bound `L`, and candidate Banach radius `R`, the checker verifies
`B L < 1` and `B D + B L R <= R`.  This proves a Cauchy-majorized analytic
remainder for supplied local data and provides checker-facing constants.  The
pointwise exact-germ theorem proves such constants exist by analyticity and
Cauchy estimates; deriving them uniformly and automatically from arbitrary
incoming interval data remains outside this lemma.

## Certificate Fields Consumed By Code

- `central_shape`
- `cubic_time_scale`
- `stable_mode_exponents`
- `selector_constants`
- `log_resonance_rows`
- `analytic_remainder_cauchy_majorant`
- `banach_defect_inverse_lipschitz_bounds`
- `punctured_isolation_radius`
- `newton_residual_tail_budget`
- `source_proof_mode`

## Failure Modes

- nonpositive mass,
- infinite-energy branch,
- binary-degenerate normalized-shape limit not excluded,
- unresolved central-configuration degeneracy after quotienting,
- infinite or nontriangular resonant normal-form data,
- no local analytic polydisc on which Cauchy estimates can be applied,
- failed Banach self-map or contraction inequality,
- missing punctured isolation certificate.
