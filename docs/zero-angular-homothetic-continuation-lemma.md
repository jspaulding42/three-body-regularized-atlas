# Zero-Angular Homothetic Continuation Lemma

## Claim

Let `m_i > 0`, and let `C = (C_1, C_2, C_3)` be a mass-centered central
configuration scaled so that

```text
sum_i m_i C_i = 0,
A(C) = -(2/9) C,
```

where `A_i(C) = sum_{j != i} m_j (C_j - C_i) / |C_j - C_i|^3` is the
Newtonian acceleration map with `G = 1`. Then

```text
t = tau^3,
q_i(tau) = C_i tau^2
```

is an analytic regularized curve through total collision at `tau = 0`. For every
`tau != 0`, the projected curve satisfies Newton's equations in physical time.
It has zero angular momentum and zero energy. The branch selector is exactly the
second regularized-time jet

```text
J_i = d^2 q_i / d tau^2 |_{tau=0} = 2 C_i.
```

Thus the `regularized_second_jet_branch` convention is not merely a label for
the homothetic parabolic branch: once the scaled central-configuration jet `J`
is supplied, it produces an explicit collision-through-ejection continuation in
the regularized parameter `tau`.

## Proof

The curve is polynomial in `tau`, so it extends analytically through `tau = 0` as
a regularized curve:

```text
q_i(0) = 0,
dq_i/dtau |_{tau=0} = 0,
d^2q_i/dtau^2 |_{tau=0} = 2 C_i.
```

For `tau != 0`, physical time satisfies:

```text
dt/dtau = 3 tau^2.
```

Therefore:

```text
dq_i/dt
  = (dq_i/dtau) / (dt/dtau)
  = (2 C_i tau) / (3 tau^2)
  = (2/3) C_i tau^(-1),
```

and

```text
d^2q_i/dt^2
  = (1 / (3 tau^2)) d/dtau ((2/3) C_i tau^(-1))
  = -(2/9) C_i tau^(-4).
```

Newtonian acceleration is homogeneous of degree `-2`, so:

```text
A_i(q(tau)) = A_i(C tau^2) = tau^(-4) A_i(C).
```

The scaling assumption `A(C) = -(2/9) C` gives:

```text
A_i(q(tau)) = -(2/9) C_i tau^(-4) = d^2q_i/dt^2.
```

Hence the projected curve solves Newton's equations on both sides of the
collision, for all `tau != 0`.

The angular momentum is zero because each physical velocity is parallel to the
same central-configuration vector as the position:

```text
q_i x dq_i/dt
  = (C_i tau^2) x ((2/3) C_i tau^(-1))
  = 0.
```

For energy, write:

```text
I_m(C) = sum_i m_i |C_i|^2,
U_m(C) = sum_{i<j} m_i m_j / |C_i - C_j|.
```

The central-configuration identity gives `U_m(C) = (2/9) I_m(C)`. Indeed,
homogeneity of `U_m` gives:

```text
-U_m(C) = sum_i C_i dot grad_{C_i} U_m(C)
        = sum_i m_i C_i dot A_i(C)
        = -(2/9) I_m(C).
```

Thus:

```text
K = (1/2) sum_i |dq_i/dt|^2
  = (1/2) sum_i m_i |dq_i/dt|^2
  = (1/2) (4/9) I_m(C) tau^(-2)
  = (2/9) I_m(C) tau^(-2),

U_m(q(tau)) = U_m(C tau^2) = U_m(C) tau^(-2).
```

So the total energy is:

```text
E = K - U_m(q(tau))
  = ((2/9) I_m(C) - U_m(C)) tau^(-2)
  = 0.
```

This proves the regularized homothetic continuation claim.

## Energy-Parametrized Homothetic Branch

The parabolic branch is the zero-energy member of a local analytic family. Keep
the same scaled central configuration `C`, so `A(C)=-(2/9)C`, and set:

```text
q_i(tau) = C_i rho(tau),
rho(tau) = tau^2 u(tau^2),
u(0) = 1.
```

Let the energy per moment of inertia of `C` be:

```text
epsilon = E / I_m(C),
I_m(C) = sum_i m_i |C_i|^2.
```

The scalar energy identity for the homothetic motion is:

```text
(1/2) (drho/dt)^2 - (2/9) / rho = epsilon.
```

Since `t=tau^3` and `z=tau^2`, this becomes the regular first-order equation:

```text
(u + z du/dz)^2 = 1/u + (9/2) epsilon z,
u(0)=1,
u + z du/dz > 0.
```

Equivalently,

```text
z du/dz = sqrt(1/u + (9/2) epsilon z) - u.
```

This is a Briot-Bouquet equation at `(z,u)=(0,1)`. Writing `y=u-1`, the right
side has linear part:

```text
(9/4) epsilon z - (3/2) y.
```

For the coefficient of `z^n`, the recurrence divisor is `n + 3/2`, never zero.
The usual majorant proof for Briot-Bouquet equations therefore gives a unique
real analytic solution `u_epsilon(z)` near `z=0`.

Equivalently, the coefficients can be constructed directly. Write:

```text
u(z) = sum_{n>=0} c_n z^n,       c_0 = 1,
1/u(z) = sum_{n>=0} d_n z^n,     d_0 = 1,
alpha = (9/2) epsilon.
```

The inverse-series coefficients satisfy:

```text
d_n = -c_n - sum_{k=1}^{n-1} c_k d_{n-k}.
```

Comparing the coefficient of `z^n` in:

```text
(u + z u')^2 = 1/u + alpha z
```

gives, for `n >= 1`:

```text
(2n + 3)c_n =
  alpha 1_{n=1}
  - sum_{k=1}^{n-1} c_k d_{n-k}
  - sum_{k=1}^{n-1} (k+1)(n-k+1)c_k c_{n-k}.
```

The divisor `2n+3` is never zero, so every coefficient is determined by earlier
ones. This recurrence is the constructive lifted-series version of the local
existence statement.

The first coefficients are:

```text
u_epsilon(z) = 1 + a z + b z^2 + c z^3 + O(z^4),
a = (9/10) epsilon,
b = -(3/7) a^2,
c = (23/63) a^3.
```

## Executable Constructor

The homothetic continuation is now executable in
`three_body_symmetry/triple_collision.py`.

```text
construct_homothetic_total_collision_branch(C, lambda, masses, epsilon, order)
```

scales the supplied central configuration by

```text
Q = ((9/2) lambda)^(1/3) C
```

so that `A(Q)=-(2/9)Q`, constructs the scalar coefficients of
`u_epsilon(z)`, and returns a branch object with:

```text
q_i(tau) = Q_i tau^2 u_epsilon(tau^2),
v_i(tau) = Q_i * (2/(3 tau)) * sum_{n>=0} (n+1)c_n tau^(2n).
```

The constructor exposes:

```text
second_regularized_jet = 2Q,
fourth_regularized_jet = 24 c_1 Q = (108/5) epsilon Q,
energy_recurrence_residual_coefficients(),
newton_residual_at_tau(tau).
```

Thus the local branch is no longer only a documented convention. The lifted
series is constructed by the recurrence above, the regularized jets are read
from that same series, and projection to punctured physical time is verified
against Newton's equation by the branch object.

For `tau != 0`, differentiating the energy identity gives:

```text
d^2rho/dt^2 = -(2/9) rho^(-2),
```

because `drho/dt` is nonzero near the collision. Hence:

```text
d^2q_i/dt^2 = C_i d^2rho/dt^2
            = -(2/9) C_i rho^(-2)
            = rho^(-2) A_i(C)
            = A_i(q).
```

Thus every sufficiently small energy-parametrized homothetic branch gives an
analytic regularized continuation through `tau=0`, and it projects to a
Newtonian solution away from the collision.

The second regularized-time jet remains:

```text
d^2q_i/dtau^2 |_{tau=0} = 2 C_i,
```

so it selects the central-configuration branch. The energy enters one order
later:

```text
d^4q_i/dtau^4 |_{tau=0} = 24 a C_i = (108/5) epsilon C_i.
```

## Scope

This does not prove a unique or global zero-angular-momentum triple-collision
continuation theorem. It proves a local analytic continuation for the
homothetic total-collision branches after the missing central-configuration
branch datum has been supplied. The parabolic branch is the special case
`epsilon=0`; the fourth regularized-time jet records the homothetic energy.

The remaining theorem gap is larger:

```text
given arbitrary zero-angular data reaching total collision,
classify or choose the admissible collision-manifold branch,
prove local existence through the selected branch,
prove the projection solves Newton away from collision,
and prove compatibility with the global Sundman/all-time construction.
```

The nonuniqueness lemma shows why the branch datum is necessary. This lemma
shows that, for homothetic total-collision branches, the second regularized jet
selects the central-configuration branch and the next even jet carries the
energy-parametrized local continuation.
