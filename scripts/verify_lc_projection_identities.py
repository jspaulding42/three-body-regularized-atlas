#!/usr/bin/env python3
"""Exact symbolic audit of the planar Levi-Civita projection algebra.

This script checks the algebra used by the regularized vector field in
``three_body_symmetry.binary_chart`` and mirrored by the exact-rational Rust
LC field/projection kernels.  It deliberately uses independent SymPy
expressions rather than floating-point evaluations.
"""

from __future__ import annotations

import sympy as sp


def _matrix_is_zero(expression: sp.Matrix) -> bool:
    return all(sp.simplify(component) == 0 for component in expression)


def verify_identities() -> dict[str, bool]:
    """Return the exact-zero status of every audited LC identity."""

    z1, z2, w1, w2 = sp.symbols("z1 z2 w1 w2", real=True)
    h, pair_mass, p1, p2 = sp.symbols("h M p1 p2", real=True)
    u1, u2 = sp.symbols("U1 U2", real=True)
    v1, v2 = sp.symbols("V1 V2", real=True)
    b1, b2, a1, a2 = sp.symbols("B1 B2 Y1 Y2", real=True)
    mass_i, mass_j, mass_k = sp.symbols("m_i m_j m_k", positive=True)
    inverse_radius_cubed = sp.symbols("r_inv_3", positive=True)
    fi1, fi2, fj1, fj2 = sp.symbols("fi1 fi2 fj1 fj2", real=True)
    y1, y2 = sp.symbols("y1 y2", real=True)
    z = sp.Matrix([z1, z2])
    w = sp.Matrix([w1, w2])
    perturbation = sp.Matrix([p1, p2])
    rho = sp.expand(z.dot(z))
    rho_prime = sp.expand(2 * z.dot(w))
    speed_square = sp.expand(w.dot(w))

    def lc_matrix(vector: sp.Matrix) -> sp.Matrix:
        a, b = vector
        return 2 * sp.Matrix([[a, -b], [b, a]])

    matrix = lc_matrix(z)
    matrix_prime = lc_matrix(w)
    q = sp.Matrix([z1**2 - z2**2, 2 * z1 * z2])
    q_prime = q.jacobian(z) * w

    w_prime = h * z / 2 + rho * matrix.T * perturbation / 4
    h_prime = sp.expand((matrix * w).dot(perturbation))
    constraint_prime = sp.expand(4 * w.dot(w_prime) - rho_prime * h - rho * h_prime)

    q_second = matrix_prime * w + matrix * w_prime
    projected_numerator = sp.expand(rho * q_second - rho_prime * q_prime)
    constrained_h = (2 * speed_square - pair_mass) / rho
    newton_relative_numerator = -pair_mass * q + rho**3 * perturbation

    # The exact nontrivial deck transformation of the planar LC cover.  The
    # force blocks P, B, and Y are invariant because they depend on z only
    # through Q(z); they are represented here by independent invariant
    # symbols so the complete displayed 14-dimensional RHS is checked.
    gamma_z = -z
    gamma_w = -w
    gamma_rho = sp.expand(gamma_z.dot(gamma_z))
    gamma_matrix = lc_matrix(gamma_z)
    gamma_q = sp.Matrix(
        [gamma_z[0] ** 2 - gamma_z[1] ** 2, 2 * gamma_z[0] * gamma_z[1]]
    )
    gamma_w_prime = h * gamma_z / 2 + gamma_rho * gamma_matrix.T * perturbation / 4
    gamma_h_prime = sp.expand((gamma_matrix * gamma_w).dot(perturbation))
    constraint = sp.expand(2 * speed_square - pair_mass - rho * h)
    gamma_constraint = sp.expand(2 * gamma_w.dot(gamma_w) - pair_mass - gamma_rho * h)

    # Full three-body reconstruction from the relative, binary-center, and
    # third-offset blocks.  The force vectors f_i and f_j point from each pair
    # body to the third body.  Keeping them symbolic makes these identities
    # generic in geometry as well as in the three positive masses.
    mass_pair = mass_i + mass_j
    alpha = mass_j / mass_pair
    beta = mass_i / mass_pair
    force_i = sp.Matrix([fi1, fi2])
    force_j = sp.Matrix([fj1, fj2])
    relative_perturbation = mass_k * (force_j - force_i)
    center_acceleration = mass_k * (mass_i * force_i + mass_j * force_j) / mass_pair
    third_acceleration = -mass_i * force_i - mass_j * force_j
    offset_acceleration = third_acceleration - center_acceleration
    relative_acceleration = (
        -mass_pair * inverse_radius_cubed * q + relative_perturbation
    )
    projected_first_acceleration = center_acceleration - alpha * relative_acceleration
    projected_second_acceleration = center_acceleration + beta * relative_acceleration
    projected_third_acceleration = center_acceleration + offset_acceleration
    newton_first_acceleration = mass_j * inverse_radius_cubed * q + mass_k * force_i
    newton_second_acceleration = -mass_i * inverse_radius_cubed * q + mass_k * force_j

    # The third-body displacement blocks depend on z only through Q(z).
    # Their equality under the deck action is the premise needed to conclude
    # invariance of the composed radial inverse-cube forces.
    third_offset = sp.Matrix([y1, y2])
    displacement_i = third_offset + alpha * q
    displacement_j = third_offset - beta * q
    gamma_displacement_i = third_offset + alpha * gamma_q
    gamma_displacement_j = third_offset - beta * gamma_q

    binary_center_velocity = sp.Matrix([u1, u2])
    third_offset_velocity = sp.Matrix([v1, v2])
    binary_center_acceleration = sp.Matrix([b1, b2])
    third_offset_acceleration = sp.Matrix([a1, a2])
    rhs = sp.Matrix(
        [
            *w,
            *w_prime,
            h_prime,
            *(rho * binary_center_velocity),
            *(rho * binary_center_acceleration),
            *(rho * third_offset_velocity),
            *(rho * third_offset_acceleration),
            rho,
        ]
    )
    gamma_rhs = sp.Matrix(
        [
            *gamma_w,
            *gamma_w_prime,
            gamma_h_prime,
            *(gamma_rho * binary_center_velocity),
            *(gamma_rho * binary_center_acceleration),
            *(gamma_rho * third_offset_velocity),
            *(gamma_rho * third_offset_acceleration),
            gamma_rho,
        ]
    )
    deck_derivative = sp.diag(-1, -1, -1, -1, *([1] * 10))

    results = {
        "matrix_gram": _matrix_is_zero(matrix.T * matrix - 4 * rho * sp.eye(2)),
        "matrix_cogram": _matrix_is_zero(matrix * matrix.T - 4 * rho * sp.eye(2)),
        "matrix_times_z": _matrix_is_zero(matrix * z - 2 * q),
        "q_derivative": _matrix_is_zero(q_prime - matrix * w),
        "kinematic_cancellation": _matrix_is_zero(
            rho * matrix_prime * w - rho_prime * matrix * w + 2 * speed_square * q
        ),
        "constraint_invariance": sp.simplify(constraint_prime) == 0,
        "projected_relative_acceleration": _matrix_is_zero(
            projected_numerator.subs(h, constrained_h) - newton_relative_numerator
        ),
        "gauge_square_projection": _matrix_is_zero(gamma_q - q),
        "gauge_rho": sp.simplify(gamma_rho - rho) == 0,
        "gauge_velocity_numerator": _matrix_is_zero(
            gamma_matrix * gamma_w - matrix * w
        ),
        "gauge_pair_energy_constraint": sp.simplify(gamma_constraint - constraint) == 0,
        "gauge_full_rhs_equivariance": _matrix_is_zero(
            gamma_rhs - deck_derivative * rhs
        ),
        "full_newton_first_body_acceleration": _matrix_is_zero(
            projected_first_acceleration - newton_first_acceleration
        ),
        "full_newton_second_body_acceleration": _matrix_is_zero(
            projected_second_acceleration - newton_second_acceleration
        ),
        "full_newton_third_body_acceleration": _matrix_is_zero(
            projected_third_acceleration - third_acceleration
        ),
        "deck_third_body_displacement_first": _matrix_is_zero(
            gamma_displacement_i - displacement_i
        ),
        "deck_third_body_displacement_second": _matrix_is_zero(
            gamma_displacement_j - displacement_j
        ),
    }
    return results


def main() -> int:
    results = verify_identities()
    failed = [name for name, valid in results.items() if not valid]
    if failed:
        print("LC projection identity verification failed: " + ", ".join(failed))
        return 1
    print(f"verified {len(results)} exact LC projection identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
