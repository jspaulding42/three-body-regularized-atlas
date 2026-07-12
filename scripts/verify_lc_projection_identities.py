#!/usr/bin/env python3
"""Exact symbolic audit of the planar Levi-Civita projection algebra.

This script checks the algebra used by the regularized vector field in
``three_body_symmetry.binary_chart``.  It deliberately uses independent
SymPy expressions rather than floating-point evaluations.
"""

from __future__ import annotations

import sympy as sp


def _matrix_is_zero(expression: sp.Matrix) -> bool:
    return all(sp.simplify(component) == 0 for component in expression)


def verify_identities() -> dict[str, bool]:
    """Return the exact-zero status of every audited LC identity."""

    z1, z2, w1, w2 = sp.symbols("z1 z2 w1 w2", real=True)
    h, pair_mass, p1, p2 = sp.symbols("h M p1 p2", real=True)
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
    constraint_prime = sp.expand(
        4 * w.dot(w_prime) - rho_prime * h - rho * h_prime
    )

    q_second = matrix_prime * w + matrix * w_prime
    projected_numerator = sp.expand(rho * q_second - rho_prime * q_prime)
    constrained_h = (2 * speed_square - pair_mass) / rho
    newton_relative_numerator = -pair_mass * q + rho**3 * perturbation

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
