#!/usr/bin/env python3
"""Certify one explicit two-sided planar binary-collision passage.

The example is deliberately small and reproducible.  It constructs all
certificates through the public production modules, runs the rigorous
checkers, and emits a machine-readable summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.certificate_checker import (
    check_planar_lc_exact_collision_anchor,
    check_planar_lc_two_sided_collision_passage,
)
from three_body_symmetry.certificate_language import (
    PlanarLCAposterioriTubeCertificate,
    PlanarLCExactCollisionAnchorCertificate,
    PlanarLCTwoSidedCollisionPassageCertificate,
    WeightedOrdinaryAposterioriTubeCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from three_body_symmetry.series import construct_taylor_solution


def certify_passage() -> dict[str, object]:
    """Build and check the explicit collision-passage certificate."""

    endpoint = 1.0e-2
    masses = np.array([1.0, 1.0, 1.2])
    initial = RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([1.0, 0.0]),
        pair_energy=0.0,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.01, -0.02]),
        third_offset=np.array([2.0, 1.0]),
        third_offset_velocity=np.array([-0.01, 0.015]),
    )
    solution = construct_regularized_binary_taylor_solution(initial, order=12)
    source_chart = planar_levi_civita_binary_chart_certificate_from_solution(
        solution,
        certificate_id="planar-lc-exact-collision-chart",
        chart_id="planar-lc-exact-collision",
        parameter_interval=(-endpoint, endpoint),
        coefficient_tolerance=1.0e-10,
        regularized_residual_tolerance=1.0e-7,
        projected_residual_tolerance=1.0,
        tail_bound=1.0e-9,
        sample_count=8,
        projection_rho_lower_bound=1.0e-12,
    )
    source_tube = PlanarLCAposterioriTubeCertificate(
        "planar-lc-tube:two-sided-passage",
        source_chart.chart_id,
        0.0,
        0.0,
        1.0e-5,
        1.0,
        10.0,
        True,
    )
    collision = check_planar_lc_exact_collision_anchor(
        PlanarLCExactCollisionAnchorCertificate(
            "planar-lc-collision:two-sided",
            source_chart.chart_id,
            source_tube.tube_id,
            0.0,
        ),
        source_tube,
        source_chart,
    )

    target_charts = []
    target_tubes = []
    for side, parameter, interval in (
        ("left", -endpoint, (-1.0e-12, 0.0)),
        ("right", endpoint, (0.0, 1.0e-12)),
    ):
        positions, velocities = regularized_binary_collision_chart_to_planar(
            solution.state_at(parameter)
        )
        ordinary_solution = construct_taylor_solution(
            positions, velocities, masses, order=12
        )
        chart = ordinary_taylor_chart_certificate_from_solution(
            ordinary_solution,
            certificate_id=f"ordinary-collision-{side}",
            chart_id=f"ordinary-collision-{side}",
            parameter_interval=interval,
            physical_time_interval=interval,
            coefficient_tolerance=1.0,
            residual_tolerance=1.0,
            tail_bound=0.0,
            sample_count=3,
        )
        tube = WeightedOrdinaryAposterioriTubeCertificate(
            f"weighted-ordinary-collision-{side}",
            chart.chart_id,
            0.0,
            5.0e-6,
            1.0e-1,
            1.0e-5,
            1.0,
            1.0e3,
            1.0e9,
        )
        target_charts.append(chart)
        target_tubes.append(tube)

    passage = check_planar_lc_two_sided_collision_passage(
        PlanarLCTwoSidedCollisionPassageCertificate(
            "planar-lc-passage:two-sided",
            source_chart.chart_id,
            collision.collision_id,
            target_charts[0].chart_id,
            target_charts[1].chart_id,
            -endpoint,
            endpoint,
            0.0,
            0.0,
        ),
        source_chart,
        collision,
        target_charts[0],
        target_charts[1],
        target_tubes[0],
        target_tubes[1],
    )
    left = passage.left_target_tube_result
    right = passage.right_target_tube_result
    return {
        "collision_certified": collision.certified,
        "passage_certified": passage.certified,
        "rho_floors": {
            "left": passage.left_rho_lower_bound,
            "right": passage.right_rho_lower_bound,
        },
        "projection_gaps": {
            "left": passage.left_projection_gap,
            "right": passage.right_projection_gap,
        },
        "weighted_tube_errors": {
            "left": {
                "position": left.proven_position_error_bound,
                "velocity": left.proven_velocity_error_bound,
            },
            "right": {
                "position": right.proven_position_error_bound,
                "velocity": right.proven_velocity_error_bound,
            },
        },
    }


def main() -> int:
    result = certify_passage()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passage_certified"] is True else 1


if __name__ == "__main__":
    sys.exit(main())
