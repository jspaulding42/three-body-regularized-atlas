from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.continuation import (
    certify_uniform_collision_free_ordinary_chart_ledgers,
    certify_uniformly_collision_free_taylor_recurrence,
)
from three_body_symmetry.dynamics import accelerations, split_state
from three_body_symmetry.general_solution import (
    _attach_finite_time_selector_trace,
    evaluate_unrestricted_solution,
)
from three_body_symmetry.global_invariants import (
    certify_nonzero_angular_momentum_excludes_triple_collision,
)
from three_body_symmetry.event_recurrence import (
    construct_primitive_cauchy_tail_input,
    derive_geometric_shell_event_isolation,
)
from three_body_symmetry.event_regime_assembler import (
    certify_local_chart_family_primitive_cauchy_inputs,
    certify_fuchsian_log_total_collision_chart_family,
    certify_uniform_noncollision_ordinary_gap_chart_family,
    certify_uniform_separated_binary_levi_civita_chart_family,
    derive_event_recurrence_from_chart_family_certificates,
    derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes,
    LocalChartFamilyCauchyCertificate,
)
from three_body_symmetry.escape_endpoint import (
    certify_positive_energy_homothetic_all_real_gluing,
    construct_homothetic_escape_dyadic_recurrence,
    construct_two_ended_scattering_atlas_recurrence,
    derive_time_reversed_homothetic_escape_recurrence,
)
from three_body_symmetry.fuchsian import (
    StableResonanceTerm,
    construct_finite_fuchsian_log_branch_from_stable_chain,
    construct_stable_log_selector_chain,
    mass_inner_product,
)
from three_body_symmetry.general_solution_theorem import (
    certify_compact_time_real_line_coverage,
    certify_global_regime_exhaustion,
    certify_positive_mass_noncollision_input_domain,
    classify_global_regime,
    certify_compact_nonzero_angular_finite_atlas,
    construct_general_solution_theorem_certificate,
    construct_global_atlas_for_regime,
    construct_nonzero_angular_first_event_shell_prefix,
    construct_nonzero_angular_compact_finite_atlas,
    construct_nonzero_angular_finite_middle_atlas_for_event_shell,
    construct_nonzero_angular_global_atlas,
    construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes,
    construct_nonzero_angular_global_atlas_from_uniform_event_envelopes,
    construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes,
    construct_maximal_classical_until_total_collision_atlas,
    construct_positive_energy_homothetic_escape_global_atlas,
    construct_prescribed_two_ended_scattering_global_atlas,
    construct_uniformly_noncollision_bounded_tail_global_atlas,
    construct_zero_angular_compact_finite_atlas_with_selector,
    construct_zero_angular_parabolic_homothetic_total_collision_atlas,
    certify_nonzero_angular_event_regime_handoff,
    certify_nonzero_angular_event_shell_invariance_from_handoff,
    certify_nonzero_angular_event_tail_margin_from_handoff,
    certify_nonzero_angular_finite_middle_atlas,
    certify_two_sided_nonzero_angular_event_budget,
    NonzeroAngularUniformPairEventEnvelopeSpec,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    spatial_to_ks_binary_chart,
)
from three_body_symmetry.ks_binary_series import (
    construct_spatial_ks_binary_taylor_solution,
    interval_spatial_ks_binary_chart_state_from_point,
)
from three_body_symmetry.triple_collision import construct_homothetic_total_collision_branch
from three_body_symmetry.triple_collision import (
    certify_homothetic_total_collision_scalar_majorant,
)
from three_body_symmetry.validated_atlas import (
    FiniteTimeChartSelectorAttempt,
    FiniteTimeChartSelectorTrace,
    partition_ks_state_by_competing_event_order,
    validated_atlas_from_spatial_ks_event_order_partition,
    validated_atlas_from_homothetic_total_collision_branch,
    validated_atlas_from_parabolic_homothetic_total_collision_branch,
)
from three_body_symmetry.zero_angular_entry import (
    certify_homothetic_finite_jet_identity_selector_entry,
)


COMPONENTS = ("value", "first_jet", "lifted_residual", "physical_residual")
ROTATING_TRIANGLE_MASSES = np.array([1.0, 0.7, 1.4])


def _mass_orthonormal_complement_basis(masses, central_shape, *, count):
    masses = np.asarray(masses, dtype=float)
    central_shape = np.asarray(central_shape, dtype=float)
    central_unit = central_shape / np.sqrt(
        mass_inner_product(masses, central_shape, central_shape)
    )
    basis = []
    for coordinate in np.eye(central_shape.size):
        candidate = coordinate.reshape(central_shape.shape)
        candidate -= mass_inner_product(masses, candidate, central_unit) * central_unit
        for previous in basis:
            candidate -= mass_inner_product(masses, candidate, previous) * previous
        norm = np.sqrt(mass_inner_product(masses, candidate, candidate))
        if norm <= 1e-12:
            continue
        basis.append(candidate / norm)
        if len(basis) == count:
            break
    if len(basis) != count:
        raise AssertionError("could not construct enough mass-orthonormal modes")
    return tuple(basis)


def _fuchsian_total_collision_chart_family():
    masses = np.ones(3)
    central_shape = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    stable_rates = (0.7, 0.9, 1.4, 1.6, 2.1)
    chain = construct_stable_log_selector_chain(
        radial_decay=1.0,
        stable_rates=stable_rates,
        source_amplitudes={0: 0.041, 1: -0.029},
        selectors={2: 0.018, 3: -0.016, 4: 0.012},
        resonant_rows={
            2: (StableResonanceTerm(0.73, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(-1.17, (1, 1, 0, 0, 0)),),
            4: (StableResonanceTerm(0.81, (1, 0, 1, 0, 0)),),
        },
    )
    mode_basis = _mass_orthonormal_complement_basis(
        masses,
        central_shape,
        count=len(stable_rates),
    )
    branch = construct_finite_fuchsian_log_branch_from_stable_chain(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        chain=chain,
        mode_shapes={index: mode_basis[index] for index in range(len(stable_rates))},
    )
    return certify_fuchsian_log_total_collision_chart_family(
        branch=branch,
        initial_radius=0.035,
        shell_contraction=0.55,
        analytic_disk_fraction=0.24,
        log_growth_factor=1.12,
        step_ratio_bounds={
            "value": 0.32,
            "first_jet": 0.34,
            "lifted_residual": 0.32,
            "physical_residual": 0.24,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        component_derivative_orders={
            "value": 0,
            "first_jet": 1,
            "lifted_residual": 1,
            "physical_residual": 0,
        },
        component_multipliers={
            "value": 1.0,
            "first_jet": 1.0,
            "lifted_residual": 0.8,
            "physical_residual": 0.15,
        },
        components=COMPONENTS,
    )


def _ordinary_gap_chart_family(masses=ROTATING_TRIANGLE_MASSES):
    return certify_uniform_noncollision_ordinary_gap_chart_family(
        masses=np.asarray(masses, dtype=float),
        pair_distance_lower_bound=0.72,
        pair_diameter_upper_bound=2.8,
        speed_upper_bound=1.35,
        step_ratio_bounds={
            "value": 0.30,
            "first_jet": 0.32,
            "lifted_residual": 0.28,
            "physical_residual": 0.22,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        components=COMPONENTS,
    )


def _separated_binary_chart_family(masses=ROTATING_TRIANGLE_MASSES):
    return certify_uniform_separated_binary_levi_civita_chart_family(
        masses=np.asarray(masses, dtype=float),
        pair=(0, 1),
        z_bound=0.36,
        z_velocity_bound=1.25,
        pair_energy_bound=1.4,
        binary_center_bound=2.6,
        binary_center_velocity_bound=0.75,
        third_offset_bound=2.4,
        third_offset_velocity_bound=0.85,
        third_body_nominal_distance_lower_bound=1.55,
        radii={
            "z": 0.055,
            "z_velocity": 0.080,
            "pair_energy": 0.100,
            "binary_center": 0.120,
            "binary_center_velocity": 0.070,
            "third_offset": 0.110,
            "third_offset_velocity": 0.070,
        },
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        components=COMPONENTS,
    )


def _uniform_binary_pair_envelopes():
    base = {
        "z_bound": 0.36,
        "z_velocity_bound": 1.25,
        "pair_energy_bound": 1.4,
        "binary_center_bound": 2.6,
        "binary_center_velocity_bound": 0.75,
        "third_offset_bound": 2.4,
        "third_offset_velocity_bound": 0.85,
        "third_body_nominal_distance_lower_bound": 1.55,
        "radii": {
            "z": 0.055,
            "z_velocity": 0.080,
            "pair_energy": 0.100,
            "binary_center": 0.120,
            "binary_center_velocity": 0.070,
            "third_offset": 0.110,
            "third_offset_velocity": 0.070,
        },
    }
    return {
        pair: {
            key: (value.copy() if isinstance(value, dict) else value)
            for key, value in base.items()
        }
        for pair in ((0, 1), (0, 2), (1, 2))
    }


def _uniform_pair_event_envelope_spec(**overrides):
    values = {
        "time_direction": "future",
        "delta_initial": 0.2,
        "theta": 0.5,
        "event_isolation_initial": 0.018,
        "boundary_clearance_initial": 0.02,
        "ordinary_pair_distance_lower_bound": 0.72,
        "ordinary_pair_diameter_upper_bound": 2.8,
        "ordinary_speed_upper_bound": 1.35,
        "binary_pair_envelopes": _uniform_binary_pair_envelopes(),
        "step_ratio_bounds": {
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        "retained_order_initials": {
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        "retained_order_increments": {
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        "checked_prefix": 6,
    }
    values.update(overrides)
    return NonzeroAngularUniformPairEventEnvelopeSpec(**values)


def _event_recurrence_from_uniform_pair_spec(masses, spec):
    return derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes(
        masses=masses,
        delta_initial=spec.delta_initial,
        theta=spec.theta,
        event_isolation_initial=spec.event_isolation_initial,
        boundary_clearance_initial=spec.boundary_clearance_initial,
        ordinary_pair_distance_lower_bound=spec.ordinary_pair_distance_lower_bound,
        ordinary_pair_diameter_upper_bound=spec.ordinary_pair_diameter_upper_bound,
        ordinary_speed_upper_bound=spec.ordinary_speed_upper_bound,
        binary_pair_envelopes=spec.binary_pair_envelopes,
        step_ratio_bounds=spec.step_ratio_bounds,
        retained_order_initials=spec.retained_order_initials,
        retained_order_increments=spec.retained_order_increments,
        checked_prefix=spec.checked_prefix,
        components=COMPONENTS,
        time_direction=spec.time_direction,
        provenance=spec.source,
    )


def _input_domain():
    return certify_positive_mass_noncollision_input_domain(
        masses=np.array([1.0, 0.7, 1.4]),
        positions=np.array([[0.8, -0.2], [-0.4, 0.6], [0.1, -0.5]]),
        velocities=np.array([[0.05, 0.11], [-0.07, 0.03], [0.02, -0.08]]),
    )


def _geometric_event_budget():
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
    )
    chart_family_certificates = (
        _ordinary_gap_chart_family(),
        _separated_binary_chart_family(),
        _fuchsian_total_collision_chart_family(),
    )
    assembly = derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=chart_family_certificates,
        checked_prefix=6,
        components=COMPONENTS,
    )
    return shell_isolation, assembly


def _binary_only_event_budget():
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        total_collision_kind=None,
    )
    chart_family_certificates = (
        _ordinary_gap_chart_family(),
        _separated_binary_chart_family(),
    )
    assembly = derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=chart_family_certificates,
        checked_prefix=6,
        components=COMPONENTS,
    )
    return shell_isolation, assembly


def _two_sided_binary_only_event_budget():
    shell_isolation, assembly = _binary_only_event_budget()
    return shell_isolation, assembly, assembly


def _rotating_triangle_data():
    masses = ROTATING_TRIANGLE_MASSES.copy()
    positions = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    angular_speed = 0.25
    velocities = angular_speed * np.column_stack((-positions[:, 1], positions[:, 0]))
    return masses, positions, velocities


def _delta_initial_for_handoff_time(target_time=1.0e-4, *, compact_time_rate=1.3):
    return 1.0 - math.tanh(float(compact_time_rate) * float(target_time))


def _nonzero_angular_finite_middle_certificate(
    *,
    bad_past_direction=False,
    future_target_time=1.0e-4,
    past_target_time=-1.0e-4,
    max_time_step=5.0e-5,
):
    masses, positions, velocities = _rotating_triangle_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    future_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        future_target_time,
        method="validated_atlas",
        order=8,
        max_compact_step=max_time_step,
        max_s_step=0.02,
        target_bisections=20,
    )
    past_atlas = (
        future_atlas
        if bad_past_direction
        else evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            past_target_time,
            method="validated_atlas",
            order=8,
            max_compact_step=max_time_step,
            max_s_step=0.02,
            target_bisections=20,
        )
    )
    return certify_nonzero_angular_finite_middle_atlas(
        input_domain_certificate=input_domain,
        future_validated_atlas=future_atlas,
        past_validated_atlas=past_atlas,
    )


def _parabolic_homothetic_total_collision_branch():
    masses = np.ones(3)
    central_shape = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    acceleration = accelerations(central_shape, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * central_shape * acceleration)
        / np.sum(masses[:, None] * central_shape * central_shape)
    )
    branch = construct_homothetic_total_collision_branch(
        central_shape,
        central_lambda,
        masses,
        energy_per_inertia=0.0,
        order=8,
    )
    return branch


def _homothetic_finite_jet_selector_entry(branch=None):
    if branch is None:
        branch = _parabolic_homothetic_total_collision_branch()
    return certify_homothetic_finite_jet_identity_selector_entry(
        branch,
        sample_taus=(-0.02, 0.02),
        tolerance=1.0e-8,
    )


def _exact_spatial_ks_collision_state():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    return SpatialKSBinaryChartState(
        masses=masses,
        pair=(0, 1),
        u=np.zeros(4),
        u_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0, 0.0, 0.0]),
        pair_energy=-0.3,
        binary_center=np.array([0.0, 0.0, 0.0]),
        binary_center_velocity=np.array([0.2, -0.1, 0.03]),
        third_offset=np.array([1.5, 0.25, -0.35]),
        third_offset_velocity=np.array([-0.03, 0.07, 0.02]),
    )


def _spatial_ks_validated_atlas():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.04)
    )
    validated_atlas = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        5.0e-5,
        method="validated_atlas",
        order=24,
        guard_order=6,
        binary_distance_threshold=1.0e-2,
    )
    return initial.masses, positions, velocities, validated_atlas


def _planar_hybrid_stop_validated_atlas():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1.0e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5.0e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    assert validated_atlas.proof_certified
    assert validated_atlas.collision_policy.total_collision_policy == (
        "finite_time_stop_before_total_collision"
    )
    return masses, positions, velocities, validated_atlas


def _event_order_branch_union_validated_atlas():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -20.0, 0.0],
        ]
    )
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)
    partition = partition_ks_state_by_competing_event_order(
        state,
        target_time_after_start_interval=FloatInterval.point(2.7e-4),
        selected_pair=(0, 1),
        competing_enter_distance=0.045,
        competing_s_upper=0.1,
        retained_order=8,
        guard_order=4,
        max_depth=2,
        max_branches=8,
    )
    atlas = validated_atlas_from_spatial_ks_event_order_partition(
        partition,
        retained_order=8,
        guard_order=4,
        competing_pair_min_distance_required=0.0,
        max_competing_repeats=0,
    )
    atlas = _attach_finite_time_selector_trace(
        atlas,
        selected_route_id="spatial_ks_event_order_branch_union",
        attempts=(
            FiniteTimeChartSelectorAttempt(
                route_id="spatial_ks_event_order_branch_union",
                attempted=True,
                selected=True,
                certified=True,
                reason="certified event-order partition consumed by member KS atlases",
            ),
        ),
    )
    state_midpoint = np.array(
        [
            sum(interval.as_tuple()) / 2.0
            for interval in atlas.initial_state_interval
        ],
        dtype=float,
    ).reshape(2, 3, 3)
    return masses, state_midpoint[0], state_midpoint[1], atlas


def _two_ended_scattering_atlas(*, mismatch_future_velocity: bool = False):
    masses = np.array([1.0, 0.7, 1.4])
    velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_velocities = velocities.copy()
    if mismatch_future_velocity:
        future_velocities[0, 0] += 0.03
    return construct_two_ended_scattering_atlas_recurrence(
        masses,
        velocities,
        offsets,
        future_velocities,
        offsets.copy(),
        retained_corrections=3,
        middle_budgets={
            "value": 5.0e-8,
            "first_jet": 5.0e-7,
            "residual": 1.0e-4,
            "physical_residual": 1.0e-10,
        },
        compact_time_rate=1.0e-8,
        initial_start_time=1.0e4,
        max_doublings=0,
    )


def _scattering_middle_state_near_endpoint_invariants(scale: float = 1.0e7):
    masses = np.array([1.0, 0.7, 1.4])
    velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    constraints = np.zeros((3, 6))
    for body_index in range(3):
        x_index = 2 * body_index
        y_index = x_index + 1
        constraints[0, x_index] = masses[body_index]
        constraints[1, y_index] = masses[body_index]
        constraints[2, x_index] = masses[body_index] * velocities[body_index, 1]
        constraints[2, y_index] = -masses[body_index] * velocities[body_index, 0]
    _, _, vh = np.linalg.svd(constraints)
    null_direction = vh[3].reshape(3, 2)
    positions = offsets + float(scale) * null_direction
    return masses, positions, velocities


def _equilateral_homothetic_escape_data(*, velocity_perturbation: float = 0.0):
    masses = np.ones(3)
    positions = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    velocities = 2.0 * positions
    velocities[1, 0] += velocity_perturbation
    return masses, positions, velocities


def test_geometric_event_tail_feeds_theorem_pipeline_without_global_overclaim():
    shell_isolation, event_assembly = _geometric_event_budget()
    ordinary_family = event_assembly.family_certificate("ordinary_gap_taylor")
    separated_binary_family = event_assembly.family_certificate(
        "separated_binary_levi_civita"
    )
    total_collision_family = event_assembly.family_certificate(
        "automatic_identity_selector_total_collision"
    )
    classification = classify_global_regime(
        input_domain_certificate=_input_domain(),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="geometric_infinite_event_tail",
        event_isolation_certificate=event_assembly,
        primitive_cauchy_inputs=event_assembly,
    )
    atlas = construct_global_atlas_for_regime(classification)
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert event_assembly.certified
    assert event_assembly.event_budget is not None
    assert event_assembly.recurrence_closes
    assert ordinary_family.source == (
        "uniform_noncollision_ordinary_gap_chart_family_constructor"
    )
    assert (
        ordinary_family.source_certificate.__class__.__name__
        == "UniformNoncollisionOrdinaryGapCauchyInputs"
    )
    assert ordinary_family.source_certificate.certified
    assert separated_binary_family.source == (
        "uniform_separated_binary_levi_civita_chart_family_constructor"
    )
    assert (
        separated_binary_family.source_certificate.__class__.__name__
        == "UniformSeparatedBinaryLeviCivitaCauchyInputs"
    )
    assert separated_binary_family.source_certificate.certified
    assert separated_binary_family.source_certificate.third_distance_floor > 0.0
    assert separated_binary_family.source_certificate.levi_civita_radius_floor > 0.0
    assert total_collision_family.source == (
        "fuchsian_log_total_collision_chart_family_constructor"
    )
    assert (
        total_collision_family.source_certificate.__class__.__name__
        == "FiniteFuchsianLogPrimitiveCauchyInputs"
    )
    assert total_collision_family.source_certificate.certified
    assert not classification.certified
    assert classification.input_domain_certified
    assert classification.compact_time_coverage_certified
    assert "geometric_event_regime_membership_from_initial_data" in (
        classification.missing_obligations
    )
    assert not atlas.certified
    assert atlas.event_budget is event_assembly
    assert not theorem.regime_theorem_certified
    assert not theorem.full_general_solution_certified
    assert "geometric_event_regime_membership_from_initial_data" in (
        theorem.missing_obligations
    )
    assert theorem.route_summary == (
        "constructive Sundman-atlas theorem assembly remains incomplete"
    )


def test_nonzero_angular_global_atlas_derives_binary_only_event_tail():
    shell_isolation, event_assembly, past_event_assembly = _two_sided_binary_only_event_budget()
    masses, positions, velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=event_assembly,
        past_event_regime_assembly=past_event_assembly,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert shell_isolation.chart_family_counts.keys() == {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
    }
    assert event_assembly.certified
    assert event_assembly.recurrence_closes
    assert atlas.event_budget.future_event_budget.time_direction == "unspecified"
    assert not atlas.event_budget.direction_provenance_certified
    assert atlas.classification.regime_id == "all_time_nonzero_angular"
    assert atlas.classification.input_domain_certified
    assert atlas.classification.compact_time_coverage_certified
    assert atlas.classification.triple_collision_exclusion_certified
    assert "finite_middle_to_event_envelope_handoff" in atlas.missing_obligations
    assert "nonzero_angular_event_family_scope" not in atlas.missing_obligations
    assert "nonzero_angular_all_pair_binary_coverage" in atlas.missing_obligations
    assert "two_sided_event_time_direction_provenance" in atlas.missing_obligations
    assert not atlas.certified
    assert atlas.event_budget.future_event_budget is event_assembly
    assert atlas.event_budget.past_event_budget is past_event_assembly
    assert not theorem.regime_theorem_certified


def test_nonzero_angular_global_atlas_requires_past_time_reversed_event_tail():
    _shell_isolation, event_assembly = _binary_only_event_budget()
    masses, positions, velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=event_assembly,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert atlas.classification.regime_id == "all_time_nonzero_angular"
    assert atlas.event_budget.future_event_budget is event_assembly
    assert atlas.event_budget.past_event_budget is None
    assert not atlas.certified
    assert "past_all_future_event_budget" in atlas.missing_obligations
    assert "two_sided_primitive_cauchy_all_time_budget" in atlas.missing_obligations
    assert not theorem.regime_theorem_certified


def test_nonzero_angular_global_atlas_derives_event_recurrence_from_uniform_envelopes():
    masses, positions, velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas_from_uniform_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        ordinary_pair_distance_lower_bound=0.72,
        ordinary_pair_diameter_upper_bound=2.8,
        ordinary_speed_upper_bound=1.35,
        binary_pair=(0, 1),
        binary_z_bound=0.36,
        binary_z_velocity_bound=1.25,
        binary_pair_energy_bound=1.4,
        binary_center_bound=2.6,
        binary_center_velocity_bound=0.75,
        binary_third_offset_bound=2.4,
        binary_third_offset_velocity_bound=0.85,
        binary_third_body_nominal_distance_lower_bound=1.55,
        binary_radii={
            "z": 0.055,
            "z_velocity": 0.080,
            "pair_energy": 0.100,
            "binary_center": 0.120,
            "binary_center_velocity": 0.070,
            "third_offset": 0.110,
            "third_offset_velocity": 0.070,
        },
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert atlas.classification.regime_id == "all_time_nonzero_angular"
    assert atlas.classification.triple_collision_exclusion_certified
    assert atlas.event_budget.certified
    assert atlas.event_budget.recurrence_closes
    assert atlas.event_budget.future_event_budget.time_direction == "time_reversal_invariant"
    assert atlas.event_budget.past_event_budget.time_direction == "time_reversal_invariant"
    assert atlas.event_budget.direction_provenance_certified
    assert set(atlas.event_budget.chart_family_counts) == {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
    }
    assert min(atlas.event_budget.chart_family_counts.values()) > 0
    assert {
        certificate.kind
        for certificate in atlas.event_budget.chart_family_certificates
    } == {"ordinary_gap_taylor", "separated_binary_levi_civita"}
    assert "nonzero_angular_all_pair_binary_coverage" in atlas.missing_obligations
    assert "two_sided_event_time_direction_provenance" not in atlas.missing_obligations
    assert "finite_middle_to_event_envelope_handoff" in atlas.missing_obligations
    assert not atlas.certified
    assert not theorem.regime_theorem_certified


def test_nonzero_angular_global_atlas_derives_pairwise_binary_event_recurrence():
    masses, positions, velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        ordinary_pair_distance_lower_bound=0.72,
        ordinary_pair_diameter_upper_bound=2.8,
        ordinary_speed_upper_bound=1.35,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert atlas.classification.regime_id == "all_time_nonzero_angular"
    assert atlas.classification.triple_collision_exclusion_certified
    assert atlas.event_budget.certified
    assert atlas.event_budget.recurrence_closes
    assert atlas.event_budget.future_event_budget.time_direction == "time_reversal_invariant"
    assert atlas.event_budget.past_event_budget.time_direction == "time_reversal_invariant"
    assert atlas.event_budget.direction_provenance_certified
    assert atlas.event_budget.future_event_budget is not atlas.event_budget.past_event_budget
    assert set(atlas.event_budget.chart_family_counts) == {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita_01",
        "separated_binary_levi_civita_02",
        "separated_binary_levi_civita_12",
    }
    assert (
        atlas.event_budget.chart_family_counts["ordinary_gap_taylor"]
        == atlas.event_budget.future_event_budget.shell_isolation.event_count_bound + 1
    )
    for family in (
        "separated_binary_levi_civita_01",
        "separated_binary_levi_civita_02",
        "separated_binary_levi_civita_12",
    ):
        assert (
            atlas.event_budget.chart_family_counts[family]
            == atlas.event_budget.future_event_budget.shell_isolation.event_count_bound
        )
        assert atlas.event_budget.family_certificate(family).source_certified
    assert "nonzero_angular_all_pair_binary_coverage" not in atlas.missing_obligations
    assert "two_sided_event_time_direction_provenance" not in atlas.missing_obligations
    assert "finite_middle_to_event_envelope_handoff" in atlas.missing_obligations
    assert not atlas.certified
    assert not theorem.regime_theorem_certified
    assert "finite_middle_to_event_envelope_handoff" in theorem.missing_obligations


def test_nonzero_angular_global_atlas_consumes_finite_middle_atlas_without_handoff_overclaim():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    finite_middle = _nonzero_angular_finite_middle_certificate()
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        ordinary_pair_distance_lower_bound=0.72,
        ordinary_pair_diameter_upper_bound=2.8,
        ordinary_speed_upper_bound=1.35,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        finite_middle_atlas=finite_middle,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert finite_middle.certified
    assert finite_middle.future_finite_atlas.certified
    assert finite_middle.past_finite_atlas.certified
    assert atlas.finite_middle_atlas is finite_middle
    assert atlas.classification.finite_middle_atlas is finite_middle
    assert atlas.classification.event_regime_handoff.certified
    assert abs(
        atlas.classification.event_regime_handoff.future_compact_parameter
        - (1.0 - delta_initial)
    ) < 1.0e-10
    assert atlas.classification.event_regime_handoff.future_metrics[
        "pair_distance_lower"
    ] >= 0.72
    assert "finite_middle_validated_atlas" not in atlas.missing_obligations
    assert "finite_middle_to_event_envelope_handoff" not in atlas.missing_obligations
    assert not atlas.classification.event_tail_margin_certificate.certified
    assert "nonzero_angular_all_future_value_tail_margin" in atlas.missing_obligations
    assert "nonzero_angular_first_event_shell_prefix" in atlas.missing_obligations
    assert "nonzero_angular_all_pair_binary_coverage" not in atlas.missing_obligations
    assert "two_sided_event_time_direction_provenance" not in atlas.missing_obligations
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        atlas.missing_obligations
    )
    assert not atlas.certified
    assert not theorem.regime_theorem_certified


def test_nonzero_angular_global_atlas_derives_finite_middle_at_event_shell_boundary():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        ordinary_pair_distance_lower_bound=0.72,
        ordinary_pair_diameter_upper_bound=2.8,
        ordinary_speed_upper_bound=1.35,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        derive_finite_middle_atlas=True,
    )

    assert atlas.finite_middle_atlas.certified
    assert atlas.classification.finite_middle_atlas is atlas.finite_middle_atlas
    assert atlas.classification.event_regime_handoff.certified
    assert "finite_middle_validated_atlas" not in atlas.missing_obligations
    assert "finite_middle_to_event_envelope_handoff" not in atlas.missing_obligations
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        atlas.missing_obligations
    )
    assert not atlas.certified


def test_nonzero_angular_finite_middle_event_shell_constructor_targets_boundary():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    future_spec = _uniform_pair_event_envelope_spec(
        delta_initial=delta_initial,
    )
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        delta_initial=delta_initial,
        source="time_reversed_past_uniform_pair_event_envelope_spec",
    )
    finite_middle = construct_nonzero_angular_finite_middle_atlas_for_event_shell(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )

    assert finite_middle.certified
    assert finite_middle.future_finite_atlas.validated_atlas.target_time > 0.0
    assert finite_middle.past_finite_atlas.validated_atlas.target_time < 0.0
    assert math.isclose(
        math.tanh(1.3 * finite_middle.future_finite_atlas.validated_atlas.target_time),
        1.0 - delta_initial,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_nonzero_angular_first_event_shell_prefix_constructor_reaches_next_boundary():
    masses, positions, velocities = _rotating_triangle_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time = certify_compact_time_real_line_coverage(1.3)
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    future_spec = _uniform_pair_event_envelope_spec(
        delta_initial=delta_initial,
        theta=0.99,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
    )
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        delta_initial=delta_initial,
        theta=0.99,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
        source="time_reversed_past_uniform_pair_event_envelope_spec",
    )
    finite_middle = construct_nonzero_angular_finite_middle_atlas_for_event_shell(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )
    handoff = certify_nonzero_angular_event_regime_handoff(
        input_domain_certificate=input_domain,
        compact_time_certificate=compact_time,
        finite_middle_atlas=finite_middle,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )
    prefix = construct_nonzero_angular_first_event_shell_prefix(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_handoff=handoff,
    )

    assert handoff.certified
    assert prefix.certified
    assert math.isclose(
        prefix.future_end_compact_parameter,
        1.0 - delta_initial * future_spec.theta,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        prefix.past_end_compact_parameter,
        -1.0 + delta_initial * past_spec.theta,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert prefix.future_prefix_atlas.validated_atlas.target_time > (
        finite_middle.future_finite_atlas.validated_atlas.target_time
    )
    assert prefix.past_prefix_atlas.validated_atlas.target_time < (
        finite_middle.past_finite_atlas.validated_atlas.target_time
    )
    assert "future_first_event_shell_prefix_extends_handoff" not in (
        prefix.missing_obligations
    )
    assert "past_first_event_shell_prefix_extends_handoff" not in (
        prefix.missing_obligations
    )

    mismatched_input_domain = replace(
        input_domain,
        velocities=tuple(
            tuple(value + 0.01 for value in row)
            for row in input_domain.velocities
        ),
    )
    mismatched_handoff = certify_nonzero_angular_event_regime_handoff(
        input_domain_certificate=mismatched_input_domain,
        compact_time_certificate=compact_time,
        finite_middle_atlas=finite_middle,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )
    assert not mismatched_handoff.certified
    assert "finite_middle_input_domain_matches_handoff" in (
        mismatched_handoff.missing_obligations
    )

    mismatched_handoff_for_prefix = replace(
        handoff,
        input_domain_certificate=mismatched_input_domain,
        compact_time_certificate=certify_compact_time_real_line_coverage(1.1),
    )
    mismatched_prefix = construct_nonzero_angular_first_event_shell_prefix(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_handoff=mismatched_handoff_for_prefix,
    )
    assert not mismatched_prefix.certified
    assert "first_event_shell_prefix_input_domain_matches_handoff" in (
        mismatched_prefix.missing_obligations
    )
    assert "first_event_shell_prefix_compact_time_matches_handoff" in (
        mismatched_prefix.missing_obligations
    )


def test_nonzero_angular_global_atlas_derives_first_event_shell_prefix_without_overclaim():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.99,
        event_isolation_initial=0.2,
        boundary_clearance_initial=0.2,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        derive_finite_middle_atlas=True,
        derive_first_event_shell_prefix=True,
    )

    assert atlas.classification.event_regime_handoff.certified
    assert atlas.classification.first_event_shell_prefix.certified
    assert "nonzero_angular_first_event_shell_prefix" not in (
        atlas.missing_obligations
    )
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        atlas.missing_obligations
    )
    assert not atlas.certified


def test_nonzero_angular_event_shell_invariance_certifies_scoped_uniform_regime():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.99,
        event_isolation_initial=0.2,
        boundary_clearance_initial=0.2,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        derive_finite_middle_atlas=True,
        derive_first_event_shell_prefix=True,
        derive_event_shell_invariance=True,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)
    invariance = atlas.classification.event_shell_invariance_certificate

    assert invariance.certified
    assert invariance.future_remaining_value_tail_bound < (
        atlas.classification.event_tail_margin_certificate.future_value_tail_bound
    )
    assert "nonzero_angular_event_shell_invariance_from_handoff" not in (
        atlas.missing_obligations
    )
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)

    foreign_atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions * 1.07,
        velocities=velocities * 0.91,
        compact_time_rate=1.3,
        event_regime_assembly=atlas.event_budget.future_event_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
        finite_middle_atlas=atlas.classification.finite_middle_atlas,
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=atlas.classification.event_tail_margin_certificate,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        event_shell_invariance_certificate=(
            atlas.classification.event_shell_invariance_certificate
        ),
    )
    assert not foreign_atlas.certified
    assert "finite_middle_input_domain_matches_classification" in (
        foreign_atlas.missing_obligations
    )
    assert "nonzero_angular_handoff_input_domain_matches_classification" in (
        foreign_atlas.missing_obligations
    )

    mismatched_margin_handoff = replace(
        atlas.classification.event_tail_margin_certificate,
        event_regime_handoff=object(),
    )
    margin_handoff_invariance = certify_nonzero_angular_event_shell_invariance_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=mismatched_margin_handoff,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        two_sided_event_budget=atlas.event_budget,
    )
    assert not margin_handoff_invariance.certified
    assert "event_tail_margin_uses_same_handoff" in (
        margin_handoff_invariance.missing_obligations
    )

    mismatched_margin_budget = replace(
        atlas.classification.event_tail_margin_certificate,
        two_sided_event_budget=object(),
    )
    margin_budget_invariance = certify_nonzero_angular_event_shell_invariance_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=mismatched_margin_budget,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        two_sided_event_budget=atlas.event_budget,
    )
    assert not margin_budget_invariance.certified
    assert "event_tail_margin_uses_same_two_sided_budget" in (
        margin_budget_invariance.missing_obligations
    )

    mismatched_prefix = replace(
        atlas.classification.first_event_shell_prefix,
        event_regime_handoff=object(),
    )
    prefix_handoff_invariance = certify_nonzero_angular_event_shell_invariance_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=atlas.classification.event_tail_margin_certificate,
        first_event_shell_prefix=mismatched_prefix,
        two_sided_event_budget=atlas.event_budget,
    )
    assert not prefix_handoff_invariance.certified
    assert "first_event_shell_prefix_uses_same_handoff" in (
        prefix_handoff_invariance.missing_obligations
    )

    mismatched_invariance = replace(invariance, event_regime_handoff=object())
    reassembled = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=atlas.event_budget.future_event_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
        finite_middle_atlas=atlas.finite_middle_atlas,
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=atlas.classification.event_tail_margin_certificate,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        event_shell_invariance_certificate=mismatched_invariance,
    )
    assert not reassembled.certified
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        reassembled.missing_obligations
    )

    rebuilt_event_budget = certify_two_sided_nonzero_angular_event_budget(
        future_event_regime_assembly=atlas.event_budget.future_event_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
    )
    swapped_budget_atlas = construct_global_atlas_for_regime(
        atlas.classification,
        event_budget=rebuilt_event_budget,
    )
    assert rebuilt_event_budget.certified
    assert not swapped_budget_atlas.certified
    assert "global_atlas_event_budget_matches_classification" in (
        swapped_budget_atlas.missing_obligations
    )

    swapped_middle_atlas = construct_global_atlas_for_regime(
        atlas.classification,
        finite_middle_atlas=replace(atlas.finite_middle_atlas),
    )
    assert swapped_middle_atlas.finite_middle_atlas.certified
    assert not swapped_middle_atlas.certified
    assert "global_atlas_finite_middle_matches_classification" in (
        swapped_middle_atlas.missing_obligations
    )

    copied_future_half_budget = replace(atlas.event_budget.future_event_budget)
    copied_half_budget = certify_two_sided_nonzero_angular_event_budget(
        future_event_regime_assembly=copied_future_half_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
    )
    copied_half_tail_margin = certify_nonzero_angular_event_tail_margin_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        two_sided_event_budget=copied_half_budget,
    )
    copied_half_invariance = certify_nonzero_angular_event_shell_invariance_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=copied_half_tail_margin,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        two_sided_event_budget=copied_half_budget,
    )
    mismatched_half_atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=atlas.event_budget.future_event_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
        finite_middle_atlas=atlas.finite_middle_atlas,
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=copied_half_tail_margin,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        event_shell_invariance_certificate=copied_half_invariance,
    )
    assert copied_half_tail_margin.certified
    assert copied_half_invariance.certified
    assert not mismatched_half_atlas.certified
    assert "nonzero_angular_all_future_value_tail_margin" in (
        mismatched_half_atlas.missing_obligations
    )
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        mismatched_half_atlas.missing_obligations
    )

    mismatched_binary_envelopes = _uniform_binary_pair_envelopes()
    mismatched_binary_envelopes[(0, 1)]["z_bound"] = 0.30
    mismatched_future_spec = replace(
        atlas.classification.event_regime_handoff.future_envelope_spec,
        binary_pair_envelopes=mismatched_binary_envelopes,
        source="mismatched_future_binary_pair_envelope_spec",
    )
    mismatched_future_budget = _event_recurrence_from_uniform_pair_spec(
        masses,
        mismatched_future_spec,
    )
    mismatched_binary_budget = certify_two_sided_nonzero_angular_event_budget(
        future_event_regime_assembly=mismatched_future_budget,
        past_event_regime_assembly=atlas.event_budget.past_event_budget,
    )
    mismatched_binary_margin = certify_nonzero_angular_event_tail_margin_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        two_sided_event_budget=mismatched_binary_budget,
    )
    mismatched_binary_invariance = certify_nonzero_angular_event_shell_invariance_from_handoff(
        event_regime_handoff=atlas.classification.event_regime_handoff,
        event_tail_margin_certificate=mismatched_binary_margin,
        first_event_shell_prefix=atlas.classification.first_event_shell_prefix,
        two_sided_event_budget=mismatched_binary_budget,
    )
    assert mismatched_future_budget.certified
    assert mismatched_binary_budget.certified
    assert mismatched_binary_margin.certified
    assert not mismatched_binary_invariance.certified
    assert "future_separated_binary_sources_match_envelope" in (
        mismatched_binary_invariance.missing_obligations
    )


def test_nonzero_angular_event_tail_margin_certifies_when_recurrence_tail_fits_envelope():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    finite_middle = _nonzero_angular_finite_middle_certificate()
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.5,
        event_isolation_initial=0.2,
        boundary_clearance_initial=0.2,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        finite_middle_atlas=finite_middle,
    )
    margin = atlas.classification.event_tail_margin_certificate

    assert atlas.classification.event_regime_handoff.certified
    assert margin.certified
    assert margin.future_value_tail_bound > 0.0
    assert margin.future_required_coordinate_margin < margin.future_metric_margins[
        "pair_distance"
    ]
    assert "nonzero_angular_all_future_value_tail_margin" not in (
        atlas.missing_obligations
    )
    assert "nonzero_angular_event_shell_invariance_from_handoff" in (
        atlas.missing_obligations
    )
    assert not atlas.certified


def test_nonzero_angular_event_tail_margin_rejects_raw_boolean_witness():
    with pytest.raises(TypeError, match="raw boolean"):
        certify_nonzero_angular_event_tail_margin_from_handoff(
            event_regime_handoff=True,
            two_sided_event_budget=True,
        )


def test_nonzero_angular_first_event_shell_prefix_rejects_raw_boolean_handoff():
    masses, positions, velocities = _rotating_triangle_data()
    with pytest.raises(TypeError, match="raw boolean"):
        construct_nonzero_angular_first_event_shell_prefix(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=1.3,
            event_regime_handoff=True,
        )


def test_nonzero_angular_event_shell_invariance_rejects_raw_boolean_witnesses():
    with pytest.raises(TypeError, match="raw boolean"):
        certify_nonzero_angular_event_shell_invariance_from_handoff(
            event_regime_handoff=True,
            event_tail_margin_certificate=True,
            first_event_shell_prefix=True,
            two_sided_event_budget=True,
        )


def test_nonzero_angular_event_handoff_rejects_endpoint_outside_ordinary_gap_envelope():
    masses, positions, velocities = _rotating_triangle_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    compact_time = certify_compact_time_real_line_coverage(1.3)
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    finite_middle = _nonzero_angular_finite_middle_certificate()
    future_spec = _uniform_pair_event_envelope_spec(
        delta_initial=delta_initial,
        ordinary_pair_distance_lower_bound=1.9,
    )
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        delta_initial=delta_initial,
        ordinary_pair_distance_lower_bound=1.9,
        source="tight_past_uniform_pair_event_envelope_spec",
    )
    handoff = certify_nonzero_angular_event_regime_handoff(
        input_domain_certificate=input_domain,
        compact_time_certificate=compact_time,
        finite_middle_atlas=finite_middle,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )
    atlas = construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
        finite_middle_atlas=finite_middle,
    )

    assert finite_middle.certified
    assert not handoff.certified
    assert handoff.future_metrics["pair_distance_lower"] < 1.9
    assert "future_endpoint_ordinary_gap_envelope_containment" in (
        handoff.missing_obligations
    )
    assert "past_endpoint_ordinary_gap_envelope_containment" in (
        handoff.missing_obligations
    )
    assert atlas.classification.event_regime_handoff is not None
    assert not atlas.classification.event_regime_handoff.certified
    assert "finite_middle_to_event_envelope_handoff" in atlas.missing_obligations
    assert "nonzero_angular_event_shell_invariance_from_handoff" not in (
        atlas.missing_obligations
    )
    assert not atlas.certified


def test_nonzero_angular_event_handoff_requires_first_shell_time_alignment():
    masses, positions, velocities = _rotating_triangle_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    finite_middle = _nonzero_angular_finite_middle_certificate()
    future_spec = _uniform_pair_event_envelope_spec()
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        source="time_reversed_past_uniform_pair_event_envelope_spec",
    )
    handoff = certify_nonzero_angular_event_regime_handoff(
        input_domain_certificate=input_domain,
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        finite_middle_atlas=finite_middle,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )

    assert finite_middle.certified
    assert not handoff.certified
    assert abs(handoff.future_compact_parameter) < 1.0e-3
    assert "future_finite_middle_reaches_first_event_shell" in (
        handoff.missing_obligations
    )
    assert "past_finite_middle_reaches_first_event_shell" in (
        handoff.missing_obligations
    )
    assert "future_endpoint_ordinary_gap_envelope_containment" not in (
        handoff.missing_obligations
    )


def test_nonzero_angular_finite_middle_rejects_one_sided_direction_reuse():
    masses, positions, velocities = _rotating_triangle_data()
    finite_middle = _nonzero_angular_finite_middle_certificate(
        bad_past_direction=True,
    )
    atlas = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        ordinary_pair_distance_lower_bound=0.72,
        ordinary_pair_diameter_upper_bound=2.8,
        ordinary_speed_upper_bound=1.35,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        finite_middle_atlas=finite_middle,
    )

    assert not finite_middle.certified
    assert "finite_middle_past_time_direction" in finite_middle.missing_obligations
    assert "finite_middle_validated_atlas" in atlas.missing_obligations
    assert not atlas.certified


def test_nonzero_angular_global_atlas_accepts_direction_specific_pair_envelopes():
    masses, positions, velocities = _rotating_triangle_data()
    future_spec = _uniform_pair_event_envelope_spec()
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        event_isolation_initial=0.011,
        boundary_clearance_initial=0.012,
        source="time_reversed_past_uniform_pair_event_envelope_spec",
    )
    atlas = construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        future_envelope_spec=future_spec,
        past_envelope_spec=past_spec,
    )
    future_counts = atlas.event_budget.future_event_budget.chart_family_counts
    past_counts = atlas.event_budget.past_event_budget.chart_family_counts

    assert not atlas.certified
    assert atlas.event_budget.future_event_budget is not atlas.event_budget.past_event_budget
    assert atlas.event_budget.future_event_budget.time_direction == "future"
    assert atlas.event_budget.past_event_budget.time_direction == "time_reversed_past"
    assert atlas.event_budget.direction_provenance_certified
    assert set(future_counts) == set(past_counts)
    assert future_counts["separated_binary_levi_civita_01"] < past_counts[
        "separated_binary_levi_civita_01"
    ]
    assert "nonzero_angular_all_pair_binary_coverage" not in atlas.missing_obligations
    assert "two_sided_event_time_direction_provenance" not in atlas.missing_obligations
    assert "finite_middle_to_event_envelope_handoff" in atlas.missing_obligations


def test_two_sided_uniform_pair_event_envelopes_require_past_all_pair_rows():
    masses, positions, velocities = _rotating_triangle_data()
    future_spec = _uniform_pair_event_envelope_spec()
    bad_past_envelopes = _uniform_binary_pair_envelopes()
    bad_past_envelopes.pop((1, 2))
    past_spec = _uniform_pair_event_envelope_spec(
        time_direction="time_reversed_past",
        binary_pair_envelopes=bad_past_envelopes,
        source="incomplete_past_pair_envelope_spec",
    )

    with pytest.raises(ValueError, match="pairs 01, 02, and 12"):
        construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=1.3,
            future_envelope_spec=future_spec,
            past_envelope_spec=past_spec,
        )


def test_two_sided_uniform_pair_event_envelopes_reject_wrong_past_direction():
    masses, positions, velocities = _rotating_triangle_data()
    future_spec = _uniform_pair_event_envelope_spec()
    wrong_past_spec = _uniform_pair_event_envelope_spec(
        time_direction="future",
        source="wrongly_oriented_past_pair_envelope_spec",
    )

    with pytest.raises(ValueError, match="time_direction"):
        construct_nonzero_angular_global_atlas_from_two_sided_uniform_pair_event_envelopes(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=1.3,
            future_envelope_spec=future_spec,
            past_envelope_spec=wrong_past_spec,
        )


def test_nonzero_angular_global_atlas_blocks_zero_angular_data():
    _shell_isolation, event_assembly, past_event_assembly = _two_sided_binary_only_event_budget()
    masses, positions, _velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=np.zeros_like(positions),
        compact_time_rate=1.3,
        event_regime_assembly=event_assembly,
        past_event_regime_assembly=past_event_assembly,
    )

    assert not atlas.classification.triple_collision_exclusion_certified
    assert not atlas.certified
    assert "nonzero_angular_triple_collision_exclusion" in atlas.missing_obligations


def test_nonzero_angular_global_atlas_rejects_total_collision_event_family():
    _shell_isolation, event_assembly = _geometric_event_budget()
    masses, positions, velocities = _rotating_triangle_data()
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=event_assembly,
        past_event_regime_assembly=event_assembly,
    )

    assert event_assembly.certified
    assert "automatic_identity_selector_total_collision" in event_assembly.chart_family_counts
    assert not atlas.certified
    assert "nonzero_angular_event_family_scope" in atlas.missing_obligations
    assert "no_total_collision_event_family" in atlas.missing_obligations


def test_nonzero_angular_global_atlas_rejects_stripped_event_family_provenance():
    _shell_isolation, event_assembly, past_event_assembly = _two_sided_binary_only_event_budget()
    masses, positions, velocities = _rotating_triangle_data()
    stripped_families = tuple(
        replace(
            certificate,
            source_certificate=None,
            source="manual_tuple_table_after_the_fact",
        )
        if certificate.kind == "separated_binary_levi_civita"
        else certificate
        for certificate in event_assembly.chart_family_certificates
    )
    stripped_assembly = replace(
        event_assembly,
        chart_family_certificates=stripped_families,
    )
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=stripped_assembly,
        past_event_regime_assembly=past_event_assembly,
    )

    assert stripped_assembly.certified
    assert stripped_assembly.recurrence_closes
    assert not atlas.certified
    assert "nonzero_angular_event_family_scope" in atlas.missing_obligations


def test_nonzero_angular_global_atlas_rejects_mismatched_event_family_masses():
    _shell_isolation, event_assembly, past_event_assembly = _two_sided_binary_only_event_budget()
    masses, positions, velocities = _rotating_triangle_data()
    mismatched_families = tuple(
        replace(
            certificate,
            source_certificate=replace(
                certificate.source_certificate,
                masses=(1.2, 0.7, 1.4),
            ),
        )
        if certificate.kind == "ordinary_gap_taylor"
        else certificate
        for certificate in event_assembly.chart_family_certificates
    )
    mismatched_assembly = replace(
        event_assembly,
        chart_family_certificates=mismatched_families,
    )
    atlas = construct_nonzero_angular_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        event_regime_assembly=mismatched_assembly,
        past_event_regime_assembly=past_event_assembly,
    )

    assert mismatched_assembly.certified
    assert mismatched_assembly.recurrence_closes
    assert "nonzero_angular_event_family_scope" not in atlas.missing_obligations
    assert "nonzero_angular_event_family_mass_consistency" in atlas.missing_obligations
    assert not atlas.certified


def test_uniform_collision_free_taylor_recurrence_derives_fixed_step_tail_bound():
    certificate = certify_uniformly_collision_free_taylor_recurrence(
        np.ones(3),
        pair_distance_lower_bound=1.5,
        position_upper_bound=1.1,
        speed_upper_bound=0.8,
        retained_order=12,
        safety=0.25,
    )
    tighter_tail = certificate.chart_tail_bound(14)
    bad_step = replace(
        certificate,
        step_size=1.01 * certificate.time_radius,
        step_ratio=1.01,
        tail_bound=float("inf"),
    )

    assert certificate.certified
    assert certificate.recurrence_closes
    assert certificate.newton_residual_certified
    assert certificate.projection_certified
    assert certificate.transition_certified
    assert certificate.tail_budget_certified
    assert 0.0 < certificate.step_ratio < 1.0
    assert certificate.tail_bound >= certificate.chart_tail_bound()
    assert tighter_tail < certificate.tail_bound
    assert not bad_step.certified
    assert "fixed_step_inside_cauchy_radius" in bad_step.missing_obligations


def test_uniform_collision_free_initial_chart_ledgers_are_constructor_derived():
    masses = np.ones(3)
    angular_speed = 3.0 ** (-0.25)
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    velocities = angular_speed * np.stack([-np.sin(angles), np.cos(angles)], axis=1)
    recurrence = certify_uniformly_collision_free_taylor_recurrence(
        masses,
        pair_distance_lower_bound=0.99 * math.sqrt(3.0),
        position_upper_bound=1.01,
        speed_upper_bound=1.01 * angular_speed,
        retained_order=12,
        safety=0.25,
    )
    ledgers = certify_uniform_collision_free_ordinary_chart_ledgers(
        positions,
        velocities,
        masses,
        recurrence,
    )
    bad_projection = replace(ledgers, chart_family="opaque_lift")

    assert ledgers.certified
    assert ledgers.newton_residual_certified
    assert ledgers.projection_certified
    assert ledgers.invariants_certified
    assert ledgers.transition_certified
    assert ledgers.tail_budget_certified
    assert ledgers.missing_obligations == ()
    assert not bad_projection.certified
    assert "ordinary_physical_projection" in bad_projection.missing_obligations


def test_uniformly_noncollision_bounded_tail_global_atlas_enters_theorem_pipeline():
    masses = np.ones(3)
    angular_speed = 3.0 ** (-0.25)
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    velocities = angular_speed * np.stack([-np.sin(angles), np.cos(angles)], axis=1)

    atlas = construct_uniformly_noncollision_bounded_tail_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0,
        pair_distance_lower_bound=0.99 * math.sqrt(3.0),
        centered_position_upper_bound=1.01,
        centered_speed_upper_bound=1.01 * angular_speed,
        retained_order=12,
        safety=0.25,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert atlas.classification.regime_id == "uniformly_noncollision_bounded_tail"
    assert atlas.certified
    assert atlas.ordinary_gap_atlas.recurrence_closes
    assert atlas.ordinary_gap_atlas.ordinary_chart_ledgers.certified
    assert atlas.ordinary_gap_atlas.recurrence.tail_bound < 1e-6
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_uniformly_noncollision_bounded_tail_rejects_bounds_missing_initial_state():
    masses = np.ones(3)
    angular_speed = 3.0 ** (-0.25)
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    velocities = angular_speed * np.stack([-np.sin(angles), np.cos(angles)], axis=1)

    atlas = construct_uniformly_noncollision_bounded_tail_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0,
        pair_distance_lower_bound=1.01 * math.sqrt(3.0),
        centered_position_upper_bound=1.01,
        centered_speed_upper_bound=1.01 * angular_speed,
        retained_order=12,
        safety=0.25,
    )
    ordinary_certificate = atlas.classification.ordinary_gap_envelope

    assert not ordinary_certificate.certified
    assert "initial_state_inside_uniform_centered_bounds" in (
        ordinary_certificate.missing_obligations
    )
    assert "ordinary_gap_envelope" in atlas.missing_obligations
    assert not atlas.certified


def test_nonzero_angular_compact_finite_atlas_consumes_validated_solution():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert validated_atlas.proof_certified
    assert atlas.classification.regime_id == "compact_nonzero_angular_finite_events"
    assert atlas.classification.triple_collision_exclusion_certified
    assert atlas.certified
    assert atlas.validated_atlas is validated_atlas
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_compact_finite_atlas_rejects_attribute_compatible_validated_atlas():
    masses, positions, velocities = _rotating_triangle_data()
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    packed_initial = np.concatenate(
        [positions.reshape(-1), velocities.reshape(-1)],
    )
    state_interval = np.asarray(
        [FloatInterval.point(float(value)) for value in packed_initial],
        dtype=object,
    )
    fake_validated_atlas = SimpleNamespace(
        proof_certified=True,
        certified=True,
        masses=tuple(float(mass) for mass in masses),
        target_time=1.0e-4,
        target_time_certified=True,
        initial_state_interval=state_interval,
        target_state_interval=state_interval,
        charts=(
            SimpleNamespace(
                chart_id="fake_chart",
                chart_type="sundman",
                dynamics_certified=True,
                residual_certified=True,
                projection_certified=True,
                invariants_certified=True,
                tail_certified=True,
                tail_bound=0.0,
                parameter_interval=FloatInterval(0.0, 1.0e-4),
                physical_time_interval=FloatInterval(0.0, 1.0e-4),
            ),
        ),
        transitions=(),
        residual_budget=SimpleNamespace(
            certified=True,
            expected_chart_count=1,
            certified_chart_count=1,
        ),
        invariants=SimpleNamespace(
            certified=True,
            expected_chart_count=1,
            certified_chart_count=1,
        ),
        tail_budget=SimpleNamespace(certified=True, finite=True),
        collision_policy=SimpleNamespace(
            certified=True,
            binary_policy="ordinary",
            total_collision_policy="maximal_classical_stop",
        ),
        proof_ledger=SimpleNamespace(
            certified=True,
            entries=(
                SimpleNamespace(name="target_time", certified=True),
            ),
        ),
        selector_trace=None,
    )

    finite_certificate = certify_compact_nonzero_angular_finite_atlas(
        input_domain_certificate=input_domain,
        validated_atlas=fake_validated_atlas,
    )

    assert not finite_certificate.certified
    assert "validated_atlas_solution" in finite_certificate.missing_obligations
    obligation = {
        item.obligation: item for item in finite_certificate.obligations
    }["validated_atlas_solution"]
    assert "actual ValidatedAtlasSolution" in obligation.detail


def test_global_atlas_assembly_rejects_swapped_validated_atlas_object():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    swapped = construct_global_atlas_for_regime(
        atlas.classification,
        validated_atlas=replace(validated_atlas),
    )

    assert atlas.certified
    assert swapped.validated_atlas.proof_certified
    assert not swapped.certified
    assert "global_atlas_validated_atlas_matches_classification" in (
        swapped.missing_obligations
    )


def test_compact_nonzero_finite_regime_rejects_foreign_embedded_input_domain():
    masses, positions, velocities = _rotating_triangle_data()
    shifted_positions = positions + np.array([0.31, -0.17])
    shifted_atlas = evaluate_unrestricted_solution(
        masses,
        shifted_positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    shifted_input = certify_positive_mass_noncollision_input_domain(
        masses,
        shifted_positions,
        velocities,
    )
    shifted_finite = certify_compact_nonzero_angular_finite_atlas(
        input_domain_certificate=shifted_input,
        validated_atlas=shifted_atlas,
    )
    classification = classify_global_regime(
        input_domain_certificate=certify_positive_mass_noncollision_input_domain(
            masses,
            positions,
            velocities,
        ),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="compact_nonzero_angular_finite_events",
        ordinary_gap_envelope=shifted_finite,
        separated_binary_envelope=shifted_finite,
        triple_collision_exclusion_certificate=(
            certify_nonzero_angular_momentum_excludes_triple_collision(
                positions,
                velocities,
                masses,
            )
        ),
    )
    rebuilt = construct_global_atlas_for_regime(
        classification,
        validated_atlas=shifted_atlas,
    )

    assert shifted_atlas.proof_certified
    assert shifted_finite.certified
    assert not classification.certified
    assert "ordinary_gap_input_domain_matches_classification" in (
        classification.missing_obligations
    )
    assert "separated_binary_input_domain_matches_classification" in (
        classification.missing_obligations
    )
    assert not rebuilt.certified
    assert "ordinary_gap_input_domain_matches_classification" in (
        rebuilt.missing_obligations
    )


def test_nonzero_angular_compact_finite_atlas_rejects_stale_selector_trace():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    bad_atlas = replace(
        validated_atlas,
        selector_trace=replace(
            validated_atlas.selector_trace,
            selected_route_id="sundman",
        ),
    )
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=bad_atlas,
    )
    finite_certificate = atlas.classification.ordinary_gap_envelope

    assert validated_atlas.selector_trace.certified
    assert not bad_atlas.selector_trace.certified
    assert not finite_certificate.certified
    assert "finite_atlas_selector_trace" in finite_certificate.missing_obligations


def test_nonzero_angular_compact_finite_atlas_rejects_selector_chart_mismatch():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    mismatched_trace = FiniteTimeChartSelectorTrace(
        selected_route_id="auto_spatial_ks",
        attempts=(
            FiniteTimeChartSelectorAttempt(
                route_id="auto_spatial_ks",
                attempted=True,
                selected=True,
                certified=True,
                reason="stale selector trace from a spatial KS route",
            ),
        ),
    )
    bad_atlas = replace(validated_atlas, selector_trace=mismatched_trace)
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=bad_atlas,
    )
    finite_certificate = atlas.classification.ordinary_gap_envelope

    assert mismatched_trace.certified
    assert "spatial_ks_binary" not in {
        chart.chart_type for chart in bad_atlas.charts
    }
    assert not finite_certificate.certified
    assert "finite_atlas_selector_trace" in finite_certificate.missing_obligations


def test_nonzero_angular_compact_finite_atlas_requires_explicit_validated_ledgers():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    assert validated_atlas.transitions

    cases = (
        (
            "finite_atlas_newton_residual_ledger",
            replace(
                validated_atlas,
                residual_budget=replace(validated_atlas.residual_budget, certified=False),
            ),
        ),
        (
            "finite_atlas_mass_consistency",
            replace(
                validated_atlas,
                masses=tuple(float(mass) * 1.03 for mass in validated_atlas.masses),
            ),
        ),
        (
            "finite_atlas_initial_state_consistency",
            replace(
                validated_atlas,
                initial_state_interval=np.asarray(
                    [
                        FloatInterval(interval.upper + 10.0, interval.upper + 11.0)
                        for interval in validated_atlas.initial_state_interval
                    ],
                    dtype=object,
                ),
            ),
        ),
        (
            "finite_atlas_invariant_ledger",
            replace(
                validated_atlas,
                invariants=replace(validated_atlas.invariants, energy_certified=False),
            ),
        ),
        (
            "finite_atlas_tail_budget",
            replace(
                validated_atlas,
                tail_budget=replace(validated_atlas.tail_budget, certified=False),
            ),
        ),
        (
            "finite_atlas_target_interval",
            replace(
                validated_atlas,
                evaluation=replace(
                    validated_atlas.evaluation,
                    target_state_interval=np.asarray([], dtype=object),
                ),
            ),
        ),
        (
            "finite_atlas_target_time_domain",
            replace(
                validated_atlas,
                target_time=validated_atlas.target_time + 0.5,
            ),
        ),
        (
            "finite_atlas_targeting_or_containment",
            replace(
                validated_atlas,
                proof_ledger=replace(
                    validated_atlas.proof_ledger,
                    entries=tuple(
                        replace(entry, certified=False)
                        if entry.name == "hybrid_target_containment"
                        else entry
                        for entry in validated_atlas.proof_ledger.entries
                    ),
                ),
            ),
        ),
        (
            "finite_atlas_transition_ledger",
            replace(
                validated_atlas,
                transitions=(
                    replace(
                        validated_atlas.transitions[0],
                        target_chart_id="not_the_next_chart",
                    ),
                    *validated_atlas.transitions[1:],
                ),
            ),
        ),
    )

    for missing_obligation, bad_atlas in cases:
        atlas = construct_nonzero_angular_compact_finite_atlas(
            masses=masses,
            positions=positions,
            velocities=velocities,
            compact_time_rate=1.3,
            validated_atlas=bad_atlas,
        )
        finite_certificate = atlas.classification.ordinary_gap_envelope

        assert not finite_certificate.certified
        assert missing_obligation in finite_certificate.missing_obligations
        assert "ordinary_gap_envelope" in atlas.missing_obligations
        assert "separated_binary_envelope" in atlas.missing_obligations


def test_nonzero_angular_compact_finite_atlas_accepts_proof_certified_spatial_ks_solution():
    masses, positions, velocities, validated_atlas = _spatial_ks_validated_atlas()
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert validated_atlas.proof_certified
    assert "spatial_ks_binary" in {
        chart.chart_type for chart in validated_atlas.charts
    }
    assert atlas.classification.regime_id == "compact_nonzero_angular_finite_events"
    assert atlas.classification.triple_collision_exclusion_certified
    assert atlas.certified
    assert atlas.classification.ordinary_gap_envelope.certified
    assert atlas.classification.ordinary_gap_envelope.missing_obligations == ()
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_nonzero_angular_compact_finite_atlas_accepts_spatial_event_order_branch_union():
    masses, positions, velocities, validated_atlas = (
        _event_order_branch_union_validated_atlas()
    )
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    finite_certificate = atlas.classification.ordinary_gap_envelope
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert validated_atlas.proof_certified
    assert validated_atlas.selector_trace.selected_route_id == (
        "spatial_ks_event_order_branch_union"
    )
    assert finite_certificate.certified
    assert finite_certificate.chart_types == ("spatial_ks_event_order_branch_union",)
    assert finite_certificate.separated_binary_chart_count == 1
    assert finite_certificate.missing_obligations == ()
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_spatial_event_order_branch_union_theorem_gate_rejects_selector_mismatch():
    masses, positions, velocities, validated_atlas = (
        _event_order_branch_union_validated_atlas()
    )
    mismatched = _attach_finite_time_selector_trace(
        validated_atlas,
        selected_route_id="planar_hybrid",
        attempts=(
            FiniteTimeChartSelectorAttempt(
                route_id="planar_hybrid",
                attempted=True,
                selected=True,
                certified=True,
                reason="deliberately mismatched selector route",
            ),
        ),
    )
    input_domain = certify_positive_mass_noncollision_input_domain(
        masses,
        positions,
        velocities,
    )
    finite_certificate = certify_compact_nonzero_angular_finite_atlas(
        input_domain_certificate=input_domain,
        validated_atlas=mismatched,
    )

    assert not finite_certificate.certified
    assert "finite_atlas_selector_trace" in finite_certificate.missing_obligations
    assert "ordinary_or_separated_binary_chart_families" not in (
        finite_certificate.missing_obligations
    )


def test_nonzero_angular_compact_finite_atlas_rejects_uncertified_spatial_ks_policy():
    masses, positions, velocities, validated_atlas = _spatial_ks_validated_atlas()
    bad_policy = replace(validated_atlas.collision_policy, certified=False)
    bad_atlas = replace(validated_atlas, collision_policy=bad_policy)
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=bad_atlas,
    )
    finite_certificate = atlas.classification.ordinary_gap_envelope

    assert not finite_certificate.certified
    assert "spatial_binary_regularization_scope" in finite_certificate.missing_obligations
    assert not atlas.certified


def test_nonzero_angular_compact_finite_atlas_rejects_total_collision_chart():
    masses, positions, velocities = _rotating_triangle_data()
    validated_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    bad_chart = replace(
        validated_atlas.charts[0],
        chart_type="automatic_identity_selector_total_collision",
    )
    bad_atlas = replace(
        validated_atlas,
        charts=(bad_chart, *validated_atlas.charts[1:]),
    )
    atlas = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=bad_atlas,
    )

    assert not atlas.certified
    assert "ordinary_gap_envelope" in atlas.missing_obligations
    assert "separated_binary_envelope" in atlas.missing_obligations


def test_maximal_classical_until_total_collision_stop_feeds_global_atlas_without_selector_continuation():
    masses, positions, velocities, validated_atlas = _planar_hybrid_stop_validated_atlas()
    atlas = construct_maximal_classical_until_total_collision_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)
    stop_certificate = atlas.classification.ordinary_gap_envelope

    assert atlas.classification.regime_id == "maximal_classical_until_total_collision"
    assert stop_certificate.certified
    assert stop_certificate.validated_atlas is validated_atlas
    assert stop_certificate.total_collision_policy == "finite_time_stop_before_total_collision"
    assert "maximal_classical_stop_before_total_collision_policy" not in (
        stop_certificate.missing_obligations
    )
    assert "maximal_classical_no_selector_continuation_policy" not in (
        stop_certificate.missing_obligations
    )
    assert atlas.ordinary_gap_atlas is stop_certificate
    assert atlas.validated_atlas is validated_atlas
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_maximal_classical_stop_rejects_selector_continuation_policy():
    masses, positions, velocities, validated_atlas = _planar_hybrid_stop_validated_atlas()
    selector_policy = replace(
        validated_atlas.collision_policy,
        total_collision_policy="finite_time_identity_selector_total_collision",
    )
    selector_atlas = replace(validated_atlas, collision_policy=selector_policy)
    atlas = construct_maximal_classical_until_total_collision_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
    )
    stop_certificate = atlas.classification.ordinary_gap_envelope

    assert not stop_certificate.certified
    assert "maximal_classical_stop_before_total_collision_policy" in (
        stop_certificate.missing_obligations
    )
    assert "maximal_classical_no_selector_continuation_policy" in (
        stop_certificate.missing_obligations
    )
    assert not atlas.certified


def test_global_regime_exhaustion_required_ids_accepts_maximal_classical_stop_candidate():
    masses, positions, velocities, validated_atlas = _planar_hybrid_stop_validated_atlas()
    maximal = construct_maximal_classical_until_total_collision_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=validated_atlas,
    )
    exhaustion = certify_global_regime_exhaustion(
        candidate_regimes=(maximal,),
        required_regime_ids=("maximal_classical_until_total_collision",),
    )
    theorem = construct_general_solution_theorem_certificate(
        maximal,
        global_regime_exhaustion_certificate=exhaustion,
    )

    assert maximal.certified
    assert not exhaustion.certified
    assert "global_exhaustion_required_regime_ids" not in (
        exhaustion.missing_obligations
    )
    assert exhaustion.missing_obligations == (
        "arbitrary_initial_data_partition_theorem",
    )
    assert not theorem.full_general_solution_certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_global_regime_exhaustion_rejects_mixed_candidate_input_domains():
    masses, positions, velocities, maximal_atlas = _planar_hybrid_stop_validated_atlas()
    maximal = construct_maximal_classical_until_total_collision_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=maximal_atlas,
    )
    shifted_positions = positions + np.array([0.27, -0.19])
    compact_atlas = evaluate_unrestricted_solution(
        masses,
        shifted_positions,
        velocities,
        1.0e-4,
        method="validated_atlas",
        order=8,
        max_compact_step=5.0e-5,
        max_s_step=0.02,
        target_bisections=20,
    )
    compact_nonzero = construct_nonzero_angular_compact_finite_atlas(
        masses=masses,
        positions=shifted_positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=compact_atlas,
    )
    exhaustion = certify_global_regime_exhaustion(
        candidate_regimes=(maximal, compact_nonzero),
        required_regime_ids=(
            "maximal_classical_until_total_collision",
            "compact_nonzero_angular_finite_events",
        ),
    )

    assert maximal.certified
    assert compact_nonzero.certified
    assert not exhaustion.candidate_input_domains_consistent
    assert not exhaustion.certified
    assert "global_exhaustion_candidate_input_domains" in (
        exhaustion.missing_obligations
    )
    assert "global_exhaustion_required_regime_ids" not in (
        exhaustion.missing_obligations
    )


def test_zero_angular_compact_finite_atlas_requires_constructor_selector_entry():
    branch = _parabolic_homothetic_total_collision_branch()
    selector_atlas = validated_atlas_from_parabolic_homothetic_total_collision_branch(
        branch,
        start_tau=-0.04,
        target_tau=0.04,
        tolerance=1.0e-8,
    )
    positions, velocities = split_state(selector_atlas.evaluation.initial_state)
    masses = np.asarray(branch.masses, dtype=float)
    selector_entry = _homothetic_finite_jet_selector_entry(branch)
    atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(selector_entry,),
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert selector_atlas.proof_certified
    assert selector_atlas.charts[0].source == "parabolic_homothetic_total_collision"
    assert selector_entry.certified
    assert atlas.classification.regime_id == "compact_zero_angular_finite_events_with_selector"
    assert atlas.classification.total_collision_selector_envelope.identity_selector_certified
    assert atlas.classification.ordinary_gap_envelope.total_collision_chart_count == 1
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)

    foreign_masses, foreign_positions, foreign_velocities = _rotating_triangle_data()
    foreign_classification = classify_global_regime(
        input_domain_certificate=certify_positive_mass_noncollision_input_domain(
            foreign_masses,
            foreign_positions,
            foreign_velocities,
        ),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="compact_zero_angular_finite_events_with_selector",
        ordinary_gap_envelope=atlas.classification.ordinary_gap_envelope,
        total_collision_selector_envelope=(
            atlas.classification.total_collision_selector_envelope
        ),
    )
    foreign_rebuilt = construct_global_atlas_for_regime(
        foreign_classification,
        validated_atlas=selector_atlas,
    )
    assert not foreign_classification.certified
    assert "ordinary_gap_input_domain_matches_classification" in (
        foreign_classification.missing_obligations
    )
    assert "total_collision_selector_input_domain_matches_classification" in (
        foreign_classification.missing_obligations
    )
    assert not foreign_rebuilt.certified

    missing_selector = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(),
    )

    assert not missing_selector.certified
    assert "finite_atlas_total_collision_selector_coverage" in (
        missing_selector.classification.ordinary_gap_envelope.missing_obligations
    )
    assert "ordinary_gap_envelope" in missing_selector.missing_obligations
    assert "total_collision_selector_envelope" in missing_selector.missing_obligations

    wrong_mass_selector = replace(selector_entry, masses=(2.0, 1.0, 1.0))
    wrong_mass_atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(wrong_mass_selector,),
    )
    assert not wrong_mass_atlas.certified
    assert "finite_atlas_total_collision_selector_mass_consistency" in (
        wrong_mass_atlas.classification.ordinary_gap_envelope.missing_obligations
    )

    wrong_energy_selector = _homothetic_finite_jet_selector_entry(
        replace(branch, energy_per_inertia=0.2),
    )
    wrong_energy_atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(wrong_energy_selector,),
    )
    assert not wrong_energy_atlas.certified
    assert "finite_atlas_total_collision_selector_branch_compatibility" in (
        wrong_energy_atlas.classification.ordinary_gap_envelope.missing_obligations
    )

    out_of_chart_selector = replace(selector_entry, sample_taus=(-0.08, 0.02))
    out_of_chart_atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(out_of_chart_selector,),
    )
    assert not out_of_chart_atlas.certified
    assert "finite_atlas_total_collision_selector_branch_compatibility" in (
        out_of_chart_atlas.classification.ordinary_gap_envelope.missing_obligations
    )

    fake_selector = SimpleNamespace(
        certified=True,
        identity_selector_certified=True,
        proof_certified=True,
        masses=tuple(float(mass) for mass in masses),
        selected_energy_limit=(
            float(branch.energy_per_inertia) * float(branch.inertia)
        ),
        selected_central_coefficient=branch.quadratic_coefficient,
        sample_taus=(-0.02, 0.02),
    )
    fake_selector_atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(fake_selector,),
    )
    assert not fake_selector_atlas.certified
    assert "finite_atlas_total_collision_selector_coverage" in (
        fake_selector_atlas.classification.ordinary_gap_envelope.missing_obligations
    )


def test_parabolic_homothetic_total_collision_constructor_derives_selector_pipeline():
    branch = _parabolic_homothetic_total_collision_branch()
    atlas = construct_zero_angular_parabolic_homothetic_total_collision_atlas(
        branch=branch,
        start_tau=-0.04,
        target_tau=0.04,
        compact_time_rate=1.3,
        atlas_tolerance=1.0e-8,
        selector_tolerance=1.0e-8,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert atlas.certified
    assert atlas.validated_atlas.proof_certified
    assert atlas.validated_atlas.charts[0].chart_type == (
        "finite_jet_identity_selector_total_collision"
    )
    assert atlas.classification.total_collision_selector_envelope.identity_selector_certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_zero_angular_compact_finite_atlas_rejects_nonselector_collision_policy():
    branch = _parabolic_homothetic_total_collision_branch()
    validated_atlas = validated_atlas_from_parabolic_homothetic_total_collision_branch(
        branch,
        start_tau=-0.04,
        target_tau=0.04,
        tolerance=1.0e-8,
    )
    bad_atlas = replace(
        validated_atlas,
        collision_policy=replace(
            validated_atlas.collision_policy,
            total_collision_policy="finite_time_stop_before_total_collision",
        ),
    )
    positions, velocities = split_state(bad_atlas.evaluation.initial_state)
    masses = np.asarray(branch.masses, dtype=float)
    atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=bad_atlas,
        total_collision_selector_envelopes=(_homothetic_finite_jet_selector_entry(branch),),
    )

    assert not atlas.certified
    assert "finite_atlas_total_collision_selector_policy" in (
        atlas.classification.ordinary_gap_envelope.missing_obligations
    )


def test_parabolic_homothetic_total_collision_adapter_rejects_nonzero_energy_tail():
    branch = _parabolic_homothetic_total_collision_branch()
    nonzero_energy_branch = replace(
        branch,
        energy_per_inertia=0.02,
    )

    with pytest.raises(ValueError, match="zero tail"):
        validated_atlas_from_parabolic_homothetic_total_collision_branch(
            nonzero_energy_branch,
            start_tau=-0.04,
            target_tau=0.04,
            tolerance=1.0e-8,
        )


def test_nonzero_energy_homothetic_total_collision_scalar_majorant_bounds_tail():
    branch = replace(
        _parabolic_homothetic_total_collision_branch(),
        energy_per_inertia=0.02,
        order=8,
    )
    majorant = certify_homothetic_total_collision_scalar_majorant(
        branch,
        z_radius=0.01,
        u_radius=0.2,
    )
    weak_majorant = certify_homothetic_total_collision_scalar_majorant(
        branch,
        z_radius=20.0,
        u_radius=0.2,
    )

    assert branch.certified_scaled_central_configuration
    assert (
        np.linalg.norm(branch.energy_recurrence_residual_coefficients(), ord=np.inf)
        < 1.0e-12
    )
    assert majorant.certified
    assert majorant.rouche_margin > 0.0
    assert (
        0.0
        < majorant.scalar_tail_bound(max_abs_tau=0.04, retained_order=branch.order)
        < 1.0e-6
    )
    assert not weak_majorant.certified
    assert np.isinf(
        weak_majorant.scalar_tail_bound(max_abs_tau=0.04, retained_order=branch.order)
    )


def test_nonzero_energy_homothetic_total_collision_adapter_feeds_selector_pipeline():
    branch = replace(
        _parabolic_homothetic_total_collision_branch(),
        energy_per_inertia=0.02,
        order=12,
    )
    selector_atlas = validated_atlas_from_homothetic_total_collision_branch(
        branch,
        start_tau=-0.04,
        target_tau=0.04,
        z_radius=0.01,
        u_radius=0.2,
        tolerance=1.0e-6,
    )
    positions, velocities = split_state(selector_atlas.evaluation.initial_state)
    selector_entry = certify_homothetic_finite_jet_identity_selector_entry(
        branch,
        sample_taus=(-0.04, 0.04),
        tolerance=1.0e-6,
    )
    atlas = construct_zero_angular_compact_finite_atlas_with_selector(
        masses=branch.masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        validated_atlas=selector_atlas,
        total_collision_selector_envelopes=(selector_entry,),
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert selector_atlas.proof_certified
    assert selector_atlas.charts[0].source == "homothetic_total_collision_energy_series"
    assert selector_atlas.charts[0].tail_bound > 0.0
    assert selector_atlas.evaluation.certificate.scalar_majorant.certified
    assert selector_entry.certified
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_prescribed_two_ended_scattering_feeds_global_atlas_with_invariant_match():
    masses, positions, velocities = _scattering_middle_state_near_endpoint_invariants()
    scattering_atlas = _two_ended_scattering_atlas()
    atlas = construct_prescribed_two_ended_scattering_global_atlas(
        masses=masses,
        middle_positions=positions,
        middle_velocities=velocities,
        compact_time_rate=1.0e-8,
        scattering_atlas=scattering_atlas,
        invariant_tolerance=2.0e-6,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert scattering_atlas.certified
    assert atlas.classification.regime_id == "prescribed_two_ended_scattering"
    assert atlas.scattering_atlas.certified
    assert atlas.scattering_atlas.invariant_match_certificate.certified
    assert atlas.scattering_atlas.middle_invariant_match_certificate.certified
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_prescribed_two_ended_scattering_rejects_middle_invariant_mismatch():
    masses, positions, velocities = _rotating_triangle_data()
    scattering_atlas = _two_ended_scattering_atlas()
    atlas = construct_prescribed_two_ended_scattering_global_atlas(
        masses=masses,
        middle_positions=positions,
        middle_velocities=velocities,
        compact_time_rate=1.0e-8,
        scattering_atlas=scattering_atlas,
    )

    assert scattering_atlas.certified
    assert atlas.scattering_atlas.invariant_match_certificate.certified
    assert not atlas.scattering_atlas.middle_invariant_match_certificate.certified
    assert "two_ended_scattering_middle_invariant_match" in (
        atlas.scattering_atlas.missing_obligations
    )
    assert not atlas.certified


def test_prescribed_two_ended_scattering_classifier_rejects_foreign_input_domain():
    source_masses, source_positions, source_velocities = (
        _scattering_middle_state_near_endpoint_invariants()
    )
    source = construct_prescribed_two_ended_scattering_global_atlas(
        masses=source_masses,
        middle_positions=source_positions,
        middle_velocities=source_velocities,
        compact_time_rate=1.0e-8,
        scattering_atlas=_two_ended_scattering_atlas(),
        invariant_tolerance=2.0e-6,
    )
    foreign_masses, foreign_positions, foreign_velocities = _rotating_triangle_data()
    foreign_input = certify_positive_mass_noncollision_input_domain(
        foreign_masses,
        foreign_positions,
        foreign_velocities,
    )
    classification = classify_global_regime(
        input_domain_certificate=foreign_input,
        compact_time_certificate=certify_compact_time_real_line_coverage(1.0e-8),
        regime_id="prescribed_two_ended_scattering",
        scattering_endpoint_envelope=source.scattering_atlas,
    )
    rebuilt = construct_global_atlas_for_regime(classification)

    assert source.certified
    assert source.scattering_atlas.certified
    assert not classification.certified
    assert "scattering_endpoint_input_domain_matches_classification" in (
        classification.missing_obligations
    )
    assert not rebuilt.certified
    assert "scattering_endpoint_input_domain_matches_classification" in (
        rebuilt.missing_obligations
    )


def test_prescribed_two_ended_scattering_rejects_invariant_mismatch():
    masses, positions, velocities = _rotating_triangle_data()
    scattering_atlas = _two_ended_scattering_atlas(mismatch_future_velocity=True)
    atlas = construct_prescribed_two_ended_scattering_global_atlas(
        masses=masses,
        middle_positions=positions,
        middle_velocities=velocities,
        compact_time_rate=1.0e-8,
        scattering_atlas=scattering_atlas,
    )

    assert scattering_atlas.certified
    assert not atlas.scattering_atlas.invariant_match_certificate.certified
    assert "two_ended_scattering_invariant_match" in (
        atlas.scattering_atlas.missing_obligations
    )
    assert not atlas.certified


def test_positive_energy_homothetic_escape_feeds_global_atlas_from_initial_data():
    masses, positions, velocities = _equilateral_homothetic_escape_data()
    atlas = construct_positive_energy_homothetic_escape_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=0.1,
        rho_radius=0.5,
        x_radius=0.4,
        retained_degree=0,
    )
    theorem = construct_general_solution_theorem_certificate(atlas)
    majorant = atlas.escape_atlas.recurrence.majorant_certificate
    projection = atlas.escape_atlas.projection_invariant_certificate
    past = atlas.escape_atlas.past_recurrence
    gluing = atlas.escape_atlas.all_real_gluing

    assert atlas.classification.regime_id == "positive_energy_homothetic_escape"
    assert atlas.escape_atlas.certified
    assert atlas.escape_atlas.branch_certificate.certified
    assert atlas.escape_atlas.branch_certificate.energy > 0.0
    assert projection.certified
    assert projection.newton_residual_certified
    assert projection.invariant_ledger_certified
    assert projection.projected_newton_residual_bound <= 1.0e-10
    assert projection.centered_angular_momentum_norm <= 1.0e-10
    assert atlas.escape_atlas.recurrence.certified
    assert majorant.certified
    assert majorant.rouche_margin > 0.0
    assert atlas.escape_atlas.recurrence.cauchy_majorant == majorant.cauchy_majorant
    assert past.certified
    assert past.recurrence_closes
    assert past.collision_time == atlas.escape_atlas.branch_certificate.endpoint.time_shift
    assert past.past_handoff_time < past.collision_time
    assert past.collision_time < past.future_handoff_time
    assert past.all_past_tail_bound == atlas.escape_atlas.recurrence.all_future_tail_bound
    assert gluing.certified
    assert gluing.middle_recurrence.certified
    assert gluing.total_collision_atlas.proof_certified
    assert gluing.selector_entry.certified
    assert gluing.middle_pair_distance_lower_bound > 0.0
    assert gluing.future_middle_chart_count_bound > 0
    assert gluing.past_middle_chart_count_bound > 0
    assert gluing.tail_budget_certified
    assert gluing.transition_budget_certified
    assert gluing.per_middle_chart_tail_bound == pytest.approx(
        gluing.middle_recurrence.tail_bound
    )
    assert gluing.total_middle_tail_budget == pytest.approx(
        gluing.per_middle_chart_tail_bound
        * (
            gluing.future_middle_chart_count_bound
            + gluing.past_middle_chart_count_bound
        )
    )
    assert gluing.all_real_tail_budget_bound == pytest.approx(
        gluing.endpoint_tail_budget
        + gluing.total_collision_tail_budget
        + gluing.total_middle_tail_budget
    )
    assert gluing.middle_transition_count_bound >= (
        gluing.future_middle_chart_count_bound
        + gluing.past_middle_chart_count_bound
        + 2
    )
    assert atlas.certified
    assert theorem.regime_theorem_certified
    assert not theorem.full_general_solution_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_positive_energy_homothetic_escape_rejects_fake_majorant_and_selector_entry():
    masses, positions, velocities = _equilateral_homothetic_escape_data()
    atlas = construct_positive_energy_homothetic_escape_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=0.1,
        rho_radius=0.5,
        x_radius=0.4,
        retained_degree=0,
    )
    endpoint = atlas.escape_atlas.branch_certificate.endpoint
    fake_majorant = SimpleNamespace(
        certified=True,
        cauchy_majorant=atlas.escape_atlas.recurrence.cauchy_majorant,
        endpoint=endpoint,
    )
    fake_selector = SimpleNamespace(
        certified=True,
        identity_selector_certified=True,
        proof_certified=True,
        masses=tuple(float(mass) for mass in masses),
        selected_energy_limit=(
            atlas.escape_atlas.all_real_gluing.selector_entry.selected_energy_limit
        ),
        selected_central_coefficient=(
            atlas.escape_atlas.all_real_gluing.selector_entry.selected_central_coefficient
        ),
        sample_taus=atlas.escape_atlas.all_real_gluing.selector_entry.sample_taus,
    )

    assert atlas.certified
    assert atlas.escape_atlas.all_real_gluing.certified
    with pytest.raises(
        ValueError,
        match="homothetic escape implicit Cauchy majorant",
    ):
        construct_homothetic_escape_dyadic_recurrence(
            endpoint,
            start_time=1.0e4,
            tau_radius=0.1,
            rho_radius=0.5,
            cauchy_majorant=fake_majorant.cauchy_majorant,
            retained_degree=0,
            majorant_certificate=fake_majorant,
        )
    with pytest.raises(
        ValueError,
        match="certified finite-jet selector entry",
    ):
        certify_positive_energy_homothetic_all_real_gluing(
            masses=masses,
            positions=positions,
            branch_certificate=atlas.escape_atlas.branch_certificate,
            projection_invariant_certificate=(
                atlas.escape_atlas.projection_invariant_certificate
            ),
            future_recurrence=atlas.escape_atlas.recurrence,
            past_recurrence=atlas.escape_atlas.past_recurrence,
            total_collision_atlas=atlas.escape_atlas.total_collision_atlas,
            selector_entry=fake_selector,
            compact_time_rate=1.0e-8,
        )
    assert not replace(
        atlas.escape_atlas.all_real_gluing,
        selector_entry=fake_selector,
    ).collision_selector_certified
    fake_middle_recurrence = SimpleNamespace(
        certified=True,
        proof_certified=True,
        tail_budget_certified=True,
        newton_residual_certified=True,
    )
    fake_total_collision_atlas = SimpleNamespace(
        proof_certified=True,
        evaluation=atlas.escape_atlas.total_collision_atlas.evaluation,
    )
    fake_branch = SimpleNamespace(
        certified=True,
        endpoint=endpoint,
        energy=atlas.escape_atlas.branch_certificate.energy,
        gravitational_parameter=(
            atlas.escape_atlas.branch_certificate.gravitational_parameter
        ),
    )
    fake_projection = SimpleNamespace(certified=True)
    stale_middle = replace(
        atlas.escape_atlas.all_real_gluing,
        middle_recurrence=fake_middle_recurrence,
    )
    stale_collision_atlas = replace(
        atlas.escape_atlas.all_real_gluing,
        total_collision_atlas=fake_total_collision_atlas,
    )
    stale_source_inputs = replace(
        atlas.escape_atlas.all_real_gluing,
        branch_certificate=fake_branch,
        projection_invariant_certificate=fake_projection,
    )

    assert not stale_middle.middle_recurrence_certified
    assert not stale_middle.certified
    assert "homothetic_finite_middle_recurrence" in stale_middle.missing_obligations
    assert not stale_collision_atlas.collision_selector_certified
    assert not stale_collision_atlas.certified
    assert "homothetic_total_collision_selector_chart" in (
        stale_collision_atlas.missing_obligations
    )
    assert not stale_source_inputs.source_inputs_certified
    assert not stale_source_inputs.certified
    assert "homothetic_source_inputs" in stale_source_inputs.missing_obligations
    spoofed_escape_certificate = replace(
        atlas.escape_atlas,
        obligations=(
            SimpleNamespace(
                obligation="fake_positive_energy_homothetic_escape",
                certified=True,
                required=True,
            ),
        ),
    )
    optional_only_escape_certificate = replace(
        atlas.escape_atlas,
        obligations=tuple(
            replace(obligation, required=False)
            for obligation in atlas.escape_atlas.obligations
        ),
    )

    assert not spoofed_escape_certificate.certified
    assert "positive_energy_homothetic_escape_obligation_type" in (
        spoofed_escape_certificate.missing_obligations
    )
    assert not optional_only_escape_certificate.certified
    assert "positive_energy_homothetic_escape_required_obligation_present" in (
        optional_only_escape_certificate.missing_obligations
    )


def test_positive_energy_homothetic_escape_classifier_rejects_foreign_input_domain():
    source_masses, source_positions, source_velocities = (
        _equilateral_homothetic_escape_data()
    )
    source = construct_positive_energy_homothetic_escape_global_atlas(
        masses=source_masses,
        positions=source_positions,
        velocities=source_velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=0.1,
        rho_radius=0.5,
        x_radius=0.4,
        retained_degree=0,
    )
    foreign_masses, foreign_positions, foreign_velocities = _rotating_triangle_data()
    foreign_input = certify_positive_mass_noncollision_input_domain(
        foreign_masses,
        foreign_positions,
        foreign_velocities,
    )
    classification = classify_global_regime(
        input_domain_certificate=foreign_input,
        compact_time_certificate=certify_compact_time_real_line_coverage(1.0e-8),
        regime_id="positive_energy_homothetic_escape",
        escape_endpoint_envelope=source.escape_atlas,
    )
    rebuilt = construct_global_atlas_for_regime(classification)

    assert source.certified
    assert source.escape_atlas.certified
    assert not classification.certified
    assert "escape_endpoint_input_domain_matches_classification" in (
        classification.missing_obligations
    )
    assert not rebuilt.certified
    assert "escape_endpoint_input_domain_matches_classification" in (
        rebuilt.missing_obligations
    )


def test_positive_energy_homothetic_escape_rejects_manual_endpoint_majorant_in_theorem():
    masses, positions, velocities = _equilateral_homothetic_escape_data()
    atlas = construct_positive_energy_homothetic_escape_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=1.0,
        rho_radius=1.0,
        cauchy_majorant=2.0,
        retained_degree=0,
    )

    assert atlas.escape_atlas.branch_certificate.certified
    assert atlas.escape_atlas.projection_invariant_certificate.certified
    assert atlas.escape_atlas.recurrence.certified
    assert atlas.escape_atlas.recurrence.majorant_certificate is None
    assert "positive_energy_homothetic_implicit_cauchy_majorant" in (
        atlas.escape_atlas.missing_obligations
    )
    assert "positive_energy_homothetic_all_real_gluing" not in (
        atlas.escape_atlas.missing_obligations
    )
    assert "escape_endpoint_envelope" in atlas.missing_obligations
    assert not atlas.certified


def test_time_reversed_homothetic_escape_recurrence_reuses_future_tail_budget():
    masses, positions, velocities = _equilateral_homothetic_escape_data()
    atlas = construct_positive_energy_homothetic_escape_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=0.1,
        rho_radius=0.5,
        x_radius=0.4,
        retained_degree=0,
    )
    past = derive_time_reversed_homothetic_escape_recurrence(
        atlas.escape_atlas.recurrence,
        compact_time_rate=1.0e-8,
    )

    assert past.certified
    assert past.past_handoff_time == (
        2.0 * past.collision_time - atlas.escape_atlas.recurrence.start_time
    )
    assert -1.0 < past.past_handoff_compact_parameter < 0.0
    assert 0.0 < past.future_handoff_compact_parameter < 1.0
    assert past.shell_tail_bound(3) == atlas.escape_atlas.recurrence.shell_tail_bound(3)
    assert past.past_tail_from_shell(2) == (
        atlas.escape_atlas.recurrence.future_tail_from_shell(2)
    )


def test_positive_energy_homothetic_escape_rejects_nonhomothetic_initial_data():
    masses, positions, velocities = _equilateral_homothetic_escape_data(
        velocity_perturbation=0.2,
    )
    atlas = construct_positive_energy_homothetic_escape_global_atlas(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.0e-8,
        start_time=1.0e4,
        tau_radius=1.0,
        rho_radius=1.0,
        cauchy_majorant=2.0,
        retained_degree=0,
    )

    assert not atlas.escape_atlas.branch_certificate.certified
    assert "homothetic_escape_branch" in atlas.escape_atlas.missing_obligations
    assert "positive_energy_homothetic_projection_newton_invariants" in (
        atlas.escape_atlas.missing_obligations
    )
    assert not atlas.classification.certified
    assert not atlas.certified
    assert "escape_endpoint_envelope" in atlas.missing_obligations


def test_general_solution_pipeline_rejects_raw_boolean_global_witnesses():
    _shell_isolation, event_assembly = _geometric_event_budget()
    compact_time = certify_compact_time_real_line_coverage(1.3)

    with pytest.raises(TypeError, match="raw boolean"):
        classify_global_regime(
            input_domain_certificate=True,
            compact_time_certificate=compact_time,
            regime_id="geometric_infinite_event_tail",
            event_isolation_certificate=event_assembly,
            primitive_cauchy_inputs=event_assembly,
        )

    classification = classify_global_regime(
        input_domain_certificate=_input_domain(),
        compact_time_certificate=compact_time,
        regime_id="geometric_infinite_event_tail",
        event_isolation_certificate=event_assembly,
        primitive_cauchy_inputs=event_assembly,
    )
    atlas = construct_global_atlas_for_regime(classification)

    with pytest.raises(TypeError, match="raw boolean"):
        construct_general_solution_theorem_certificate(
            atlas,
            global_regime_exhaustion_certificate=np.bool_(True),
        )

    class LooseGlobalExhaustion:
        certified = True
        proof_certified = True

    theorem = construct_general_solution_theorem_certificate(
        atlas,
        global_regime_exhaustion_certificate=LooseGlobalExhaustion(),
    )

    assert not theorem.full_general_solution_certified
    assert "global_regime_exhaustion" in theorem.missing_obligations
    assert "full theorem accepts only a typed" in theorem.obligations[-1].detail


def test_required_constructor_obligations_reject_truthy_attribute_fields():
    classification = classify_global_regime(
        input_domain_certificate=_input_domain(),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="uniformly_noncollision_bounded_tail",
        ordinary_gap_envelope=SimpleNamespace(
            certified="yes",
            proof_certified="yes",
        ),
    )
    atlas = construct_global_atlas_for_regime(
        classification,
        ordinary_gap_atlas=classification.ordinary_gap_envelope,
    )

    assert not classification.certified
    assert "ordinary_gap_envelope" in classification.missing_obligations
    assert not atlas.certified
    assert "ordinary_gap_envelope" in atlas.missing_obligations


def test_global_regime_exhaustion_constructor_reports_partition_theorem_gap():
    masses, positions, velocities = _rotating_triangle_data()
    delta_initial = _delta_initial_for_handoff_time(
        target_time=1.0e-4,
        compact_time_rate=1.3,
    )
    scoped_nonzero = construct_nonzero_angular_global_atlas_from_uniform_pair_event_envelopes(
        masses=masses,
        positions=positions,
        velocities=velocities,
        compact_time_rate=1.3,
        delta_initial=delta_initial,
        theta=0.99,
        event_isolation_initial=0.2,
        boundary_clearance_initial=0.2,
        ordinary_pair_distance_lower_bound=0.1,
        ordinary_pair_diameter_upper_bound=5.0,
        ordinary_speed_upper_bound=5.0,
        binary_pair_envelopes=_uniform_binary_pair_envelopes(),
        step_ratio_bounds={
            "value": 0.33,
            "first_jet": 0.34,
            "lifted_residual": 0.31,
            "physical_residual": 0.23,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        checked_prefix=6,
        derive_finite_middle_atlas=True,
        derive_first_event_shell_prefix=True,
        derive_event_shell_invariance=True,
    )
    exhaustion = certify_global_regime_exhaustion(
        candidate_regimes=(scoped_nonzero,),
    )
    theorem = construct_general_solution_theorem_certificate(
        scoped_nonzero,
        global_regime_exhaustion_certificate=exhaustion,
    )

    assert scoped_nonzero.certified
    assert exhaustion.theorem_role == "optional_endpoint_compression_theorem"
    assert not exhaustion.certified
    assert "arbitrary_initial_data_partition_theorem" in (
        exhaustion.missing_obligations
    )
    assert "global_exhaustion_required_regime_ids" in (
        exhaustion.missing_obligations
    )
    assert not theorem.full_general_solution_certified
    assert theorem.regime_theorem_certified
    assert theorem.missing_obligations == ("global_regime_exhaustion",)


def test_missing_regime_inputs_propagate_to_theorem_obligations():
    shell_isolation, _event_budget = _geometric_event_budget()
    classification = classify_global_regime(
        input_domain_certificate=_input_domain(),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="geometric_infinite_event_tail",
        event_isolation_certificate=shell_isolation,
    )
    atlas = construct_global_atlas_for_regime(classification)
    theorem = construct_general_solution_theorem_certificate(atlas)

    assert not classification.certified
    assert "primitive_cauchy_all_future_budget" in classification.missing_obligations
    assert "primitive_cauchy_all_future_budget" in atlas.missing_obligations
    assert not atlas.certified
    assert not theorem.regime_theorem_certified
    assert "primitive_cauchy_all_future_budget" in theorem.missing_obligations


def test_event_regime_assembler_rejects_manual_total_collision_cauchy_tuples():
    primitive_inputs = {
        "value": (1.8e-5, 1.12, 0.32, 5, 3),
        "first_jet": (4.0e-5, 1.13, 0.35, 5, 3),
        "lifted_residual": (7.8e-5, 1.09, 0.32, 6, 3),
        "physical_residual": (4.9e-9, 1.05, 0.24, 6, 3),
    }

    with pytest.raises(ValueError, match="constructor source certificate"):
        certify_local_chart_family_primitive_cauchy_inputs(
            kind="automatic_identity_selector_total_collision",
            primitive_inputs=primitive_inputs,
            components=COMPONENTS,
        )


def test_event_regime_assembler_rejects_manual_ordinary_gap_cauchy_tuples():
    primitive_inputs = {
        "value": (0.9e-5, 1.06, 0.30, 5, 3),
        "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
        "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
        "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
    }

    with pytest.raises(ValueError, match="constructor source certificate"):
        certify_local_chart_family_primitive_cauchy_inputs(
            kind="ordinary_gap_taylor",
            primitive_inputs=primitive_inputs,
            components=COMPONENTS,
        )


def test_event_regime_assembler_rejects_manual_separated_binary_cauchy_tuples():
    primitive_inputs = {
        "value": (1.4e-5, 1.10, 0.33, 5, 3),
        "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
        "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
        "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
    }

    with pytest.raises(ValueError, match="constructor source certificate"):
        certify_local_chart_family_primitive_cauchy_inputs(
            kind="separated_binary_levi_civita",
            primitive_inputs=primitive_inputs,
            components=COMPONENTS,
        )


def test_event_regime_assembler_rejects_manual_custom_chart_family_tuples():
    primitive_inputs = {
        "value": (1.1e-5, 1.07, 0.31, 5, 3),
        "first_jet": (2.7e-5, 1.09, 0.33, 5, 3),
        "lifted_residual": (5.8e-5, 1.06, 0.29, 6, 3),
        "physical_residual": (3.2e-9, 1.04, 0.23, 6, 3),
    }

    with pytest.raises(ValueError, match="constructor source certificate"):
        certify_local_chart_family_primitive_cauchy_inputs(
            kind="custom_event_family",
            primitive_inputs=primitive_inputs,
            components=COMPONENTS,
        )


def test_event_regime_assembler_rejects_direct_source_less_chart_family_certificate():
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
        binary_kind="custom_event_family",
        total_collision_kind=None,
    )
    component_inputs = {
        component: construct_primitive_cauchy_tail_input(
            majorant_initial=1.0e-5,
            majorant_growth=1.05,
            step_ratio_bound=0.28,
            retained_order_initial=5,
            retained_order_increment=3,
        )
        for component in COMPONENTS
    }
    source_less_custom_family = LocalChartFamilyCauchyCertificate(
        kind="custom_event_family",
        component_inputs=component_inputs,
        source_certificate=None,
    )
    assembly = derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=(
            _ordinary_gap_chart_family(),
            source_less_custom_family,
        ),
        checked_prefix=6,
        components=COMPONENTS,
    )

    assert not source_less_custom_family.source_certified
    assert not source_less_custom_family.certified_for_components(COMPONENTS)
    assert not assembly.certified
    assert assembly.event_budget is None
    assert "chart_family:custom_event_family" in assembly.missing_obligations
    assert "all_future_event_budget" in assembly.missing_obligations


def test_event_regime_assembler_reports_missing_chart_family_obligations():
    shell_isolation, event_assembly = _geometric_event_budget()
    incomplete_family_certificates = tuple(
        certificate
        for certificate in event_assembly.chart_family_certificates
        if certificate.kind != "automatic_identity_selector_total_collision"
    )
    incomplete = derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=incomplete_family_certificates,
        checked_prefix=6,
        components=COMPONENTS,
    )
    classification = classify_global_regime(
        input_domain_certificate=_input_domain(),
        compact_time_certificate=certify_compact_time_real_line_coverage(1.3),
        regime_id="geometric_infinite_event_tail",
        event_isolation_certificate=incomplete,
        primitive_cauchy_inputs=incomplete,
    )
    atlas = construct_global_atlas_for_regime(classification)

    assert not incomplete.certified
    assert incomplete.event_budget is None
    assert "chart_family:automatic_identity_selector_total_collision" in (
        incomplete.missing_obligations
    )
    assert "all_future_event_budget" in incomplete.missing_obligations
    assert not classification.certified
    assert "event_isolation" in classification.missing_obligations
    assert "primitive_cauchy_all_future_budget" in classification.missing_obligations
    assert not atlas.certified
