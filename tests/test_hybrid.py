from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from three_body_symmetry.continuation import choose_step_size
from three_body_symmetry.hybrid import (
    _ordinary_end_state_interval_union,
    _ordinary_end_state_interval_over_time_interval,
    _hull_state_intervals,
    _regularized_binary_atlas_for_state_interval_union,
    _pair_distance_squared_polynomial_intervals,
    _regularized_binary_atlas_end_state_interval,
    _regularized_binary_atlas_end_state_intervals,
    certify_ordinary_binary_entry_event_union,
    choose_chart,
    choose_interval_chart,
    choose_interval_chart_union,
    choose_interval_ordinary_step_size,
    choose_interval_union_ordinary_step_size,
    choose_transition_interval_chart,
    choose_transition_interval_chart_union,
    closest_pair,
    certify_regularized_binary_interval_taylor_equations,
    certify_regularized_binary_projection_domain,
    continue_hybrid,
    continue_hybrid_from_regularized_binary_collision,
    event_limited_ordinary_taylor_step,
    first_interval_polynomial_crossing,
    first_ordinary_binary_entry_event_union,
    HybridStep,
    interval_body_acceleration_upper_bounds,
    interval_body_speed_upper_bounds,
    interval_min_pair_distance_lower_bound,
    interval_pairwise_distance_bounds,
    pack_planar_state,
    certify_polynomial_root,
    polynomial_real_roots_in_interval,
    regularized_binary_taylor_step,
)
from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    exact_binary_collision_interval_chart,
    planar_interval_to_regularized_binary_collision_chart_atlas,
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import (
    construct_interval_regularized_binary_taylor_solution_from_intervals,
    construct_regularized_binary_taylor_solution,
)
from three_body_symmetry.intervals import FloatInterval, point_interval_coefficients
from three_body_symmetry.series import construct_interval_taylor_solution, construct_taylor_solution, integrate_reference
from three_body_symmetry.validated_atlas import (
    time_reverse_validated_atlas_solution,
    validated_atlas_from_hybrid_solution,
)


def _ordinary_planar_data():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array(
        [
            [0.8, -0.2],
            [-0.4, 0.6],
            [0.1, -0.5],
        ]
    )
    velocities = np.array(
        [
            [0.05, 0.11],
            [-0.07, 0.03],
            [0.02, -0.08],
        ]
    )
    return masses, positions, velocities


def _exact_collision_state():
    masses = np.array([0.8, 1.2, 1.7])
    pair_mass = masses[0] + masses[1]
    return RegularizedBinaryCollisionChartState(
        masses=masses,
        pair=(0, 1),
        z=np.array([0.0, 0.0]),
        z_velocity=np.array([np.sqrt(pair_mass / 2.0), 0.0]),
        pair_energy=-0.3,
        binary_center=np.array([0.0, 0.0]),
        binary_center_velocity=np.array([0.2, -0.1]),
        third_offset=np.array([1.5, 0.25]),
        third_offset_velocity=np.array([-0.03, 0.07]),
    )


def _exact_collision_interval_state():
    initial = _exact_collision_state()
    return exact_binary_collision_interval_chart(
        initial.masses,
        pair=initial.pair,
        z_velocity=(
            (initial.z_velocity[0] - 1e-14, initial.z_velocity[0] + 1e-14),
            (initial.z_velocity[1], initial.z_velocity[1]),
        ),
        pair_energy=(initial.pair_energy - 1e-14, initial.pair_energy + 1e-14),
        binary_center=tuple((value - 1e-14, value + 1e-14) for value in initial.binary_center),
        binary_center_velocity=tuple(
            (value - 1e-14, value + 1e-14) for value in initial.binary_center_velocity
        ),
        third_offset=tuple((value - 1e-14, value + 1e-14) for value in initial.third_offset),
        third_offset_velocity=tuple(
            (value - 1e-14, value + 1e-14) for value in initial.third_offset_velocity
        ),
    )


def _exact_collision_exit_time(initial, exit_distance, *, order=30):
    solution = construct_regularized_binary_taylor_solution(initial, order=order)
    root = brentq(lambda s_value: solution.state_at(s_value).rho - exit_distance, 0.0, 0.01)
    return solution.physical_time_at(root)


def _close_pair_planar_data():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array(
        [
            [0.0, 0.0],
            [1e-3, 0.0],
            [1.0, 0.4],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.02],
            [-0.01, 0.0],
        ]
    )
    return masses, positions, velocities


def _close_pair_for(pair):
    masses = np.array([1.0, 1.2, 0.7])
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    i, j = pair
    positions[j] = positions[i] + np.array([1e-3, 0.0])
    return masses, positions, velocities


def test_choose_chart_identifies_closest_pair_and_mode():
    _masses, positions, _velocities = _close_pair_planar_data()

    pair, distance = closest_pair(positions)
    ordinary_mode = choose_chart(positions, binary_distance_threshold=1e-4)
    binary_mode = choose_chart(positions, binary_distance_threshold=1e-2)

    assert pair == (0, 1)
    assert distance == 1e-3
    assert ordinary_mode == ("ordinary", None, 1e-3)
    assert binary_mode == ("binary", (0, 1), 1e-3)


def test_interval_chart_decision_certifies_ordinary_state():
    masses, positions, velocities = _ordinary_planar_data()
    state_interval = tuple(
        (float(value), float(value)) for value in pack_planar_state(positions, velocities)
    )

    decision = choose_interval_chart(state_interval, binary_distance_threshold=0.05)

    assert decision.chart == "ordinary"
    assert decision.pair is None
    assert decision.certified
    assert all(lower > 0.05 for _pair, (lower, _upper) in decision.pair_distance_bounds)


def test_interval_chart_decision_certifies_unique_binary_pair():
    masses, positions, velocities = _close_pair_planar_data()
    state_interval = tuple(
        (float(value), float(value)) for value in pack_planar_state(positions, velocities)
    )

    decision = choose_interval_chart(state_interval, binary_distance_threshold=1e-2)

    assert decision.chart == "binary"
    assert decision.pair == (0, 1)
    assert decision.certified
    assert dict(decision.pair_distance_bounds)[(0, 1)][1] <= 1e-2


def test_interval_chart_decision_reports_ambiguous_threshold_overlap():
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.009, 0.011),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6

    decision = choose_interval_chart(state_interval, binary_distance_threshold=1e-2)

    assert decision.chart == "ambiguous"
    assert decision.pair is None
    assert not decision.certified
    lower_01, upper_01 = dict(decision.pair_distance_bounds)[(0, 1)]
    assert lower_01 < 1e-2 < upper_01


def test_interval_chart_union_certifies_ordinary_when_hull_is_ambiguous():
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.21),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.21, -0.2),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6

    hull_state = tuple(
        (min(first_state[index][0], second_state[index][0]), max(first_state[index][1], second_state[index][1]))
        for index in range(len(first_state))
    )
    hull_decision = choose_interval_chart(hull_state, binary_distance_threshold=0.1)
    union_decision = choose_interval_chart_union(
        (first_state, second_state),
        binary_distance_threshold=0.1,
    )

    assert hull_decision.chart == "ambiguous"
    assert not hull_decision.certified
    assert union_decision.chart == "ordinary"
    assert union_decision.certified


def test_interval_chart_union_certifies_common_binary_when_hull_is_ambiguous():
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.004, 0.006),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.006, -0.004),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6

    union_decision = choose_interval_chart_union(
        (first_state, second_state),
        binary_distance_threshold=0.01,
    )

    assert union_decision.chart == "binary"
    assert union_decision.pair == (0, 1)
    assert union_decision.certified


def test_interval_union_ordinary_step_size_uses_split_members_not_hull():
    masses = np.array([1.0, 1.0, 0.2])
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.21),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.21, -0.2),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    hull_state = _hull_state_intervals((first_state, second_state))

    hull_step = choose_interval_ordinary_step_size(
        hull_state,
        masses,
        1e-3,
        max_step=1e-3,
    )
    union_step = choose_interval_union_ordinary_step_size(
        (first_state, second_state),
        masses,
        1e-3,
        max_step=1e-3,
    )

    assert hull_step == np.finfo(float).eps
    assert union_step == pytest.approx(1e-3)


def test_hybrid_accepts_initial_state_union_when_hull_chart_is_ambiguous():
    masses = np.array([1.0, 1.0, 0.2])
    positions = np.array([[0.0, 0.0], [0.205, 0.0], [2.0, 0.0]])
    velocities = np.zeros((3, 2))
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.21),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.21, -0.2),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    hull_state = _hull_state_intervals((first_state, second_state))

    with pytest.raises(RuntimeError, match="interval chart decision is not certified"):
        continue_hybrid(
            positions,
            velocities,
            masses,
            1e-4,
            binary_distance_threshold=0.1,
            max_time_step=1e-3,
            ordinary_order=8,
            tail_certificate_mode="cauchy",
            initial_state_interval=hull_state,
            require_interval_chart_certification=True,
        )

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        1e-4,
        binary_distance_threshold=0.1,
        max_time_step=1e-3,
        ordinary_order=8,
        tail_certificate_mode="cauchy",
        initial_state_interval_union=(first_state, second_state),
        require_interval_chart_certification=True,
    )

    assert len(continued.steps) == 1
    assert continued.proof_certified
    step = continued.steps[0]
    assert step.chart == "ordinary"
    assert step.interval_chart_certified
    assert step.interval_min_pair_distance > 0.19
    assert step.start_state_interval == hull_state
    assert step.start_state_interval_union == (first_state, second_state)
    assert step.residual_certificate.certified
    assert step.residual_certificate.member_count == 2
    assert step.residual_certificate.coefficient_source == "ordinary_interval_taylor_union"
    assert step.invariants_certified
    assert step.energy_certificate.member_count == 2

    atlas = validated_atlas_from_hybrid_solution(continued)

    assert atlas.proof_certified
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified


def test_hybrid_proof_rejects_missing_residual_or_invariant_certificate():
    masses, positions, velocities = _ordinary_planar_data()
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.01,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=8,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
    )
    step = continued.steps[0]

    assert step.proof_certified
    assert not replace(step, residual_certificate=None).proof_certified
    assert not replace(step, energy_certificate=None).proof_certified


def test_transition_chart_decision_uses_certified_binary_entry_event():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.0201,
        binary_distance_threshold=0.1,
        binary_exit_distance=0.13,
        max_time_step=0.05,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
    )
    entry_step = continued.steps[0]
    ambiguous_state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.099, 0.101),
        (0.0, 0.0),
        (5.0, 5.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    ambiguous_distance_decision = choose_interval_chart(
        ambiguous_state_interval,
        binary_distance_threshold=0.1,
    )
    transition_decision = choose_transition_interval_chart(
        ambiguous_state_interval,
        entry_step,
        binary_distance_threshold=0.1,
    )

    assert entry_step.event == "enter_binary"
    assert ambiguous_distance_decision.chart == "ambiguous"
    assert not ambiguous_distance_decision.certified
    assert transition_decision.chart == "binary"
    assert transition_decision.pair == (0, 1)
    assert transition_decision.certified
    assert transition_decision.reason.startswith("certified ordinary binary-entry")
    binary_steps = [step for step in continued.steps if step.chart == "binary"]
    assert binary_steps
    assert all(step.binary_lc_atlas_certified for step in binary_steps)
    assert all(step.binary_lc_atlas_propagated for step in binary_steps)
    assert all(
        step.binary_interval_lift_reason == "planar interval state lifted into regularized binary chart"
        for step in binary_steps
    )


def test_ordinary_binary_entry_union_certificate_covers_each_member():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    (
        _end_positions,
        _end_velocities,
        event_time,
        _indicator,
        event_pair,
        _certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.05,
        order=20,
        binary_enter_distance=0.1,
    )
    state = pack_planar_state(positions, velocities)
    state_union = (
        tuple((float(value - 1e-14), float(value + 1e-14)) for value in state),
        tuple((float(value - 2e-14), float(value + 2e-14)) for value in state),
    )

    union_certificate = certify_ordinary_binary_entry_event_union(
        positions,
        velocities,
        masses,
        trial_physical_step=0.05,
        event_root=event_time,
        event_pair=event_pair,
        binary_enter_distance=0.1,
        order=20,
        start_state_interval_union=state_union,
    )

    assert union_certificate.member_count == 2
    assert union_certificate.pair == (0, 1)
    assert union_certificate.interval_is_isolated
    assert union_certificate.earliest_interval_is_certified
    assert union_certificate.event_time_interval is not None
    assert union_certificate.event_time_interval[0] <= event_time <= union_certificate.event_time_interval[1]
    assert all(
        certificate.coefficient_source == "ordinary_interval_taylor_union_member"
        for certificate in union_certificate.member_certificates
    )


def test_ordinary_binary_entry_union_search_requires_common_isolated_event():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    (
        _end_positions,
        _end_velocities,
        event_time,
        _indicator,
        event_pair,
        _certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.05,
        order=20,
        binary_enter_distance=0.1,
    )
    state = pack_planar_state(positions, velocities)
    state_union = (
        tuple((float(value - 1e-14), float(value + 1e-14)) for value in state),
        tuple((float(value - 2e-14), float(value + 2e-14)) for value in state),
    )

    crossing = first_ordinary_binary_entry_event_union(
        positions,
        velocities,
        masses,
        trial_physical_step=0.05,
        binary_enter_distance=0.1,
        order=20,
        start_state_interval_union=state_union,
    )

    assert crossing is not None
    root, pair, union_certificate = crossing
    assert pair == event_pair == (0, 1)
    assert abs(root - event_time) < 1e-14
    assert union_certificate.member_count == 2
    assert union_certificate.interval_is_isolated
    assert union_certificate.earliest_interval_is_certified
    assert union_certificate.event_time_interval is not None
    assert union_certificate.event_time_interval[0] <= root <= union_certificate.event_time_interval[1]
    assert union_certificate.event_time_interval[1] - union_certificate.event_time_interval[0] < 0.01
    assert all(
        certificate.root_enclosure is not None
        and certificate.root_enclosure.certifies_earliest_root
        and certificate.root_enclosure.excludes_earlier_roots
        and certificate.root_enclosure.coefficient_source == "ordinary_interval_taylor_union_member_interval_search"
        for certificate in union_certificate.member_certificates
    )


def test_transition_chart_union_uses_union_event_certificate_when_hull_is_ambiguous():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    (
        _end_positions,
        _end_velocities,
        event_time,
        _indicator,
        event_pair,
        _certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.05,
        order=20,
        binary_enter_distance=0.1,
    )
    state = pack_planar_state(positions, velocities)
    state_union = (
        tuple((float(value - 1e-14), float(value + 1e-14)) for value in state),
        tuple((float(value - 2e-14), float(value + 2e-14)) for value in state),
    )
    union_certificate = certify_ordinary_binary_entry_event_union(
        positions,
        velocities,
        masses,
        trial_physical_step=0.05,
        event_root=event_time,
        event_pair=event_pair,
        binary_enter_distance=0.1,
        order=20,
        start_state_interval_union=state_union,
    )
    previous_step = HybridStep(
        chart="ordinary",
        start_time=0.0,
        physical_step=event_time,
        parameter_step=event_time,
        min_pair_distance=0.1,
        pair=event_pair,
        event="enter_binary",
        event_union_certificate=union_certificate,
    )
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.099, 0.101),
        (0.0, 0.0),
        (5.0, 5.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.101, -0.099),
        (0.0, 0.0),
        (5.0, 5.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6

    decision = choose_transition_interval_chart_union(
        (first_state, second_state),
        previous_step,
        binary_distance_threshold=0.1,
    )

    assert decision.chart == "binary"
    assert decision.pair == (0, 1)
    assert decision.certified
    assert decision.reason.startswith("certified union ordinary binary-entry")


def test_binary_atlas_endpoint_hull_contains_each_branch_solution():
    masses = np.array([1.0, 1.0, 1.0])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    velocities = np.zeros((3, 2))
    upper_positions = np.array([[0.0, 0.0], [-1.0, 0.1], [3.0, 0.0]])
    lower_positions = np.array([[0.0, 0.0], [-1.0, -0.1], [3.0, 0.0]])
    upper_initial = planar_to_regularized_binary_collision_chart(upper_positions, velocities, masses, pair=(0, 1))
    lower_initial = planar_to_regularized_binary_collision_chart(lower_positions, velocities, masses, pair=(0, 1))
    atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
        state_interval,
        masses,
        pair=(0, 1),
    )
    s_step = 0.001

    hull = _regularized_binary_atlas_end_state_interval(
        upper_initial,
        atlas,
        s_step,
        order=8,
    )
    branch_intervals = _regularized_binary_atlas_end_state_intervals(
        upper_initial,
        atlas,
        s_step,
        order=8,
    )
    upper_state = construct_regularized_binary_taylor_solution(upper_initial, order=8).state_at(s_step)
    lower_state = construct_regularized_binary_taylor_solution(lower_initial, order=8).state_at(s_step)
    upper_planar = pack_planar_state(*regularized_binary_collision_chart_to_planar(upper_state))
    lower_planar = pack_planar_state(*regularized_binary_collision_chart_to_planar(lower_state))

    assert len(atlas) == 2
    assert len(branch_intervals) == 2
    assert all(chart.branch_certificate is not None and chart.branch_certificate.certified for chart in atlas)
    for point in (upper_planar, lower_planar):
        for value, (lower, upper) in zip(point, hull):
            assert lower <= value <= upper


def test_binary_atlas_lifts_interval_state_union_when_hull_contains_collision():
    masses = np.array([1.0, 1.0, 0.1])
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.09, 0.11),
        (0.0, 0.0),
        (5.0, 5.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.11, -0.09),
        (0.0, 0.0),
        (5.0, 5.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    state_union = (first_state, second_state)
    hull_state = _hull_state_intervals(state_union)

    with pytest.raises(ValueError, match="selected pair interval contains binary collision"):
        planar_interval_to_regularized_binary_collision_chart_atlas(
            hull_state,
            masses,
            pair=(0, 1),
        )
    atlas = _regularized_binary_atlas_for_state_interval_union(
        state_union,
        masses,
        pair=(0, 1),
    )

    # Each member lies on the closed real-axis boundary and now uses one
    # canonical closed-half-plane patch instead of a redundant boundary copy.
    assert len(atlas) == 2
    assert all(chart.branch_certificate is not None for chart in atlas)
    assert all(chart.branch_certificate.certified for chart in atlas)


def test_ordinary_interval_union_propagates_members_separately():
    masses, positions, velocities = _ordinary_planar_data()
    base_state = pack_planar_state(positions, velocities)
    first_state = tuple((float(value - 1e-13), float(value + 1e-13)) for value in base_state)
    second_state = tuple((float(value - 2e-13), float(value + 2e-13)) for value in base_state)

    propagated = _ordinary_end_state_interval_union(
        positions,
        velocities,
        masses,
        0.001,
        order=8,
        start_state_interval_union=(first_state, second_state),
    )

    assert len(propagated) == 2
    assert propagated[0] != propagated[1]
    for interval_state in propagated:
        assert len(interval_state) == 12


def test_ordinary_interval_endpoint_can_propagate_over_time_interval():
    masses, positions, velocities = _ordinary_planar_data()
    state = pack_planar_state(positions, velocities)
    state_interval = tuple((float(value - 1e-14), float(value + 1e-14)) for value in state)
    point_series = construct_taylor_solution(positions, velocities, masses, order=10)
    time_interval = (0.001, 0.002)

    propagated = _ordinary_end_state_interval_over_time_interval(
        positions,
        velocities,
        masses,
        time_interval,
        order=10,
        start_state_interval=state_interval,
    )

    for time in time_interval:
        point_state = point_series.state_at(time)
        for value, (lower, upper) in zip(point_state, propagated):
            assert lower <= value <= upper


def test_polynomial_event_root_isolation_finds_all_interval_roots():
    roots = [0.125, 0.25, 0.5]
    coefficients = np.polynomial.polynomial.polyfromroots(roots)

    isolated = polynomial_real_roots_in_interval(coefficients, 0.0, 0.4)

    assert len(isolated) == 2
    assert np.linalg.norm(np.array(isolated) - np.array(roots[:2]), ord=np.inf) < 1e-12


def test_interval_polynomial_crossing_finds_earliest_root_without_point_seed():
    coefficients = np.polynomial.polynomial.polyfromroots([0.25, 0.75])
    coefficient_intervals = point_interval_coefficients(coefficients)

    enclosure = first_interval_polynomial_crossing(
        coefficient_intervals,
        0.0,
        0.5,
        direction="decreasing",
        coefficient_source="test_interval_search",
    )

    assert enclosure is not None
    assert enclosure.certifies_earliest_root
    assert enclosure.interval[0] <= 0.25 <= enclosure.interval[1]
    assert enclosure.interval[1] < 0.75
    assert enclosure.coefficient_source == "test_interval_search"


def test_polynomial_root_certificate_isolates_simple_crossing():
    coefficients = np.polynomial.polynomial.polyfromroots([0.25, 0.75])
    coefficient_intervals = point_interval_coefficients(coefficients)

    certificate = certify_polynomial_root(
        coefficients,
        0.25,
        0.0,
        0.5,
        direction="decreasing",
        coefficient_intervals=coefficient_intervals,
    )

    assert certificate.is_isolated
    assert certificate.bracket[0] < certificate.root < certificate.bracket[1]
    assert certificate.values[0] > 0.0
    assert certificate.values[1] < 0.0
    assert certificate.derivative_sign < 0
    assert certificate.interval_is_isolated
    assert certificate.interval_derivative_sign < 0
    assert certificate.coefficient_intervals is not None
    assert certificate.earliest_interval_is_certified
    assert certificate.root_enclosure is not None
    assert certificate.root_enclosure.interval[0] <= certificate.root <= certificate.root_enclosure.interval[1]
    assert certificate.root_enclosure.excludes_earlier_roots


def test_polynomial_root_certificate_allows_small_coefficient_intervals():
    coefficients = np.polynomial.polynomial.polyfromroots([0.25, 0.75])
    coefficient_intervals = tuple(
        FloatInterval(float(coefficient - 1e-16), float(coefficient + 1e-16))
        for coefficient in coefficients
    )

    certificate = certify_polynomial_root(
        coefficients,
        0.25,
        0.0,
        0.5,
        direction="decreasing",
        coefficient_intervals=coefficient_intervals,
    )

    assert certificate.interval_is_isolated
    assert certificate.earliest_interval_is_certified


def test_pair_distance_polynomial_intervals_accept_interval_taylor_coefficients():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])

    point_solution = construct_taylor_solution(positions, velocities, masses, order=8)
    interval_solution = construct_interval_taylor_solution(positions, velocities, masses, order=8)

    point_widths = [
        coefficient.upper - coefficient.lower
        for coefficient in _pair_distance_squared_polynomial_intervals(point_solution.position, (0, 1))
    ]
    interval_widths = [
        coefficient.upper - coefficient.lower
        for coefficient in _pair_distance_squared_polynomial_intervals(interval_solution.position, (0, 1))
    ]

    assert max(interval_widths) > max(point_widths)


def test_interval_pairwise_distance_bounds_enclose_position_boxes():
    state_interval = (
        (-0.1, 0.1),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (2.0, 2.0),
    ) + ((0.0, 0.0),) * 6

    bounds = dict(interval_pairwise_distance_bounds(state_interval))

    lower_01, upper_01 = bounds[(0, 1)]
    assert 0.89 < lower_01 <= 0.9
    assert 1.1 <= upper_01 < 1.11
    assert interval_min_pair_distance_lower_bound(state_interval) == min(
        lower for lower, _upper in bounds.values()
    )


def test_distance_margin_override_shrinks_ordinary_step_choice():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    velocities = np.zeros((3, 2))

    point_step = abs(
        choose_step_size(
            positions,
            velocities,
            masses,
            10.0,
            max_step=10.0,
            safety=0.2,
        )
    )
    interval_step = abs(
        choose_step_size(
            positions,
            velocities,
            masses,
            10.0,
            max_step=10.0,
            safety=0.2,
            distance_margin=0.8,
        )
    )

    assert interval_step < point_step


def test_interval_speed_and_acceleration_bounds_enclose_planar_state():
    masses = np.array([1.0, 2.0, 3.0])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (2.0, 2.0),
        (-2.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (-0.5, 0.5),
    )

    speed_bounds = interval_body_speed_upper_bounds(state_interval)
    acceleration_bounds = interval_body_acceleration_upper_bounds(state_interval, masses)

    assert 2.0 <= speed_bounds[0] < 2.01
    assert 1.0 <= speed_bounds[1] < 1.01
    assert 0.5 <= speed_bounds[2] < 0.51
    assert 2.75 <= acceleration_bounds[0] < 2.76


def test_interval_ordinary_step_choice_uses_speed_and_acceleration_bounds():
    masses = np.array([1.0, 0.7, 1.4])
    tight_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (3.0, 3.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    wide_velocity_state = tight_state[:6] + (
        (-3.0, 3.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    tight_step = abs(
        choose_interval_ordinary_step_size(
            tight_state,
            masses,
            10.0,
            max_step=10.0,
            safety=0.2,
        )
    )
    wide_velocity_step = abs(
        choose_interval_ordinary_step_size(
            wide_velocity_state,
            masses,
            10.0,
            max_step=10.0,
            safety=0.2,
        )
    )

    assert wide_velocity_step < tight_step


def test_hybrid_uses_ordinary_charts_and_matches_reference_when_well_separated():
    masses, positions, velocities = _ordinary_planar_data()
    t_final = 0.05

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=16,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)

    assert {step.chart for step in continued.steps} == {"ordinary"}
    assert {step.interval_chart for step in continued.steps} == {"ordinary"}
    assert all(step.interval_chart_certified for step in continued.steps)
    assert abs(continued.times[-1] - t_final) < 1e-14
    assert np.linalg.norm(continued.final_state - reference, ord=np.inf) < 1e-11


def test_hybrid_cauchy_ordinary_path_is_proof_certified():
    masses, positions, velocities = _ordinary_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.03,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
    )

    assert continued.steps
    assert continued.proof_certified
    assert continued.interval_chart_certified_step_count == len(continued.steps)
    assert continued.proof_certified_step_count == len(continued.steps)
    assert all(step.proof_certified for step in continued.steps)
    assert all(
        step.truncation_certificate.coefficient_source == "interval_cauchy_majorant_union"
        for step in continued.steps
    )


def test_hybrid_can_reject_uncertified_interval_chart_fallback():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.4]])
    velocities = np.zeros((3, 2))
    initial_state = list((float(value), float(value)) for value in pack_planar_state(positions, velocities))
    initial_state[2] = (0.009, 0.011)
    initial_state_interval = tuple(initial_state)

    fallback = continue_hybrid(
        positions,
        velocities,
        masses,
        1e-5,
        binary_distance_threshold=0.01,
        binary_exit_distance=0.02,
        max_binary_s_step=1e-3,
        ordinary_order=10,
        binary_order=10,
        initial_state_interval=initial_state_interval,
    )

    assert fallback.steps
    assert fallback.steps[0].interval_chart == "ambiguous"
    assert not fallback.steps[0].interval_chart_certified
    assert not fallback.proof_certified

    with pytest.raises(RuntimeError, match="interval chart decision is not certified"):
        continue_hybrid(
            positions,
            velocities,
            masses,
            1e-5,
            binary_distance_threshold=0.01,
            binary_exit_distance=0.02,
            max_binary_s_step=1e-3,
            ordinary_order=10,
            binary_order=10,
            initial_state_interval=initial_state_interval,
            require_interval_chart_certification=True,
        )


def test_hybrid_ordinary_steps_record_end_state_interval_enclosures():
    masses, positions, velocities = _ordinary_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.05,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=12,
    )

    assert continued.steps
    for index, step in enumerate(continued.steps):
        assert step.end_state_interval is not None
        assert step.start_state_interval is not None
        assert step.start_state_interval_union is not None
        assert step.end_state_interval_union is not None
        assert len(step.start_state_interval_union) == 1
        assert len(step.end_state_interval_union) == 1
        assert step.start_state_interval_contains(continued.states[index])
        assert step.end_state_interval_contains(continued.states[index + 1])


def test_hybrid_ordinary_step_uses_recorded_interval_margins():
    masses, positions, velocities = _ordinary_planar_data()
    t_final = 0.05
    max_time_step = 1.0
    safety = 0.08

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=max_time_step,
        ordinary_order=12,
        safety=safety,
    )
    step = continued.steps[0]
    state = continued.states[0]
    start_velocities = state[6:].reshape(3, 2)
    expected_step = choose_interval_ordinary_step_size(
        step.start_state_interval,
        masses,
        t_final,
        max_step=max_time_step,
        safety=safety,
    )

    assert step.chart == "ordinary"
    assert step.interval_min_pair_distance is not None
    assert step.interval_max_speed is not None
    assert step.interval_max_acceleration is not None
    assert step.interval_min_pair_distance <= step.min_pair_distance
    assert step.interval_max_speed >= np.max(np.linalg.norm(start_velocities, axis=1))
    assert abs(step.physical_step - expected_step) < 1e-14
    assert step.start_state_interval == tuple(
        (float(value), float(value)) for value in pack_planar_state(positions, velocities)
    )


def test_hybrid_ordinary_interval_enclosures_feed_next_step():
    masses, positions, velocities = _ordinary_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.05,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=10,
    )

    assert len(continued.steps) > 1
    for index, (previous, current) in enumerate(zip(continued.steps, continued.steps[1:]), start=1):
        assert current.start_state_interval == previous.end_state_interval
        assert current.start_state_interval_union == previous.end_state_interval_union
        assert current.start_state_interval_contains(continued.states[index])


def test_hybrid_uses_binary_charts_for_close_pair_and_matches_reference():
    masses, positions, velocities = _close_pair_planar_data()
    t_final = 2e-5

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=1e-2,
        max_binary_s_step=0.005,
        binary_order=18,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)

    assert {step.chart for step in continued.steps} == {"binary"}
    assert {step.pair for step in continued.steps} == {(0, 1)}
    assert continued.steps[0].interval_chart == "binary"
    assert continued.steps[0].interval_chart_pair == (0, 1)
    assert continued.steps[0].interval_chart_certified
    assert continued.steps[0].binary_interval_lift_certified
    assert continued.steps[0].binary_lc_branch == "principal_upper_half"
    assert continued.steps[0].binary_lc_branch_certified
    assert continued.steps[0].binary_lc_atlas_chart_count == 1
    assert continued.steps[0].binary_lc_atlas_certified
    assert continued.steps[0].binary_lc_atlas_propagated
    assert continued.steps[0].end_state_interval_union is not None
    assert len(continued.steps[0].end_state_interval_union) == 1
    assert abs(continued.times[-1] - t_final) < 1e-14
    assert np.linalg.norm(continued.final_state - reference, ord=np.inf) < 1e-10


def test_hybrid_binary_steps_record_projected_end_state_interval_enclosures():
    masses, positions, velocities = _close_pair_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        2e-5,
        binary_distance_threshold=1e-2,
        max_binary_s_step=0.005,
        binary_order=16,
    )

    assert continued.steps
    assert {step.chart for step in continued.steps} == {"binary"}
    for index, step in enumerate(continued.steps):
        assert step.end_state_interval is not None
        assert step.end_state_interval_contains(continued.states[index + 1])


def test_hybrid_can_select_each_binary_pair_chart():
    for pair in [(0, 1), (0, 2), (1, 2)]:
        masses, positions, velocities = _close_pair_for(pair)

        continued = continue_hybrid(
            positions,
            velocities,
            masses,
            5e-7,
            binary_distance_threshold=1e-2,
            max_binary_s_step=0.002,
            binary_order=16,
        )

        assert continued.steps
        assert continued.steps[0].chart == "binary"
        assert continued.steps[0].pair == pair


def test_binary_step_solves_for_parameter_when_physical_target_is_inside_trial_step():
    masses, positions, velocities = _close_pair_planar_data()
    target_time = 1e-6

    end_positions, end_velocities, s_step, physical_step, certificate = regularized_binary_taylor_step(
        positions,
        velocities,
        masses,
        (0, 1),
        target_time,
        order=18,
        max_s_step=0.01,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    projected = np.concatenate([end_positions.reshape(-1), end_velocities.reshape(-1)])

    assert 0.0 < s_step < 0.01
    assert abs(physical_step - target_time) < 1e-18
    assert certificate is None
    assert np.linalg.norm(projected - reference, ord=np.inf) < 1e-11


def test_ordinary_step_stops_at_binary_entry_event():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])

    (
        end_positions,
        _end_velocities,
        physical_step,
        _indicator,
        event_pair,
        certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.05,
        order=20,
        binary_enter_distance=0.1,
    )
    distance = np.linalg.norm(end_positions[0] - end_positions[1])

    assert event_pair == (0, 1)
    assert certificate is not None
    assert certificate.is_isolated
    assert certificate.interval_is_isolated
    assert certificate.earliest_interval_is_certified
    assert certificate.coefficient_intervals is not None
    assert certificate.coefficient_source == "ordinary_interval_taylor"
    assert 0.0 < physical_step < 0.05
    assert abs(distance - 0.1) < 1e-12


def test_hybrid_records_binary_entry_event():
    masses = np.array([1.0, 1.0, 0.1])
    positions = np.array([[0.0, 0.0], [0.12, 0.0], [5.0, 0.0]])
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    (
        _end_positions,
        _end_velocities,
        event_time,
        _indicator,
        _event_pair,
        _certificate,
    ) = event_limited_ordinary_taylor_step(
        positions,
        velocities,
        masses,
        0.05,
        order=20,
        binary_enter_distance=0.1,
    )

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        event_time + 1e-4,
        binary_distance_threshold=0.1,
        binary_exit_distance=0.13,
        max_time_step=0.05,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
    )

    assert continued.steps[0].chart == "ordinary"
    assert continued.steps[0].event == "enter_binary"
    assert continued.steps[0].pair == (0, 1)
    assert continued.steps[0].event_certificate is not None
    assert continued.steps[0].event_certificate.is_isolated
    assert continued.steps[0].event_certificate.interval_is_isolated
    assert continued.steps[0].event_certificate.coefficient_intervals is not None
    assert continued.steps[0].event_certificate.coefficient_source == "ordinary_interval_taylor"
    assert continued.steps[0].event_union_certificate is not None
    assert continued.steps[0].event_union_certificate.interval_is_isolated
    assert continued.steps[0].event_union_certificate.earliest_interval_is_certified
    assert continued.steps[0].event_union_certificate.member_count == 1
    assert continued.steps[0].event_union_certificate.event_time_interval is not None
    assert continued.steps[0].event_time_interval == continued.steps[0].event_union_certificate.event_time_interval
    assert continued.steps[0].event_time_interval[0] <= continued.steps[0].physical_step <= continued.steps[0].event_time_interval[1]
    point_series = construct_taylor_solution(positions, velocities, masses, order=20)
    for time in continued.steps[0].event_time_interval:
        assert continued.steps[0].end_state_interval_contains(point_series.state_at(time))
    assert abs(continued.steps[0].end_time - event_time) < 1e-14
    assert len(continued.steps) >= 2
    assert continued.steps[1].chart == "binary"
    assert continued.steps[1].pair == (0, 1)
    assert continued.steps[1].interval_chart == "binary"
    assert continued.steps[1].interval_chart_pair == (0, 1)
    assert continued.steps[1].interval_chart_certified
    assert continued.steps[1].binary_interval_lift_certified
    assert continued.steps[1].binary_lc_branch_certified
    assert continued.steps[1].binary_lc_atlas_certified
    assert continued.steps[1].binary_lc_atlas_propagated


def test_binary_step_stops_at_exit_event():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [1e-3, 0.0], [1.0, 0.4]])
    velocities = np.array([[-40.0, 0.0], [40.0, 0.0], [-0.01, 0.0]])

    end_positions, _end_velocities, s_step, physical_step, certificate = regularized_binary_taylor_step(
        positions,
        velocities,
        masses,
        (0, 1),
        1.0,
        order=30,
        max_s_step=0.05,
        binary_exit_distance=0.004,
    )
    distance = np.linalg.norm(end_positions[0] - end_positions[1])

    assert 0.0 < s_step < 0.05
    assert physical_step > 0.0
    assert certificate is not None
    assert certificate.is_isolated
    assert certificate.interval_is_isolated
    assert certificate.coefficient_intervals is not None
    assert certificate.coefficient_source == "regularized_interval_taylor"
    assert abs(distance - 0.004) < 1e-12


def test_hybrid_continues_from_lifted_exact_binary_collision():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    s_target = 0.002
    point_solution = construct_regularized_binary_taylor_solution(initial, order=20)
    t_final = point_solution.physical_time_at(s_target)
    expected_end = point_solution.state_at(s_target)
    expected_positions, expected_velocities = regularized_binary_collision_chart_to_planar(expected_end)
    expected_state = pack_planar_state(expected_positions, expected_velocities)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.002,
        binary_exit_distance=0.004,
        max_binary_s_step=0.01,
        binary_order=20,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=3,
    )
    set_enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=20)

    assert len(continued.steps) == 1
    assert continued.proof_certified
    assert continued.steps[0].chart == "binary"
    assert continued.steps[0].event is None
    assert continued.steps[0].pair == initial.pair
    assert continued.steps[0].interval_chart_certified
    assert continued.steps[0].binary_interval_lift_certified
    assert continued.steps[0].binary_lc_branch == "exact_binary_collision"
    assert continued.steps[0].binary_lc_branch_certified
    assert continued.steps[0].start_regularized_interval_state is interval_initial
    assert continued.steps[0].truncation_certificate.coefficient_source == "regularized_interval_cauchy_majorant"
    assert continued.steps[0].truncation_certificate.ratio_bound < 1.0
    assert abs(continued.steps[0].parameter_step - s_target) < 1e-12
    assert np.linalg.norm(continued.final_state - expected_state, ord=np.inf) < 1e-10
    assert continued.steps[0].end_state_interval_contains(expected_state)
    assert set_enclosure.certified_step_count == 1
    assert set_enclosure.final_state_contains(expected_state)


def test_hybrid_exact_collision_start_stays_proof_certified_across_binary_steps():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    s_target = 0.01
    point_solution = construct_regularized_binary_taylor_solution(initial, order=30)
    t_final = point_solution.physical_time_at(s_target)
    expected_end = point_solution.state_at(s_target)
    expected_positions, expected_velocities = regularized_binary_collision_chart_to_planar(expected_end)
    expected_state = pack_planar_state(expected_positions, expected_velocities)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.002,
        binary_exit_distance=0.004,
        max_binary_s_step=0.003,
        binary_order=20,
        ordinary_order=20,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=8,
    )
    set_enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=20)

    assert len(continued.steps) == 4
    assert continued.proof_certified
    assert continued.proof_certified_step_count == len(continued.steps)
    assert all(step.chart == "binary" and step.event is None for step in continued.steps)
    assert continued.steps[0].start_regularized_interval_state is interval_initial
    assert all(step.start_regularized_interval_state is None for step in continued.steps[1:])
    assert all(
        step.truncation_certificate.coefficient_source == "regularized_interval_cauchy_majorant"
        for step in continued.steps
    )
    assert all(step.truncation_certificate.ratio_bound < 1.0 for step in continued.steps)
    assert np.linalg.norm(continued.final_state - expected_state, ord=np.inf) < 1e-10
    assert len(set_enclosure.steps) == len(continued.steps)
    assert set_enclosure.final_state_contains(expected_state)
    assert set_enclosure.final_state_contains(continued.final_state)


def test_hybrid_exact_collision_start_certifies_binary_exit_event():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 1e-9)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        binary_order=24,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=1,
    )
    set_enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=24)
    step = continued.steps[0]

    assert len(continued.steps) == 1
    assert continued.proof_certified
    assert step.event == "exit_binary"
    assert step.event_certificate is not None
    assert step.event_certificate.earliest_interval_is_certified
    assert step.event_certificate.coefficient_source == "regularized_interval_taylor"
    assert step.event_certificate.root_enclosure.interval[0] <= step.parameter_step <= step.event_certificate.root_enclosure.interval[1]
    assert set_enclosure.steps[0].event == "exit_binary"
    assert set_enclosure.steps[0].event_parameter_interval == step.event_certificate.root_enclosure.interval
    assert set_enclosure.final_state_contains(continued.final_state)


def test_hybrid_exact_collision_start_hands_off_to_proof_certified_ordinary_chart():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 1e-6)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        max_time_step=1e-12,
        binary_order=16,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=4,
    )
    set_enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=16)

    assert len(continued.steps) == 2
    assert continued.proof_certified
    assert continued.proof_certified_step_count == len(continued.steps)
    assert [step.chart for step in continued.steps] == ["binary", "ordinary"]
    assert continued.steps[0].event == "exit_binary"
    assert continued.steps[0].event_certificate.earliest_interval_is_certified
    assert continued.steps[1].event is None
    assert continued.steps[1].interval_chart_certified
    assert continued.steps[1].start_time == continued.steps[0].end_time
    assert continued.steps[1].truncation_certificate.coefficient_source == "interval_cauchy_majorant_union"
    assert continued.steps[1].truncation_certificate.ratio_bound < 1.0
    assert len(set_enclosure.steps) == len(continued.steps)
    assert [step.chart for step in set_enclosure.steps] == ["binary", "ordinary"]
    assert set_enclosure.steps[0].event == "exit_binary"
    assert set_enclosure.steps[1].tail_coefficient_source == "interval_cauchy_majorant"
    assert set_enclosure.final_state_contains(continued.final_state)


def test_validated_atlas_from_hybrid_solution_closes_binary_handoff_ledgers():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 1e-6)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        max_time_step=1e-12,
        binary_order=16,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=4,
    )
    atlas = validated_atlas_from_hybrid_solution(continued)

    assert continued.proof_certified
    assert atlas.chart_count == 2
    assert atlas.transition_count == 1
    assert [chart.chart_type for chart in atlas.charts] == [
        "planar_levi_civita_binary",
        "planar_ordinary_taylor",
    ]
    assert atlas.transitions[0].transition_type == "binary_to_ordinary_event_handoff"
    assert atlas.transitions[0].certified
    assert atlas.tail_budget.certified
    assert atlas.collision_policy.binary_policy == "planar_levi_civita_binary_regularized"
    assert atlas.target_state_contains(continued.final_state)

    assert atlas.proof_certified
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified
    assert atlas.proof_ledger.missing_required_obligations == ()
    assert all(chart.dynamics_certified and chart.tail_certified for chart in atlas.charts)
    assert all(chart.residual_certified for chart in atlas.charts)
    assert all(chart.projection_certified for chart in atlas.charts)
    assert [chart.invariants_certified for chart in atlas.charts] == [True, True]
    assert continued.steps[0].residual_certificate.projection_certified
    assert continued.steps[0].residual_certificate.projection_domain_certificate is not None
    assert continued.steps[0].residual_certificate.projection_domain == (
        "punctured_initial_collision_rho_positive_interval"
    )
    assert continued.steps[0].residual_certificate.projection_domain_certificate.certified
    assert continued.steps[0].residual_certificate.projection_domain_certificate.endpoint_projection_certified
    assert (
        continued.steps[0]
        .residual_certificate
        .projection_domain_certificate
        .third_body_separation_certified
    )
    assert (
        continued.steps[0]
        .residual_certificate
        .projection_domain_certificate
        .min_competing_pair_squared_distance_lower_bound
        > 0.0
    )
    assert continued.steps[0].center_of_mass_certificate.certified
    assert continued.steps[0].linear_momentum_certificate.certified
    assert continued.steps[0].angular_momentum_certificate.certified
    assert continued.steps[0].energy_certificate.certified
    assert continued.steps[0].energy_certificate.coefficient_source == (
        "regularized_binary_interval_series"
    )

    interior_time = continued.steps[-1].start_time + 0.5 * continued.steps[-1].physical_step
    interior_time_atlas = validated_atlas_from_hybrid_solution(
        continued,
        target_time=interior_time,
    )

    assert interior_time_atlas.target_time_certified
    assert interior_time_atlas.proof_certified
    assert interior_time_atlas.evaluation.target_step_index == 1
    assert interior_time_atlas.evaluation.target_interval_source == (
        "hybrid_chart_target_enclosure"
    )
    assert interior_time_atlas.target_state_contains(
        interior_time_atlas.evaluation.final_state
    )
    assert interior_time_atlas.chart_count == 2
    assert interior_time_atlas.proof_ledger.missing_required_obligations == ()

    binary_interior_time = continued.steps[0].start_time + 0.5 * continued.steps[0].physical_step
    binary_interior_time_atlas = validated_atlas_from_hybrid_solution(
        continued,
        target_time=binary_interior_time,
    )

    assert binary_interior_time_atlas.target_time_certified
    assert binary_interior_time_atlas.proof_certified
    assert binary_interior_time_atlas.evaluation.target_step_index == 0
    assert binary_interior_time_atlas.evaluation.target_interval_source == (
        "hybrid_chart_target_enclosure"
    )
    assert binary_interior_time_atlas.chart_count == 1
    assert binary_interior_time_atlas.target_state_contains(
        binary_interior_time_atlas.evaluation.final_state
    )
    assert binary_interior_time_atlas.proof_ledger.missing_required_obligations == ()

    wrong_time_atlas = validated_atlas_from_hybrid_solution(
        continued,
        target_time=t_final + 1.0e-3,
    )

    assert wrong_time_atlas.target_state_contains(continued.final_state)
    assert not wrong_time_atlas.target_time_certified
    assert not wrong_time_atlas.proof_certified
    assert "hybrid_requested_target_time" in (
        wrong_time_atlas.proof_ledger.missing_required_obligations
    )
    assert "target_time_domain" in (
        wrong_time_atlas.proof_ledger.missing_required_obligations
    )


def test_binary_projection_domain_rejects_closed_interval_containing_rho_zero():
    interval_initial = _exact_collision_interval_state()
    series = construct_interval_regularized_binary_taylor_solution_from_intervals(
        interval_initial,
        order=12,
    )

    rejected = certify_regularized_binary_projection_domain(
        series,
        parameter_interval=(0.0, 0.01),
        endpoint_parameter_interval=(0.01, 0.01),
        branch_or_atlas_certified=True,
        starts_at_binary_collision=False,
    )
    accepted = certify_regularized_binary_projection_domain(
        series,
        parameter_interval=(0.0, 0.01),
        endpoint_parameter_interval=(0.01, 0.01),
        branch_or_atlas_certified=True,
        starts_at_binary_collision=True,
        initial_collision_selector_certified=True,
    )
    residual_without_domain = certify_regularized_binary_interval_taylor_equations(
        series,
        coefficient_count=12,
    )
    residual_with_domain = certify_regularized_binary_interval_taylor_equations(
        series,
        coefficient_count=12,
        projection_domain_certificate=accepted,
    )

    assert not rejected.certified
    assert "projection_rho_interval_contains_zero" in rejected.missing_obligations
    assert accepted.certified
    assert accepted.projection_domain == (
        "punctured_initial_collision_rho_positive_interval"
    )
    assert accepted.third_body_separation_certified
    assert residual_without_domain.certified
    assert not residual_without_domain.projection_certified
    assert residual_without_domain.projection_domain == (
        "missing_projection_domain_certificate"
    )
    assert residual_with_domain.projection_certified


def test_binary_projection_domain_rejects_uncertified_third_body_separation():
    interval_initial = _exact_collision_interval_state()
    series = construct_interval_regularized_binary_taylor_solution_from_intervals(
        interval_initial,
        order=12,
    )
    bad_third_offset = np.empty_like(series.third_offset)
    for index in np.ndindex(bad_third_offset.shape):
        bad_third_offset[index] = FloatInterval.point(0.0)
    bad_series = replace(series, third_offset=bad_third_offset)

    certificate = certify_regularized_binary_projection_domain(
        bad_series,
        parameter_interval=(0.0, 0.01),
        endpoint_parameter_interval=(0.01, 0.01),
        branch_or_atlas_certified=True,
        starts_at_binary_collision=True,
        initial_collision_selector_certified=True,
    )

    assert not certificate.certified
    assert not certificate.third_body_separation_certified
    assert certificate.min_competing_pair_squared_distance_lower_bound == pytest.approx(0.0)
    assert "third_body_separation_not_certified" in certificate.missing_obligations


def test_branch_cut_lc_atlas_projection_domain_has_member_certificates():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array([[0.0, 0.0], [-1.0, 0.1], [3.0, 0.0]])
    velocities = np.zeros((3, 2))
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        1.0e-5,
        binary_distance_threshold=2.0,
        binary_exit_distance=2.5,
        max_binary_s_step=1.0e-3,
        binary_order=12,
        tail_certificate_mode="cauchy",
        initial_state_interval=state_interval,
        require_interval_chart_certification=True,
        max_steps=2,
    )
    step = continued.steps[0]
    member_certificates = tuple(step.residual_certificate.member_certificates)
    domain_certificates = tuple(
        certificate.projection_domain_certificate
        for certificate in member_certificates
    )

    assert continued.proof_certified
    assert step.chart == "binary"
    assert step.binary_lc_atlas_chart_count == 2
    assert step.binary_lc_atlas_certified
    assert step.binary_lc_atlas_propagated
    assert step.residual_certificate.member_count == 2
    assert step.residual_certificate.projection_certified
    assert step.residual_certificate.coefficient_source == (
        "regularized_binary_interval_taylor_atlas"
    )
    assert step.center_of_mass_certificate.member_count == 2
    assert step.energy_certificate.member_count == 2
    assert all(certificate is not None for certificate in domain_certificates)
    assert all(certificate.certified for certificate in domain_certificates)
    assert all(
        certificate.third_body_separation_certified
        for certificate in domain_certificates
    )
    assert all(
        certificate.projection_domain == "rho_positive_interval"
        for certificate in domain_certificates
    )


def test_validated_atlas_planar_binary_requires_projection_domain_certificate():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 1e-6)
    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        max_time_step=1e-12,
        binary_order=16,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=4,
    )
    steps = list(continued.steps)
    stale_residual = replace(
        steps[0].residual_certificate,
        projection_domain_certificate=None,
    )
    steps[0] = replace(steps[0], residual_certificate=stale_residual)
    stale_continued = replace(continued, steps=tuple(steps))
    stale_atlas = validated_atlas_from_hybrid_solution(stale_continued)

    assert continued.proof_certified
    assert continued.steps[0].residual_certificate.projection_certified
    assert not stale_continued.proof_certified
    assert not stale_atlas.proof_certified
    assert "projection_ledger" in stale_atlas.proof_ledger.missing_required_obligations


def test_time_reversal_preserves_proof_certified_binary_hybrid_atlas():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 1e-6)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        max_time_step=1e-12,
        binary_order=16,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=4,
    )
    forward_atlas = validated_atlas_from_hybrid_solution(continued)
    reversed_atlas = time_reverse_validated_atlas_solution(forward_atlas)
    reversed_final = continued.final_state.copy()
    reversed_final[reversed_final.size // 2 :] *= -1.0

    assert forward_atlas.proof_certified
    assert reversed_atlas.proof_certified
    assert reversed_atlas.target_time == pytest.approx(-forward_atlas.target_time)
    assert [chart.chart_type for chart in reversed_atlas.charts] == [
        "planar_levi_civita_binary",
        "planar_ordinary_taylor",
    ]
    assert all(chart.physical_time_interval.upper <= 0.0 for chart in reversed_atlas.charts)
    assert reversed_atlas.target_state_contains(reversed_final)
    assert reversed_atlas.proof_ledger.well_formed
    assert reversed_atlas.proof_ledger.missing_required_obligations == ()


def test_validated_atlas_from_ordinary_hybrid_solution_closes_residual_and_invariant_ledgers():
    masses, positions, velocities = _ordinary_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.03,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
    )
    atlas = validated_atlas_from_hybrid_solution(continued)

    assert continued.proof_certified
    assert atlas.chart_count == len(continued.steps)
    assert {chart.chart_type for chart in atlas.charts} == {"planar_ordinary_taylor"}
    assert atlas.residual_budget.certified
    assert atlas.invariants.certified
    assert atlas.proof_certified
    assert atlas.proof_ledger.missing_required_obligations == ()
    assert all(chart.residual_certified for chart in atlas.charts)
    assert all(chart.invariants_certified for chart in atlas.charts)
    assert atlas.target_state_contains(continued.final_state)


def test_hybrid_exact_collision_post_exit_ordinary_set_propagation_substeps():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    exit_distance = 1e-5
    t_final = _exact_collision_exit_time(initial, exit_distance) * (1.0 + 3e-6)

    continued = continue_hybrid_from_regularized_binary_collision(
        initial,
        interval_initial,
        t_final,
        binary_distance_threshold=0.5 * exit_distance,
        binary_exit_distance=exit_distance,
        max_binary_s_step=0.1,
        max_time_step=1e-12,
        binary_order=16,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
        require_interval_chart_certification=True,
        max_steps=5,
    )
    set_enclosure = continued.ordinary_set_propagated_interval_enclosure(
        retained_order=12,
        ordinary_substeps=4,
    )

    assert len(continued.steps) == 3
    assert continued.proof_certified
    assert [step.chart for step in continued.steps] == ["binary", "ordinary", "ordinary"]
    assert len(set_enclosure.steps) == len(continued.steps)
    assert [step.ordinary_substeps for step in set_enclosure.steps] == [1, 4, 4]
    assert set_enclosure.steps[1].tail_coefficient_source == "interval_cauchy_majorant_substeps"
    assert set_enclosure.steps[2].tail_coefficient_source == "interval_cauchy_majorant_substeps"
    assert interval_pairwise_distance_bounds(set_enclosure.final_state_interval)[0][1][0] > 0.0
    assert set_enclosure.final_state_contains(continued.final_state)


def test_hybrid_records_binary_exit_event():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [1e-3, 0.0], [1.0, 0.4]])
    velocities = np.array([[-40.0, 0.0], [40.0, 0.0], [-0.01, 0.0]])
    _end_positions, _end_velocities, _s_step, physical_step, _certificate = regularized_binary_taylor_step(
        positions,
        velocities,
        masses,
        (0, 1),
        1.0,
        order=30,
        max_s_step=0.05,
        binary_exit_distance=0.004,
    )

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        physical_step,
        binary_distance_threshold=0.002,
        binary_exit_distance=0.004,
        max_binary_s_step=0.05,
        binary_order=30,
        max_steps=5,
    )

    assert len(continued.steps) == 1
    assert continued.steps[0].chart == "binary"
    assert continued.steps[0].event == "exit_binary"
    assert continued.steps[0].pair == (0, 1)
    assert continued.steps[0].event_certificate is not None
    assert continued.steps[0].event_certificate.is_isolated
    assert continued.steps[0].event_certificate.interval_is_isolated
    assert continued.steps[0].event_certificate.coefficient_intervals is not None
    assert continued.steps[0].event_certificate.coefficient_source == "regularized_interval_taylor"
    assert abs(continued.times[-1] - physical_step) < 1e-14


def test_hybrid_aggregates_local_tail_budget():
    masses, positions, velocities = _ordinary_planar_data()

    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.05,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=10,
        tail_guard_order=8,
    )
    step_bounds = [step.truncation_certificate.tail_bound for step in continued.steps]

    assert continued.certified_step_count == len(continued.steps)
    assert continued.local_tail_bound == sum(step_bounds)
    assert continued.max_step_tail_bound == max(step_bounds)
    assert continued.local_tail_bound > 0.0
