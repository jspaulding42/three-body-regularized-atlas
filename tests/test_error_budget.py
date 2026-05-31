from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from three_body_symmetry.binary_chart import (
    RegularizedBinaryCollisionChartState,
    exact_binary_collision_interval_chart,
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from three_body_symmetry.binary_series import construct_regularized_binary_taylor_solution
from three_body_symmetry.error_budget import (
    LohnerOrdinaryPropagatedEnclosure,
    LohnerOrdinaryTaylorStepCertificate,
    LohnerStateSet,
    center_of_mass_reduction_lohner_map,
    inflate_state_interval,
    interval_contains_state,
    lohner_state_from_interval_box,
    newtonian_planar_lipschitz_bound,
    propagate_error_budget,
    propagate_lohner_affine_map,
    propagate_lohner_ordinary_set_enclosures,
    propagate_lohner_ordinary_taylor_step,
    propagate_ordinary_set_enclosures,
    reduce_lohner_state_to_center_of_mass_frame,
    propagate_sundman_target_set_enclosure,
    state_interval_subset,
)
from three_body_symmetry.hybrid import (
    certify_ordinary_binary_entry_event_union,
    continue_hybrid,
    event_limited_ordinary_taylor_step,
    pack_planar_state,
    regularized_binary_taylor_step,
)
from three_body_symmetry.series import construct_taylor_solution, integrate_reference
from three_body_symmetry.sundman import continue_interval_sundman_to_time, continue_sundman_to_s
from three_body_symmetry.tail_bounds import sundman_interval_cauchy_majorant_tail_certificate


def _ordinary_data():
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


def _exact_collision_position_summary_interval(initial):
    first, second = initial.pair
    third = initial.third_index
    positions = np.empty((3, 2), dtype=float)
    velocities = np.empty((3, 2), dtype=float)
    positions[first] = initial.binary_center
    positions[second] = initial.binary_center
    positions[third] = initial.binary_center + initial.third_offset
    velocities[first] = initial.binary_center_velocity
    velocities[second] = initial.binary_center_velocity
    velocities[third] = initial.binary_center_velocity + initial.third_offset_velocity
    return tuple((float(value), float(value)) for value in pack_planar_state(positions, velocities))


def _max_interval_width(state_interval):
    return max(upper - lower for lower, upper in state_interval)


def _hull_interval_states(states):
    return tuple(
        (
            min(state[index][0] for state in states),
            max(state[index][1] for state in states),
        )
        for index in range(len(states[0]))
    )


def _naive_affine_interval_map(state_interval, matrix, bias=None):
    matrix = np.asarray(matrix, dtype=float)
    if bias is None:
        bias = np.zeros(matrix.shape[0])
    out = []
    for row, shift in zip(matrix, bias):
        lower = float(shift)
        upper = float(shift)
        for coefficient, (interval_lower, interval_upper) in zip(row, state_interval):
            products = (
                coefficient * interval_lower,
                coefficient * interval_upper,
            )
            lower += min(products)
            upper += max(products)
        out.append((lower, upper))
    return tuple(out)


def test_lohner_affine_map_preserves_correlation_lost_by_interval_hull():
    state_set = LohnerStateSet(
        center=np.array([0.0, 0.0]),
        linear_shape=np.array([[1.0], [1.0]]),
        remainder_box=((0.0, 0.0), (0.0, 0.0)),
    )
    matrix = np.array([[1.0, -1.0], [1.0, 1.0]])
    certificate = propagate_lohner_affine_map(state_set, matrix)
    target = certificate.target_set
    naive = _naive_affine_interval_map(state_set.interval_hull, matrix)

    assert certificate.certified
    lohner_cancelled_width = target.interval_hull[0][1] - target.interval_hull[0][0]
    naive_cancelled_width = naive[0][1] - naive[0][0]
    assert lohner_cancelled_width < naive_cancelled_width
    assert target.interval_hull[0][0] <= 0.0 <= target.interval_hull[0][1]
    assert lohner_cancelled_width < 1e-14
    assert naive_cancelled_width == pytest.approx(4.0)
    for parameter in (-1.0, -0.25, 0.5, 1.0):
        source_state = np.array([parameter, parameter])
        assert target.contains_state(matrix @ source_state)


def test_lohner_center_of_mass_reduction_preserves_zero_mass_moment_correlation():
    masses, positions, velocities = _ordinary_data()
    state = pack_planar_state(positions, velocities)
    state_interval = tuple((float(value - 1e-4), float(value + 1e-4)) for value in state)
    source_set = lohner_state_from_interval_box(state_interval)
    reduction = reduce_lohner_state_to_center_of_mass_frame(
        source_set,
        masses,
        dimension=2,
    )
    reduced = reduction.target_set
    matrix = center_of_mass_reduction_lohner_map(masses, 2)

    assert reduction.certified
    assert matrix.shape == (12, 12)
    for axis in range(2):
        position_weights = np.zeros(12)
        velocity_weights = np.zeros(12)
        for body, mass in enumerate(masses):
            position_weights[body * 2 + axis] = mass
            velocity_weights[6 + body * 2 + axis] = mass
        position_moment = reduced.linear_observable_interval(position_weights)
        velocity_moment = reduced.linear_observable_interval(velocity_weights)
        assert max(abs(position_moment[0]), abs(position_moment[1])) < 1e-12
        assert max(abs(velocity_moment[0]), abs(velocity_moment[1])) < 1e-12

    center_position = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    center_velocity = np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    point_reduced_positions = positions - center_position
    point_reduced_velocities = velocities - center_velocity
    assert reduced.contains_state(pack_planar_state(point_reduced_positions, point_reduced_velocities))


def test_lohner_ordinary_taylor_step_keeps_kinematic_cancellation_tighter_than_box_hull():
    masses, positions, velocities = _ordinary_data()
    h = 0.02
    epsilon = 1e-4
    center = pack_planar_state(positions, velocities)
    shape = np.zeros((12, 1))
    shape[0, 0] = epsilon
    shape[6, 0] = -epsilon / h
    state_set = LohnerStateSet(
        center=center,
        linear_shape=shape,
        remainder_box=tuple((0.0, 0.0) for _index in range(12)),
    )
    certificate = propagate_lohner_ordinary_taylor_step(
        state_set,
        masses,
        h,
        retained_order=8,
    )
    box_step = SimpleNamespace(
        chart="ordinary",
        start_time=0.0,
        physical_step=h,
        start_state_interval=state_set.interval_hull,
        start_state_interval_union=(),
        event=None,
    )
    box_enclosure = propagate_ordinary_set_enclosures(masses, (box_step,), retained_order=8)

    assert isinstance(certificate, LohnerOrdinaryTaylorStepCertificate)
    assert certificate.proof_certified
    lohner_width = certificate.target_set.interval_hull[0][1] - certificate.target_set.interval_hull[0][0]
    box_width = box_enclosure.final_state_interval[0][1] - box_enclosure.final_state_interval[0][0]
    assert lohner_width < 0.1 * box_width


def test_lohner_ordinary_taylor_step_contains_sampled_chart_endpoints():
    masses, positions, velocities = _ordinary_data()
    h = 0.015
    epsilon = 5e-5
    center = pack_planar_state(positions, velocities)
    shape = np.zeros((12, 2))
    shape[0, 0] = epsilon
    shape[6, 0] = -epsilon / h
    shape[3, 1] = -epsilon
    shape[9, 1] = epsilon / h
    state_set = LohnerStateSet(
        center=center,
        linear_shape=shape,
        remainder_box=tuple((0.0, 0.0) for _index in range(12)),
    )
    certificate = propagate_lohner_ordinary_taylor_step(
        state_set,
        masses,
        h,
        retained_order=8,
    )

    assert certificate.certified
    for parameters in (
        np.array([-1.0, -1.0]),
        np.array([-0.5, 0.25]),
        np.array([0.0, 0.0]),
        np.array([0.25, -0.5]),
        np.array([1.0, 1.0]),
    ):
        source_state = center + shape @ parameters
        sample_positions = source_state[:6].reshape(3, 2)
        sample_velocities = source_state[6:].reshape(3, 2)
        endpoint = construct_taylor_solution(
            sample_positions,
            sample_velocities,
            masses,
            order=8,
        ).state_at(h)
        assert certificate.target_set.contains_state(endpoint)


def test_lohner_ordinary_chain_preserves_correlation_across_multiple_steps():
    masses, positions, velocities = _ordinary_data()
    h = 0.01
    epsilon = 1e-4
    center = pack_planar_state(positions, velocities)
    shape = np.zeros((12, 1))
    shape[0, 0] = epsilon
    shape[6, 0] = -epsilon / (2.0 * h)
    state_set = LohnerStateSet(
        center=center,
        linear_shape=shape,
        remainder_box=tuple((0.0, 0.0) for _index in range(12)),
    )
    steps = (
        SimpleNamespace(
            chart="ordinary",
            event=None,
            start_time=0.0,
            physical_step=h,
            start_state_interval=state_set.interval_hull,
        ),
        SimpleNamespace(
            chart="ordinary",
            event=None,
            start_time=h,
            physical_step=h,
            start_state_interval=None,
        ),
    )
    lohner = propagate_lohner_ordinary_set_enclosures(
        masses,
        steps,
        retained_order=8,
        initial_state_set=state_set,
    )
    box = propagate_ordinary_set_enclosures(masses, steps[:1], retained_order=8)
    second_box_step = SimpleNamespace(
        chart="ordinary",
        event=None,
        start_time=h,
        physical_step=h,
        start_state_interval=box.final_state_interval,
        start_state_interval_union=(),
    )
    box = propagate_ordinary_set_enclosures(masses, (second_box_step,), retained_order=8)

    assert isinstance(lohner, LohnerOrdinaryPropagatedEnclosure)
    assert lohner.proof_certified
    assert lohner.chain_certified
    assert lohner.certified_step_count == 2
    lohner_width = lohner.final_state_interval[0][1] - lohner.final_state_interval[0][0]
    box_width = box.final_state_interval[0][1] - box.final_state_interval[0][0]
    assert lohner_width < 0.1 * box_width


def test_lohner_ordinary_chain_contains_sampled_multi_step_endpoints_and_rejects_uncertified_events():
    masses, positions, velocities = _ordinary_data()
    h = 0.01
    epsilon = 4e-5
    center = pack_planar_state(positions, velocities)
    shape = np.zeros((12, 2))
    shape[0, 0] = epsilon
    shape[6, 0] = -epsilon / (2.0 * h)
    shape[4, 1] = epsilon
    shape[10, 1] = -epsilon / h
    state_set = LohnerStateSet(
        center=center,
        linear_shape=shape,
        remainder_box=tuple((0.0, 0.0) for _index in range(12)),
    )
    steps = (
        SimpleNamespace(chart="ordinary", event=None, start_time=0.0, physical_step=h),
        SimpleNamespace(chart="ordinary", event=None, start_time=h, physical_step=h),
    )
    enclosure = propagate_lohner_ordinary_set_enclosures(
        masses,
        steps,
        retained_order=8,
        initial_state_set=state_set,
    )

    assert enclosure.proof_certified
    assert enclosure.chain_certified
    bad_second = replace(
        enclosure.steps[1],
        start_state_set=enclosure.steps[0].start_state_set,
    )
    corrupted = replace(enclosure, steps=(enclosure.steps[0], bad_second))
    assert bad_second.proof_certified
    assert not corrupted.chain_certified
    assert not corrupted.proof_certified
    for parameters in (
        np.array([-1.0, -1.0]),
        np.array([-0.25, 0.5]),
        np.array([0.5, -0.25]),
        np.array([1.0, 1.0]),
    ):
        source_state = center + shape @ parameters
        sample_positions = source_state[:6].reshape(3, 2)
        sample_velocities = source_state[6:].reshape(3, 2)
        first = construct_taylor_solution(
            sample_positions,
            sample_velocities,
            masses,
            order=8,
        ).state_at(h)
        second = construct_taylor_solution(
            first[:6].reshape(3, 2),
            first[6:].reshape(3, 2),
            masses,
            order=8,
        ).state_at(h)
        assert enclosure.final_state_contains(second)

    with pytest.raises(ValueError, match="event-localized"):
        propagate_lohner_ordinary_set_enclosures(
            masses,
            (
                SimpleNamespace(chart="ordinary", event="enter_binary", start_time=0.0, physical_step=h),
            ),
            retained_order=8,
            initial_state_set=state_set,
        )


def test_lohner_ordinary_event_step_uses_certified_time_interval():
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
        0.01,
        order=20,
        binary_enter_distance=0.1199,
    )
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        event_time + 1e-8,
        binary_distance_threshold=0.1199,
        binary_exit_distance=0.13,
        max_time_step=0.01,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
        tail_certificate_mode="cauchy",
        max_steps=5,
    )

    enclosure = propagate_lohner_ordinary_set_enclosures(
        masses,
        (continued.steps[0],),
        retained_order=20,
    )
    event_step = enclosure.steps[0]

    assert enclosure.proof_certified
    assert event_step.event == "enter_binary"
    assert event_step.event_time_interval == continued.steps[0].event_time_interval
    assert event_step.step_certificate.time_interval == continued.steps[0].event_time_interval
    assert event_step.end_state_contains(continued.states[1])
    point_series = construct_taylor_solution(positions, velocities, masses, order=20)
    for time in event_step.event_time_interval:
        assert event_step.truncated_end_state_set.contains_state(point_series.state_at(time))


def test_lipschitz_bound_grows_as_pair_distance_shrinks():
    masses = np.array([1.0, 0.7, 1.4])

    far = newtonian_planar_lipschitz_bound(masses, 1.0)
    close = newtonian_planar_lipschitz_bound(masses, 0.5)

    assert far >= 1.0
    assert close > far


def test_propagated_error_budget_dominates_local_tail_ledger():
    masses, positions, velocities = _ordinary_data()
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

    budget = continued.propagated_error_budget()

    assert budget.steps
    assert budget.local_tail_bound == continued.local_tail_bound
    assert budget.final_bound >= budget.local_tail_bound
    assert budget.final_bound >= continued.max_step_tail_bound
    assert budget.max_lipschitz_bound >= 1.0


def test_inflated_state_interval_contains_radius_ball():
    state_interval = ((1.0, 1.0), (-2.0, -2.0), (0.5, 0.75))
    inflated = inflate_state_interval(state_interval, 0.25)

    assert interval_contains_state(inflated, np.array([1.2, -2.2, 0.6]))
    assert not interval_contains_state(inflated, np.array([1.3, -2.2, 0.6]))


def test_propagated_interval_enclosure_materializes_error_budget():
    masses, positions, velocities = _ordinary_data()
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

    budget = continued.propagated_error_budget()
    enclosure = continued.propagated_interval_enclosure()

    assert len(enclosure.steps) == len(continued.steps)
    assert enclosure.local_tail_bound == budget.local_tail_bound
    assert enclosure.final_radius == budget.final_bound
    assert enclosure.max_lipschitz_bound == budget.max_lipschitz_bound
    for index, step in enumerate(enclosure.steps):
        assert step.start_state_contains(continued.states[index])
        assert step.end_state_contains(continued.states[index + 1])


def test_propagated_budget_covers_lower_order_difference_on_ordinary_path():
    masses, positions, velocities = _ordinary_data()
    t_final = 0.05
    low_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=10,
        tail_guard_order=8,
    )
    high_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=18,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)
    budget = propagate_error_budget(masses, low_order.steps)

    assert np.linalg.norm(low_order.final_state - high_order.final_state, ord=np.inf) <= budget.final_bound
    assert np.linalg.norm(low_order.final_state - reference, ord=np.inf) <= budget.final_bound


def test_propagated_interval_enclosure_contains_higher_order_and_reference_endpoints():
    masses, positions, velocities = _ordinary_data()
    t_final = 0.05
    low_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=10,
        tail_guard_order=8,
    )
    high_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.02,
        ordinary_order=18,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)

    enclosure = low_order.propagated_interval_enclosure()

    assert enclosure.final_state_contains(low_order.final_state)
    assert enclosure.final_state_contains(high_order.final_state)
    assert enclosure.final_state_contains(reference)


def test_ordinary_set_propagation_rebuilds_each_step_from_incoming_box():
    masses, positions, velocities = _ordinary_data()
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        0.03,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
    )

    enclosure = continued.ordinary_set_propagated_interval_enclosure()

    assert len(enclosure.steps) == len(continued.steps) == 3
    assert enclosure.certified_step_count == 3
    for index, step in enumerate(enclosure.steps):
        assert step.tail_coefficient_source == "interval_cauchy_majorant"
        assert step.tail_ratio_bound < 1.0
        assert step.start_state_contains(continued.states[index])
        assert step.end_state_contains(continued.states[index + 1])
        if index > 0:
            assert step.start_state_interval == enclosure.steps[index - 1].end_state_interval
            assert step.start_state_interval != continued.steps[index].start_state_interval
    assert enclosure.proof_certified
    assert enclosure.chain_certified

    bad_time_step = replace(
        enclosure.steps[1],
        start_time=enclosure.steps[1].start_time + 0.25,
    )
    corrupted = replace(
        enclosure,
        steps=(enclosure.steps[0], bad_time_step, *enclosure.steps[2:]),
    )
    assert bad_time_step.proof_certified
    assert not corrupted.chain_certified
    assert not corrupted.proof_certified


def test_ordinary_set_propagation_preserves_start_union_without_hulling_through_collision():
    masses = np.ones(3)
    velocities = np.zeros((3, 2))
    left_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(
            np.array([[0.0, 0.0], [-1.0, 0.0], [4.0, 0.0]]),
            velocities,
        )
    )
    right_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(
            np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
            velocities,
        )
    )
    start_hull = _hull_interval_states((left_state, right_state))
    step = SimpleNamespace(
        chart="ordinary",
        event=None,
        start_time=0.0,
        physical_step=1e-3,
        parameter_step=1e-3,
        start_state_interval=start_hull,
        start_state_interval_union=(left_state, right_state),
        truncation_certificate=None,
    )

    enclosure = propagate_ordinary_set_enclosures(masses, (step,), retained_order=8)
    propagated_step = enclosure.steps[0]

    assert state_interval_subset(start_hull, propagated_step.start_state_interval)
    assert propagated_step.start_state_interval_union == (left_state, right_state)
    assert propagated_step.union_member_count == 2
    assert len(propagated_step.truncated_end_state_interval_union) == 2
    assert propagated_step.tail_coefficient_source == "interval_cauchy_majorant"
    assert propagated_step.tail_ratio_bound < 1.0
    for state in (left_state, right_state):
        positions = np.array([value for value, _upper in state[:6]], dtype=float).reshape(3, 2)
        point_endpoint = construct_taylor_solution(positions, velocities, masses, order=10).state_at(1e-3)
        assert propagated_step.end_state_contains(point_endpoint)

    hull_only_step = SimpleNamespace(
        chart="ordinary",
        event=None,
        start_time=0.0,
        physical_step=1e-3,
        parameter_step=1e-3,
        start_state_interval=start_hull,
        truncation_certificate=None,
    )
    with pytest.raises(ValueError, match="positive"):
        propagate_ordinary_set_enclosures(masses, (hull_only_step,), retained_order=8)


def test_ordinary_set_propagation_proof_certifies_union_chain_not_just_hulls():
    masses = np.ones(3)
    velocities = np.zeros((3, 2))
    left_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(
            np.array([[0.0, 0.0], [-1.0, 0.0], [4.0, 0.0]]),
            velocities,
        )
    )
    right_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(
            np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
            velocities,
        )
    )
    start_hull = _hull_interval_states((left_state, right_state))
    steps = (
        SimpleNamespace(
            chart="ordinary",
            event=None,
            start_time=0.0,
            physical_step=1e-3,
            parameter_step=1e-3,
            start_state_interval=start_hull,
            start_state_interval_union=(left_state, right_state),
            truncation_certificate=None,
        ),
        SimpleNamespace(
            chart="ordinary",
            event=None,
            start_time=1e-3,
            physical_step=1e-3,
            parameter_step=1e-3,
            start_state_interval=start_hull,
            truncation_certificate=None,
        ),
    )

    enclosure = propagate_ordinary_set_enclosures(masses, steps, retained_order=8)
    second = enclosure.steps[1]
    bad_second = replace(
        second,
        start_state_interval_union=(second.start_state_interval,),
    )
    corrupted = replace(enclosure, steps=(enclosure.steps[0], bad_second))

    assert enclosure.proof_certified
    assert enclosure.chain_certified
    assert len(second.start_state_interval_union) == 2
    assert bad_second.proof_certified
    assert not corrupted.chain_certified
    assert not corrupted.proof_certified


def test_ordinary_set_propagation_contains_reference_and_tightens_scalar_box():
    masses, positions, velocities = _ordinary_data()
    t_final = 0.05
    low_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=12,
        tail_certificate_mode="cauchy",
    )
    high_order = continue_hybrid(
        positions,
        velocities,
        masses,
        t_final,
        binary_distance_threshold=0.05,
        max_time_step=0.01,
        ordinary_order=18,
    )
    reference = integrate_reference(positions, velocities, masses, t_final)

    scalar_enclosure = low_order.propagated_interval_enclosure()
    set_enclosure = low_order.ordinary_set_propagated_interval_enclosure()

    assert set_enclosure.final_state_contains(low_order.final_state)
    assert set_enclosure.final_state_contains(high_order.final_state)
    assert set_enclosure.final_state_contains(reference)
    assert _max_interval_width(set_enclosure.final_state_interval) < _max_interval_width(
        scalar_enclosure.final_state_interval
    )


def test_ordinary_set_propagation_supports_certified_event_step_prefix():
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
        0.01,
        order=20,
        binary_enter_distance=0.1199,
    )
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        event_time + 1e-8,
        binary_distance_threshold=0.1199,
        binary_exit_distance=0.13,
        max_time_step=0.01,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
        tail_certificate_mode="cauchy",
        max_steps=5,
    )

    enclosure = propagate_ordinary_set_enclosures(masses, (continued.steps[0],), retained_order=20)
    event_step = enclosure.steps[0]

    assert event_step.event == "enter_binary"
    assert event_step.event_time_interval == continued.steps[0].event_time_interval
    assert event_step.tail_coefficient_source == "interval_cauchy_majorant"
    assert event_step.end_state_contains(continued.states[1])
    point_series = construct_taylor_solution(positions, velocities, masses, order=20)
    for time in event_step.event_time_interval:
        assert event_step.truncated_end_state_contains(point_series.state_at(time))


def test_ordinary_event_set_propagation_uses_member_event_time_intervals():
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
        0.01,
        order=20,
        binary_enter_distance=0.1199,
    )
    state = pack_planar_state(positions, velocities)
    start_state_interval_union = (
        tuple((float(value - 1e-14), float(value + 1e-14)) for value in state),
        tuple((float(value - 2e-14), float(value + 2e-14)) for value in state),
    )
    event_union_certificate = certify_ordinary_binary_entry_event_union(
        positions,
        velocities,
        masses,
        trial_physical_step=0.01,
        event_root=event_time,
        event_pair=event_pair,
        binary_enter_distance=0.1199,
        order=20,
        start_state_interval_union=start_state_interval_union,
    )
    member_intervals = tuple(
        enclosure.interval for enclosure in event_union_certificate.member_root_enclosures
    )
    start_hull = _hull_interval_states(start_state_interval_union)
    step = SimpleNamespace(
        chart="ordinary",
        event="enter_binary",
        pair=event_pair,
        start_time=0.0,
        physical_step=event_time,
        parameter_step=event_time,
        start_state_interval=start_hull,
        start_state_interval_union=start_state_interval_union,
        event_time_interval=event_union_certificate.event_time_interval,
        event_union_certificate=event_union_certificate,
        truncation_certificate=None,
    )

    enclosure = propagate_ordinary_set_enclosures(masses, (step,), retained_order=20)
    event_step = enclosure.steps[0]

    assert event_union_certificate.earliest_interval_is_certified
    assert event_step.event_time_interval == event_union_certificate.event_time_interval
    assert event_step.event_time_interval_union == member_intervals
    assert len(event_step.truncated_end_state_interval_union) == len(member_intervals)
    assert event_step.tail_coefficient_source == "interval_cauchy_majorant"
    assert event_step.tail_ratio_bound < 1.0
    point_series = construct_taylor_solution(positions, velocities, masses, order=20)
    for member_interval in event_step.event_time_interval_union:
        for time in member_interval:
            assert event_step.truncated_end_state_contains(point_series.state_at(time))


def test_ordinary_set_propagation_handles_direct_binary_handoff_after_event():
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
        0.01,
        order=20,
        binary_enter_distance=0.1199,
    )
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        event_time + 1e-8,
        binary_distance_threshold=0.1199,
        binary_exit_distance=0.13,
        max_time_step=0.01,
        safety=1.0,
        ordinary_order=20,
        binary_order=20,
        tail_certificate_mode="cauchy",
        max_steps=5,
    )

    assert continued.steps[0].event == "enter_binary"
    assert continued.steps[1].chart == "binary"
    enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=20)
    event_step, binary_step = enclosure.steps

    assert len(enclosure.steps) == len(continued.steps) == 2
    assert event_step.chart == "ordinary"
    assert event_step.event == "enter_binary"
    assert event_step.end_state_interval == binary_step.start_state_interval
    assert binary_step.chart == "binary"
    assert binary_step.pair == (0, 1)
    assert binary_step.parameter_step == continued.steps[1].parameter_step
    assert binary_step.tail_coefficient_source == "regularized_interval_cauchy_majorant"
    assert binary_step.tail_ratio_bound < 1.0
    assert binary_step.start_state_contains(continued.states[1])
    assert binary_step.truncated_end_state_contains(continued.states[2])
    assert binary_step.end_state_contains(continued.states[2])
    assert enclosure.final_state_contains(continued.final_state)


def test_set_propagation_handles_certified_binary_exit_event():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [1e-3, 0.0], [1.0, 0.4]])
    velocities = np.array([[-40.0, 0.0], [40.0, 0.0], [-0.01, 0.0]])
    (
        _end_positions,
        _end_velocities,
        _s_step,
        physical_step,
        _certificate,
    ) = regularized_binary_taylor_step(
        positions,
        velocities,
        masses,
        (0, 1),
        1.0,
        order=30,
        max_s_step=0.05,
        binary_exit_distance=0.00101,
    )
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        physical_step,
        binary_distance_threshold=0.001005,
        binary_exit_distance=0.00101,
        max_binary_s_step=0.05,
        binary_order=30,
        tail_certificate_mode="cauchy",
        max_steps=3,
    )

    assert len(continued.steps) == 1
    assert continued.steps[0].event == "exit_binary"
    assert continued.steps[0].event_certificate.root_enclosure is not None
    enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=30)
    binary_step = enclosure.steps[0]

    assert binary_step.chart == "binary"
    assert binary_step.event == "exit_binary"
    assert binary_step.pair == (0, 1)
    assert binary_step.event_parameter_interval == continued.steps[0].event_certificate.root_enclosure.interval
    assert binary_step.event_parameter_interval[0] <= binary_step.parameter_step <= binary_step.event_parameter_interval[1]
    assert binary_step.tail_coefficient_source == "regularized_interval_cauchy_majorant"
    assert binary_step.tail_ratio_bound < 1.0
    assert binary_step.start_state_contains(continued.states[0])
    assert binary_step.truncated_end_state_contains(continued.states[1])
    assert binary_step.end_state_contains(continued.states[1])
    assert enclosure.final_state_contains(continued.final_state)


def test_set_propagation_rebuilds_multi_step_binary_charts_from_incoming_box():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [1e-3, 0.0], [1.0, 0.4]])
    velocities = np.array([[-40.0, 0.0], [40.0, 0.0], [-0.01, 0.0]])
    continued = continue_hybrid(
        positions,
        velocities,
        masses,
        5e-8,
        binary_distance_threshold=0.002,
        binary_exit_distance=0.01,
        max_binary_s_step=2e-5,
        binary_order=20,
        tail_certificate_mode="cauchy",
        max_steps=10,
    )

    assert len(continued.steps) == 3
    assert all(step.chart == "binary" and step.event is None for step in continued.steps)
    enclosure = continued.ordinary_set_propagated_interval_enclosure(retained_order=20)

    assert len(enclosure.steps) == len(continued.steps) == 3
    assert enclosure.proof_certified
    assert enclosure.chain_certified
    assert enclosure.final_state_contains(continued.final_state)
    for index, step in enumerate(enclosure.steps):
        assert step.chart == "binary"
        assert step.tail_coefficient_source == "regularized_interval_cauchy_majorant"
        assert step.tail_ratio_bound < 1.0
        assert step.start_state_contains(continued.states[index])
        assert step.truncated_end_state_contains(continued.states[index + 1])
        assert step.end_state_contains(continued.states[index + 1])
        if index > 0:
            assert step.start_state_interval == enclosure.steps[index - 1].end_state_interval
            assert step.start_state_interval != continued.steps[index].start_state_interval
    bad_time_step = replace(
        enclosure.steps[1],
        start_time=enclosure.steps[1].start_time + 0.25,
    )
    corrupted = replace(
        enclosure,
        steps=(enclosure.steps[0], bad_time_step, *enclosure.steps[2:]),
    )
    assert bad_time_step.proof_certified
    assert not corrupted.chain_certified
    assert not corrupted.proof_certified


def test_set_propagation_uses_certified_binary_atlas_for_branch_cut_overlap():
    masses = np.array([1.0, 1.0, 1.0])
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.2, -0.8),
        (-0.2, 0.2),
        (3.0, 3.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 6
    s_step = 0.001
    step = SimpleNamespace(
        chart="binary",
        pair=(0, 1),
        event=None,
        start_state_interval=state_interval,
        start_time=0.0,
        parameter_step=s_step,
        physical_step=0.0,
        truncation_certificate=None,
    )

    enclosure = propagate_ordinary_set_enclosures(masses, (step,), retained_order=8)
    binary_step = enclosure.steps[0]

    assert binary_step.chart == "binary"
    assert binary_step.binary_atlas_chart_count == 2
    assert binary_step.tail_coefficient_source == "regularized_interval_cauchy_majorant_atlas"
    assert binary_step.tail_ratio_bound < 1.0

    velocities = np.zeros((3, 2))
    for positions in (
        np.array([[0.0, 0.0], [-1.0, 0.1], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [-1.0, -0.1], [3.0, 0.0]]),
    ):
        initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
        end_state = construct_regularized_binary_taylor_solution(initial, order=8).state_at(s_step)
        endpoint = np.concatenate(
            [component.reshape(-1) for component in regularized_binary_collision_chart_to_planar(end_state)]
        )
        assert binary_step.truncated_end_state_contains(endpoint)
        assert binary_step.end_state_contains(endpoint)


def test_binary_set_propagation_accepts_lifted_exact_collision_interval_start():
    initial = _exact_collision_state()
    interval_initial = _exact_collision_interval_state()
    s_step = 0.002
    start_state_interval = _exact_collision_position_summary_interval(initial)
    step = SimpleNamespace(
        chart="binary",
        pair=initial.pair,
        event=None,
        start_state_interval=start_state_interval,
        start_regularized_interval_state=interval_initial,
        start_time=0.0,
        parameter_step=s_step,
        physical_step=0.0,
        truncation_certificate=None,
    )

    enclosure = propagate_ordinary_set_enclosures(initial.masses, (step,), retained_order=8)
    binary_step = enclosure.steps[0]
    end_state = construct_regularized_binary_taylor_solution(initial, order=8).state_at(s_step)
    endpoint = np.concatenate(
        [component.reshape(-1) for component in regularized_binary_collision_chart_to_planar(end_state)]
    )

    assert interval_initial.branch_certificate.branch == "exact_binary_collision"
    assert binary_step.chart == "binary"
    assert binary_step.start_state_interval == start_state_interval
    assert binary_step.union_member_count == 1
    assert binary_step.binary_atlas_chart_count == 0
    assert binary_step.tail_coefficient_source == "regularized_interval_cauchy_majorant"
    assert binary_step.tail_ratio_bound < 1.0
    assert binary_step.truncated_end_state_contains(endpoint)
    assert binary_step.end_state_contains(endpoint)


def test_binary_set_propagation_preserves_start_union_without_hulling_through_collision():
    masses = np.ones(3)
    velocities = np.zeros((3, 2))
    upper_positions = np.array([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0]])
    lower_positions = np.array([[0.0, 0.0], [0.0, -1.0], [4.0, 0.0]])
    upper_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(upper_positions, velocities)
    )
    lower_state = tuple(
        (float(value), float(value))
        for value in pack_planar_state(lower_positions, velocities)
    )
    start_hull = _hull_interval_states((upper_state, lower_state))
    s_step = 1e-3
    step = SimpleNamespace(
        chart="binary",
        pair=(0, 1),
        event=None,
        start_state_interval=start_hull,
        start_state_interval_union=(upper_state, lower_state),
        start_time=0.0,
        parameter_step=s_step,
        physical_step=0.0,
        truncation_certificate=None,
    )

    enclosure = propagate_ordinary_set_enclosures(masses, (step,), retained_order=8)
    binary_step = enclosure.steps[0]

    assert state_interval_subset(start_hull, binary_step.start_state_interval)
    assert binary_step.start_state_interval_union == (upper_state, lower_state)
    assert binary_step.union_member_count == 2
    assert binary_step.binary_atlas_chart_count == 0
    assert binary_step.tail_coefficient_source == "regularized_interval_cauchy_majorant_union"
    assert binary_step.tail_ratio_bound < 1.0
    for positions in (upper_positions, lower_positions):
        initial = planar_to_regularized_binary_collision_chart(positions, velocities, masses, pair=(0, 1))
        end_state = construct_regularized_binary_taylor_solution(initial, order=8).state_at(s_step)
        endpoint = np.concatenate(
            [component.reshape(-1) for component in regularized_binary_collision_chart_to_planar(end_state)]
        )
        assert binary_step.truncated_end_state_contains(endpoint)
        assert binary_step.end_state_contains(endpoint)

    hull_only_step = SimpleNamespace(
        chart="binary",
        pair=(0, 1),
        event=None,
        start_state_interval=start_hull,
        start_time=0.0,
        parameter_step=s_step,
        physical_step=0.0,
        truncation_certificate=None,
    )
    with pytest.raises(ValueError, match="contains binary collision"):
        propagate_ordinary_set_enclosures(masses, (hull_only_step,), retained_order=8)


def test_sundman_target_set_propagation_rebuilds_multi_step_target_charts():
    masses, positions, velocities = _ordinary_data()
    retained_order = 10
    first_limit = 0.5 * sundman_interval_cauchy_majorant_tail_certificate(
        positions,
        velocities,
        masses,
        retained_order=retained_order,
        step_size=0.0,
    ).s_radius

    for target_multiplier, expected_full_steps in ((1.5, 1), (2.5, 2), (3.0, 3)):
        for direction in (1.0, -1.0):
            point_s_target = direction * target_multiplier * first_limit
            point = continue_sundman_to_s(
                positions,
                velocities,
                masses,
                point_s_target,
                order=18,
                max_s_step=0.5 * first_limit,
            )
            target_solution = continue_interval_sundman_to_time(
                positions,
                velocities,
                masses,
                point.times[-1],
                order=retained_order,
                max_s_step=1.0,
                tail_certificate_mode="cauchy",
            )

            enclosure = propagate_sundman_target_set_enclosure(target_solution)
            method_enclosure = target_solution.set_propagated_interval_enclosure()

            assert target_solution.proof_certified
            assert enclosure.proof_certified
            assert enclosure.chain_certified
            assert method_enclosure.proof_certified
            assert method_enclosure.chain_certified
            assert len(target_solution.steps) == expected_full_steps
            assert len(enclosure.steps) == expected_full_steps + 1
            assert enclosure.certified_step_count == expected_full_steps + 1
            assert method_enclosure.final_state_interval == enclosure.final_state_interval
            assert all(step.kind == "full" for step in enclosure.steps[:-1])
            assert enclosure.steps[-1].kind == "target"
            for index in range(1, len(enclosure.steps)):
                assert enclosure.steps[index].start_state_interval == enclosure.steps[index - 1].end_state_interval
            assert all(step.equation_residual_certified for step in enclosure.steps)
            assert all(step.time_monotone_certified for step in enclosure.steps)
            assert all(
                step.tail_coefficient_source == "sundman_interval_cauchy_majorant"
                for step in enclosure.steps
            )
            assert all(step.tail_ratio_bound <= 0.5 + 1e-12 for step in enclosure.steps[:-1])
            assert enclosure.steps[-1].tail_ratio_bound < 1.0
            assert enclosure.steps[-1].target_certified
            assert enclosure.steps[-1].global_s_interval[0] <= point_s_target <= enclosure.steps[-1].global_s_interval[1]
            bad_time_target = replace(
                enclosure.steps[-1],
                start_time_interval=(
                    enclosure.steps[-1].start_time_interval[0] + 0.25,
                    enclosure.steps[-1].start_time_interval[1] + 0.25,
                ),
            )
            bad_state_target = replace(
                enclosure.steps[-1],
                start_state_interval=enclosure.steps[0].start_state_interval,
            )
            for bad_target in (bad_time_target, bad_state_target):
                corrupted = replace(
                    enclosure,
                    steps=(*enclosure.steps[:-1], bad_target),
                )
                assert bad_target.proof_certified
                assert not corrupted.chain_certified
                assert not corrupted.proof_certified
            for full_step in enclosure.steps[:-1]:
                full_step_end_s = (
                    full_step.global_s_interval[1] if direction > 0.0 else full_step.global_s_interval[0]
                )
                point_at_step_end = continue_sundman_to_s(
                    positions,
                    velocities,
                    masses,
                    full_step_end_s,
                    order=18,
                    max_s_step=0.5 * first_limit,
                )
                assert full_step.end_state_contains(point_at_step_end.final_state)
            assert enclosure.steps[-1].end_state_contains(point.final_state)
            assert enclosure.final_state_contains(point.final_state)
