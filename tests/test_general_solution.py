from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import three_body_symmetry.finite_time_regime as finite_time_regime_module
import three_body_symmetry.general_solution as general_solution_module
import three_body_symmetry.validated_atlas as validated_atlas_module
from three_body_symmetry.compact_time import compact_parameter_from_physical_time
from three_body_symmetry.finite_time_regime import classify_finite_time_regime
from three_body_symmetry.general_solution import (
    FiniteTimeChartSelectorError,
    evaluate_compact_time_reduced_compactified_sundman_solution,
    evaluate_compactified_reduced_sundman_solution,
    evaluate_reduced_compactified_sundman_solution,
    evaluate_reduced_sundman_solution,
    evaluate_unrestricted_solution,
)
from three_body_symmetry.intervals import FloatInterval, interval_array_contains_point
from three_body_symmetry.ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    spatial_to_ks_binary_chart,
)
from three_body_symmetry.ks_binary_series import (
    certify_spatial_ks_competing_binary_entry_event,
    certify_spatial_ks_binary_rho_exit_event,
    construct_interval_spatial_ks_binary_taylor_solution,
    construct_spatial_ks_binary_taylor_solution,
    interval_spatial_ks_binary_chart_state_from_point,
    project_spatial_ks_binary_interval_chart_state_to_physical,
)
from three_body_symmetry.reduction import reduce_to_center_of_mass_frame
from three_body_symmetry.series import construct_taylor_solution, integrate_reference
from three_body_symmetry.sundman import construct_sundman_taylor_solution
from three_body_symmetry.validated_atlas import (
    CollisionPolicyWitness,
    FiniteTimeChartSelectorAttempt,
    FiniteTimeChartSelectorTrace,
    GlobalInvariantLedger,
    NewtonResidualLedger,
    ProofLedger,
    ProofLedgerEntry,
    TailBudgetLedger,
    ValidatedAtlasSolution,
    ValidatedChart,
    certify_simultaneous_close_pair_partition,
    validated_atlas_from_hybrid_solution,
)


class _HybridEnclosureOverride:
    def __init__(
        self,
        base,
        *,
        lohner_enclosure=None,
        set_enclosure=None,
    ):
        self._base = base
        self._lohner_enclosure = lohner_enclosure
        self._set_enclosure = set_enclosure

    def __getattr__(self, name):
        return getattr(self._base, name)

    def lohner_ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order=None,
    ):
        if retained_order is None and self._lohner_enclosure is not None:
            return self._lohner_enclosure
        return self._base.lohner_ordinary_set_propagated_interval_enclosure(
            retained_order=retained_order,
        )

    def ordinary_set_propagated_interval_enclosure(
        self,
        *,
        retained_order=None,
        ordinary_substeps=1,
    ):
        if (
            retained_order is None
            and int(ordinary_substeps) == 1
            and self._set_enclosure is not None
        ):
            return self._set_enclosure
        return self._base.ordinary_set_propagated_interval_enclosure(
            retained_order=retained_order,
            ordinary_substeps=ordinary_substeps,
        )


class _TargetStateIntervalOverride:
    def __init__(self, base, *, target_state_interval, final_state):
        self._base = base
        self.target_state_interval = target_state_interval
        self.final_state = np.asarray(final_state, dtype=float).reshape(-1)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def target_state_contains(self, state):
        return interval_array_contains_point(
            self.target_state_interval,
            np.asarray(state, dtype=float).reshape(-1),
        )


def _general_initial_data():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array(
        [
            [0.8, -0.2, 0.1],
            [-0.4, 0.6, -0.3],
            [0.1, -0.5, 0.7],
        ]
    )
    velocities = np.array(
        [
            [0.05, 0.11, -0.02],
            [-0.07, 0.03, 0.04],
            [0.02, -0.08, 0.01],
        ]
    )
    return masses, positions, velocities


def _target_time_from_reduced_s(reduced, s_value):
    chart = construct_sundman_taylor_solution(
        reduced.positions,
        reduced.velocities,
        reduced.masses,
        order=18,
    )
    return chart.physical_time_at_s(s_value)


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


def test_reduced_sundman_evaluator_contains_reference_path_in_inertial_frame():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)

    evaluation = evaluate_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        target_time,
        initial_radius=1e-15,
        order=10,
        max_s_step=0.02,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.reduction_certified
    assert evaluation.target_time_certified
    assert evaluation.dynamics_certified
    assert evaluation.tail_certified
    assert evaluation.proof_certified
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.proof_certified
    assert evaluation.target_state_contains(reference)
    assert evaluation.local_tail_bound > 0.0


def test_reduced_sundman_evaluator_handles_negative_target_time():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, -0.006)

    evaluation = evaluate_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        target_time,
        initial_radius=1e-15,
        order=10,
        max_s_step=0.02,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.proof_certified
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.target_s_interval.upper < 0.0
    assert evaluation.target_state_contains(reference)


def test_reduced_sundman_evaluator_marks_zero_angular_momentum_branch_undecided():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    velocities = np.zeros_like(positions)
    target_time = 1e-3

    evaluation = evaluate_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        target_time,
        initial_radius=0.0,
        order=10,
        max_s_step=0.02,
        target_bisections=36,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.proof_certified
    assert not evaluation.triple_collision_excluded
    assert evaluation.triple_collision_status == "undecided"
    assert evaluation.triple_collision_exclusion_reason == "centered angular momentum interval contains zero"
    assert evaluation.triple_collision_undecided
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.triple_collision_undecided
    assert evaluation.target_state_contains(reference)


def test_compactified_reduced_sundman_evaluator_contains_reference_path():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)
    compact_parameter = compact_parameter_from_physical_time(target_time, rate=2.0)

    evaluation = evaluate_compactified_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        compact_parameter,
        time_rate=2.0,
        initial_radius=1e-15,
        order=10,
        max_s_step=0.02,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, evaluation.target_time)

    assert evaluation.compactification_certificate.certified
    assert evaluation.time_taylor_certificate.certified
    assert (
        evaluation.time_taylor_certificate.time_enclosure.lower
        <= evaluation.target_time
        <= evaluation.time_taylor_certificate.time_enclosure.upper
    )
    assert evaluation.target_time == pytest.approx(target_time)
    assert evaluation.proof_certified
    assert evaluation.evaluation.reduced_target is not None
    assert evaluation.evaluation.reduced_target.proof_certified
    assert evaluation.target_state_contains(reference)


def test_reduced_compactified_sundman_evaluator_contains_reference_path_forward_and_backward():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    for s_value in (0.006, -0.006):
        target_time = _target_time_from_reduced_s(reduced, s_value)
        evaluation = evaluate_reduced_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            target_time,
            initial_radius=1e-15,
            order=10,
            sundman_rate=1.15,
            max_compact_step=0.025,
            radius_fraction=0.2,
            guard_order=6,
            target_bisections=42,
        )
        reference = integrate_reference(positions, velocities, masses, target_time)

        assert evaluation.reduction_certified
        assert evaluation.target_time_certified
        assert evaluation.dynamics_certified
        assert evaluation.tail_certified
        assert evaluation.proof_certified
        assert evaluation.reduced_target is not None
        assert evaluation.reduced_target.proof_certified
        assert evaluation.triple_collision_excluded
        assert evaluation.triple_collision_status == "excluded"
        assert evaluation.triple_collision_exclusion_reason == "centered angular momentum is bounded away from zero"
        assert evaluation.reduced_target.angular_momentum_certified_step_count == len(evaluation.reduced_target.steps) + 1
        assert evaluation.reduced_target.energy_certified_step_count == len(evaluation.reduced_target.steps) + 1
        assert evaluation.reduced_target.linear_momentum_certified_step_count == len(evaluation.reduced_target.steps) + 1
        assert evaluation.reduced_target.center_of_mass_certified_step_count == len(evaluation.reduced_target.steps) + 1
        assert evaluation.reduced_target.target_certificate.factor_interval.lower > 0.0
        assert evaluation.target_compact_parameter_interval is not None
        assert evaluation.local_tail_bound > 0.0
        assert evaluation.target_state_contains(reference)


def test_unrestricted_solution_entrypoint_contains_spatial_reference_for_finite_times():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_times = [
        _target_time_from_reduced_s(reduced, 0.006),
        _target_time_from_reduced_s(reduced, -0.006),
        0.0,
    ]

    for target_time in target_times:
        evaluation = evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            target_time,
            initial_radius=1e-15,
            order=10,
            sundman_rate=1.15,
            max_compact_step=0.025,
            radius_fraction=0.2,
            guard_order=6,
            target_bisections=42,
        )
        reference = (
            np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            if target_time == 0.0
            else integrate_reference(positions, velocities, masses, target_time)
        )

        assert evaluation.proof_certified
        assert evaluation.reduction_certified
        assert evaluation.target_time_certified
        assert evaluation.target_state_contains(reference)
        if target_time == 0.0:
            assert evaluation.reduced_target is None
        else:
            assert evaluation.reduced_target is not None
            assert evaluation.reduced_target.proof_certified


def test_unrestricted_solution_validated_atlas_method_derives_single_proof_pipeline():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "ReducedCompactifiedSundmanEvaluation"
    assert solution.proof_certified
    assert solution.target_state_contains(reference)
    assert solution.proof_ledger.missing_required_obligations == ()
    assert solution.missing_certification_obligations == ()
    assert solution.physical_time_chain_progress_certified
    assert solution.tail_budget.certified
    assert solution.tail_budget.finite
    assert solution.tail_budget.local_tail_bound == pytest.approx(solution.evaluation.local_tail_bound)
    assert solution.residual_budget.certified
    assert solution.invariants.certified
    assert solution.collision_policy.certified
    assert solution.collision_policy.well_formed
    assert solution.collision_policy.missing_obligations == ()
    assert solution.collision_policy.total_collision_policy == "stop_or_require_explicit_selector"
    assert solution.collision_policy.triple_collision_status == "excluded"
    assert solution.selector_trace is not None
    assert solution.selector_trace.certified
    assert solution.selector_trace.selected_route_id == "compactified_sundman"
    assert [attempt.route_id for attempt in solution.selector_trace.attempts] == [
        "auto_spatial_ks",
        "planar_hybrid",
        "compactified_sundman",
    ]
    assert solution.evaluation.reduced_target is not None
    assert solution.chart_count == len(solution.evaluation.reduced_target.steps) + 1
    assert solution.transition_count == solution.chart_count - 1
    assert solution.residual_budget.certified_chart_count == solution.chart_count
    assert solution.invariants.certified_chart_count == solution.chart_count
    chart_types = {chart.chart_type for chart in solution.charts}
    assert "compactified_sundman_target" in chart_types
    assert chart_types <= {"compactified_sundman", "compactified_sundman_target"}
    assert all(
        entry.source == "evaluate_unrestricted_solution"
        for entry in solution.proof_ledger.entries
    )
    assert solution.proof_ledger.well_formed

    nameless_ledger_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=(
                replace(solution.proof_ledger.entries[0], name=""),
                *solution.proof_ledger.entries[1:],
            ),
        ),
    )
    sourceless_ledger_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=(
                replace(solution.proof_ledger.entries[0], source=""),
                *solution.proof_ledger.entries[1:],
            ),
        ),
    )
    duplicate_ledger_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=(
                solution.proof_ledger.entries[0],
                *solution.proof_ledger.entries,
            ),
        ),
    )
    assert not nameless_ledger_solution.proof_ledger.well_formed
    assert not nameless_ledger_solution.proof_certified
    assert not sourceless_ledger_solution.proof_ledger.well_formed
    assert not sourceless_ledger_solution.proof_certified
    assert not duplicate_ledger_solution.proof_ledger.well_formed
    assert not duplicate_ledger_solution.proof_certified

    selector_only_ledger_solution = replace(
        solution,
        proof_ledger=ProofLedger(
            entries=(
                ProofLedgerEntry(
                    "finite_time_chart_selector",
                    True,
                    "evaluate_unrestricted_solution",
                ),
            )
        ),
    )
    assert selector_only_ledger_solution.proof_ledger.certified
    assert not selector_only_ledger_solution.proof_ledger_covers_solution
    assert any(
        obligation.startswith("proof_ledger_coverage:")
        for obligation in selector_only_ledger_solution.missing_certification_obligations
    )
    assert not selector_only_ledger_solution.proof_certified
    missing_projection_ledger_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=tuple(
                entry
                for entry in solution.proof_ledger.entries
                if entry.name != "center_of_mass_reduction"
            ),
        ),
    )
    missing_target_evidence_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=tuple(
                entry
                for entry in solution.proof_ledger.entries
                if entry.name != "target_time"
            ),
        ),
    )
    assert missing_projection_ledger_solution.proof_ledger.certified
    assert not missing_projection_ledger_solution.proof_ledger_covers_solution
    assert any(
        obligation.startswith("proof_ledger_coverage:any_of(projection_ledger|center_of_mass_reduction")
        for obligation in missing_projection_ledger_solution.missing_certification_obligations
    )
    assert not missing_projection_ledger_solution.proof_certified
    assert missing_target_evidence_solution.proof_ledger.certified
    assert not missing_target_evidence_solution.proof_ledger_covers_solution
    assert any(
        obligation.startswith("proof_ledger_coverage:any_of(target_time|target_containment")
        for obligation in missing_target_evidence_solution.missing_certification_obligations
    )
    assert not missing_target_evidence_solution.proof_certified

    empty_target_evaluation = replace(
        solution.evaluation,
        target_position_interval=np.asarray([], dtype=object),
        target_velocity_interval=np.asarray([], dtype=object),
    )
    empty_target_solution = replace(solution, evaluation=empty_target_evaluation)
    assert not empty_target_solution.target_state_interval_certified
    assert not empty_target_solution.proof_certified

    stale_initial_interval = np.asarray(
        [
            FloatInterval.point(1.0e9)
            for _interval in solution.initial_state_interval
        ],
        dtype=object,
    )
    stale_initial_solution = replace(
        solution,
        initial_state_interval=stale_initial_interval,
    )
    assert not stale_initial_solution.initial_state_interval_certified
    assert not stale_initial_solution.proof_certified

    missing_parameter_chart = replace(solution.charts[0], parameter_interval=None)
    missing_parameter_solution = replace(
        solution,
        charts=(missing_parameter_chart, *solution.charts[1:]),
    )
    negative_tail_chart = replace(solution.charts[0], tail_bound=-1.0)
    negative_tail_solution = replace(
        solution,
        charts=(negative_tail_chart, *solution.charts[1:]),
    )
    assert not missing_parameter_chart.certified
    assert not missing_parameter_solution.proof_certified
    assert not negative_tail_chart.certified
    assert not negative_tail_solution.proof_certified

    stale_residual_budget = replace(
        solution.residual_budget,
        certified=True,
        expected_chart_count=solution.chart_count + 1,
    )
    stale_residual_solution = replace(
        solution,
        residual_budget=stale_residual_budget,
    )
    internally_certified_stale_residual_budget = replace(
        solution.residual_budget,
        certified=True,
        certified_chart_count=solution.chart_count,
        expected_chart_count=solution.chart_count,
    )
    extra_chart = replace(solution.charts[-1], chart_id="extra_stale_chart")
    internally_certified_stale_residual_solution = replace(
        solution,
        charts=(*solution.charts, extra_chart),
        residual_budget=internally_certified_stale_residual_budget,
    )
    internally_certified_stale_invariants = replace(
        solution.invariants,
        certified_chart_count=solution.chart_count,
        expected_chart_count=solution.chart_count,
    )
    internally_certified_stale_invariant_solution = replace(
        solution,
        charts=(*solution.charts, extra_chart),
        invariants=internally_certified_stale_invariants,
    )
    negative_tail_budget = replace(
        solution.tail_budget,
        certified=True,
        local_tail_bound=-1.0,
    )
    negative_tail_budget_solution = replace(
        solution,
        tail_budget=negative_tail_budget,
    )
    assert solution.residual_budget.coverage_certified
    assert not stale_residual_budget.coverage_certified
    assert not stale_residual_solution.proof_certified
    assert internally_certified_stale_residual_budget.coverage_certified
    assert not internally_certified_stale_residual_solution.residual_coverage_matches_charts
    assert not internally_certified_stale_residual_solution.proof_certified
    assert internally_certified_stale_invariants.certified
    assert not internally_certified_stale_invariant_solution.invariant_coverage_matches_charts
    assert not internally_certified_stale_invariant_solution.proof_certified
    assert solution.tail_budget.admissible
    assert not negative_tail_budget.admissible
    assert not negative_tail_budget_solution.proof_certified

    first_physical_interval = solution.charts[0].physical_time_interval
    stale_start = FloatInterval(
        0.5 * (first_physical_interval.lower + first_physical_interval.upper),
        first_physical_interval.upper,
    )
    stale_start_chart = replace(
        solution.charts[0],
        physical_time_interval=stale_start,
    )
    stale_start_solution = replace(
        solution,
        charts=(stale_start_chart, *solution.charts[1:]),
    )
    assert not stale_start_solution.physical_time_chain_progress_certified
    assert "physical_time_chain_progress" in (
        stale_start_solution.missing_certification_obligations
    )
    assert not stale_start_solution.proof_certified

    nameless_collision_policy_solution = replace(
        solution,
        collision_policy=replace(solution.collision_policy, policy_id=""),
    )
    missing_collision_reason_solution = replace(
        solution,
        collision_policy=replace(solution.collision_policy, triple_collision_reason=None),
    )
    assert not nameless_collision_policy_solution.collision_policy.well_formed
    assert "collision_policy:policy_id" in (
        nameless_collision_policy_solution.missing_certification_obligations
    )
    assert not nameless_collision_policy_solution.proof_certified
    assert not missing_collision_reason_solution.collision_policy.well_formed
    assert "collision_policy:triple_collision_reason" in (
        missing_collision_reason_solution.missing_certification_obligations
    )
    assert not missing_collision_reason_solution.proof_certified


def test_finite_time_regime_classifier_constructs_validated_atlas_route():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        target_time,
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert classification.certified
    assert classification.input_domain_certified
    assert classification.target_time_certified
    assert classification.atlas_certified
    assert classification.selector_certified
    assert classification.selected_route_id == "compactified_sundman"
    assert classification.validated_atlas is not None
    assert classification.validated_atlas.proof_certified
    assert classification.validated_atlas.target_state_contains(reference)
    assert "compactified_sundman_target" in classification.chart_types
    assert classification.missing_obligations == ()


def test_finite_time_regime_classifier_rejects_foreign_validated_atlas_input_domain(
    monkeypatch,
):
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)
    source_atlas = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )

    foreign_positions = positions.copy()
    foreign_positions[0, 0] += 0.25
    monkeypatch.setattr(
        finite_time_regime_module,
        "evaluate_unrestricted_solution",
        lambda *_args, **_kwargs: source_atlas,
    )
    classification = finite_time_regime_module.classify_finite_time_regime(
        masses,
        foreign_positions,
        velocities,
        target_time,
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )

    assert source_atlas.proof_certified
    assert classification.atlas_certified
    assert not classification.certified
    assert "finite_time_atlas_input_domain_matches_classifier" in (
        classification.missing_obligations
    )


def test_finite_time_regime_classifier_reports_required_regularized_route_failure():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ]
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        1.0,
        order=8,
        binary_distance_threshold=0.08,
        max_compact_step=0.01,
        target_bisections=20,
    )

    assert not classification.certified
    assert classification.input_domain_certified
    assert classification.validated_atlas is None
    assert classification.selected_route_id is None
    assert "finite_time_validated_atlas" in classification.missing_obligations
    assert "finite_time_chart_selector" in classification.missing_obligations
    assert "spatial_close_binary_requires_regularized_chart_or_split" in (
        classification.missing_obligations
    )
    assert "spatial_close_pair_not_separated_from_third_body" in (
        classification.failure_obligations
    )
    assert classification.failure_reason is not None
    assert "spatial_close_binary_guard" in classification.failure_reason


def test_finite_time_regime_classifier_reports_competing_ks_binary_obligation():
    initial = replace(
        _exact_spatial_ks_collision_state(),
        third_offset=np.array([1.0e-3, 2.0e-4, -1.0e-4]),
        third_offset_velocity=np.zeros(3),
    )
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(solution.state_at(0.005))

    classification = classify_finite_time_regime(
        initial.masses,
        positions,
        velocities,
        1.0e-8,
        order=18,
        guard_order=6,
        binary_distance_threshold=1.0e-2,
    )

    assert not classification.certified
    assert classification.validated_atlas is None
    assert "ks_competing_close_binary_requires_next_regularized_chart_or_split" in (
        classification.failure_obligations
    )
    assert "ks_competing_close_binary_requires_next_regularized_chart_or_split" in (
        classification.missing_obligations
    )
    assert classification.failure_reason is not None
    assert "auto_spatial_ks" in classification.failure_reason


def test_finite_time_regime_classifier_surfaces_close_pair_branch_partition():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 3))

    with pytest.raises(FiniteTimeChartSelectorError) as exc_info:
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            0.25,
            method="validated_atlas",
            initial_radius=0.04,
            order=8,
            binary_distance_threshold=0.05,
            max_compact_step=0.01,
            target_bisections=20,
        )
    assert exc_info.value.blocking_attempts[0].branch_partition is not None
    assert exc_info.value.blocking_attempts[0].branch_partition.certified
    assert "finite_time_branch_union_consumption" in (
        exc_info.value.missing_obligations
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        0.25,
        initial_radius=0.04,
        order=8,
        binary_distance_threshold=0.05,
        max_compact_step=0.01,
        target_bisections=20,
    )

    assert not classification.certified
    assert classification.input_domain_certified
    assert classification.branch_partition is not None
    assert classification.branch_partition_certified
    assert classification.branch_partition.certified_branch_count == len(
        classification.branch_partition.branches
    )
    assert all(
        len(branch.possible_close_pairs) <= 1
        for branch in classification.branch_partition.branches
    )
    assert any(
        branch.selected_pair == (0, 1)
        for branch in classification.branch_partition.branches
    )
    assert any(
        branch.selected_pair == (1, 2)
        for branch in classification.branch_partition.branches
    )
    assert "simultaneous_close_pair_partition" not in (
        classification.missing_obligations
    )
    assert "finite_time_branch_union_consumption" in (
        classification.missing_obligations
    )
    assert "finite_time_branch_union_consumption" in (
        classification.failure_obligations
    )


def test_finite_time_regime_classifier_surfaces_nonzero_angular_cluster_exclusion():
    masses = np.ones(3)
    threshold = 0.045
    half_binary = 0.00465
    third_y = float(np.sqrt(threshold**2 - half_binary**2))
    positions = np.array(
        [
            [-half_binary, 0.0, 0.0],
            [half_binary, 0.0, 0.0],
            [0.0, third_y, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        0.01,
        initial_radius=0.0,
        order=8,
        binary_distance_threshold=threshold,
        spatial_binary_auto=False,
        max_compact_step=0.01,
        target_bisections=8,
    )

    assert not classification.certified
    assert classification.input_domain_certified
    assert classification.branch_partition is not None
    assert not classification.branch_partition_certified
    branch = classification.branch_partition.branches[0]
    assert branch.leaf_type == "threshold_adjacent_triple_close_cluster"
    assert branch.triple_collision_excluded
    assert branch.jacobi_cluster_coordinates_certified
    assert branch.interval_jacobi_cluster_coordinates_certified
    assert branch.jacobi_cluster_coordinate_certificate.pair == (0, 1)
    assert branch.interval_jacobi_cluster_coordinate_certificate.pair == (0, 1)
    assert branch.jacobi_cluster_coordinate_certificate.reconstruction_error < 1.0e-12
    assert branch.interval_jacobi_cluster_coordinate_certificate.reconstruction_containment_certified
    assert branch.interval_jacobi_cluster_coordinate_certificate.absolute_reconstruction_containment_certified
    assert (
        branch.interval_jacobi_cluster_coordinate_certificate.angular_decomposition_residual_contains_zero
    )
    assert (
        branch.jacobi_cluster_coordinate_certificate.angular_decomposition_error
        < 1.0e-12
    )
    assert (
        branch.triple_collision_exclusion_certificate.angular_momentum_norm_squared_lower_bound
        > 0.0
    )
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion"
        in classification.missing_obligations
    )
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector"
        not in classification.missing_obligations
    )


def test_event_order_branch_union_preserves_nonzero_angular_cluster_blocker():
    missing = general_solution_module._event_order_branch_union_failure_obligations(
        RuntimeError(
            "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion"
        )
    )

    assert missing == (
        "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion",
    )


def test_unrestricted_solution_rejects_close_pair_branch_union_when_members_cannot_keep_threshold():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.12, 0.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 3))
    options = dict(
        method="validated_atlas",
        initial_radius=0.005,
        order=8,
        guard_order=4,
        binary_distance_threshold=0.05,
        max_s_step=0.02,
    )

    with pytest.raises(FiniteTimeChartSelectorError) as exc_info:
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            1.0e-6,
            **options,
        )

    assert "finite_time_branch_union_consumption" in (
        exc_info.value.missing_obligations
    )
    branch_union_attempt = next(
        attempt
        for attempt in exc_info.value.blocking_attempts
        if attempt.route_id == "spatial_branch_union"
    )
    assert branch_union_attempt.attempted
    assert branch_union_attempt.branch_partition is not None
    assert branch_union_attempt.branch_partition.certified
    assert "spatial_collision_policy_scope" in (
        branch_union_attempt.missing_obligations
    )
    assert "collision_policy" in branch_union_attempt.missing_obligations
    assert any(
        attempt.branch_partition is not None and attempt.branch_partition.certified
        for attempt in exc_info.value.blocking_attempts
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        1.0e-6,
        initial_radius=0.005,
        order=8,
        guard_order=4,
        binary_distance_threshold=0.05,
        max_s_step=0.02,
    )

    assert not classification.certified
    assert classification.branch_partition is not None
    assert classification.branch_partition_certified
    assert "finite_time_branch_union_consumption" in (
        classification.missing_obligations
    )


def test_unrestricted_solution_spatial_branch_union_blocks_ordinary_handoff_when_members_cannot_keep_threshold():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.12, 0.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 3))

    with pytest.raises(FiniteTimeChartSelectorError) as exc_info:
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            5.0e-5,
            method="validated_atlas",
            initial_radius=0.005,
            order=8,
            guard_order=4,
            binary_distance_threshold=0.05,
            max_s_step=2.0e-4,
            spatial_binary_ordinary_handoff_min_pair_distance_required=0.0,
        )

    assert "finite_time_branch_union_consumption" in (
        exc_info.value.missing_obligations
    )
    assert any(
        attempt.branch_partition is not None and attempt.branch_partition.certified
        for attempt in exc_info.value.blocking_attempts
    )


def test_close_pair_branch_union_retries_members_through_initial_ks_competing_loop(
    monkeypatch,
):
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.zeros((3, 3))
    velocities = np.zeros((3, 3))
    branch = SimpleNamespace(
        branch_id="leaf_0",
        certified=True,
        selected_pair=(0, 1),
        state_interval=tuple((0.0, 0.0) for _ in range(18)),
    )
    partition = SimpleNamespace(
        certified=True,
        branches=(branch,),
        binary_distance_threshold=0.05,
    )
    member_atlas = SimpleNamespace(proof_certified=True)
    final_atlas = SimpleNamespace(proof_certified=True)
    calls = []

    def fake_competing_loop(*args, **kwargs):
        calls.append(("competing_loop", args, kwargs))
        return member_atlas

    def fake_branch_union_constructor(received_partition, received_masses, **kwargs):
        calls.append(("branch_union_constructor", received_partition, received_masses, kwargs))
        builder = kwargs["member_atlas_builder"]
        built_member = builder(
            branch=received_partition.branches[0],
            branch_label="positive_x",
            target_time_interval=FloatInterval.point(kwargs["target_time"]),
            s_endpoint=kwargs["s_endpoint"],
            retained_order=kwargs["retained_order"],
            guard_order=kwargs["guard_order"],
            source="test_member_builder",
        )
        assert built_member is member_atlas
        return final_atlas

    monkeypatch.setattr(
        general_solution_module,
        "_try_evaluate_auto_initial_spatial_ks_competing_handoff",
        fake_competing_loop,
    )
    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_close_pair_branch_partition",
        fake_branch_union_constructor,
    )

    atlas, missing = general_solution_module._try_evaluate_spatial_close_pair_branch_union(
        masses,
        positions,
        velocities,
        1.0e-4,
        branch_partition=partition,
        order=8,
        guard_order=4,
        s_endpoint=0.02,
        ordinary_handoff_min_pair_distance_required=0.01,
        max_competing_events=7,
    )

    assert atlas is final_atlas
    assert missing == ()
    assert calls[0][0] == "branch_union_constructor"
    assert calls[1][0] == "competing_loop"
    assert calls[1][1][0] is masses
    assert calls[1][1][1] is positions
    assert calls[1][1][2] is velocities
    assert calls[1][1][3] == 1.0e-4
    assert calls[1][2]["state_interval"] == branch.state_interval
    assert calls[1][2]["selected_pair"] == branch.selected_pair
    assert calls[1][2]["selected_branch"] == "positive_x"
    assert calls[1][2]["binary_distance_threshold"] == 0.05
    assert calls[1][2]["s_upper"] == 0.02
    assert calls[1][2]["max_competing_events"] == 7


def _repeat_budget_block(max_competing_events):
    return general_solution_module.FiniteTimeKSLoopBlocked(
        general_solution_module.FiniteTimeKSLoopProgressCertificate(
            initial_pair=(0, 1),
            target_time_after_start=FloatInterval.point(1.0e-4),
            max_competing_events=max_competing_events,
            steps=(),
            missing_obligations=("finite_time_atlas_loop_repeat_budget",),
        )
    )


def test_default_spatial_ks_competing_budget_retries_pure_repeat_budget():
    calls = []
    final = object()

    def builder(max_competing_events):
        calls.append(max_competing_events)
        if max_competing_events == 3:
            raise _repeat_budget_block(max_competing_events)
        return final

    result = general_solution_module._try_with_spatial_competing_event_budget_retries(
        builder,
        general_solution_module._spatial_competing_event_budgets(None),
    )

    assert result is final
    assert calls == [3, 7]


def test_explicit_spatial_ks_competing_budget_remains_strict():
    calls = []

    def builder(max_competing_events):
        calls.append(max_competing_events)
        raise _repeat_budget_block(max_competing_events)

    with pytest.raises(general_solution_module.FiniteTimeKSLoopBlocked) as exc_info:
        general_solution_module._try_with_spatial_competing_event_budget_retries(
            builder,
            general_solution_module._spatial_competing_event_budgets(0),
        )

    assert calls == [0]
    assert exc_info.value.certificate.max_competing_events == 0
    assert exc_info.value.missing_obligations == (
        "finite_time_atlas_loop_repeat_budget",
    )


def test_default_spatial_ks_competing_budget_does_not_retry_other_obstructions():
    calls = []

    def builder(max_competing_events):
        calls.append(max_competing_events)
        raise general_solution_module.FiniteTimeKSLoopBlocked(
            general_solution_module.FiniteTimeKSLoopProgressCertificate(
                initial_pair=(0, 1),
                target_time_after_start=FloatInterval.point(1.0e-4),
                max_competing_events=max_competing_events,
                steps=(),
                missing_obligations=("finite_time_event_order_requires_split",),
            )
        )

    with pytest.raises(general_solution_module.FiniteTimeKSLoopBlocked) as exc_info:
        general_solution_module._try_with_spatial_competing_event_budget_retries(
            builder,
            general_solution_module._spatial_competing_event_budgets(None),
        )

    assert calls == [3]
    assert exc_info.value.missing_obligations == (
        "finite_time_event_order_requires_split",
    )


def test_close_pair_branch_union_uses_default_budget_retry_for_members(
    monkeypatch,
):
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.zeros((3, 3))
    velocities = np.zeros((3, 3))
    branch = SimpleNamespace(
        branch_id="leaf_0",
        certified=True,
        selected_pair=(0, 1),
        state_interval=tuple((0.0, 0.0) for _ in range(18)),
    )
    partition = SimpleNamespace(
        certified=True,
        branches=(branch,),
        binary_distance_threshold=0.05,
    )
    member_atlas = SimpleNamespace(proof_certified=True)
    final_atlas = SimpleNamespace(proof_certified=True)
    budgets = []

    def fake_competing_loop(*args, **kwargs):
        max_competing_events = kwargs["max_competing_events"]
        budgets.append(max_competing_events)
        if max_competing_events == 3:
            raise _repeat_budget_block(max_competing_events)
        return member_atlas

    def fake_branch_union_constructor(received_partition, _received_masses, **kwargs):
        builder = kwargs["member_atlas_builder"]
        built_member = builder(
            branch=received_partition.branches[0],
            branch_label="positive_x",
            target_time_interval=FloatInterval.point(kwargs["target_time"]),
            s_endpoint=kwargs["s_endpoint"],
            retained_order=kwargs["retained_order"],
            guard_order=kwargs["guard_order"],
            source="test_member_builder",
        )
        assert built_member is member_atlas
        return final_atlas

    monkeypatch.setattr(
        general_solution_module,
        "_try_evaluate_auto_initial_spatial_ks_competing_handoff",
        fake_competing_loop,
    )
    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_close_pair_branch_partition",
        fake_branch_union_constructor,
    )

    atlas, missing = general_solution_module._try_evaluate_spatial_close_pair_branch_union(
        masses,
        positions,
        velocities,
        1.0e-4,
        branch_partition=partition,
        order=8,
        guard_order=4,
        s_endpoint=0.02,
        max_competing_events=None,
    )

    assert atlas is final_atlas
    assert missing == ()
    assert budgets == [3, 7]


def test_finite_time_regime_classifier_stops_before_invalid_input_domain():
    masses = np.array([1.0, 1.0, 0.5])
    positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 2))

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        0.01,
        order=8,
    )

    assert not classification.certified
    assert classification.input_domain_certificate is None
    assert classification.validated_atlas is None
    assert classification.selected_route_id is None
    assert "positive_mass_noncollision_input_domain" in (
        classification.missing_obligations
    )
    assert classification.failure_reason == (
        "input domain is not positive-mass noncollision data"
    )


def test_validated_atlas_proof_rejects_stale_selector_trace():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    second_solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        1.25 * target_time,
        method="validated_atlas",
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    stale_trace = replace(solution.selector_trace, selected_route_id="sundman")
    stale_solution = replace(solution, selector_trace=stale_trace)
    transplanted_trace_solution = replace(
        second_solution,
        selector_trace=solution.selector_trace,
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
    mismatched_solution = replace(solution, selector_trace=mismatched_trace)
    missing_trace_solution = replace(solution, selector_trace=None)
    missing_selector_ledger_solution = replace(
        solution,
        proof_ledger=replace(
            solution.proof_ledger,
            entries=tuple(
                entry
                for entry in solution.proof_ledger.entries
                if entry.name != "finite_time_chart_selector"
            ),
        ),
    )
    nameless_attempt_trace = replace(
        solution.selector_trace,
        attempts=(
            replace(solution.selector_trace.selected_attempt, route_id=""),
            *(
                attempt
                for attempt in solution.selector_trace.attempts
                if attempt is not solution.selector_trace.selected_attempt
            ),
        ),
    )
    nameless_attempt_solution = replace(solution, selector_trace=nameless_attempt_trace)
    duplicate_attempt_trace = replace(
        solution.selector_trace,
        attempts=(
            replace(
                solution.selector_trace.selected_attempt,
                selected=False,
                certified=False,
                reason="duplicate route id should not certify selector provenance",
            ),
            *solution.selector_trace.attempts,
        ),
    )
    duplicate_attempt_solution = replace(solution, selector_trace=duplicate_attempt_trace)

    assert solution.proof_certified
    assert second_solution.proof_certified
    assert solution.selector_trace.well_formed
    assert solution.selector_trace.missing_obligations == ()
    assert not stale_trace.certified
    assert not stale_solution.proof_certified
    assert transplanted_trace_solution.selector_trace.certified
    assert not transplanted_trace_solution.proof_certified
    assert "selector_trace_matches_validated_atlas" in (
        transplanted_trace_solution.missing_certification_obligations
    )
    assert mismatched_trace.certified
    assert "spatial_ks_binary" not in {chart.chart_type for chart in solution.charts}
    assert not mismatched_solution.proof_certified
    assert not missing_trace_solution.proof_certified
    assert missing_selector_ledger_solution.proof_ledger.certified
    assert not missing_selector_ledger_solution.proof_certified
    assert not nameless_attempt_trace.well_formed
    assert not nameless_attempt_trace.certified
    assert "selector_trace_attempt_well_formed:0" in (
        nameless_attempt_solution.missing_certification_obligations
    )
    assert not nameless_attempt_solution.proof_certified
    assert not duplicate_attempt_trace.well_formed
    assert "selector_trace_route_ids_unique" in (
        duplicate_attempt_solution.missing_certification_obligations
    )
    assert not duplicate_attempt_solution.proof_certified


def test_unrestricted_solution_validated_atlas_exposes_lohner_ordinary_chain():
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
    target_time = 0.02

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=12,
        max_compact_step=0.01,
        binary_distance_threshold=0.05,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    lohner = solution.evaluation.lohner_ordinary_set_propagated_interval_enclosure()

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "HybridValidatedEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "planar_hybrid"
    assert solution.selector_trace.certified
    assert {chart.chart_type for chart in solution.charts} == {"planar_ordinary_taylor"}
    assert solution.evaluation.target_interval_source == "lohner_ordinary_set_propagation"
    assert solution.evaluation.lohner_enclosure is lohner
    assert lohner.proof_certified
    assert lohner.certified_step_count == solution.chart_count
    assert lohner.final_state_contains(solution.evaluation.final_state)
    assert lohner.final_state_contains(reference)
    assert solution.transitions
    broken_transition = replace(
        solution.transitions[0],
        target_chart_id="not_the_next_chart",
    )
    broken_solution = replace(
        solution,
        transitions=(broken_transition, *solution.transitions[1:]),
    )
    stale_time_chart = replace(
        solution.charts[0],
        physical_time_interval=FloatInterval(100.0, 101.0),
    )
    stale_time_solution = replace(
        solution,
        charts=(stale_time_chart, *solution.charts[1:]),
    )
    missing_transition_type = replace(
        solution.transitions[0],
        transition_type="",
    )
    missing_transition_type_solution = replace(
        solution,
        transitions=(missing_transition_type, *solution.transitions[1:]),
    )
    duplicate_chart_id = replace(
        solution.charts[1],
        chart_id=solution.charts[0].chart_id,
    )
    duplicate_chart_transition = replace(
        solution.transitions[0],
        target_chart_id=solution.charts[0].chart_id,
    )
    duplicate_chart_id_solution = replace(
        solution,
        charts=(solution.charts[0], duplicate_chart_id),
        transitions=(duplicate_chart_transition,),
    )
    assert not broken_solution.proof_certified
    assert stale_time_chart.certified
    assert stale_time_solution.target_time_certified
    assert not stale_time_solution.proof_certified
    assert missing_transition_type.certified
    assert not missing_transition_type_solution.proof_certified
    assert duplicate_chart_id.certified
    assert duplicate_chart_transition.certified
    assert not duplicate_chart_id_solution.proof_certified
    for target_interval, lohner_interval in zip(
        solution.target_state_interval,
        lohner.final_state_interval,
        strict=True,
    ):
        assert target_interval.as_tuple() == lohner_interval

    stale_lohner_steps = list(lohner.steps)
    stale_lohner_steps[-1] = replace(
        stale_lohner_steps[-1],
        physical_step=stale_lohner_steps[-1].physical_step + 0.25,
    )
    stale_lohner = replace(lohner, steps=tuple(stale_lohner_steps))
    assert stale_lohner.proof_certified

    stale_wrapper = _HybridEnclosureOverride(
        solution.evaluation.hybrid_solution,
        lohner_enclosure=stale_lohner,
    )
    stale_enclosure_solution = validated_atlas_from_hybrid_solution(
        stale_wrapper,
        target_time=target_time,
        source="test_stale_lohner_target_enclosure",
    )

    assert stale_enclosure_solution.target_state_contains(solution.evaluation.final_state)
    assert "hybrid_target_enclosure_chain" in (
        stale_enclosure_solution.proof_ledger.missing_required_obligations
    )
    assert not stale_enclosure_solution.proof_certified

    stale_target_interval = np.asarray(
        [
            FloatInterval.point(1.0e9)
            for _interval in solution.target_state_interval
        ],
        dtype=object,
    )
    stale_target_solution = replace(
        solution,
        evaluation=_TargetStateIntervalOverride(
            solution.evaluation,
            target_state_interval=stale_target_interval,
            final_state=solution.evaluation.final_state,
        ),
    )
    assert not stale_target_solution.target_state_interval_certified
    assert not stale_target_solution.proof_certified

    stale_initial_interval = np.asarray(
        [
            FloatInterval.point(1.0e9)
            for _interval in solution.initial_state_interval
        ],
        dtype=object,
    )
    stale_initial_solution = replace(
        solution,
        initial_state_interval=stale_initial_interval,
    )
    assert not stale_initial_solution.initial_state_interval_certified
    assert not stale_initial_solution.proof_certified


def test_planar_hybrid_transition_requires_interval_handoff_not_only_point():
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
    target_time = 0.02

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=12,
        max_compact_step=0.01,
        binary_distance_threshold=0.05,
        target_bisections=42,
    )
    hybrid_solution = solution.evaluation.hybrid_solution
    steps = list(hybrid_solution.steps)
    stale_start = list(steps[1].start_state_interval)
    lower, upper = stale_start[0]
    stale_start[0] = (float(lower) - 1.0e-6, float(upper))
    steps[1] = replace(steps[1], start_state_interval=tuple(stale_start))
    stale_hybrid = replace(hybrid_solution, steps=tuple(steps))

    stale_solution = validated_atlas_from_hybrid_solution(
        stale_hybrid,
        target_time=target_time,
        source="test_stale_handoff",
    )

    assert solution.proof_certified
    assert solution.transitions[0].certified
    assert not stale_solution.transitions[0].certified
    assert "hybrid_chart_transitions" in (
        stale_solution.proof_ledger.missing_required_obligations
    )
    assert not stale_solution.proof_certified


def test_unrestricted_solution_validated_atlas_accepts_split_planar_initial_union():
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
    target_time = 1e-4

    with pytest.raises(RuntimeError, match="planar_hybrid"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            target_time,
            method="validated_atlas",
            initial_radius=0.21,
            order=8,
            max_compact_step=1e-3,
            binary_distance_threshold=0.1,
            spatial_binary_auto=False,
            tail_certificate_mode="cauchy",
        )

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=8,
        max_compact_step=1e-3,
        binary_distance_threshold=0.1,
        spatial_binary_auto=False,
        tail_certificate_mode="cauchy",
        planar_initial_state_interval_union=(first_state, second_state),
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    step = solution.evaluation.hybrid_solution.steps[0]

    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "planar_hybrid"
    assert solution.selector_trace.certified
    assert solution.target_state_contains(reference)
    assert solution.residual_budget.certified
    assert solution.invariants.certified
    assert solution.initial_state_interval_union is not None
    assert len(solution.initial_state_interval_union) == 2
    assert solution.evaluation.target_interval_source == "hybrid_set_propagation"
    assert solution.evaluation.lohner_enclosure is None
    assert solution.evaluation.set_enclosure is not None
    assert solution.evaluation.set_enclosure.proof_certified
    assert solution.target_state_interval_union is not None
    assert len(solution.target_state_interval_union) == 2
    assert solution.evaluation.set_enclosure.steps[-1].end_state_interval_union
    assert len(solution.evaluation.set_enclosure.steps[-1].end_state_interval_union) == 2
    assert step.start_state_interval_union == (first_state, second_state)
    assert step.residual_certificate.member_count == 2
    assert step.energy_certificate.member_count == 2

    collapsed_solution = replace(solution, initial_state_interval_union=None)
    stale_union_solution = replace(
        solution,
        initial_state_interval_union=(solution.initial_state_interval_union[1],),
    )
    in_hull_initial_union_solution = replace(
        solution,
        initial_state_interval_union=(solution.initial_state_interval,),
    )
    collapsed_target_evaluation = replace(
        solution.evaluation,
        target_state_interval_union=None,
    )
    collapsed_target_solution = replace(
        solution,
        evaluation=collapsed_target_evaluation,
    )
    stale_target_union = (
        np.asarray(
            [
                FloatInterval.point(1.0e9)
                for _interval in solution.target_state_interval
            ],
            dtype=object,
        ),
    )
    stale_target_evaluation = replace(
        solution.evaluation,
        target_state_interval_union=stale_target_union,
    )
    stale_target_solution = replace(
        solution,
        evaluation=stale_target_evaluation,
    )
    in_hull_target_evaluation = replace(
        solution.evaluation,
        target_state_interval_union=(solution.target_state_interval,),
    )
    in_hull_target_solution = replace(
        solution,
        evaluation=in_hull_target_evaluation,
    )

    assert not collapsed_solution.proof_certified
    assert not stale_union_solution.proof_certified
    assert not in_hull_initial_union_solution.proof_certified
    assert not collapsed_target_solution.proof_certified
    assert not stale_target_solution.proof_certified
    assert not in_hull_target_solution.proof_certified

    set_enclosure = solution.evaluation.set_enclosure
    stale_set_steps = list(set_enclosure.steps)
    stale_set_steps[-1] = replace(
        stale_set_steps[-1],
        physical_step=stale_set_steps[-1].physical_step + 0.25,
    )
    stale_set_enclosure = replace(set_enclosure, steps=tuple(stale_set_steps))
    assert stale_set_enclosure.proof_certified

    stale_wrapper = _HybridEnclosureOverride(
        solution.evaluation.hybrid_solution,
        set_enclosure=stale_set_enclosure,
    )
    stale_enclosure_solution = validated_atlas_from_hybrid_solution(
        stale_wrapper,
        target_time=target_time,
        source="test_stale_set_target_enclosure",
    )

    assert stale_enclosure_solution.target_state_contains(solution.evaluation.final_state)
    assert "hybrid_target_enclosure_chain" in (
        stale_enclosure_solution.proof_ledger.missing_required_obligations
    )
    assert not stale_enclosure_solution.proof_certified


def test_simultaneous_close_pair_split_produces_certified_branch_union():
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.01, 0.21),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.20, 0.20),
        (0.0, 0.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 9

    partition = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.05,
        max_depth=5,
    )

    assert partition.well_formed
    assert partition.certified
    assert partition.split_count > 0
    assert partition.recursive_bisection_cover_certified
    assert partition.branch_cover_certified
    assert partition.certified_branch_count == len(partition.branches)
    assert partition.ambiguous_branch_count == 0
    assert all(len(branch.possible_close_pairs) <= 1 for branch in partition.branches)
    assert any(branch.selected_pair == (0, 1) for branch in partition.branches)
    assert any(branch.selected_pair == (1, 2) for branch in partition.branches)
    assert all(branch.leaf_type == "single_possible_binary" for branch in partition.branches)
    for branch in partition.branches:
        for branch_interval, original_interval in zip(
            branch.state_interval,
            partition.original_state_interval,
            strict=True,
        ):
            assert (
                original_interval[0]
                <= branch_interval[0]
                <= branch_interval[1]
                <= original_interval[1]
            )


def test_simultaneous_close_pair_split_refuses_genuine_triple_close_cluster():
    state_interval = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.006, 0.010),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.012, 0.016),
        (0.0, 0.0),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 9

    partition = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.02,
        max_depth=4,
    )

    assert partition.well_formed
    assert not partition.certified
    assert partition.recursive_bisection_cover_certified
    assert partition.branch_cover_certified
    assert partition.split_count == 0
    assert len(partition.branches) == 1
    assert partition.ambiguous_branch_count >= 1
    assert "simultaneous_close_pair_partitioning" in partition.missing_obligations
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector"
        in partition.missing_obligations
    )
    assert partition.branches[0].leaf_type == "certified_triple_close_cluster"
    assert len(partition.branches[0].certified_close_pairs) >= 2

    partition_with_masses = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.02,
        max_depth=4,
        masses=np.ones(3),
    )
    branch = partition_with_masses.branches[0]
    assert not partition_with_masses.certified
    assert branch.jacobi_cluster_coordinate_certificate is None
    assert len(branch.jacobi_cluster_coordinate_certificates) == len(
        branch.certified_close_pairs
    )
    assert len(branch.interval_jacobi_cluster_coordinate_certificates) == len(
        branch.certified_close_pairs
    )
    assert all(
        certificate.certified
        for certificate in branch.jacobi_cluster_coordinate_certificates
    )
    assert all(
        certificate.certified
        for certificate in branch.interval_jacobi_cluster_coordinate_certificates
    )


def test_simultaneous_close_pair_split_stops_threshold_adjacent_triple_cluster():
    state_interval = (
        (-0.00465, -0.00465),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.00465, 0.00465),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.04465, 0.04515),
        (0.0, 0.0),
    ) + ((0.0, 0.0),) * 9

    partition = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.045,
        max_depth=8,
    )

    assert partition.well_formed
    assert not partition.certified
    assert partition.recursive_bisection_cover_certified
    assert partition.branch_cover_certified
    assert partition.split_count == 0
    assert len(partition.branches) == 1
    assert partition.branches[0].leaf_type == "threshold_adjacent_triple_close_cluster"
    assert partition.branches[0].certified_close_pairs == ((0, 1),)
    assert set(partition.branches[0].possible_close_pairs) == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert "simultaneous_close_pair_partitioning" in partition.missing_obligations
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector"
        in partition.missing_obligations
    )


def test_simultaneous_close_pair_split_excludes_nonzero_angular_total_collision():
    state_interval = (
        (-0.00465, -0.00465),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.00465, 0.00465),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.04465, 0.04515),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    partition = certify_simultaneous_close_pair_partition(
        state_interval,
        binary_distance_threshold=0.045,
        max_depth=8,
        masses=np.ones(3),
    )

    branch = partition.branches[0]
    assert partition.well_formed
    assert not partition.certified
    assert partition.split_count == 0
    assert branch.leaf_type == "threshold_adjacent_triple_close_cluster"
    assert branch.triple_collision_excluded
    assert branch.jacobi_cluster_coordinates_certified
    assert branch.interval_jacobi_cluster_coordinates_certified
    assert branch.jacobi_cluster_coordinate_certificate.pair == (0, 1)
    assert branch.interval_jacobi_cluster_coordinate_certificate.pair == (0, 1)
    assert branch.jacobi_cluster_coordinate_certificate.reconstruction_error < 1.0e-12
    assert branch.interval_jacobi_cluster_coordinate_certificate.reconstruction_containment_certified
    assert branch.interval_jacobi_cluster_coordinate_certificate.absolute_reconstruction_containment_certified
    assert (
        branch.interval_jacobi_cluster_coordinate_certificate.angular_decomposition_residual_contains_zero
    )
    assert (
        branch.jacobi_cluster_coordinate_certificate.angular_decomposition_error
        < 1.0e-12
    )
    assert (
        branch.triple_collision_exclusion_certificate.angular_momentum_norm_squared_lower_bound
        > 0.0
    )
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_after_nonzero_angular_exclusion"
        in partition.missing_obligations
    )
    assert (
        "spatial_triple_close_cluster_requires_cluster_blowup_or_total_collision_selector"
        not in partition.missing_obligations
    )


def test_unrestricted_solution_validated_atlas_time_reverses_split_planar_union():
    masses = np.array([1.0, 1.0, 0.2])
    positions = np.array([[0.0, 0.0], [0.205, 0.0], [2.0, 0.0]])
    velocities = np.zeros((3, 2))
    velocity_boxes = (
        (-2.0e-5, 1.0e-5),
        (-1.0e-5, 2.0e-5),
        (-2.0e-5, 1.0e-5),
        (-1.0e-5, 2.0e-5),
        (-2.0e-5, 1.0e-5),
        (-1.0e-5, 2.0e-5),
    )
    first_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.21),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
        *velocity_boxes,
    )
    second_state = (
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.21, -0.2),
        (0.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
        *velocity_boxes,
    )
    target_time = -1.0e-4

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=8,
        max_compact_step=1e-3,
        binary_distance_threshold=0.1,
        spatial_binary_auto=False,
        tail_certificate_mode="cauchy",
        planar_initial_state_interval_union=(first_state, second_state),
    )
    reference = integrate_reference(positions, velocities, masses, target_time)
    initial_union_bounds = tuple(
        tuple(
            interval.as_tuple()
            for interval in np.asarray(member, dtype=object).reshape(-1)
        )
        for member in solution.initial_state_interval_union
    )

    assert solution.evaluation.__class__.__name__ == "TimeReversedValidatedEvaluation"
    assert solution.proof_certified
    assert solution.target_state_contains(reference)
    assert solution.initial_state_interval_union is not None
    assert solution.target_state_interval_union is not None
    assert initial_union_bounds == (first_state, second_state)


def test_unrestricted_solution_validated_atlas_routes_negative_planar_time_by_reversal():
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
    target_time = -0.02

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=12,
        max_compact_step=0.01,
        binary_distance_threshold=0.05,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "TimeReversedValidatedEvaluation"
    assert solution.proof_certified
    assert solution.target_time == pytest.approx(target_time)
    assert {chart.chart_type for chart in solution.charts} == {"planar_ordinary_taylor"}
    assert all(chart.physical_time_interval.upper <= 0.0 for chart in solution.charts)
    assert solution.target_state_contains(reference)
    assert "time_reversal_symmetry" not in solution.proof_ledger.missing_required_obligations
    assert solution.proof_ledger.missing_required_obligations == ()


def test_unrestricted_solution_validated_atlas_routes_planar_binary_through_hybrid():
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
    target_time = 1e-8

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=16,
        max_compact_step=1e-4,
        max_s_step=0.02,
        binary_distance_threshold=1e-2,
        binary_exit_distance=1.25e-2,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "HybridValidatedEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "planar_hybrid"
    assert solution.selector_trace.certified
    assert solution.target_state_contains(reference)
    assert solution.proof_ledger.missing_required_obligations == ()
    assert solution.collision_policy.binary_policy == "planar_levi_civita_binary_regularized"
    assert solution.chart_count == 1
    assert solution.charts[0].chart_type == "planar_levi_civita_binary"
    assert solution.charts[0].residual_certified
    assert solution.charts[0].projection_certified
    assert solution.charts[0].invariants_certified
    assert solution.residual_budget.certified
    assert solution.invariants.certified
    assert solution.evaluation.target_interval_source == "hybrid_set_propagation"
    assert solution.evaluation.lohner_enclosure is None
    assert solution.evaluation.set_enclosure is not None
    assert solution.evaluation.set_enclosure.proof_certified
    assert solution.evaluation.set_enclosure.final_state_contains(solution.evaluation.final_state)
    with pytest.raises(ValueError, match="only supports ordinary"):
        solution.evaluation.lohner_ordinary_set_propagated_interval_enclosure(
            retained_order=16,
        )
    set_enclosure = solution.evaluation.ordinary_set_propagated_interval_enclosure()
    assert set_enclosure is solution.evaluation.set_enclosure
    assert all(
        entry.source == "evaluate_unrestricted_solution"
        for entry in solution.proof_ledger.entries
    )


def test_unrestricted_solution_validated_atlas_routes_explicit_spatial_ks_handoff():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    exit_rho = 2.0 * enter_distance
    target_time = 5.0e-5

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=24,
        guard_order=6,
        spatial_binary_pair=(0, 1),
        spatial_binary_enter_distance=enter_distance,
        spatial_binary_entry_time_upper=entry_time_upper,
        spatial_binary_exit_rho=exit_rho,
        spatial_binary_s_upper=0.1,
    )

    ordinary_point = construct_taylor_solution(positions, velocities, initial.masses, order=32)
    entry_time = solution.evaluation.entry_event_certificate.root
    lifted_entry = spatial_to_ks_binary_chart(
        ordinary_point.positions_at(entry_time),
        ordinary_point.velocities_at(entry_time),
        initial.masses,
        pair=(0, 1),
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_entry, order=32)
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(solution.evaluation.ks_evaluation.endpoint_projection.s_value)
    )
    reference = np.concatenate([target_positions.reshape(-1), target_velocities.reshape(-1)])

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "SpatialOrdinaryKSHandoffEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "explicit_spatial_ks"
    assert solution.selector_trace.certified
    assert solution.target_state_contains(reference)
    assert solution.proof_ledger.missing_required_obligations == ()
    assert [chart.chart_type for chart in solution.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert solution.evaluation.ks_evaluation.ordinary_solution is None
    assert solution.collision_policy.binary_policy == (
        "spatial_ks_selected_binary_entry_and_local_regularization"
    )
    assert solution.collision_policy.triple_collision_status == (
        "locally_excluded_on_handoff_charts"
    )
    assert solution.evaluation.collision_policy_certificate.certified
    assert (
        solution.evaluation.collision_policy_certificate.min_squared_distance_lower_bound
        > 0.0
    )
    assert all(
        entry.source == "evaluate_unrestricted_solution"
        for entry in solution.proof_ledger.entries
    )


def test_unrestricted_solution_validated_atlas_default_spatial_ks_handoff_floor_stays_regularized():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    exit_rho = 0.1

    lifted_entry = pre_collision_solution.state_at(s_entry)
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_entry, order=32)
    interval_ks_solution = construct_interval_spatial_ks_binary_taylor_solution(
        lifted_entry,
        order=32,
    )
    original_exit = certify_spatial_ks_binary_rho_exit_event(
        interval_ks_solution,
        exit_rho=exit_rho,
        s_upper=0.5,
        coefficient_count=24,
    )
    retried_exit = certify_spatial_ks_binary_rho_exit_event(
        interval_ks_solution,
        exit_rho=1.5 * (2.0 * 0.08),
        s_upper=1.5,
        coefficient_count=24,
    )
    entry_time = (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    target_s = original_exit.root + 1.0e-4
    target_time = entry_time + ks_point_solution.physical_time_at(target_s)

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=24,
        guard_order=6,
        binary_distance_threshold=0.08,
        spatial_binary_pair=(0, 1),
        spatial_binary_enter_distance=enter_distance,
        spatial_binary_entry_time_upper=entry_time_upper,
        spatial_binary_exit_rho=exit_rho,
        spatial_binary_s_upper=0.5,
    )

    assert original_exit.certified
    assert retried_exit.certified
    assert target_s > original_exit.root
    assert target_s < retried_exit.root
    assert solution.proof_certified
    assert [chart.chart_type for chart in solution.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert solution.evaluation.ks_evaluation.ordinary_solution is None
    assert solution.evaluation.ks_evaluation.ordinary_handoff_admissibility is None
    assert solution.evaluation.ks_evaluation.endpoint_projection.s_value > original_exit.root
    assert "ordinary_handoff_admissibility" not in (
        solution.proof_ledger.missing_required_obligations
    )


def test_unrestricted_solution_validated_atlas_retries_spatial_ks_exit_but_refuses_inadmissible_target_handoff():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    s_start = -0.04
    s_entry = -0.03
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_start)
    )
    entry_positions, _entry_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(s_entry)
    )
    entry_time_upper = 1.5 * (
        pre_collision_solution.physical_time_at(s_entry)
        - pre_collision_solution.physical_time_at(s_start)
    )
    enter_distance = float(np.linalg.norm(entry_positions[1] - entry_positions[0]))
    initial_exit_rho = 0.1
    binary_distance_threshold = 0.08
    handoff_floor = 2.0 * binary_distance_threshold
    retried_exit_rho = 1.5 * handoff_floor

    lifted_entry = pre_collision_solution.state_at(s_entry)
    interval_ks_solution = construct_interval_spatial_ks_binary_taylor_solution(
        lifted_entry,
        order=32,
    )
    retried_exit = certify_spatial_ks_binary_rho_exit_event(
        interval_ks_solution,
        exit_rho=retried_exit_rho,
        s_upper=1.5,
        coefficient_count=24,
    )
    state_interval = tuple(
        (float(component), float(component))
        for component in np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    )
    retry_probe = validated_atlas_module.validated_atlas_from_spatial_ordinary_ks_handoff(
        state_interval,
        initial.masses,
        pair=(0, 1),
        enter_distance=enter_distance,
        entry_time_upper=entry_time_upper,
        branch="positive_x",
        exit_rho=retried_exit_rho,
        s_upper=1.5,
        ordinary_step_size=1.0e-4,
        retained_order=24,
        guard_order=6,
        ordinary_handoff_min_pair_distance_required=handoff_floor,
    )
    target_time = (
        retry_probe.evaluation.entry_event_certificate.root_interval[1]
        + retry_probe.evaluation.ks_evaluation.endpoint_projection.physical_time.upper
        + 1.0e-4
    )

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=24,
        guard_order=6,
        binary_distance_threshold=binary_distance_threshold,
        spatial_binary_pair=(0, 1),
        spatial_binary_enter_distance=enter_distance,
        spatial_binary_entry_time_upper=entry_time_upper,
        spatial_binary_exit_rho=initial_exit_rho,
        spatial_binary_s_upper=0.5,
    )
    ks_evaluation = solution.evaluation.ks_evaluation

    assert retried_exit.certified
    assert general_solution_module._spatial_ks_ordinary_handoff_exit_rho_candidates(
        initial_exit_rho,
        ordinary_handoff_min_pair_distance_required=handoff_floor,
    ) == pytest.approx((initial_exit_rho, retried_exit_rho))
    assert retry_probe.evaluation.ks_evaluation.ordinary_handoff_admissibility.certified
    assert [chart.chart_type for chart in retry_probe.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
        "spatial_ordinary_taylor_after_ks",
    ]
    assert (
        retry_probe.evaluation.ks_evaluation.ordinary_handoff_admissibility.min_pair_distance_lower_bound
        >= handoff_floor
    )
    assert (
        retry_probe.evaluation.ks_evaluation.ordinary_handoff_admissibility.cauchy_radius
        < retry_probe.evaluation.ks_evaluation.endpoint_projection.physical_time.upper
        - retry_probe.evaluation.ks_evaluation.endpoint_projection.physical_time.lower
    )
    assert solution.proof_certified
    assert [chart.chart_type for chart in solution.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert ks_evaluation.exit_event_certificate.exit_rho == pytest.approx(initial_exit_rho)
    assert ks_evaluation.ordinary_solution is None
    assert ks_evaluation.ordinary_handoff_admissibility is None
    assert "ordinary_handoff_admissibility" not in (
        solution.proof_ledger.missing_required_obligations
    )


def test_unrestricted_solution_validated_atlas_auto_starts_closing_close_binary_in_ks():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.04)
    )

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        5.0e-5,
        method="validated_atlas",
        order=24,
        guard_order=6,
        binary_distance_threshold=1.0e-2,
    )

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "SpatialKSValidatedEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "auto_spatial_ks"
    assert solution.selector_trace.certified
    assert solution.proof_ledger.missing_required_obligations == ()
    assert solution.charts[0].chart_type == "spatial_ks_binary"
    assert "spatial_ordinary_taylor_before_ks" not in {
        chart.chart_type for chart in solution.charts
    }
    assert solution.evaluation.ordinary_solution is None
    assert solution.evaluation.exit_event_certificate.certified
    assert solution.collision_policy.binary_policy == (
        "spatial_ks_initial_close_binary_local_regularization"
    )


def test_unrestricted_solution_validated_atlas_auto_selects_future_spatial_ks_entry():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=48)
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.2)
    )
    binary_distance_threshold = 2.0e-2

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        3.0e-3,
        method="validated_atlas",
        order=28,
        guard_order=8,
        binary_distance_threshold=binary_distance_threshold,
    )

    assert np.linalg.norm(positions[1] - positions[0]) > binary_distance_threshold
    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "SpatialOrdinaryKSHandoffEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "auto_spatial_ks"
    assert solution.selector_trace.certified
    assert solution.evaluation.entry_event_certificate.root > 0.0
    assert solution.evaluation.entry_event_certificate.enter_distance == pytest.approx(
        binary_distance_threshold
    )
    assert [chart.chart_type for chart in solution.charts[:2]] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
    ]
    assert solution.collision_policy.binary_policy == (
        "spatial_ks_selected_binary_entry_and_local_regularization"
    )


def test_unrestricted_solution_validated_atlas_auto_starts_receding_close_binary_in_ks():
    initial = _exact_spatial_ks_collision_state()
    post_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=48)
    positions, velocities = ks_binary_chart_to_spatial(
        post_collision_solution.state_at(0.04)
    )
    relative_position = positions[1] - positions[0]
    relative_velocity = velocities[1] - velocities[0]

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        5.0e-5,
        method="validated_atlas",
        order=24,
        guard_order=8,
        binary_distance_threshold=1.0e-2,
    )

    assert np.linalg.norm(relative_position) < 1.0e-2
    assert np.dot(relative_position, relative_velocity) > 0.0
    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "SpatialKSValidatedEvaluation"
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "auto_spatial_ks"
    assert solution.selector_trace.certified
    assert solution.charts[0].chart_type == "spatial_ks_binary"
    assert "spatial_ordinary_taylor_before_ks" not in {
        chart.chart_type for chart in solution.charts
    }
    assert solution.collision_policy.binary_policy == (
        "spatial_ks_initial_close_binary_local_regularization"
    )


def test_unrestricted_solution_validated_atlas_blocks_future_spatial_binary_without_ks_auto():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=48)
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.2)
    )

    with pytest.raises(RuntimeError, match="spatial_future_close_binary"):
        evaluate_unrestricted_solution(
            initial.masses,
            positions,
            velocities,
            3.0e-3,
            method="validated_atlas",
            order=14,
            guard_order=4,
            binary_distance_threshold=2.0e-2,
            spatial_binary_auto=False,
        )


def test_unrestricted_solution_validated_atlas_blocks_receding_close_binary_without_ks_auto():
    initial = _exact_spatial_ks_collision_state()
    post_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=48)
    positions, velocities = ks_binary_chart_to_spatial(
        post_collision_solution.state_at(0.04)
    )

    with pytest.raises(RuntimeError, match="spatial_close_binary"):
        evaluate_unrestricted_solution(
            initial.masses,
            positions,
            velocities,
            5.0e-5,
            method="validated_atlas",
            order=14,
            guard_order=4,
            binary_distance_threshold=1.0e-2,
            spatial_binary_auto=False,
        )


def test_unrestricted_solution_validated_atlas_blocks_competing_close_binary_inside_ks():
    initial = replace(
        _exact_spatial_ks_collision_state(),
        third_offset=np.array([1.0e-3, 2.0e-4, -1.0e-4]),
        third_offset_velocity=np.zeros(3),
    )
    solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(solution.state_at(0.005))

    with pytest.raises(
        FiniteTimeChartSelectorError,
        match="ks_competing_close_binary_requires_next_regularized_chart_or_split",
    ) as exc_info:
        evaluate_unrestricted_solution(
            initial.masses,
            positions,
            velocities,
            1.0e-8,
            method="validated_atlas",
            order=18,
            guard_order=6,
            binary_distance_threshold=1.0e-2,
        )
    assert "ks_competing_close_binary_requires_next_regularized_chart_or_split" in (
        exc_info.value.missing_obligations
    )


def test_unrestricted_solution_validated_atlas_blocks_second_competing_close_after_ks_repeat():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.06, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0],
        ]
    )
    binary_distance_threshold = 2.0e-2
    initial = spatial_to_ks_binary_chart(positions, velocities, masses, pair=(0, 1))
    first_interval = construct_interval_spatial_ks_binary_taylor_solution(
        initial,
        order=32,
    )
    first_point = construct_spatial_ks_binary_taylor_solution(initial, order=32)
    competing_event = certify_spatial_ks_competing_binary_entry_event(
        first_interval,
        pair=(0, 2),
        enter_distance=binary_distance_threshold,
        s_upper=0.1,
        coefficient_count=24,
    )
    target_time = first_point.physical_time_at(competing_event.root) + 1.0e-8

    with pytest.raises(
        FiniteTimeChartSelectorError,
        match="ks_competing_close_binary_requires_next_regularized_chart_or_split",
    ) as exc_info:
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            target_time,
            method="validated_atlas",
            order=24,
            guard_order=8,
            binary_distance_threshold=binary_distance_threshold,
        )
    assert "ks_competing_close_binary_requires_next_regularized_chart_or_split" in (
        exc_info.value.missing_obligations
    )
    assert "simultaneous_close_pair_partitioning" in exc_info.value.missing_obligations
    assert (
        "spatial_close_binary_requires_regularized_chart_or_split"
        in exc_info.value.missing_obligations
    )
    with pytest.raises(
        FiniteTimeChartSelectorError,
        match="finite_time_atlas_loop_repeat_budget",
    ) as budget_exc:
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            target_time,
            method="validated_atlas",
            order=24,
            guard_order=8,
            binary_distance_threshold=binary_distance_threshold,
            spatial_binary_max_competing_events=0,
        )
    assert "finite_time_atlas_loop_repeat_budget" in (
        budget_exc.value.missing_obligations
    )
    assert budget_exc.value.blocking_attempts[0].route_id == "auto_spatial_ks"
    assert budget_exc.value.blocking_attempts[0].missing_obligations == (
        "finite_time_atlas_loop_repeat_budget",
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        target_time,
        order=24,
        guard_order=8,
        binary_distance_threshold=binary_distance_threshold,
    )
    assert not classification.certified
    assert "ks_competing_close_binary_requires_next_regularized_chart_or_split" in (
        classification.missing_obligations
    )
    assert "simultaneous_close_pair_partitioning" in classification.missing_obligations
    assert (
        "spatial_close_binary_requires_regularized_chart_or_split"
        in classification.missing_obligations
    )


def test_finite_time_loop_rejects_branch_with_uncertified_event_order():
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
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )

    loop_progress = general_solution_module._certify_spatial_ks_competing_loop_progress(
        initial,
        target_time_after_ks_start=1.0e-3,
        binary_distance_threshold=0.045,
        s_upper=0.1,
        order=8,
        guard_order=4,
        max_competing_events=2,
    )

    assert not loop_progress.certified
    assert len(loop_progress.steps) == 1
    assert loop_progress.steps[0].decision == "event_order_blocked"
    assert (
        loop_progress.steps[0]
        .event_set_certificate
        .multiple_possible_first_events_requiring_split
    )
    assert "finite_time_event_order_requires_split" in (
        loop_progress.missing_obligations
    )
    partition = loop_progress.steps[0].event_order_partition_certificate
    assert partition is not None
    assert partition.well_formed
    assert not partition.certified
    assert "ks_event_order_leaf_decisions" in loop_progress.missing_obligations
    assert "ks_event_order_state_width_exhausted" in (
        loop_progress.steps[0].missing_obligations
    )


def test_finite_time_loop_surfaces_certified_event_order_partition_for_branch_union():
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
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)

    loop_progress = general_solution_module._certify_spatial_ks_competing_loop_progress(
        state,
        target_time_after_ks_start=1.0e-3,
        binary_distance_threshold=0.045,
        s_upper=0.1,
        order=4,
        guard_order=2,
        max_competing_events=2,
    )

    assert not loop_progress.certified
    assert len(loop_progress.steps) == 1
    assert loop_progress.steps[0].decision == "event_order_blocked"
    partition = loop_progress.steps[0].event_order_partition_certificate
    assert partition is not None
    assert partition.certified
    assert partition.certified_leaf_count == len(partition.leaves)
    assert "finite_time_event_order_partition_requires_branch_union_constructor" in (
        loop_progress.missing_obligations
    )


def test_loop_progress_attachment_accepts_certified_event_order_branch_union():
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
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)
    loop_progress = general_solution_module._certify_spatial_ks_competing_loop_progress(
        state,
        target_time_after_ks_start=2.7e-4,
        binary_distance_threshold=0.045,
        s_upper=0.1,
        order=8,
        guard_order=4,
        max_competing_events=2,
    )
    partition = loop_progress.steps[0].event_order_partition_certificate
    atlas = validated_atlas_module.validated_atlas_from_spatial_ks_event_order_partition(
        partition,
        retained_order=8,
        guard_order=4,
        competing_pair_min_distance_required=0.0,
        max_competing_repeats=0,
    )

    attached = general_solution_module._attach_spatial_ks_loop_progress(
        atlas,
        loop_progress,
        source="test",
    )
    loop_entries = tuple(
        entry
        for entry in attached.proof_ledger.entries
        if entry.name == "finite_time_atlas_loop_progress"
    )

    assert not loop_progress.certified
    assert partition.certified
    assert atlas.proof_certified
    assert attached.proof_certified
    assert len(loop_entries) == 1
    assert loop_entries[0].certified
    assert "event_order_branch_union_consumed=True" in loop_entries[0].detail


def test_initial_spatial_ks_loop_rejects_event_order_union_with_close_old_binary():
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
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)
    projection = project_spatial_ks_binary_interval_chart_state_to_physical(
        state,
        physical_time=FloatInterval.point(0.0),
    )

    assert projection.certified
    with pytest.raises(general_solution_module.FiniteTimeKSLoopBlocked) as exc_info:
        general_solution_module._try_evaluate_auto_initial_spatial_ks_competing_handoff(
            masses,
            positions,
            velocities,
            2.7e-4,
            state_interval=projection.state_interval,
            selected_pair=(0, 1),
            selected_branch="positive_x",
            binary_distance_threshold=0.045,
            s_upper=0.1,
            order=8,
            guard_order=4,
            max_competing_events=2,
        )

    assert "finite_time_event_order_requires_split" in (
        exc_info.value.missing_obligations
    )
    assert (
        "ks_competing_close_binary_requires_next_regularized_chart_or_split"
        in exc_info.value.missing_obligations
    )
    assert "spatial_collision_policy_scope" in (
        exc_info.value.missing_obligations
    )


def test_spatial_ks_competing_handoff_can_consume_projected_close_pair_suffix(
    monkeypatch,
):
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
    initial = spatial_to_ks_binary_chart(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )
    state = interval_spatial_ks_binary_chart_state_from_point(initial)
    third_offset = state.third_offset.copy()
    third_offset[0] = FloatInterval(1.0e-4, 5.0e-4)
    state = replace(state, third_offset=third_offset)
    calls = []

    class FakePartition:
        certified = True
        well_formed = True

    class FakeBranchEvaluation:
        def __init__(self, target_box, target_interval, partition):
            self.branch_partition = partition
            self.branch_atlases = ()
            self.initial_state_interval = target_box
            self.initial_state_interval_union = ()
            self.target_state_interval = target_box
            self.target_state_interval_union = (target_box,)
            self.initial_state = np.zeros(18)
            self.final_state = np.zeros(18)
            self.target_time_interval = target_interval

        def target_state_contains(self, state):
            return interval_array_contains_point(
                self.target_state_interval,
                np.asarray(state, dtype=float).reshape(-1),
            )

    def fake_next_chart(*_args, **_kwargs):
        raise ValueError("force projected close-pair suffix")

    def fake_partition(state_interval, *, binary_distance_threshold, **_kwargs):
        calls.append(("partition", len(state_interval), binary_distance_threshold))
        return FakePartition()

    def fake_branch_union(partition, masses, *, target_time, source, **_kwargs):
        calls.append(("branch_union", partition, target_time, source))
        target_interval = target_time if isinstance(target_time, FloatInterval) else FloatInterval.point(float(target_time))
        target_box = np.asarray([FloatInterval.point(0.0) for _ in range(18)], dtype=object)
        chart = ValidatedChart(
            chart_id="spatial_branch_union_0",
            chart_type="spatial_branch_union",
            source=source,
            parameter_name="t",
            parameter_interval=FloatInterval(0.0, target_interval.upper),
            physical_time_interval=FloatInterval(0.0, target_interval.upper),
            dynamics_certified=True,
            residual_certified=True,
            projection_certified=True,
            invariants_certified=True,
            tail_certified=True,
            tail_bound=0.0,
        )
        invariants = GlobalInvariantLedger(
            center_of_mass_certified=True,
            linear_momentum_certified=True,
            angular_momentum_certified=True,
            energy_certified=True,
            certified_chart_count=1,
            expected_chart_count=1,
        )
        residual_budget = NewtonResidualLedger(
            certified=True,
            certified_chart_count=1,
            expected_chart_count=1,
        )
        tail_budget = TailBudgetLedger(
            local_tail_bound=0.0,
            max_step_tail_bound=0.0,
            certified=True,
        )
        collision_policy = CollisionPolicyWitness(
            policy_id="fake_close_pair_branch_union",
            binary_policy="fake",
            total_collision_policy="fake",
            triple_collision_status="fake",
            triple_collision_reason="fake proof-certified suffix for routing test",
            certified=True,
        )
        proof_ledger = ProofLedger(
            entries=(
                ProofLedgerEntry("mass_domain", True, source),
                ProofLedgerEntry("initial_state_domain", True, source),
                ProofLedgerEntry("target_time_domain", True, source),
                ProofLedgerEntry("finite_time_physical_targeting", True, source),
                ProofLedgerEntry("projection_ledger", True, source),
                ProofLedgerEntry("newton_residuals", True, source),
                ProofLedgerEntry("invariant_ledger", True, source),
                ProofLedgerEntry("local_tail_budget", True, source),
                ProofLedgerEntry("collision_policy", True, source),
                ProofLedgerEntry("simultaneous_close_pair_partition", True, source),
                ProofLedgerEntry("finite_time_branch_union_consumption", True, source),
            )
        )
        return ValidatedAtlasSolution(
            masses=tuple(float(mass) for mass in np.asarray(masses, dtype=float)),
            target_time=0.5 * (target_interval.lower + target_interval.upper),
            initial_state_interval=target_box,
            charts=(chart,),
            transitions=(),
            invariants=invariants,
            tail_budget=tail_budget,
            residual_budget=residual_budget,
            collision_policy=collision_policy,
            proof_ledger=proof_ledger,
            evaluation=FakeBranchEvaluation(target_box, target_interval, partition),
        )

    monkeypatch.setattr(
        validated_atlas_module,
        "validated_atlas_from_spatial_ks_binary_chart",
        fake_next_chart,
    )
    monkeypatch.setattr(
        validated_atlas_module,
        "certify_simultaneous_close_pair_partition",
        fake_partition,
    )
    monkeypatch.setattr(
        validated_atlas_module,
        "validated_atlas_from_spatial_close_pair_branch_partition",
        fake_branch_union,
    )

    atlas = validated_atlas_module.validated_atlas_from_spatial_ks_competing_binary_handoff(
        state,
        competing_pair=(1, 2),
        enter_distance=0.045,
        entry_s_upper=0.1,
        branch="positive_x",
        next_s_endpoint=0.1,
        target_time_after_ks_start_interval=FloatInterval.point(2.7e-4),
        retained_order=8,
        guard_order=4,
        competing_pair_min_distance_required=0.045,
    )

    assert atlas.proof_certified
    assert any(call[0] == "partition" and call[2] == 0.045 for call in calls)
    assert any(call[0] == "branch_union" for call in calls)
    assert [chart.chart_type for chart in atlas.charts] == [
        "spatial_ks_binary",
        "spatial_branch_union",
    ]
    assert any(
        entry.name == "spatial_ks_prefix_close_pair_branch_union_transition"
        and entry.certified
        for entry in atlas.proof_ledger.entries
    )
    assert any(
        entry.name == "finite_time_branch_union_consumption" and entry.certified
        for entry in atlas.proof_ledger.entries
    )


def test_finite_time_regime_classifier_recovers_nested_prefix_close_pair_partition(
    monkeypatch,
):
    class FakePartition:
        certified = True
        well_formed = True
        binary_distance_threshold = 0.045
        branches = (object(),)
        missing_obligations = ()

    partition = FakePartition()
    trace = SimpleNamespace(
        certified=True,
        selected_route_id="spatial_ks_prefix_close_pair_branch_union",
        attempts=(
            SimpleNamespace(
                route_id="spatial_ks_prefix_close_pair_branch_union",
                branch_partition=None,
            ),
        ),
    )
    atlas = SimpleNamespace(
        proof_certified=True,
        selector_trace=trace,
        evaluation=SimpleNamespace(
            branch_union_evaluation=SimpleNamespace(branch_partition=partition),
        ),
        charts=(
            SimpleNamespace(chart_type="spatial_ks_binary"),
            SimpleNamespace(chart_type="spatial_branch_union"),
        ),
    )

    monkeypatch.setattr(
        finite_time_regime_module,
        "evaluate_unrestricted_solution",
        lambda *_args, **_kwargs: atlas,
    )

    classification = finite_time_regime_module.classify_finite_time_regime(
        np.ones(3),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        np.zeros((3, 3)),
        0.1,
    )

    assert not classification.certified
    assert not classification.atlas_certified
    assert classification.branch_partition is partition
    assert classification.branch_partition_certified
    assert classification.selected_route_id == "spatial_ks_prefix_close_pair_branch_union"
    assert "finite_time_validated_atlas_type" in classification.missing_obligations
    assert "finite_time_validated_atlas" in classification.missing_obligations
    assert "simultaneous_close_pair_partition" not in (
        classification.missing_obligations
    )


def test_finite_time_regime_classifier_mirrors_consumed_branch_union_ledger(
    monkeypatch,
):
    masses = np.ones(3)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    velocities = np.zeros((3, 3))
    flat_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
    interval_state = np.asarray(
        [FloatInterval.point(value) for value in flat_state],
        dtype=object,
    )
    chart = ValidatedChart(
        chart_id="spatial_branch_union_0",
        chart_type="spatial_branch_union",
        source="test",
        parameter_name="t",
        parameter_interval=FloatInterval(0.0, 0.1),
        physical_time_interval=FloatInterval(0.0, 0.1),
        dynamics_certified=True,
        residual_certified=True,
        projection_certified=True,
        invariants_certified=True,
        tail_certified=True,
        tail_bound=0.0,
    )
    ledgers = dict(
        invariants=GlobalInvariantLedger(
            center_of_mass_certified=True,
            linear_momentum_certified=True,
            angular_momentum_certified=True,
            energy_certified=True,
            certified_chart_count=1,
            expected_chart_count=1,
        ),
        tail_budget=TailBudgetLedger(
            local_tail_bound=0.0,
            max_step_tail_bound=0.0,
            certified=True,
        ),
        residual_budget=NewtonResidualLedger(
            certified=True,
            certified_chart_count=1,
            expected_chart_count=1,
        ),
    )
    collision_policy = CollisionPolicyWitness(
        policy_id="test_branch_union",
        binary_policy="spatial_ks_branch_union",
        total_collision_policy="local_branch_union_excludes_total_collision",
        triple_collision_status="locally_excluded",
        triple_collision_reason="proof-ledger branch-union test fixture",
        certified=True,
    )
    evaluation = SimpleNamespace(
        initial_state=flat_state,
        final_state=flat_state,
        target_state_interval=interval_state,
        target_state_interval_union=(interval_state,),
    )
    base_entries = (
        ProofLedgerEntry("mass_domain", True, "test"),
        ProofLedgerEntry("initial_state_domain", True, "test"),
        ProofLedgerEntry("target_time_domain", True, "test"),
        ProofLedgerEntry("finite_time_physical_targeting", True, "test"),
        ProofLedgerEntry("projection_ledger", True, "test"),
        ProofLedgerEntry("newton_residuals", True, "test"),
        ProofLedgerEntry("invariant_ledger", True, "test"),
        ProofLedgerEntry("local_tail_budget", True, "test"),
        ProofLedgerEntry("collision_policy", True, "test"),
        ProofLedgerEntry("simultaneous_close_pair_partition", True, "test"),
        ProofLedgerEntry("finite_time_branch_union_consumption", True, "test"),
    )
    atlas = ValidatedAtlasSolution(
        masses=tuple(float(mass) for mass in masses),
        target_time=0.1,
        initial_state_interval=interval_state,
        charts=(chart,),
        transitions=(),
        proof_ledger=ProofLedger(entries=base_entries),
        evaluation=evaluation,
        collision_policy=collision_policy,
        **ledgers,
    )
    trace = FiniteTimeChartSelectorTrace(
        selected_route_id="spatial_branch_union",
        attempts=(
            FiniteTimeChartSelectorAttempt(
                route_id="spatial_branch_union",
                attempted=True,
                selected=True,
                certified=True,
                reason="test atlas consumed local branch union",
            ),
        ),
        atlas_binding_token=validated_atlas_module.finite_time_selector_trace_binding_token(
            atlas,
        ),
    )
    atlas = replace(
        atlas,
        selector_trace=trace,
        proof_ledger=ProofLedger(
            entries=(
                *base_entries,
                ProofLedgerEntry("finite_time_chart_selector", True, "test"),
            )
        ),
    )
    assert atlas.proof_certified

    monkeypatch.setattr(
        finite_time_regime_module,
        "evaluate_unrestricted_solution",
        lambda *_args, **_kwargs: atlas,
    )

    classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        0.1,
    )

    assert classification.certified
    details = {obligation.obligation: obligation for obligation in classification.obligations}
    assert details["simultaneous_close_pair_partition"].certified
    assert details["finite_time_branch_union_consumption"].certified
    assert "ValidatedAtlasSolution.proof_ledger" not in (
        details["finite_time_branch_union_consumption"].source
    )
    assert "recursive_set_valued_branch_partition_consumption" not in (
        classification.missing_obligations
    )

    event_chart = replace(
        chart,
        chart_id="spatial_ks_event_order_branch_union_0",
        chart_type="spatial_ks_event_order_branch_union",
    )
    event_entries = (
        *base_entries[:9],
        ProofLedgerEntry("ks_event_order_partition", True, "test"),
        ProofLedgerEntry(
            "finite_time_event_order_branch_union_consumption",
            True,
            "test",
        ),
    )
    event_atlas = replace(
        atlas,
        charts=(event_chart,),
        proof_ledger=ProofLedger(entries=event_entries),
        selector_trace=None,
    )
    event_trace = FiniteTimeChartSelectorTrace(
        selected_route_id="spatial_ks_event_order_branch_union",
        attempts=(
            FiniteTimeChartSelectorAttempt(
                route_id="spatial_ks_event_order_branch_union",
                attempted=True,
                selected=True,
                certified=True,
                reason="test atlas consumed local event-order partition",
            ),
        ),
        atlas_binding_token=validated_atlas_module.finite_time_selector_trace_binding_token(
            event_atlas,
        ),
    )
    event_atlas = replace(
        event_atlas,
        selector_trace=event_trace,
        proof_ledger=ProofLedger(
            entries=(
                *event_entries,
                ProofLedgerEntry("finite_time_chart_selector", True, "test"),
            )
        ),
    )
    assert event_atlas.proof_certified
    monkeypatch.setattr(
        finite_time_regime_module,
        "evaluate_unrestricted_solution",
        lambda *_args, **_kwargs: event_atlas,
    )

    event_classification = classify_finite_time_regime(
        masses,
        positions,
        velocities,
        0.1,
    )

    assert event_classification.certified
    event_details = {
        obligation.obligation: obligation
        for obligation in event_classification.obligations
    }
    assert event_details["ks_event_order_partition"].certified
    assert event_details["finite_time_event_order_branch_union_consumption"].certified
    assert "event_order_partition_consumption_theorem" not in (
        event_classification.missing_obligations
    )


def test_spatial_ks_loop_composes_later_event_order_partition_with_prefix(
    monkeypatch,
):
    class FakePartition:
        certified = True

    class FakeAtlas:
        proof_certified = True

    suffix_atlas = FakeAtlas()
    final_atlas = FakeAtlas()
    partition = FakePartition()
    prefix_step = general_solution_module.FiniteTimeKSLoopProgressStep(
        index=0,
        active_pair=(0, 1),
        remaining_time_interval=FloatInterval.point(0.2),
        event_set_certificate=object(),
        decision="unique_competing_event",
        event_id="spatial_ks_competing_binary_entry:0-2",
        event_pair=(0, 2),
        event_time_interval=FloatInterval.point(0.1),
        next_branch="positive_x",
    )
    blocked_step = general_solution_module.FiniteTimeKSLoopProgressStep(
        index=1,
        active_pair=(0, 2),
        remaining_time_interval=FloatInterval.point(0.1),
        event_set_certificate=object(),
        decision="event_order_blocked",
        event_order_partition_certificate=partition,
        missing_obligations=(
            "finite_time_event_order_partition_requires_branch_union_constructor",
        ),
    )
    loop_progress = general_solution_module.FiniteTimeKSLoopProgressCertificate(
        initial_pair=(0, 1),
        target_time_after_start=FloatInterval.point(0.2),
        max_competing_events=3,
        steps=(prefix_step, blocked_step),
        missing_obligations=(
            "finite_time_event_order_partition_requires_branch_union_constructor",
        ),
    )
    calls = []

    def build_suffix(received_partition, **kwargs):
        calls.append(("suffix", received_partition, kwargs["max_competing_repeats"]))
        return suffix_atlas

    def derive_prefix_states(initial_state, prefix_steps, **_kwargs):
        calls.append(("prefix_states", initial_state, tuple(prefix_steps)))
        return ("prefix_state",)

    def wrap_prefix(prefix_state, **kwargs):
        calls.append(
            (
                "wrap",
                prefix_state,
                kwargs["next_atlas"],
                kwargs["competing_pair"],
                kwargs["branch"],
                kwargs["target_time_after_ks_start_interval"],
            )
        )
        return final_atlas

    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_ks_event_order_partition",
        build_suffix,
    )
    monkeypatch.setattr(
        general_solution_module,
        "_derive_spatial_ks_loop_prefix_states",
        derive_prefix_states,
    )
    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_ks_competing_binary_handoff_to_atlas",
        wrap_prefix,
    )

    result = general_solution_module._try_evaluate_spatial_ks_event_order_branch_union(
        "initial_state",
        loop_progress,
        order=8,
        guard_order=4,
        binary_distance_threshold=0.045,
        s_upper=0.1,
    )

    assert result is final_atlas
    assert calls[0] == ("suffix", partition, 1)
    assert calls[1] == ("prefix_states", "initial_state", (prefix_step,))
    assert calls[2][0:5] == (
        "wrap",
        "prefix_state",
        suffix_atlas,
        (0, 2),
        "positive_x",
    )
    assert calls[2][5] == FloatInterval.point(0.2)


def test_ordinary_entry_spatial_ks_loop_consumes_event_order_branch_union(monkeypatch):
    class FakeAtlas:
        proof_certified = True

    suffix_atlas = FakeAtlas()
    final_atlas = FakeAtlas()
    partition = SimpleNamespace(certified=True)
    loop_progress = general_solution_module.FiniteTimeKSLoopProgressCertificate(
        initial_pair=(0, 1),
        target_time_after_start=FloatInterval.point(0.2),
        max_competing_events=2,
        steps=(
            general_solution_module.FiniteTimeKSLoopProgressStep(
                index=0,
                active_pair=(0, 1),
                remaining_time_interval=FloatInterval.point(0.2),
                event_set_certificate=object(),
                decision="event_order_blocked",
                event_order_partition_certificate=partition,
                missing_obligations=(
                    "finite_time_event_order_partition_requires_branch_union_constructor",
                ),
            ),
        ),
        missing_obligations=(
            "finite_time_event_order_partition_requires_branch_union_constructor",
        ),
    )
    calls = []

    monkeypatch.setattr(
        general_solution_module,
        "_derive_certified_spatial_ordinary_entry_ks_state",
        lambda *_args, **_kwargs: ("entry_ks_state", 0.05),
    )
    monkeypatch.setattr(
        general_solution_module,
        "_certify_spatial_ks_competing_loop_progress",
        lambda *_args, **_kwargs: loop_progress,
    )

    def fake_event_order_branch_union(initial_ks_state, received_loop_progress, **kwargs):
        calls.append(("branch_union", initial_ks_state, received_loop_progress, kwargs))
        return suffix_atlas

    def fake_ordinary_prefix(state_interval, masses, **kwargs):
        calls.append(("ordinary_prefix", state_interval, masses, kwargs))
        return final_atlas

    monkeypatch.setattr(
        general_solution_module,
        "_try_evaluate_spatial_ks_event_order_branch_union",
        fake_event_order_branch_union,
    )
    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_ordinary_ks_suffix_atlas",
        fake_ordinary_prefix,
    )
    monkeypatch.setattr(
        general_solution_module,
        "_attach_spatial_ks_loop_progress",
        lambda atlas, *_args, **_kwargs: atlas,
    )

    result = general_solution_module._try_evaluate_auto_spatial_ordinary_ks_competing_handoff(
        np.ones(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        0.25,
        state_interval=tuple((0.0, 0.0) for _ in range(18)),
        selected_pair=(0, 1),
        selected_enter_distance=0.02,
        selected_entry_time_upper=0.1,
        selected_branch="positive_x",
        binary_distance_threshold=0.045,
        s_upper=0.1,
        order=8,
        guard_order=4,
    )

    assert result is final_atlas
    assert calls[0][0:3] == ("branch_union", "entry_ks_state", loop_progress)
    assert calls[1][0] == "ordinary_prefix"
    assert calls[1][3]["ks_suffix_atlas"] is suffix_atlas
    assert calls[1][3]["target_time"] == pytest.approx(0.25)
    assert calls[1][3]["pair"] == (0, 1)


def test_initial_spatial_ks_competing_handoff_backtracks_across_certified_candidates(
    monkeypatch,
):
    competing_obligation = "ks_competing_close_binary_requires_next_regularized_chart_or_split"

    class FakeKSState:
        certified = True

    class FakeAtlas:
        def __init__(self, *, proof_certified, missing=()):
            self.proof_certified = bool(proof_certified)
            self.missing_certification_obligations = tuple(missing)
            self.proof_ledger = ProofLedger(entries=())

    def derive_candidates(state, **_kwargs):
        if isinstance(state, FakeKSState):
            return (
                (0.1, (0, 1), "first_branch", 0.02, 0.1, 0.1),
                (0.2, (0, 2), "second_branch", 0.02, 0.1, 0.1),
            )
        raise AssertionError(f"unexpected selector state {state!r}")

    tried_pairs = []

    def build_handoff(_initial_state, *, competing_pair, **_kwargs):
        tried_pairs.append(competing_pair)
        if competing_pair == (0, 1):
            return FakeAtlas(
                proof_certified=False,
                missing=(competing_obligation,),
            )
        if competing_pair == (0, 2):
            return FakeAtlas(proof_certified=True)
        raise AssertionError(f"unexpected competing pair {competing_pair!r}")

    monkeypatch.setattr(
        general_solution_module,
        "spatial_interval_to_ks_binary_chart_state",
        lambda *_args, **_kwargs: FakeKSState(),
    )

    monkeypatch.setattr(
        general_solution_module,
        "_derive_certified_spatial_ks_competing_selectors_from_ks_state",
        derive_candidates,
    )
    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_ks_competing_binary_handoff",
        build_handoff,
    )
    monkeypatch.setattr(
        general_solution_module,
        "_certify_spatial_ks_competing_loop_progress",
        lambda *_args, **_kwargs: type(
            "FakeLoopProgress",
            (),
            {
                "certified": True,
                "event_count": 1,
                "required_recursive_repeat_budget": 0,
                "steps": (object(),),
            },
        )(),
    )
    monkeypatch.setattr(
        general_solution_module,
        "_attach_spatial_ks_loop_progress",
        lambda atlas, _loop_progress, **_kwargs: atlas,
    )

    atlas = general_solution_module._try_evaluate_auto_initial_spatial_ks_competing_handoff(
        np.ones(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        1.0,
        state_interval=tuple((0.0, 0.0) for _ in range(18)),
        selected_pair=(0, 1),
        selected_branch="positive_x",
        binary_distance_threshold=0.02,
        s_upper=0.1,
        order=8,
        guard_order=2,
    )

    assert atlas is not None
    assert atlas.proof_certified
    assert tried_pairs == [(0, 1), (0, 2)]


def test_unrestricted_solution_validated_atlas_explicit_ordinary_ks_repeats_for_competing_binary():
    masses = np.array([0.8, 1.2, 1.7])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.151, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [-120.0, 0.0, 0.0],
        ]
    )
    target_time = 9.5e-4

    solution = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=24,
        guard_order=8,
        binary_distance_threshold=1.8e-2,
        spatial_binary_pair=(0, 1),
        spatial_binary_enter_distance=2.0e-2,
        spatial_binary_entry_time_upper=1.0e-2,
        spatial_binary_exit_rho=4.0e-2,
        spatial_binary_s_upper=1.0,
    )

    ordinary_point = construct_taylor_solution(positions, velocities, masses, order=32)
    entry_time = solution.evaluation.entry_event_certificate.root
    entry_ks = spatial_to_ks_binary_chart(
        ordinary_point.positions_at(entry_time),
        ordinary_point.velocities_at(entry_time),
        masses,
        pair=(0, 1),
    )
    first_ks_point = construct_spatial_ks_binary_taylor_solution(entry_ks, order=32)
    competing_s = (
        solution.evaluation.ks_competing_evaluation.competing_entry_event_certificate.root
    )
    event_positions, event_velocities = ks_binary_chart_to_spatial(
        first_ks_point.state_at(competing_s)
    )
    next_ks = spatial_to_ks_binary_chart(
        event_positions,
        event_velocities,
        masses,
        pair=(1, 2),
    )
    next_ks_point = construct_spatial_ks_binary_taylor_solution(next_ks, order=32)
    target_s = (
        solution.evaluation.ks_competing_evaluation.next_ks_evaluation.endpoint_projection.s_value
    )
    target_positions, target_velocities = ks_binary_chart_to_spatial(
        next_ks_point.state_at(target_s)
    )
    reference = np.concatenate(
        [target_positions.reshape(-1), target_velocities.reshape(-1)]
    )

    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == (
        "SpatialOrdinaryKSCompetingHandoffEvaluation"
    )
    assert solution.proof_certified
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "explicit_spatial_ks"
    assert solution.selector_trace.certified
    assert [chart.chart_type for chart in solution.charts] == [
        "spatial_ordinary_taylor_before_ks",
        "spatial_ks_binary",
        "spatial_ks_binary",
    ]
    assert [transition.transition_type for transition in solution.transitions] == [
        "spatial_ordinary_to_ks_decreasing_distance_entry",
        "spatial_ks_to_ks_competing_binary_entry",
    ]
    assert solution.evaluation.ks_competing_evaluation.next_ks_state.pair == (1, 2)
    loop_progress = solution.evaluation.loop_progress_certificate
    assert loop_progress is not None
    assert loop_progress.certified
    assert loop_progress.event_count == 1
    assert loop_progress.required_recursive_repeat_budget == 0
    assert [step.decision for step in loop_progress.steps] == [
        "unique_competing_event",
        "target_before_all_events",
    ]
    assert any(
        entry.name == "finite_time_atlas_loop_progress" and entry.certified
        for entry in solution.proof_ledger.entries
    )
    assert solution.target_state_contains(reference)
    assert solution.missing_certification_obligations == ()


def test_unrestricted_solution_validated_atlas_auto_selects_backward_spatial_ks_by_reversal():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, forward_velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.04)
    )
    velocities = -forward_velocities
    target_time = -5.0e-5

    solution = evaluate_unrestricted_solution(
        initial.masses,
        positions,
        velocities,
        target_time,
        method="validated_atlas",
        order=24,
        guard_order=6,
        binary_distance_threshold=1.0e-2,
    )
    forward_evaluation = solution.evaluation.forward_evaluation
    assert solution.__class__.__name__ == "ValidatedAtlasSolution"
    assert solution.evaluation.__class__.__name__ == "TimeReversedValidatedEvaluation"
    assert forward_evaluation.__class__.__name__ == "SpatialKSValidatedEvaluation"
    assert solution.proof_certified
    assert solution.target_time == pytest.approx(target_time)
    assert solution.selector_trace is not None
    assert solution.selector_trace.selected_route_id == "auto_spatial_ks"
    assert solution.selector_trace.certified
    assert [chart.chart_type for chart in solution.charts] == ["spatial_ks_binary"]
    assert all(chart.physical_time_interval.upper <= 0.0 for chart in solution.charts)
    lifted_initial = spatial_to_ks_binary_chart(
        positions,
        forward_velocities,
        initial.masses,
        pair=(0, 1),
    )
    ks_point_solution = construct_spatial_ks_binary_taylor_solution(lifted_initial, order=32)
    target_positions, forward_target_velocities = ks_binary_chart_to_spatial(
        ks_point_solution.state_at(forward_evaluation.endpoint_projection.s_value)
    )
    reference = np.concatenate(
        [target_positions.reshape(-1), (-forward_target_velocities).reshape(-1)]
    )
    assert solution.target_state_contains(reference)
    assert solution.collision_policy.policy_id.startswith("time_reversed_")
    assert "time_reversal_symmetry" not in (
        solution.proof_ledger.missing_required_obligations
    )
    assert solution.proof_ledger.missing_required_obligations == ()


def test_explicit_spatial_ks_negative_time_preserves_competing_pair_distance_requirement(monkeypatch):
    calls = []
    forward_atlas = SimpleNamespace(charts=(), proof_certified=True)

    def fake_handoff(*_args, **kwargs):
        calls.append(kwargs)
        return forward_atlas

    def fake_time_reverse(atlas, *, source):
        return SimpleNamespace(forward_atlas=atlas, source=source)

    monkeypatch.setattr(
        general_solution_module,
        "validated_atlas_from_spatial_ordinary_ks_handoff",
        fake_handoff,
    )
    monkeypatch.setattr(
        general_solution_module,
        "time_reverse_validated_atlas_solution",
        fake_time_reverse,
    )

    result = general_solution_module.evaluate_spatial_ks_validated_atlas_solution(
        np.ones(3),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -0.1, 0.0],
            ]
        ),
        -0.01,
        pair=(0, 1),
        enter_distance=0.02,
        entry_time_upper=0.1,
        branch="positive_x",
        exit_rho=0.03,
        s_upper=0.1,
        order=6,
        guard_order=2,
        competing_pair_min_distance_required=0.045,
    )

    assert result.forward_atlas is forward_atlas
    assert result.source == "evaluate_unrestricted_solution_time_reversal"
    assert calls
    assert calls[0]["competing_pair_min_distance_required"] == pytest.approx(0.045)


def test_unrestricted_solution_validated_atlas_stops_when_spatial_ks_route_fails():
    initial = _exact_spatial_ks_collision_state()
    pre_collision_solution = construct_spatial_ks_binary_taylor_solution(initial, order=40)
    positions, velocities = ks_binary_chart_to_spatial(
        pre_collision_solution.state_at(-0.04)
    )

    with pytest.raises(RuntimeError, match="required regularized route failed"):
        evaluate_unrestricted_solution(
            initial.masses,
            positions,
            velocities,
            2.0e-4,
            method="validated_atlas",
            order=10,
            guard_order=1,
            binary_distance_threshold=1.0e-2,
            max_compact_step=0.01,
            target_bisections=30,
        )


def test_unrestricted_solution_validated_atlas_stops_ambiguous_close_spatial_binary():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.03, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ]
    )

    with pytest.raises(RuntimeError, match="spatial_close_binary_guard"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            1.0,
            method="validated_atlas",
            order=8,
            binary_distance_threshold=0.08,
            max_compact_step=0.01,
            target_bisections=20,
        )


def test_unrestricted_solution_validated_atlas_stops_backward_close_spatial_binary():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.8, 0.5, 0.0],
        ]
    )
    velocities = np.array(
        [
            [-0.03, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(RuntimeError, match="spatial_close_binary_guard"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            -1.0,
            method="validated_atlas",
            order=8,
            binary_distance_threshold=0.08,
            max_compact_step=0.01,
            target_bisections=20,
        )


def test_unrestricted_solution_validated_atlas_blocks_interval_close_spatial_binary():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.11, 0.0, 0.0],
            [1.0, 0.7, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.02, 0.0, 0.0],
            [-0.02, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(RuntimeError, match="spatial_interval_close_binary"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            0.25,
            method="validated_atlas",
            initial_radius=0.02,
            order=8,
            binary_distance_threshold=0.08,
            max_compact_step=0.01,
            target_bisections=20,
        )


def test_unrestricted_solution_validated_atlas_stops_ambiguous_planar_hybrid():
    masses = np.array([1.0, 1.0, 0.7])
    positions = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.4]])
    velocities = np.zeros((3, 2))

    with pytest.raises(RuntimeError, match="planar_hybrid"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            1.0e-5,
            method="validated_atlas",
            initial_radius=1.0e-3,
            order=10,
            binary_distance_threshold=0.01,
            binary_exit_distance=0.02,
            max_binary_s_step=1.0e-3,
            max_compact_step=0.01,
            target_bisections=20,
        )


def test_unrestricted_solution_validated_atlas_rejects_incomplete_spatial_ks_options():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="spatial KS validated atlas requires"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            1.0e-5,
            method="validated_atlas",
            spatial_binary_pair=(0, 1),
        )


def test_unrestricted_solution_entrypoint_certifies_close_approach_noncollision():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.07, 0.0, 0.0],
            [1.0, 0.7, -0.2],
        ]
    )
    velocities = np.array(
        [
            [-0.02, 0.04, 0.0],
            [0.03, -0.01, 0.02],
            [-0.01, 0.02, -0.03],
        ]
    )
    target_time = 1e-5
    pair_distances = [
        np.linalg.norm(positions[j] - positions[i])
        for i in range(3)
        for j in range(i + 1, 3)
    ]

    evaluation = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        initial_radius=0.0,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.01,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=48,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert 0.0 < min(pair_distances) < 0.1
    assert evaluation.proof_certified
    assert evaluation.target_state_contains(reference)
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.proof_certified
    assert evaluation.local_tail_bound > 0.0


def test_unrestricted_solution_entrypoint_certifies_tighter_close_approach_with_sundman_method():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.0, 0.7, -0.2],
        ]
    )
    velocities = np.array(
        [
            [-0.02, 0.04, 0.0],
            [0.03, -0.01, 0.02],
            [-0.01, 0.02, -0.03],
        ]
    )
    target_time = 1e-5

    evaluation = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method="sundman",
        initial_radius=0.0,
        order=10,
        max_s_step=0.001,
        target_bisections=48,
        tail_certificate_mode="cauchy",
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.proof_certified
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.proof_certified
    assert evaluation.target_state_contains(reference)
    assert evaluation.local_tail_bound > 0.0


def test_unrestricted_solution_entrypoint_auto_falls_back_for_tighter_close_approach():
    masses = np.array([1.0, 0.8, 1.3])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.0, 0.7, -0.2],
        ]
    )
    velocities = np.array(
        [
            [-0.02, 0.04, 0.0],
            [0.03, -0.01, 0.02],
            [-0.01, 0.02, -0.03],
        ]
    )
    target_time = 1e-5

    evaluation = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        initial_radius=0.0,
        order=10,
        max_s_step=0.001,
        max_compact_step=0.01,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=48,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.__class__.__name__ == "ReducedSundmanEvaluation"
    assert evaluation.proof_certified
    assert evaluation.target_state_contains(reference)


def test_unrestricted_solution_entrypoint_rejects_unknown_method():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="method must be"):
        evaluate_unrestricted_solution(
            masses,
            positions,
            velocities,
            0.0,
            method="raw_newton",
        )


def test_compact_time_reduced_compactified_sundman_evaluator_contains_reference_path():
    masses, positions, velocities = _general_initial_data()
    reduced = reduce_to_center_of_mass_frame(positions, velocities, masses)
    target_time = _target_time_from_reduced_s(reduced, 0.006)
    compact_parameter = compact_parameter_from_physical_time(target_time, rate=2.0)

    evaluation = evaluate_compact_time_reduced_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        compact_parameter,
        time_rate=2.0,
        initial_radius=1e-15,
        order=10,
        sundman_rate=1.15,
        max_compact_step=0.025,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=42,
    )
    reference = integrate_reference(positions, velocities, masses, evaluation.target_time)

    assert evaluation.compactification_certificate.certified
    assert evaluation.time_taylor_certificate.certified
    assert evaluation.target_time == pytest.approx(target_time)
    assert evaluation.proof_certified
    assert evaluation.evaluation.reduced_target is not None
    assert evaluation.evaluation.reduced_target.proof_certified
    assert evaluation.evaluation.triple_collision_excluded
    assert evaluation.evaluation.triple_collision_status == "excluded"
    assert evaluation.target_state_contains(reference)


def test_reduced_compactified_sundman_evaluator_marks_zero_angular_momentum_branch_undecided():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    velocities = np.zeros_like(positions)
    target_time = 1e-3

    evaluation = evaluate_reduced_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        target_time,
        initial_radius=0.0,
        order=10,
        sundman_rate=1.1,
        max_compact_step=0.02,
        radius_fraction=0.2,
        guard_order=6,
        target_bisections=36,
    )
    reference = integrate_reference(positions, velocities, masses, target_time)

    assert evaluation.proof_certified
    assert not evaluation.triple_collision_excluded
    assert evaluation.triple_collision_status == "undecided"
    assert evaluation.triple_collision_exclusion_reason == "centered angular momentum interval contains zero"
    assert evaluation.reduced_target is not None
    assert evaluation.reduced_target.triple_collision_undecided
    assert evaluation.target_state_contains(reference)


def test_reduced_sundman_evaluator_zero_time_is_the_initial_identity():
    masses, positions, velocities = _general_initial_data()
    evaluation = evaluate_reduced_sundman_solution(
        positions,
        velocities,
        masses,
        0.0,
        initial_radius=1e-12,
    )
    initial_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    assert evaluation.reduced_target is None
    assert evaluation.reduction_certified
    assert evaluation.proof_certified
    assert evaluation.target_time_certified
    assert evaluation.target_state_contains(initial_state)
    assert interval_array_contains_point(evaluation.target_position_interval, positions)
    assert interval_array_contains_point(evaluation.target_velocity_interval, velocities)


def test_reduced_compactified_sundman_evaluator_zero_time_is_the_initial_identity():
    masses, positions, velocities = _general_initial_data()
    evaluation = evaluate_reduced_compactified_sundman_solution(
        positions,
        velocities,
        masses,
        0.0,
        initial_radius=1e-12,
    )
    initial_state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    assert evaluation.reduced_target is None
    assert evaluation.reduction_certified
    assert evaluation.proof_certified
    assert evaluation.target_time_certified
    assert evaluation.target_compact_parameter_interval is None
    assert evaluation.target_state_contains(initial_state)
    assert interval_array_contains_point(evaluation.target_position_interval, positions)
    assert interval_array_contains_point(evaluation.target_velocity_interval, velocities)


def test_reduced_sundman_evaluator_rejects_uncertified_collision_boxes():
    masses, positions, velocities = _general_initial_data()
    positions[1] = positions[0] + np.array([1e-4, 0.0, 0.0])

    with pytest.raises(ValueError, match="non-collision"):
        evaluate_reduced_sundman_solution(
            positions,
            velocities,
            masses,
            0.001,
            initial_radius=1e-3,
            order=8,
            max_s_step=0.01,
        )


def test_reduced_compactified_sundman_evaluator_rejects_uncertified_collision_boxes():
    masses, positions, velocities = _general_initial_data()
    positions[1] = positions[0] + np.array([1e-4, 0.0, 0.0])

    with pytest.raises(ValueError, match="non-collision"):
        evaluate_reduced_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            0.001,
            initial_radius=1e-3,
            order=8,
            max_compact_step=0.01,
        )


def test_compactified_reduced_sundman_evaluator_rejects_time_boundary():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="strictly between"):
        evaluate_compactified_reduced_sundman_solution(
            positions,
            velocities,
            masses,
            1.0,
            order=8,
        )


def test_compact_time_reduced_compactified_sundman_evaluator_rejects_time_boundary():
    masses, positions, velocities = _general_initial_data()

    with pytest.raises(ValueError, match="strictly between"):
        evaluate_compact_time_reduced_compactified_sundman_solution(
            positions,
            velocities,
            masses,
            1.0,
            order=8,
        )
