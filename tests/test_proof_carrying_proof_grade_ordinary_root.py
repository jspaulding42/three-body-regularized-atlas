from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest

from three_body_symmetry.certificate_checker import check_validated_ordinary_ivp_chart
from three_body_symmetry.certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
)
from three_body_symmetry.planar_chain_review_artifact import (
    strict_load_raw_planar_chain,
)
from three_body_symmetry.proof_carrying_proof_grade_ordinary_root import (
    PROOF_GRADE_VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS,
    ProofGradeValidatedOrdinaryRootResult,
    check_proof_grade_validated_ordinary_root,
)


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_RAW = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain" / "success.raw.json"


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True


class _BindingSubclass(InitialValueProblemBindingCertificate):
    pass


class _TubeSubclass(OrdinaryAposterioriTubeCertificate):
    pass


class _ChartSubclass(OrdinaryTaylorChartCertificate):
    pass


class _ResultEqualitySpoof(ProofGradeValidatedOrdinaryRootResult):
    def __eq__(self, other: object) -> bool:
        return True


@lru_cache(maxsize=1)
def _raw_root() -> tuple[
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    OrdinaryTaylorChartCertificate,
]:
    raw = strict_load_raw_planar_chain(SUCCESS_RAW)
    return raw.root_binding, raw.initial_tube, raw.initial_chart


def _check(
    binding: InitialValueProblemBindingCertificate | None = None,
    tube: OrdinaryAposterioriTubeCertificate | None = None,
    chart: OrdinaryTaylorChartCertificate | None = None,
) -> ProofGradeValidatedOrdinaryRootResult:
    raw_binding, raw_tube, raw_chart = _raw_root()
    result = check_proof_grade_validated_ordinary_root(
        raw_binding if binding is None else binding,
        raw_tube if tube is None else tube,
        raw_chart if chart is None else chart,
    )
    assert type(result) is ProofGradeValidatedOrdinaryRootResult
    return result


def _rows(result: ProofGradeValidatedOrdinaryRootResult) -> tuple[bool, ...]:
    return tuple(item.certified for item in result.obligations)


def test_success_raw_root_replays_the_eight_ordered_obligations_and_exact_clock():
    binding, _, _ = _raw_root()
    result = _check()

    assert result.profile_id == "binary64_outward_validated_ordinary_root_v04"
    assert result.checker_id == "proof_grade_validated_ordinary_root_checker_v04"
    assert tuple(item.obligation for item in result.obligations) == (
        PROOF_GRADE_VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS
    )
    assert _rows(result) == (True,) * 8
    assert result.actual_initial_error == Fraction(0)
    assert result.root_clock_origin == (
        Fraction.from_float(binding.initial_time)
        - Fraction.from_float(binding.chart_parameter)
    )
    assert result.validated_root_satisfied
    assert result.certified
    assert result.missing_obligations == ()


def test_negative_claimed_tail_is_an_authenticated_false_diagnostic_only():
    binding, tube, chart = _raw_root()
    baseline = _check()
    negative_chart = replace(chart, tail_bound=-1.0)
    result = _check(binding, tube, negative_chart)
    compatibility = check_validated_ordinary_ivp_chart(binding, tube, negative_chart)

    assert not result.chart_diagnostic.certified
    assert result.binding_result == baseline.binding_result
    assert result.tube_result == baseline.tube_result
    assert result.actual_initial_error == baseline.actual_initial_error
    assert result.root_clock_origin == baseline.root_clock_origin
    assert _rows(result) == _rows(baseline) == (True,) * 8
    assert result.certified
    assert not compatibility.certified


def test_fresh_replay_rejects_diagnostic_direct_result_ledger_and_clock_forgery():
    result = _check()

    assert not replace(
        result,
        chart_diagnostic=replace(
            result.chart_diagnostic,
            max_coefficient_residual=123.0,
        ),
    ).certified
    assert not replace(result, binding_result=replace(result.binding_result, time_gap=1.0)).certified
    assert not replace(result, tube_result=replace(result.tube_result, tube_id="fake")).certified
    assert not replace(result, obligations=result.obligations[:-1]).certified
    assert not replace(result, root_clock_origin=Fraction(1)).certified


def test_hostile_equality_and_exact_input_class_subclasses_are_rejected():
    binding, tube, chart = _raw_root()
    result = _check()

    assert not replace(result, actual_initial_error=_AlwaysEqual()).certified
    assert not _ResultEqualitySpoof(**result.__dict__).certified
    binding_subclass = _BindingSubclass(**binding.to_dict())
    tube_subclass = _TubeSubclass(**tube.to_dict())
    chart_subclass = _ChartSubclass(**chart.to_dict())
    with pytest.raises(TypeError, match="exact classes"):
        check_proof_grade_validated_ordinary_root(binding_subclass, tube, chart)
    with pytest.raises(TypeError, match="exact classes"):
        check_proof_grade_validated_ordinary_root(binding, tube_subclass, chart)
    with pytest.raises(TypeError, match="exact classes"):
        check_proof_grade_validated_ordinary_root(binding, tube, chart_subclass)


def test_exact_unit_speed_and_time_anchor_rows_are_independent_and_clear_clock():
    binding, tube, chart = _raw_root()
    unequal = _check(
        binding,
        tube,
        replace(chart, physical_time_interval=(0.0, chart.parameter_interval[1] * 2.0)),
    )
    shifted_binding = replace(
        binding,
        initial_time=0.25,
        time_tolerance=0.25,
    )
    shifted = _check(shifted_binding, tube, chart)

    assert _rows(unequal) == (True, False, True, True, True, True, True, True)
    assert unequal.root_clock_origin is None
    assert _rows(shifted) == (True, True, False, True, True, True, True, True)
    assert shifted.root_clock_origin is None


def test_id_anchor_and_uncovered_error_mutations_only_clear_their_root_rows():
    binding, tube, chart = _raw_root()
    wrong_id = _check(binding, replace(tube, chart_id="other-chart"), chart)
    wrong_anchor = _check(binding, replace(tube, anchor_parameter=5e-324), chart)
    changed_positions = [list(row) for row in binding.positions]
    changed_positions[0][0] += 0.25
    uncovered_binding = replace(
        binding,
        positions=tuple(tuple(row) for row in changed_positions),
        position_tolerance=0.25,
    )
    uncovered = _check(uncovered_binding, tube, chart)

    assert _rows(wrong_id) == (True, True, True, True, False, False, True, True)
    assert _rows(wrong_anchor) == (True, True, True, True, True, True, False, True)
    assert uncovered.actual_initial_error == Fraction(1, 4)
    assert _rows(uncovered) == (True, True, True, True, True, True, True, False)


def test_strict_schema_and_direct_replay_fail_closed_without_misusing_chart_diagnostic():
    binding, tube, chart = _raw_root()
    malformed_chart = replace(
        chart,
        position_coefficients=chart.position_coefficients[:-1],
    )
    malformed_tube = replace(tube, initial_error_bound=-1.0)
    malformed_binding = replace(binding, positions=((0.0,), (0.0, 0.0), (0.0, 0.0)))

    chart_result = _check(binding, tube, malformed_chart)
    tube_result = _check(binding, malformed_tube, chart)
    binding_result = _check(malformed_binding, tube, chart)

    assert _rows(chart_result) == (False, True, True, True, True, True, True, True)
    assert _rows(tube_result) == (True, True, True, True, False, True, True, False)
    assert _rows(binding_result) == (True, True, False, False, True, True, True, False)
    assert not chart_result.certified
    assert not tube_result.certified
    assert not binding_result.certified
