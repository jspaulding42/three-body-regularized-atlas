from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from three_body_symmetry.certificate_checker import check_ordinary_taylor_chart
from three_body_symmetry.planar_chain_review_artifact import (
    strict_load_raw_planar_chain_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "conformance" / "raw-v1"
EXPECTATION_PATH = CORPUS_ROOT / "ordinary-chart-profile-expectations.json"

PYTHON_PROFILE_ID = "frozen_python_ordinary_chart_direct_object_v03"
RUST_PROFILE_ID = "exact_rational_ordinary_chart_claimed_tail_v04"
PYTHON_CHECKER_ID = "independent_ordinary_taylor_checker_interval_v2"

OBLIGATION_IDS = (
    "ordinary_chart_type",
    "certificate_identity_present",
    "coefficient_array_shape",
    "finite_coefficients",
    "positive_masses",
    "finite_nonempty_time_intervals",
    "ordinary_physical_parameter_unit_speed",
    "finite_checker_tolerances",
    "initial_noncollision",
    "ordinary_taylor_coefficient_recurrence",
    "ordinary_taylor_exact_rational_residual_polynomials",
    "interval_taylor_model_newton_residual",
    "tail_bound_admissible",
)

# This table is deliberately independent of the expectation document.
CANONICAL_CASES = (
    ("failed-revisit", "inputs/failed-revisit.raw.json"),
    ("success", "inputs/success.raw.json"),
)


def _load_expectation() -> dict[str, Any]:
    value = json.loads(EXPECTATION_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _case_by_id(document: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [case for case in document["cases"] if case["case_id"] == case_id]
    assert len(matches) == 1
    return matches[0]


def _profile_by_id(case: dict[str, Any], profile_id: str) -> dict[str, Any]:
    matches = [
        profile for profile in case["profiles"] if profile["profile_id"] == profile_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_profile_expectation_has_the_exact_versioned_comparison_surface() -> None:
    document = _load_expectation()

    assert (
        document["expectation_schema"]
        == "raw-v1-ordinary-chart-profile-expectation-v1"
    )
    assert tuple(document["obligation_ids"]) == OBLIGATION_IDS
    assert len(OBLIGATION_IDS) == 13
    assert [case["case_id"] for case in document["cases"]] == [
        case_id for case_id, _input_file in CANONICAL_CASES
    ]


@pytest.mark.parametrize(("case_id", "input_file"), CANONICAL_CASES)
def test_frozen_python_profile_replays_without_rust_or_independence_claims(
    case_id: str,
    input_file: str,
) -> None:
    document = _load_expectation()
    case = _case_by_id(document, case_id)

    # Admission and hashing are performed from the explicit local table, not
    # discovered from the expectation that supplies the expected outcomes.
    payload = (CORPUS_ROOT / input_file).read_bytes()
    assert case["input_file"] == input_file
    assert case["input_sha256"] == hashlib.sha256(payload).hexdigest()
    certificate = strict_load_raw_planar_chain_bytes(payload)

    ordered_charts = (certificate.initial_chart,) + tuple(
        segment.target_chart for segment in certificate.segments
    )
    expected_paths = ("$.initial_chart",) + tuple(
        f"$.segments[{index}].target_chart"
        for index in range(len(certificate.segments))
    )
    assert len(ordered_charts) == len(expected_paths) == 6
    assert case["chart_order"] == [
        {
            "path": path,
            "certificate_id": chart.certificate_id,
            "chart_id": chart.chart_id,
        }
        for path, chart in zip(expected_paths, ordered_charts, strict=True)
    ]

    observed_python_ledgers: list[list[bool]] = []
    for chart in ordered_charts:
        result = check_ordinary_taylor_chart(chart)
        assert result.checker_id == PYTHON_CHECKER_ID
        assert tuple(obligation.obligation for obligation in result.obligations) == (
            OBLIGATION_IDS
        )
        observed_python_ledgers.append(
            [obligation.certified for obligation in result.obligations]
        )

    assert [profile["profile_id"] for profile in case["profiles"]] == [
        PYTHON_PROFILE_ID,
        RUST_PROFILE_ID,
    ]
    python_profile = _profile_by_id(case, PYTHON_PROFILE_ID)
    rust_profile = _profile_by_id(case, RUST_PROFILE_ID)
    assert python_profile["runtime"] == "python-v0.3-baseline"
    assert python_profile["profile_scope"] == "frozen_direct_object_primitive"
    assert python_profile["chart_satisfaction"] == observed_python_ledgers

    # The Rust rows are checked-in observations only.  This test neither runs
    # Rust nor treats equal Boolean arrays as independent mathematical agreement.
    observed_arrays_equal = (
        python_profile["chart_satisfaction"] == rust_profile["chart_satisfaction"]
    )
    assert observed_arrays_equal is True
    assert rust_profile["runtime"] == "rust-v1"
    assert rust_profile["profile_scope"] == "conditional_claimed_tail"

    comparison = case["comparison"]
    assert comparison["left_profile_id"] == PYTHON_PROFILE_ID
    assert comparison["right_profile_id"] == RUST_PROFILE_ID
    assert comparison["observed_ordered_boolean_ledgers_equal"] is (
        observed_arrays_equal
    )
    assert comparison["mathematical_outcomes_comparable"] is False
    assert comparison["release_gate_status"] == "BLOCKED"
    assert comparison["blocker_id"] == "profile_semantics_differ_open_v1_06"
