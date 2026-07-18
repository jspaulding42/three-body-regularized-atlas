from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "raw_v1_differential_gate.py"
SPEC = importlib.util.spec_from_file_location("raw_v1_differential_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def fraction(numerator: str, denominator: str = "1") -> dict[str, str]:
    return {"numerator": numerator, "denominator": denominator}


def interval(left: str, right: str) -> list[dict[str, str]]:
    return [fraction(left), fraction(right)]


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def obligation_rows(certified: bool = True) -> list[dict[str, object]]:
    return [
        {"obligation": f"obligation-{index:02d}", "certified": certified}
        for index in range(13)
    ]


def final_enclosure() -> dict[str, object]:
    point = interval("0", "0")
    return {
        "enclosure_type": "validated_planar_chain_fixed_physical_time",
        "physical_time_interval": point,
        "parameter_preimage_interval": point,
        "position_intervals": [[point, point] for _ in range(3)],
        "velocity_intervals": [[point, point] for _ in range(3)],
    }


def rust_final_enclosure() -> dict[str, object]:
    point = interval("0", "0")
    return {
        "enclosure_type": "validated_planar_chain_fixed_physical_time",
        "coordinate_system": "planar_cartesian_12",
        "physical_time_interval": point,
        "parameter_interval": point,
        "components": [
            {"component": name, "interval": point}
            for name in gate.FINAL_COMPONENT_NAMES
        ],
    }


def accepted_fixture(*, failed_prefix: bool = False):
    input_bytes = b'{"synthetic":true}'
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    point = interval("0", "0")
    request = {
        "target_physical_time": fraction("1"),
        "maximum_component_width": fraction("1", "2"),
    }
    status = "UNRESOLVED" if failed_prefix else "CERTIFIED_TO_T"
    failed_index = 1 if failed_prefix else None
    missing = ["local-failure"] if failed_prefix else []
    first_failure = "segment[1]:local-failure" if failed_prefix else None
    rows = obligation_rows(not failed_prefix)
    python_cocycle = {
        "segment_index": 0,
        "cocycle_type": "planar_lc_passage_v1"
        if failed_prefix
        else "ordinary_bridge_v1",
    }
    checker = (
        "carried_planar_lc_exit_checker_v2"
        if failed_prefix
        else "carried_ordinary_bridge_checker_v1"
    )
    kind = "planar_lc_passage" if failed_prefix else "ordinary_bridge"
    pair = [0, 1] if failed_prefix else None
    if failed_prefix:
        python_cocycle["canonical_pair"] = pair

    python = {
        "artifact_schema": gate.PYTHON_TRANSCRIPT_SCHEMA,
        "artifact_version": gate.PYTHON_TRANSCRIPT_VERSION,
        "status": status,
        "certificate_id": "synthetic-certificate",
        "theorem_evidence_sha256": input_sha,
        "request": request,
        "obligations": rows,
        "first_failed_obligation": first_failure,
        "certified_segment_count": 1,
        "failed_segment_index": failed_index,
        "failed_segment_missing_obligations": missing,
        "segment_checker_ids": [checker],
        "clock_origin_ledger": [
            {"vertex_index": 0, "chart_id": "chart-0", "clock_origin_interval": point},
            {"vertex_index": 1, "chart_id": "chart-1", "clock_origin_interval": point},
        ],
        "cocycle_ledger": [python_cocycle],
        "current_chart_id": "chart-1",
        "current_clock_origin_interval": point,
        "covered_physical_time_interval": point,
        "target_parameter_preimage_interval": None if failed_prefix else point,
        "maximum_final_component_width": None if failed_prefix else fraction("0"),
        "final_enclosure": None if failed_prefix else final_enclosure(),
        "retained_regions": [],
    }
    if failed_prefix:
        python["retained_regions"] = [
            {
                "region_type": "certified_lifted_lc_right_frontier",
                "coordinate_system": "planar_lc_lifted_14",
                "physical_time_interval": point,
                "parameter_interval": point,
                "component_intervals": [point for _ in gate.LC_COMPONENT_NAMES],
                "certified_segment_count": 1,
                "provenance_checker_id": "python-checker",
            }
        ]

    rust_semantic = {
        "schema": gate.RUST_SEMANTIC_SCHEMA,
        "profile": gate.RUST_SEMANTIC_PROFILE,
        "status": status,
        "certificate_id": "synthetic-certificate",
        "evidence_sha256": input_sha,
        "request": request,
        "top_level_obligations": rows,
        "first_failed_obligation": first_failure,
        "certified_segment_count": 1,
        "failed_segment_index": failed_index,
        "failed_segment_missing_obligations": missing,
        "segment_profiles": [
            {"segment_index": 0, "kind": kind, "profile": "ignored", "pair": pair}
        ],
        "clock_ledger": [
            {"vertex_index": 0, "chart_id": "chart-0", "clock_origin": point},
            {"vertex_index": 1, "chart_id": "chart-1", "clock_origin": point},
        ],
        "current_chart": {
            "certificate_id": "chart-cert",
            "chart_id": "chart-1",
            "parameter_interval": point,
            "physical_time_interval": point,
        },
        "current_clock_origin": point,
        "covered_physical_time_interval": point,
        "target_parameter_preimage_interval": None if failed_prefix else point,
        "maximum_final_component_width": None if failed_prefix else fraction("0"),
        "final_enclosure": None if failed_prefix else rust_final_enclosure(),
        "retained_region": None,
    }
    if failed_prefix:
        rust_semantic["segment_profiles"].append(
            {
                "segment_index": 1,
                "kind": "ordinary_bridge",
                "profile": "uncommitted-and-ignored",
                "pair": None,
            }
        )
        rust_semantic["retained_region"] = {
            "region_type": "certified_lifted_lc_right_frontier",
            "coordinate_system": "planar_lc_lifted_14",
            "certified_segment_count": 1,
            "failed_segment_index": 1,
            "chart_id": "lc-chart",
            "pair": [0, 1],
            "parameter_interval": point,
            "physical_time_interval": point,
            "components": [
                {"component": name, "interval": point}
                for name in gate.LC_COMPONENT_NAMES
            ],
        }
    rust_execution = {
        "schema": gate.RUST_EXECUTION_SCHEMA,
        "profile": gate.RUST_EXECUTION_PROFILE,
        "parse_outcome": "ACCEPT",
        "rejection_stage": None,
        "evaluation_outcome": "RESULT",
        "evaluation_error_stage": None,
        "semantic_result": rust_semantic,
    }
    python_bytes = json_bytes(python)
    retained_summaries = []
    if failed_prefix:
        retained_summaries = [
            {
                "region_type": "certified_lifted_lc_right_frontier",
                "coordinate_system": "planar_lc_lifted_14",
                "physical_time_interval": point,
                "parameter_interval": point,
                "component_count": 14,
                "certified_segment_count": 1,
                "provenance_checker_id": "python-checker",
            }
        ]
    facts = {
        "certificate_id": "synthetic-certificate",
        "theorem_evidence_sha256": input_sha,
        "request": request,
        "certified_segment_count": 1,
        "failed_segment_index": failed_index,
        "failed_segment_missing_obligations": missing,
        "first_failed_obligation": first_failure,
        "top_level_obligations": rows,
        "segment_checker_ids": [checker],
        "clock_vertex_count": 2,
        "cocycle_kinds": [python_cocycle["cocycle_type"]],
        "cocycle_pairs": [pair] if failed_prefix else [],
        "current_chart_id": "chart-1",
        "current_clock_origin_interval": point,
        "covered_physical_time_interval": point,
        "target_parameter_preimage_interval": None if failed_prefix else point,
        "maximum_final_component_width_present": not failed_prefix,
        "final_enclosure_summary": None,
        "retained_region_summaries": retained_summaries,
    }
    if not failed_prefix:
        facts["final_enclosure_summary"] = {
            "enclosure_type": "validated_planar_chain_fixed_physical_time",
            "physical_time_interval": point,
            "parameter_preimage_interval": point,
            "position_shape": [3, 2],
            "velocity_shape": [3, 2],
        }
    expectation = {
        "expectation_schema": gate.EXPECTATION_SCHEMA,
        "case_id": "synthetic-failed" if failed_prefix else "synthetic-success",
        "input_file": "inputs/synthetic.raw.json",
        "input_sha256": input_sha,
        "expected_parse_outcome": "ACCEPT",
        "expected_replay_outcome": status,
        "source_replay_transcript": "replays/synthetic.json",
        "source_replay_sha256": hashlib.sha256(python_bytes).hexdigest(),
        "facts": facts,
    }
    return (
        input_bytes,
        json_bytes(expectation),
        python_bytes,
        json_bytes(rust_execution),
    )


def mutate_accepted_fixture(fixture, mutation):
    input_bytes, expectation_bytes, python_bytes, rust_bytes = fixture
    expectation = json.loads(expectation_bytes)
    python = json.loads(python_bytes)
    rust = json.loads(rust_bytes)
    mutation(expectation, python, rust)
    python_bytes = json_bytes(python)
    expectation["source_replay_sha256"] = hashlib.sha256(python_bytes).hexdigest()
    return input_bytes, json_bytes(expectation), python_bytes, json_bytes(rust)


def rejected_fixture():
    input_bytes = b"not-json"
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    expectation = {
        "expectation_schema": gate.EXPECTATION_SCHEMA,
        "case_id": "synthetic-reject",
        "input_file": "inputs/synthetic.json",
        "input_sha256": input_sha,
        "expected_parse_outcome": "REJECT",
        "expected_replay_outcome": None,
        "source_replay_transcript": None,
        "source_replay_sha256": None,
        "facts": None,
        "mutation": {"description": "synthetic"},
    }
    rust = {
        "schema": gate.RUST_EXECUTION_SCHEMA,
        "profile": gate.RUST_EXECUTION_PROFILE,
        "parse_outcome": "REJECT",
        "rejection_stage": "CANONICAL_WIRE",
        "evaluation_outcome": "NOT_RUN",
        "evaluation_error_stage": None,
        "semantic_result": None,
    }
    return input_bytes, json_bytes(expectation), None, json_bytes(rust)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (interval("0", "2"), interval("0", "2"), "EQUAL"),
        (interval("0", "3"), interval("1", "2"), "LEFT_CONTAINS_RIGHT"),
        (interval("1", "2"), interval("0", "3"), "RIGHT_CONTAINS_LEFT"),
        (interval("0", "2"), interval("1", "3"), "OVERLAP"),
        (interval("0", "1"), interval("2", "3"), "DISJOINT"),
    ),
)
def test_classify_interval_exact_relations(left, right, expected) -> None:
    assert gate.classify_interval(left, right) == expected


@pytest.mark.parametrize(
    "bad_fraction",
    (
        {"numerator": "01", "denominator": "1"},
        {"numerator": "-0", "denominator": "1"},
        {"numerator": "1", "denominator": "01"},
        {"numerator": "2", "denominator": "4"},
        {"numerator": "1", "denominator": "-2"},
    ),
)
def test_classify_interval_rejects_noncanonical_fractions(bad_fraction) -> None:
    with pytest.raises(gate.DifferentialGateError):
        gate.classify_interval([bad_fraction, fraction("1")], interval("0", "1"))


def test_classify_interval_accepts_5000_digit_canonical_numerator() -> None:
    numerator = "1" + "0" * 4_999
    point = [fraction(numerator), fraction(numerator)]
    assert gate.classify_interval(point, point) == "EQUAL"


def test_classify_interval_rejects_more_than_100000_decimal_digits() -> None:
    numerator = "1" + "0" * gate.MAX_DECIMAL_DIGITS
    point = [fraction(numerator), fraction(numerator)]
    with pytest.raises(
        gate.DifferentialGateError,
        match="FRACTION_DECIMAL_DIGIT_LIMIT_EXCEEDED",
    ):
        gate.classify_interval(point, interval("0", "1"))


def test_compare_case_accept_success_maps_discrete_surfaces() -> None:
    fixture = accepted_fixture()
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is True
    assert result["mismatch_ids"] == []
    assert result["comparison_status"] == "PROFILE_COMPATIBILITY_OBSERVED"
    assert result["theorem_status"] == "CERTIFIED_TO_T"
    assert result == gate.compare_case("synthetic-success", *fixture)
    assert result["profiles"]["python"]["profile_id"] == gate.PYTHON_PROFILE
    assert (
        result["profiles"]["rust_semantic"]["profile_id"] == gate.RUST_SEMANTIC_PROFILE
    )
    assert all(len(digest) == 64 for digest in result["sha256_bindings"].values())
    assert len(result["interval_relations"]) == 24
    assert {record["relation"] for record in result["interval_relations"]} == {"EQUAL"}


def test_compare_case_failed_prefix_uses_only_committed_segments() -> None:
    fixture = accepted_fixture(failed_prefix=True)
    result = gate.compare_case("synthetic-failed", *fixture)
    assert result["differential_gate_passed"] is True
    assert result["mismatch_ids"] == []
    assert result["comparison_status"] == "PROFILE_COMPATIBILITY_OBSERVED"
    assert result["theorem_status"] == "UNRESOLVED"
    assert len(result["interval_relations"]) == 24


def test_compare_case_parser_reject_has_no_theorem_status_or_python_hash() -> None:
    result = gate.compare_case("synthetic-reject", *rejected_fixture())
    assert result["differential_gate_passed"] is True
    assert result["comparison_status"] == "PARSER_EXPECTATION_COMPATIBILITY_OBSERVED"
    assert result["theorem_status"] is None
    assert result["profiles"]["rust_semantic"] is None
    assert result["sha256_bindings"]["python_transcript_sha256"] is None
    assert result["sha256_bindings"]["theorem_evidence_sha256"] is None


def test_compare_case_records_stable_semantic_mismatch_without_exception() -> None:
    input_bytes, expectation_bytes, python_bytes, rust_bytes = accepted_fixture()
    rust = json.loads(rust_bytes)
    rust["semantic_result"]["certificate_id"] = "different-certificate"
    result = gate.compare_case(
        "synthetic-success",
        input_bytes,
        expectation_bytes,
        python_bytes,
        json_bytes(rust),
    )
    assert result["differential_gate_passed"] is False
    assert result["mismatch_ids"] == ["CERTIFICATE_ID_MISMATCH"]
    assert "exception" not in json.dumps(result).lower()


def test_cross_profile_interval_records_cover_all_non_disjoint_relations() -> None:
    def mutation(expectation, python, rust):
        python["clock_origin_ledger"][1]["clock_origin_interval"] = interval("0", "2")
        rust["semantic_result"]["clock_ledger"][1]["clock_origin"] = interval("0", "3")

        python["current_clock_origin_interval"] = interval("0", "3")
        expectation["facts"]["current_clock_origin_interval"] = interval("0", "3")
        rust["semantic_result"]["current_clock_origin"] = interval("1", "2")

        python["covered_physical_time_interval"] = interval("0", "2")
        expectation["facts"]["covered_physical_time_interval"] = interval("0", "2")
        rust["semantic_result"]["covered_physical_time_interval"] = interval("1", "3")

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is True
    assert len(result["interval_relations"]) == 24
    counts = {
        relation: sum(
            record["relation"] == relation for record in result["interval_relations"]
        )
        for relation in (
            "EQUAL",
            "LEFT_CONTAINS_RIGHT",
            "RIGHT_CONTAINS_LEFT",
            "OVERLAP",
            "DISJOINT",
        )
    }
    assert counts == {
        "EQUAL": 21,
        "LEFT_CONTAINS_RIGHT": 1,
        "RIGHT_CONTAINS_LEFT": 1,
        "OVERLAP": 1,
        "DISJOINT": 0,
    }
    serialized = json.dumps(result, sort_keys=True)
    assert '"numerator"' not in serialized
    assert '"denominator"' not in serialized
    for record in result["interval_relations"]:
        assert set(record) == {
            "id",
            "left_path",
            "relation",
            "right_path",
            "scope",
        }
        assert record["scope"] in {"EXPECTATION_PYTHON", "CROSS_PROFILE"}


def test_cross_profile_disjoint_interval_is_a_hard_mismatch() -> None:
    def mutation(_expectation, _python, rust):
        rust["semantic_result"]["current_clock_origin"] = interval("2", "3")

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is False
    assert "CROSS_PROFILE_INTERVAL_DISJOINT" in result["mismatch_ids"]
    assert any(
        record["id"] == "cross_profile.current_clock_origin"
        and record["relation"] == "DISJOINT"
        for record in result["interval_relations"]
    )


def test_cross_profile_null_interval_mismatch_is_hard() -> None:
    def mutation(_expectation, _python, rust):
        rust["semantic_result"]["target_parameter_preimage_interval"] = None

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is False
    assert "CROSS_PROFILE_INTERVAL_PRESENCE_MISMATCH" in result["mismatch_ids"]


def test_profile_local_maximum_width_must_not_exceed_its_request() -> None:
    def mutation(_expectation, _python, rust):
        rust["semantic_result"]["maximum_final_component_width"] = fraction("1")

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is False
    assert result["mismatch_ids"] == ["RUST_MAXIMUM_FINAL_WIDTH_EXCEEDS_REQUEST"]


def test_profile_local_maximum_width_must_be_nonnegative() -> None:
    def mutation(_expectation, _python, rust):
        rust["semantic_result"]["maximum_final_component_width"] = fraction("-1")

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is False
    assert result["mismatch_ids"] == ["RUST_MAXIMUM_FINAL_WIDTH_NEGATIVE"]


def test_expectation_interval_must_equal_frozen_python_interval() -> None:
    def mutation(expectation, _python, _rust):
        expectation["facts"]["covered_physical_time_interval"] = interval("0", "1")

    fixture = mutate_accepted_fixture(accepted_fixture(), mutation)
    result = gate.compare_case("synthetic-success", *fixture)
    assert result["differential_gate_passed"] is False
    assert "EXPECTATION_PYTHON_INTERVAL_MISMATCH" in result["mismatch_ids"]
    record = next(
        item
        for item in result["interval_relations"]
        if item["id"] == "expectation_python.covered_physical_time"
    )
    assert record["relation"] == "LEFT_CONTAINS_RIGHT"


def test_build_report_is_canonical_blocked_and_explicitly_nonclaiming() -> None:
    accepted = gate.compare_case("synthetic-success", *accepted_fixture())
    rejected = gate.compare_case("synthetic-reject", *rejected_fixture())
    report_bytes = gate.build_report([accepted, rejected])
    assert report_bytes == gate.build_report([accepted, rejected])
    assert not report_bytes.endswith(b"\n")
    assert report_bytes == json_bytes(json.loads(report_bytes))
    report = json.loads(report_bytes)
    assert report["report_schema"] == gate.REPORT_SCHEMA
    assert report["comparator_id"] == gate.COMPARATOR_ID
    assert report["comparison_kind"] == gate.COMPARISON_KIND
    assert [case["case_id"] for case in report["cases"]] == [
        "synthetic-success",
        "synthetic-reject",
    ]
    aggregate = report["aggregate"]
    assert aggregate["differential_gate_passed"] is True
    assert aggregate["case_count"] == 2
    assert aggregate["mismatch_count"] == 0
    assert aggregate["disjoint_interval_count"] == 0
    assert aggregate["interval_relation_counts"] == {
        "all": {
            "DISJOINT": 0,
            "EQUAL": 24,
            "LEFT_CONTAINS_RIGHT": 0,
            "OVERLAP": 0,
            "RIGHT_CONTAINS_LEFT": 0,
        },
        "cross_profile": {
            "DISJOINT": 0,
            "EQUAL": 19,
            "LEFT_CONTAINS_RIGHT": 0,
            "OVERLAP": 0,
            "RIGHT_CONTAINS_LEFT": 0,
        },
        "expectation_python": {
            "DISJOINT": 0,
            "EQUAL": 5,
            "LEFT_CONTAINS_RIGHT": 0,
            "OVERLAP": 0,
            "RIGHT_CONTAINS_LEFT": 0,
        },
    }
    assert aggregate["mathematical_outcomes_comparable"] is False
    assert aggregate["release_gate_status"] == "BLOCKED"
    assert aggregate["blockers"] == list(gate.FIXED_BLOCKERS)
    assert aggregate["nonclaims"] == list(gate.FIXED_NONCLAIMS)
    lowered = report_bytes.decode().lower()
    assert "timestamp" not in lowered
    assert "exception" not in lowered
    assert str(ROOT).lower() not in lowered


def test_build_report_adds_differential_blocker_and_disjoint_count() -> None:
    def mutation(_expectation, _python, rust):
        rust["semantic_result"]["current_clock_origin"] = interval("2", "3")

    mismatch = gate.compare_case(
        "synthetic-success",
        *mutate_accepted_fixture(accepted_fixture(), mutation),
    )
    report = json.loads(gate.build_report([mismatch]))
    aggregate = report["aggregate"]
    assert aggregate["differential_gate_passed"] is False
    assert aggregate["mismatch_case_count"] == 1
    assert aggregate["mismatch_count"] == 1
    assert aggregate["disjoint_interval_count"] == 1
    assert aggregate["blockers"] == [
        *gate.FIXED_BLOCKERS,
        "DIFFERENTIAL-DISAGREEMENT",
    ]


def test_build_report_rejects_duplicate_case_ids() -> None:
    accepted = gate.compare_case("synthetic-success", *accepted_fixture())
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.build_report([accepted, accepted])
    assert error.value.code == "DUPLICATE_CASE_ID"


def test_build_report_rejects_duplicate_interval_relation_ids() -> None:
    accepted = gate.compare_case("synthetic-success", *accepted_fixture())
    accepted["interval_relations"][1]["id"] = accepted["interval_relations"][0]["id"]
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.build_report([accepted])
    assert error.value.code == "DUPLICATE_INTERVAL_RELATION_ID"


def test_json_artifact_loader_enforces_bounded_payload_before_decode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gate, "MAX_ARTIFACT_BYTES", 4)
    with pytest.raises(gate.DifferentialGateError) as error:
        gate._load_json_object(b'{"a":1}', "artifact")
    assert error.value.code == "JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED"


def test_compare_case_rejects_surrogate_case_id_with_stable_error() -> None:
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.compare_case("\ud800", *accepted_fixture())
    assert error.value.code == "STRING_NOT_UTF8_ENCODABLE"


@pytest.mark.parametrize(
    ("payload", "code"),
    (
        (b'{"a":1,"a":2}', "JSON_DUPLICATE_KEY"),
        (b'{"a":NaN}', "JSON_NONFINITE_CONSTANT_NOT_PERMITTED"),
        (b"\xff", "JSON_UTF8_INVALID"),
        (b"[]", "JSON_TOP_LEVEL_OBJECT_REQUIRED"),
    ),
)
def test_json_artifact_loader_fails_closed(payload, code) -> None:
    with pytest.raises(gate.DifferentialGateError) as error:
        gate._load_json_object(payload, "artifact")
    assert error.value.code == code
