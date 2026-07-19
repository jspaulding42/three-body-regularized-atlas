from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "raw_v1_differential_gate.py"
CORPUS_ROOT = ROOT / "conformance" / "raw-v1"
RUST_VERIFIER = ROOT / "verifiers" / "rust-v1" / "target" / "debug" / "raw_v1_verify"
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


def rust_stdout_bytes(payload: bytes) -> bytes:
    value = json.loads(payload)
    field_order = (
        "schema",
        "profile",
        "parse_outcome",
        "rejection_stage",
        "evaluation_outcome",
        "evaluation_error_stage",
        "semantic_result",
    )
    ordered = {field: value[field] for field in field_order}
    return json.dumps(
        ordered,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


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


def python_reject_capture(case_id: str, input_bytes: bytes):
    observation = {
        "capture_kind": "PARSER_REJECT_OBSERVATION",
        "capture_schema": gate.PYTHON_REJECT_CAPTURE_SCHEMA,
        "case_id": case_id,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "parse_outcome": "REJECT",
        "profile": gate.PYTHON_REJECT_PROFILE,
    }
    return gate.PythonLiveCapture(
        kind="PARSER_REJECT_OBSERVATION",
        payload=json_bytes(observation),
        profile_id=gate.PYTHON_REJECT_PROFILE,
    )


def write_fake_live_corpus(root: Path):
    expectation_root = root / "expectations"
    input_root = root / "inputs"
    expectation_root.mkdir(parents=True)
    input_root.mkdir()
    (root / "manifest.json").write_bytes(
        json_bytes(
            {
                "manifest_schema": "synthetic-raw-v1-corpus-manifest-v1",
                "status": "test-only",
            }
        )
    )
    captures = {}
    rust_executions = {}
    for case_id, fixture in (
        ("synthetic-success", accepted_fixture()),
        ("synthetic-reject", rejected_fixture()),
    ):
        input_bytes, expectation_bytes, python_bytes, rust_bytes = fixture
        expectation = json.loads(expectation_bytes)
        input_name = Path(expectation["input_file"]).name
        (expectation_root / f"{case_id}.expected.json").write_bytes(expectation_bytes)
        (input_root / input_name).write_bytes(input_bytes)
        captures[case_id] = (
            gate.PythonLiveCapture(
                kind="REPLAY_TRANSCRIPT",
                payload=python_bytes,
                profile_id=gate.PYTHON_PROFILE,
            )
            if python_bytes is not None
            else python_reject_capture(case_id, input_bytes)
        )
        rust_executions[hashlib.sha256(input_bytes).hexdigest()] = rust_bytes
    return captures, rust_executions


def write_fake_rust_executable(root: Path) -> Path:
    executable = root / "fake-rust-verifier"
    executable.write_bytes(b"deterministic-fake-rust-verifier-v1")
    executable.chmod(0o755)
    return executable


def tree_file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


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


@pytest.mark.parametrize(
    ("fixture_factory", "updates", "removed_field", "code"),
    (
        (
            accepted_fixture,
            {"unexpected": None},
            None,
            "RUST_EXECUTION_FIELDS_INVALID",
        ),
        (
            accepted_fixture,
            {},
            "semantic_result",
            "RUST_EXECUTION_FIELDS_INVALID",
        ),
        (
            rejected_fixture,
            {"rejection_stage": None},
            None,
            "RUST_REJECTION_STAGE_INVALID",
        ),
        (
            rejected_fixture,
            {"rejection_stage": "OTHER"},
            None,
            "RUST_REJECTION_STAGE_INVALID",
        ),
        (
            rejected_fixture,
            {"evaluation_outcome": "RESULT"},
            None,
            "RUST_REJECT_ENVELOPE_INCONSISTENT",
        ),
        (
            rejected_fixture,
            {"evaluation_error_stage": "NAMESPACE"},
            None,
            "RUST_REJECT_ENVELOPE_INCONSISTENT",
        ),
        (
            rejected_fixture,
            {"semantic_result": {}},
            None,
            "RUST_REJECT_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {"rejection_stage": "SCHEMA"},
            None,
            "RUST_ACCEPT_REJECTION_STAGE_NOT_NULL",
        ),
        (
            accepted_fixture,
            {"evaluation_outcome": "NOT_RUN"},
            None,
            "RUST_ACCEPT_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {"evaluation_error_stage": "NAMESPACE"},
            None,
            "RUST_RESULT_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {"semantic_result": None},
            None,
            "RUST_RESULT_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {
                "evaluation_outcome": "ERROR",
                "evaluation_error_stage": None,
                "semantic_result": None,
            },
            None,
            "RUST_ERROR_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {
                "evaluation_outcome": "ERROR",
                "evaluation_error_stage": "NAMESPACE",
            },
            None,
            "RUST_ERROR_ENVELOPE_INCONSISTENT",
        ),
        (
            accepted_fixture,
            {"evaluation_error_stage": "OTHER"},
            None,
            "RUST_EVALUATION_ERROR_STAGE_INVALID",
        ),
        (
            accepted_fixture,
            {"semantic_result": "not-an-object"},
            None,
            "RUST_SEMANTIC_RESULT_OBJECT_OR_NULL_REQUIRED",
        ),
    ),
)
def test_compare_case_rejects_mutated_rust_execution_envelope(
    fixture_factory, updates, removed_field, code
) -> None:
    input_bytes, expectation_bytes, python_bytes, rust_bytes = fixture_factory()
    rust_execution = json.loads(rust_bytes)
    rust_execution.update(updates)
    if removed_field is not None:
        del rust_execution[removed_field]
    case_id = json.loads(expectation_bytes)["case_id"]
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.compare_case(
            case_id,
            input_bytes,
            expectation_bytes,
            python_bytes,
            json_bytes(rust_execution),
        )
    assert error.value.code == code


def test_compare_case_accepts_consistent_rust_error_envelope_as_a_mismatch() -> None:
    input_bytes, expectation_bytes, python_bytes, rust_bytes = accepted_fixture()
    rust_execution = json.loads(rust_bytes)
    rust_execution.update(
        {
            "evaluation_outcome": "ERROR",
            "evaluation_error_stage": "MIXED_REPLAY",
            "semantic_result": None,
        }
    )
    result = gate.compare_case(
        "synthetic-success",
        input_bytes,
        expectation_bytes,
        python_bytes,
        json_bytes(rust_execution),
    )
    assert result["differential_gate_passed"] is False
    assert result["mismatch_ids"] == [
        "RUST_ACCEPT_EVALUATION_OUTCOME_MISMATCH",
        "RUST_ACCEPT_SEMANTIC_RESULT_ABSENT",
        "RUST_ACCEPT_EVALUATION_ERROR_STAGE_PRESENT",
    ]
    assert result["profiles"]["rust_semantic"] is None


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


def test_offline_v1_report_shape_and_bytes_match_pushed_1ac2175_contract() -> None:
    cases = [
        gate.compare_case("synthetic-success", *accepted_fixture()),
        gate.compare_case("synthetic-reject", *rejected_fixture()),
    ]
    report_bytes = gate.build_report(cases)
    report = json.loads(report_bytes)

    assert hashlib.sha256(report_bytes).hexdigest() == (
        "63bc1903bdb5385149afc697ed9fe454aa00b5999ea4f48dcec17c5ca4a14b43"
    )
    assert set(report) == {
        "aggregate",
        "cases",
        "comparator_id",
        "comparison_kind",
        "report_schema",
    }
    assert report["report_schema"] == gate.REPORT_SCHEMA
    assert "capture_mode" not in report
    for case in report["cases"]:
        assert set(case) == {
            "case_id",
            "comparison_status",
            "differential_gate_passed",
            "expected_parse_outcome",
            "interval_relations",
            "mismatch_ids",
            "profiles",
            "sha256_bindings",
            "theorem_status",
        }
        assert set(case["profiles"]["python"]) == {
            "profile_id",
            "source_marker",
            "transcript_present",
        }
        assert set(case["sha256_bindings"]) == {
            "expectation_sha256",
            "input_sha256",
            "python_transcript_sha256",
            "rust_execution_sha256",
            "theorem_evidence_sha256",
        }
        assert "python_capture_kind" not in case


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


def test_live_corpus_fake_adapters_are_fresh_bound_ordered_and_deterministic(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    def rust_runner(_executable, input_path):
        captured = input_path.read_bytes()
        return rust_executions[hashlib.sha256(captured).hexdigest()]

    first = gate.run_live_corpus(
        corpus,
        "fake-rust",
        jobs=2,
        python_capture_adapter=python_adapter,
        rust_execution_runner=rust_runner,
    )
    second = gate.run_live_corpus(
        corpus,
        "fake-rust",
        jobs=2,
        python_capture_adapter=python_adapter,
        rust_execution_runner=rust_runner,
    )
    assert first == second
    assert not first.endswith(b"\n")
    report = json.loads(first)
    assert report["capture_mode"] == gate.LIVE_CAPTURE_MODE
    assert report["report_schema"] == gate.LIVE_REPORT_SCHEMA
    assert report["aggregate"]["differential_gate_passed"] is True
    assert [case["case_id"] for case in report["cases"]] == [
        "synthetic-reject",
        "synthetic-success",
    ]
    rejected, accepted = report["cases"]
    assert rejected["python_capture_kind"] == "PARSER_REJECT_OBSERVATION"
    assert rejected["profiles"]["python"] == {
        "capture_kind": "PARSER_REJECT_OBSERVATION",
        "capture_schema": gate.PYTHON_REJECT_CAPTURE_SCHEMA,
        "profile_id": gate.PYTHON_REJECT_PROFILE,
        "source_marker": gate.PYTHON_LIVE_REJECT_SOURCE_MARKER,
        "transcript_present": False,
    }
    assert (
        rejected["sha256_bindings"]["python_capture_sha256"]
        == hashlib.sha256(captures["synthetic-reject"].payload).hexdigest()
    )
    assert rejected["sha256_bindings"]["python_transcript_sha256"] is None
    assert accepted["python_capture_kind"] == "REPLAY_TRANSCRIPT"
    assert (
        accepted["sha256_bindings"]["python_capture_sha256"]
        == accepted["sha256_bindings"]["python_transcript_sha256"]
    )
    lowered = first.decode().lower()
    assert str(tmp_path).lower() not in lowered
    assert "timestamp" not in lowered
    assert "exception" not in lowered


def test_live_evidence_bundle_is_deterministic_complete_and_fully_bound(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)
    rust_executions = {
        input_sha: rust_stdout_bytes(payload)
        for input_sha, payload in rust_executions.items()
    }
    executable = write_fake_rust_executable(tmp_path)
    input_sha_to_case = {
        hashlib.sha256(
            (corpus / "inputs" / "synthetic.json").read_bytes()
        ).hexdigest(): "synthetic-reject",
        hashlib.sha256(
            (corpus / "inputs" / "synthetic.raw.json").read_bytes()
        ).hexdigest(): "synthetic-success",
    }
    lock = threading.Lock()
    python_counts = dict.fromkeys(captures, 0)
    rust_counts = dict.fromkeys(captures, 0)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        with lock:
            python_counts[case_id] += 1
        return captures[case_id]

    def rust_runner(_executable, input_path):
        input_bytes = input_path.read_bytes()
        input_sha = hashlib.sha256(input_bytes).hexdigest()
        with lock:
            rust_counts[input_sha_to_case[input_sha]] += 1
        return rust_executions[input_sha]

    bundle_a = tmp_path / "bundle-a"
    bundle_b = tmp_path / "bundle-b"
    report_a = gate.run_live_evidence_bundle(
        corpus,
        executable,
        bundle_a,
        jobs=2,
        python_capture_adapter=python_adapter,
        rust_execution_runner=rust_runner,
    )
    report_b = gate.run_live_evidence_bundle(
        corpus,
        executable,
        bundle_b,
        jobs=2,
        python_capture_adapter=python_adapter,
        rust_execution_runner=rust_runner,
    )
    assert report_a == report_b
    assert python_counts == {"synthetic-success": 2, "synthetic-reject": 2}
    assert rust_counts == {"synthetic-success": 2, "synthetic-reject": 2}

    files_a = tree_file_bytes(bundle_a)
    files_b = tree_file_bytes(bundle_b)
    assert files_a == files_b
    assert set(files_a) == {
        "manifest.json",
        "report.json",
        "cases/0000/python-capture.json",
        "cases/0000/rust-execution.json",
        "cases/0001/python-capture.json",
        "cases/0001/rust-execution.json",
    }
    assert files_a["report.json"] == report_a
    for relative, payload in files_a.items():
        assert not payload.endswith(b"\n")
        if relative.endswith("/rust-execution.json"):
            assert payload != json_bytes(json.loads(payload))
        else:
            assert payload == json_bytes(json.loads(payload))

    report = json.loads(report_a)
    manifest = json.loads(files_a["manifest.json"])
    gate._validate_live_bundle_manifest(manifest, 2)
    assert manifest["manifest_schema"] == gate.LIVE_BUNDLE_SCHEMA
    assert manifest["comparator_id"] == gate.COMPARATOR_ID
    assert manifest["live_driver_id"] == gate.LIVE_DRIVER_ID
    assert manifest["capture_mode"] == gate.LIVE_CAPTURE_MODE
    assert manifest["report"] == {
        "bundle_relative_path": "report.json",
        "byte_length": len(report_a),
        "report_schema": gate.LIVE_REPORT_SCHEMA,
        "sha256": hashlib.sha256(report_a).hexdigest(),
    }
    assert manifest["frozen_specification"] == {
        "byte_length": len((ROOT / gate.FROZEN_SPECIFICATION_PATH).read_bytes()),
        "repository_relative_path": gate.FROZEN_SPECIFICATION_PATH,
        "sha256": gate.FROZEN_SPECIFICATION_SHA256,
    }
    assert manifest["corpus_manifest"] == {
        "byte_length": len((corpus / "manifest.json").read_bytes()),
        "corpus_relative_path": "manifest.json",
        "sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
    }
    assert manifest["comparator_source"] == {
        "byte_length": len(SCRIPT.read_bytes()),
        "repository_relative_path": gate.COMPARATOR_SOURCE_PATH,
        "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }
    assert manifest["rust_executable"] == {
        "byte_length": len(executable.read_bytes()),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    assert manifest["claim_scope"] == {
        "implementation_or_verifier_independence_attested": False,
        "loaded_code_matches_comparator_source_attested": False,
        "nonclaims_report_field": "aggregate.nonclaims",
        "release_gate_status": "BLOCKED",
        "release_gate_status_report_field": "aggregate.release_gate_status",
    }
    assert report["aggregate"]["release_gate_status"] == "BLOCKED"
    assert report["aggregate"]["nonclaims"] == list(gate.FIXED_NONCLAIMS)

    assert [item["case_id"] for item in manifest["cases"]] == [
        "synthetic-reject",
        "synthetic-success",
    ]
    for index, case_record in enumerate(manifest["cases"]):
        case_id = case_record["case_id"]
        for field in ("input", "expectation"):
            binding = case_record[field]
            payload = (corpus / binding["corpus_relative_path"]).read_bytes()
            assert binding["byte_length"] == len(payload)
            assert binding["sha256"] == hashlib.sha256(payload).hexdigest()
        python_binding = case_record["python_capture"]
        python_payload = files_a[python_binding["bundle_relative_path"]]
        assert python_payload == captures[case_id].payload
        assert python_binding["byte_length"] == len(python_payload)
        assert python_binding["sha256"] == hashlib.sha256(python_payload).hexdigest()
        assert python_binding["capture_kind"] == captures[case_id].kind
        assert python_binding["profile_id"] == captures[case_id].profile_id
        rust_binding = case_record["rust_execution"]
        rust_payload = files_a[rust_binding["bundle_relative_path"]]
        input_payload = (
            corpus / case_record["input"]["corpus_relative_path"]
        ).read_bytes()
        expected_rust = rust_executions[hashlib.sha256(input_payload).hexdigest()]
        assert rust_payload == expected_rust
        assert rust_payload == rust_stdout_bytes(rust_payload)
        assert rust_payload != json_bytes(json.loads(rust_payload))
        assert not rust_payload.endswith(b"\n")
        assert rust_binding["byte_length"] == len(rust_payload)
        assert rust_binding["sha256"] == hashlib.sha256(rust_payload).hexdigest()
        assert rust_binding["evidence_kind"] == gate.RUST_EXECUTION_EVIDENCE_KIND
        assert rust_binding["profile_id"] == gate.RUST_EXECUTION_PROFILE
        if index == 0:
            assert json.loads(python_payload)["capture_kind"] == (
                "PARSER_REJECT_OBSERVATION"
            )
        else:
            assert json.loads(python_payload)["artifact_schema"] == (
                gate.PYTHON_TRANSCRIPT_SCHEMA
            )

    combined = b"\n".join(files_a.values()).lower()
    assert str(tmp_path).encode().lower() not in combined
    assert b"timestamp" not in combined
    assert b"exception" not in combined
    assert b"hostname" not in combined

    extra_field = json.loads(files_a["manifest.json"])
    extra_field["unexpected"] = None
    with pytest.raises(gate.DifferentialGateError) as extra_error:
        gate._validate_live_bundle_manifest(extra_field, 2)
    assert extra_error.value.code == "LIVE_BUNDLE_MANIFEST_FIELDS_INVALID"

    traversal = json.loads(files_a["manifest.json"])
    traversal["cases"][0]["input"]["corpus_relative_path"] = "../input.json"
    with pytest.raises(gate.DifferentialGateError) as traversal_error:
        gate._validate_live_bundle_manifest(traversal, 2)
    assert traversal_error.value.code == "LIVE_BUNDLE_RELATIVE_PATH_INVALID"


def test_live_case_evidence_retains_valid_unsorted_rust_stdout_exactly(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)
    case = next(
        item
        for item in gate.enumerate_live_corpus(corpus)
        if item.case_id == "synthetic-reject"
    )
    input_sha = hashlib.sha256(case.input_bytes).hexdigest()
    rust_stdout = rust_stdout_bytes(rust_executions[input_sha])
    assert rust_stdout != json_bytes(json.loads(rust_stdout))
    assert not rust_stdout.endswith(b"\n")

    capture = captures[case.case_id]
    evidence = gate.LiveCaseEvidence(
        comparison_result=gate._live_comparison_from_evidence(
            case, capture, rust_stdout
        ),
        python_capture=capture,
        rust_execution_bytes=rust_stdout,
    )
    gate._validate_live_case_evidence(case, evidence)
    assert evidence.rust_execution_bytes == rust_stdout
    assert evidence.comparison_result["sha256_bindings"]["rust_execution_sha256"] == (
        hashlib.sha256(rust_stdout).hexdigest()
    )


def test_live_corpus_rejects_unbound_input_and_path_traversal(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    write_fake_live_corpus(corpus)
    (corpus / "inputs" / "unbound.json").write_bytes(b"{}")
    with pytest.raises(gate.DifferentialGateError) as unbound:
        gate.enumerate_live_corpus(corpus)
    assert unbound.value.code == "CORPUS_INPUT_SET_NOT_EXACTLY_BOUND"

    (corpus / "inputs" / "unbound.json").unlink()
    expectation_path = corpus / "expectations" / "synthetic-reject.expected.json"
    expectation = json.loads(expectation_path.read_bytes())
    expectation["input_file"] = "inputs/../synthetic.json"
    expectation_path.write_bytes(json_bytes(expectation))
    with pytest.raises(gate.DifferentialGateError) as traversal:
        gate.enumerate_live_corpus(corpus)
    assert traversal.value.code == "CORPUS_INPUT_BINDING_PATH_INVALID"


def test_live_corpus_rejects_symlinked_input(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    write_fake_live_corpus(corpus)
    input_path = corpus / "inputs" / "synthetic.json"
    payload = input_path.read_bytes()
    target = tmp_path / "target.json"
    target.write_bytes(payload)
    input_path.unlink()
    try:
        input_path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.enumerate_live_corpus(corpus)
    assert error.value.code == "CORPUS_SYMLINK_REJECTED"


def test_report_writer_resolves_symlinked_parent_but_rejects_output_symlink(
    tmp_path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    payload = b'{"report":"through-resolved-parent"}'
    gate._write_report_file(alias_parent / "report.json", payload)
    assert (real_parent / "report.json").read_bytes() == payload

    symlink_target = real_parent / "ordinary-file.json"
    symlink_target.write_bytes(b"preserve-me")
    output_symlink = real_parent / "output-link.json"
    output_symlink.symlink_to(symlink_target)
    with pytest.raises(gate.DifferentialGateError) as error:
        gate._write_report_file(alias_parent / "output-link.json", payload)
    assert error.value.code == "OUTPUT_SYMLINK_REJECTED"
    assert symlink_target.read_bytes() == b"preserve-me"


def test_live_bundle_refuses_existing_or_symlink_target_before_capture(
    tmp_path,
) -> None:
    corpus = tmp_path / "corpus"
    write_fake_live_corpus(corpus)
    executable = write_fake_rust_executable(tmp_path)
    capture_called = False

    def forbidden_capture(*_args):
        nonlocal capture_called
        capture_called = True
        pytest.fail("capture must not run for a forbidden bundle target")

    existing = tmp_path / "existing-bundle"
    existing.mkdir()
    with pytest.raises(gate.DifferentialGateError) as existing_error:
        gate.run_live_evidence_bundle(
            corpus,
            executable,
            existing,
            jobs=1,
            python_capture_adapter=forbidden_capture,
            rust_execution_runner=lambda *_args: b"{}",
        )
    assert existing_error.value.code == "LIVE_BUNDLE_TARGET_ALREADY_EXISTS"
    assert capture_called is False

    symlink = tmp_path / "bundle-symlink"
    try:
        symlink.symlink_to(existing, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(gate.DifferentialGateError) as symlink_error:
        gate.run_live_evidence_bundle(
            corpus,
            executable,
            symlink,
            jobs=1,
            python_capture_adapter=forbidden_capture,
            rust_execution_runner=lambda *_args: b"{}",
        )
    assert symlink_error.value.code == "LIVE_BUNDLE_TARGET_SYMLINK_REJECTED"
    assert capture_called is False


def test_live_bundle_resolves_symlinked_parent_once(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)
    executable = write_fake_rust_executable(tmp_path)
    real_parent = tmp_path / "real-bundles"
    real_parent.mkdir()
    alias_parent = tmp_path / "bundle-alias"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    def rust_runner(_executable, input_path):
        payload = input_path.read_bytes()
        return rust_executions[hashlib.sha256(payload).hexdigest()]

    report = gate.run_live_evidence_bundle(
        corpus,
        executable,
        alias_parent / "bundle",
        jobs=1,
        python_capture_adapter=python_adapter,
        rust_execution_runner=rust_runner,
    )
    assert (real_parent / "bundle" / "report.json").read_bytes() == report
    assert not (alias_parent / "bundle").is_symlink()


@pytest.mark.parametrize(
    ("changed_label", "code"),
    (
        ("comparator_source", "COMPARATOR_SOURCE_MUTATED_DURING_CAPTURE"),
        (
            "frozen_specification",
            "FROZEN_SPECIFICATION_MUTATED_DURING_CAPTURE",
        ),
    ),
)
def test_live_bundle_detects_bound_source_drift_during_fake_capture(
    tmp_path, monkeypatch, changed_label, code
) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)
    executable = write_fake_rust_executable(tmp_path)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    def rust_runner(_executable, input_path):
        payload = input_path.read_bytes()
        return rust_executions[hashlib.sha256(payload).hexdigest()]

    original_reader = gate._read_regular_file_bounded
    matching_reads = 0

    def drifting_reader(path, label):
        nonlocal matching_reads
        payload = original_reader(path, label)
        if label == changed_label:
            matching_reads += 1
            if matching_reads == 2:
                return payload + b"drift"
        return payload

    monkeypatch.setattr(gate, "_read_regular_file_bounded", drifting_reader)
    bundle = tmp_path / "bundle"
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.run_live_evidence_bundle(
            corpus,
            executable,
            bundle,
            jobs=1,
            python_capture_adapter=python_adapter,
            rust_execution_runner=rust_runner,
        )
    assert error.value.code == code
    assert matching_reads == 2
    assert not bundle.exists()


def test_live_bundle_cleans_private_sibling_after_build_failure(
    tmp_path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)
    executable = write_fake_rust_executable(tmp_path)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    def rust_runner(_executable, input_path):
        payload = input_path.read_bytes()
        return rust_executions[hashlib.sha256(payload).hexdigest()]

    def failed_bundle_write(_path, _payload):
        raise gate.DifferentialGateError("LIVE_BUNDLE_FILE_WRITE_FAILED", "bundle")

    monkeypatch.setattr(gate, "_write_new_bundle_file", failed_bundle_write)
    bundle = tmp_path / "bundle"
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.run_live_evidence_bundle(
            corpus,
            executable,
            bundle,
            jobs=1,
            python_capture_adapter=python_adapter,
            rust_execution_runner=rust_runner,
        )
    assert error.value.code == "LIVE_BUNDLE_FILE_WRITE_FAILED"
    assert not bundle.exists()
    assert list(tmp_path.glob(".bundle.private-*")) == []


def test_live_jobs_are_bounded_before_capture(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    write_fake_live_corpus(corpus)
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.run_live_corpus(
            corpus,
            "fake-rust",
            jobs=3,
            python_capture_adapter=lambda *_args: None,
            rust_execution_runner=lambda *_args: b"{}",
        )
    assert error.value.code == "LIVE_JOBS_OUT_OF_RANGE"


def test_live_rust_handoff_detects_private_input_mutation(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    captures, rust_executions = write_fake_live_corpus(corpus)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    def mutating_rust_runner(_executable, input_path):
        original = input_path.read_bytes()
        execution = rust_executions[hashlib.sha256(original).hexdigest()]
        input_path.write_bytes(b"mutated")
        return execution

    with pytest.raises(gate.DifferentialGateError) as error:
        gate.run_live_corpus(
            corpus,
            "fake-rust",
            jobs=1,
            python_capture_adapter=python_adapter,
            rust_execution_runner=mutating_rust_runner,
        )
    assert error.value.code == "RUST_PRIVATE_INPUT_MUTATED"


def test_live_rust_handoff_rejects_zero_byte_private_input_write(
    tmp_path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    captures, _rust_executions = write_fake_live_corpus(corpus)

    def python_adapter(case_id, _input_bytes, _expected_parse):
        return captures[case_id]

    monkeypatch.setattr(gate.os, "write", lambda _descriptor, _payload: 0)
    with pytest.raises(gate.DifferentialGateError) as error:
        gate.run_live_corpus(
            corpus,
            "fake-rust",
            jobs=1,
            python_capture_adapter=python_adapter,
            rust_execution_runner=lambda *_args: pytest.fail(
                "Rust must not run after a failed private input write"
            ),
        )
    assert error.value.code == "RUST_PRIVATE_INPUT_WRITE_FAILED"


def test_python_reject_adapter_catches_only_public_artifact_error(monkeypatch) -> None:
    from three_body_symmetry import planar_chain_review_artifact as artifact

    def internal_failure(_payload):
        raise RuntimeError("internal")

    monkeypatch.setattr(
        artifact, "strict_load_raw_planar_chain_bytes", internal_failure
    )
    with pytest.raises(RuntimeError, match="internal"):
        gate.capture_python_raw_v1("reject", b"bad", "REJECT")

    def public_reject(_payload):
        raise artifact.ReviewArtifactError("expected")

    monkeypatch.setattr(artifact, "strict_load_raw_planar_chain_bytes", public_reject)
    capture = gate.capture_python_raw_v1("reject", b"bad", "REJECT")
    assert capture.kind == "PARSER_REJECT_OBSERVATION"
    assert "expected" not in capture.payload.decode()

    with pytest.raises(gate.DifferentialGateError) as accepted_error:
        gate.capture_python_raw_v1("accept", b"bad", "ACCEPT")
    assert accepted_error.value.code == "PYTHON_EXPECTED_ACCEPT_WAS_REJECTED"


def test_real_reject_live_capture_and_rust_envelope_are_compatible() -> None:
    if not RUST_VERIFIER.is_file():
        pytest.skip("debug Rust verifier executable is not built")
    case = next(
        item
        for item in gate.enumerate_live_corpus(CORPUS_ROOT)
        if item.case_id == "reject-trailing-newline"
    )
    result = gate._compare_live_case(
        case,
        RUST_VERIFIER,
        gate.capture_python_raw_v1,
        gate.run_rust_raw_v1,
    )
    assert result["differential_gate_passed"] is True
    assert result["comparison_status"] == "PARSER_PROFILE_COMPATIBILITY_OBSERVED"
    assert result["python_capture_kind"] == "PARSER_REJECT_OBSERVATION"
    assert result["sha256_bindings"]["python_capture_sha256"] is not None
    assert result["sha256_bindings"]["rust_execution_sha256"] is not None


def test_direct_script_live_command_runs_one_cheap_real_reject(tmp_path) -> None:
    if not RUST_VERIFIER.is_file():
        pytest.skip("debug Rust verifier executable is not built")
    corpus = tmp_path / "corpus"
    (corpus / "expectations").mkdir(parents=True)
    (corpus / "inputs").mkdir()
    expectation_name = "reject-trailing-newline.expected.json"
    input_name = "reject-trailing-newline.json"
    (corpus / "expectations" / expectation_name).write_bytes(
        (CORPUS_ROOT / "expectations" / expectation_name).read_bytes()
    )
    (corpus / "inputs" / input_name).write_bytes(
        (CORPUS_ROOT / "inputs" / input_name).read_bytes()
    )
    output = tmp_path / "live-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "live",
            "--corpus-root",
            corpus,
            "--rust-verifier",
            RUST_VERIFIER,
            "--output",
            output,
            "--jobs",
            "1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""
    report_bytes = output.read_bytes()
    assert not report_bytes.endswith(b"\n")
    report = json.loads(report_bytes)
    assert report["capture_mode"] == gate.LIVE_CAPTURE_MODE
    assert report["cases"][0]["python_capture_kind"] == "PARSER_REJECT_OBSERVATION"


@pytest.mark.parametrize(
    ("passed", "expected_exit", "expected_stderr"),
    (
        (True, 0, ""),
        (False, 1, "raw-v1-live-bundle:DIFFERENTIAL_DISAGREEMENT\n"),
    ),
)
def test_live_bundle_cli_has_distinct_command_and_gate_exit_semantics(
    tmp_path, capsys, passed, expected_exit, expected_stderr
) -> None:
    case = gate.compare_case("synthetic-reject", *rejected_fixture())
    if not passed:
        case["differential_gate_passed"] = False
        case["mismatch_ids"] = ["SYNTHETIC_DIFFERENTIAL_MISMATCH"]
    report = gate.build_report([case], capture_mode=gate.LIVE_CAPTURE_MODE)
    bundle = tmp_path / f"bundle-{passed}"
    calls = []

    def fake_bundle_runner(corpus, rust, bundle_dir, *, jobs):
        calls.append((corpus, rust, bundle_dir, jobs))
        bundle_dir.mkdir()
        (bundle_dir / "report.json").write_bytes(report)
        return report

    def forbidden_live_runner(*_args, **_kwargs):
        pytest.fail("the live command runner must not handle live-bundle")

    exit_code = gate.main(
        [
            "live-bundle",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--rust-verifier",
            str(tmp_path / "rust"),
            "--bundle-dir",
            str(bundle),
            "--jobs",
            "1",
        ],
        live_runner=forbidden_live_runner,
        live_bundle_runner=fake_bundle_runner,
    )
    assert exit_code == expected_exit
    assert calls == [
        (tmp_path / "corpus", tmp_path / "rust", bundle, 1),
    ]
    assert (bundle / "report.json").read_bytes() == report
    assert capsys.readouterr().err == expected_stderr


def test_live_bundle_cli_capture_failure_has_stable_distinct_stderr(
    tmp_path, capsys
) -> None:
    def failed_runner(*_args, **_kwargs):
        raise gate.DifferentialGateError("FAKE_BUNDLE_FAILURE", "bundle")

    exit_code = gate.main(
        [
            "live-bundle",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--rust-verifier",
            str(tmp_path / "rust"),
            "--bundle-dir",
            str(tmp_path / "bundle"),
        ],
        live_bundle_runner=failed_runner,
    )
    assert exit_code == 2
    assert capsys.readouterr().err == "raw-v1-live-bundle:FAKE_BUNDLE_FAILURE\n"


def test_live_cli_writes_canonical_report_and_uses_gate_result_for_exit(
    tmp_path, capsys
) -> None:
    output = tmp_path / "report.json"
    case = gate.compare_case("synthetic-reject", *rejected_fixture())
    report = gate.build_report([case], capture_mode=gate.LIVE_CAPTURE_MODE)

    def fake_live_runner(_corpus, _rust, *, jobs):
        assert jobs == 1
        return report

    exit_code = gate.main(
        [
            "live",
            "--corpus-root",
            str(tmp_path),
            "--rust-verifier",
            str(tmp_path / "rust"),
            "--output",
            str(output),
            "--jobs",
            "1",
        ],
        live_runner=fake_live_runner,
    )
    assert exit_code == 0
    assert output.read_bytes() == report
    assert not output.read_bytes().endswith(b"\n")
    assert capsys.readouterr().err == ""


def test_live_cli_capture_failure_has_stable_stderr(tmp_path, capsys) -> None:
    def failed_runner(*_args, **_kwargs):
        raise gate.DifferentialGateError("FAKE_CAPTURE_FAILURE", "fake")

    exit_code = gate.main(
        [
            "live",
            "--corpus-root",
            str(tmp_path),
            "--rust-verifier",
            str(tmp_path / "rust"),
            "--output",
            str(tmp_path / "report.json"),
        ],
        live_runner=failed_runner,
    )
    assert exit_code == 2
    assert capsys.readouterr().err == "raw-v1-live:FAKE_CAPTURE_FAILURE\n"


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
