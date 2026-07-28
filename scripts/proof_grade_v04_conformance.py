#!/usr/bin/env python3
"""Validate and replay the bounded proof-grade raw-v1 v0.4 foundation.

This module deliberately imports no project verifier code.  It validates a
hash-bound corpus contract and, when asked, treats a supplied executable as an
opaque process whose compact JSON output must match that contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parent.parent
CORPUS_RELATIVE = Path("conformance/raw-v1/proof-grade-v04")
CORPUS_ROOT = REPOSITORY_ROOT / CORPUS_RELATIVE

CASES_SCHEMA = "raw-v1-proof-grade-v04-cases-v1"
MANIFEST_SCHEMA = "raw-v1-proof-grade-v04-manifest-v1"
CORPUS_STATUS = "foundation_incomplete"
EXECUTION_SCHEMA = "raw-v1-rust-proof-grade-execution-v1"
EXECUTION_PROFILE = "exact_rational_proof_grade_raw_v1_execution_v04"
SEMANTIC_SCHEMA = "raw-v1-rust-proof-grade-semantic-outcome-v1"
SEMANTIC_PROFILE = "exact_rational_proof_grade_admitted_raw_v1_outcome_v04"
FROZEN_SPEC_PATH = "docs/raw-v1-certificate-specification.md"
FROZEN_SPEC_SHA256 = "18a131cb559dc74d86c33a61de8b252402b2b2e41755f789f66bc67f61238cac"

PROOF_OBLIGATION_IDS = [
    "proof_grade_raw_planar_chain_outer_schema_exact",
    "proof_grade_raw_planar_chain_global_identifier_namespace_unique",
    "proof_grade_raw_planar_chain_canonical_evidence_serializable",
    "proof_grade_raw_planar_chain_requested_target_finite",
    "proof_grade_raw_planar_chain_requested_width_admissible",
    "proof_grade_raw_planar_chain_root_exact_point_left_anchor",
    "proof_grade_raw_planar_chain_root_direct_tube_and_binding_certified",
    "proof_grade_raw_planar_chain_all_segments_direct_evidence_folded",
    "proof_grade_raw_planar_chain_target_not_before_current_left_clock",
    "proof_grade_raw_planar_chain_fixed_time_preimage_exactly_derived",
    "proof_grade_raw_planar_chain_fixed_time_preimage_inside_forward_current_domain",
    "proof_grade_raw_planar_chain_target_state_directly_evaluated_and_inflated",
    "proof_grade_raw_planar_chain_final_component_width_within_requested_bound",
]

SEGMENT_PROFILES = [
    {
        "segment_index": 0,
        "kind": "ordinary_bridge",
        "profile": "exact_rational_carried_ordinary_bridge_v04",
        "pair": None,
    },
    {
        "segment_index": 1,
        "kind": "planar_lc_passage",
        "profile": "exact_rational_proof_grade_carried_planar_lc_exit_v04",
        "pair": [0, 1],
    },
    {
        "segment_index": 2,
        "kind": "planar_lc_passage",
        "profile": "exact_rational_proof_grade_carried_planar_lc_exit_v04",
        "pair": [0, 2],
    },
    {
        "segment_index": 3,
        "kind": "planar_lc_passage",
        "profile": "exact_rational_proof_grade_carried_planar_lc_exit_v04",
        "pair": [1, 2],
    },
    {
        "segment_index": 4,
        "kind": "planar_lc_passage",
        "profile": "exact_rational_proof_grade_carried_planar_lc_exit_v04",
        "pair": [0, 1],
    },
]

REJECTION_STAGES = {
    "reject-alternate-real-spelling.json": "CANONICAL_WIRE",
    "reject-decoded-duplicate-unicode-key.json": "CANONICAL_WIRE",
    "reject-duplicate-outer-key.json": "CANONICAL_WIRE",
    "reject-invalid-utf8.bin": "CANONICAL_WIRE",
    "reject-leading-whitespace.json": "CANONICAL_WIRE",
    "reject-lone-surrogate.json": "CANONICAL_WIRE",
    "reject-missing-outer-field.json": "SCHEMA",
    "reject-nonfinite-token.json": "CANONICAL_WIRE",
    "reject-overflow-real.json": "CANONICAL_WIRE",
    "reject-trailing-newline.json": "CANONICAL_WIRE",
    "reject-unknown-outer-field.json": "SCHEMA",
    "reject-unknown-segment-tag.json": "SCHEMA",
    "reject-unsorted-outer-keys.json": "CANONICAL_WIRE",
    "reject-wrong-interval-length.json": "SCHEMA",
    "reject-wrong-outer-scalar-class.json": "SCHEMA",
    "reject-wrong-pair-length.json": "SCHEMA",
}

CASE_ID_BY_INPUT = {
    "success.raw.json": "success",
    "failed-revisit.raw.json": "failed-revisit",
    **{name: name.rsplit(".", 1)[0] for name in REJECTION_STAGES},
}

LOCAL_CORPUS_FILES = {
    "README.md",
    "cases.schema.json",
    "cases.json",
    "manifest.json",
}
RATIONAL_NUMERATOR = re.compile(r"^(0|[1-9][0-9]*|-[1-9][0-9]*)$")
RATIONAL_DENOMINATOR = re.compile(r"^[1-9][0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SEMANTIC_KEYS = {
    "diagnostics",
    "tail_bound",
    "resource",
    "required",
    "limit",
    "source_error",
}


class ConformanceError(ValueError):
    """A deterministic corpus or executable-contract violation."""


def _fail(path: str, detail: str) -> None:
    raise ConformanceError(f"{path}: {detail}")


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConformanceError(f"JSON duplicate key: {key}")
        result[key] = value
    return result


def _forbid_nonfinite(value: str) -> None:
    raise ConformanceError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(str(path), f"cannot read: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_forbid_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ConformanceError) as error:
        _fail(str(path), f"invalid JSON: {error}")
    if not isinstance(value, dict):
        _fail(str(path), "top level must be an object")
    return value


def _exact_keys(value: Any, required: Iterable[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    required_set = set(required)
    actual_set = set(value)
    if actual_set != required_set:
        _fail(path, f"keys must be exactly {sorted(required_set)}, found {sorted(actual_set)}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        _fail(str(path), f"cannot hash: {error}")
    raise AssertionError("unreachable")


def _relative_path(path: str, field: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or path != candidate.as_posix():
        _fail(field, "must be a normalized repository-relative path")
    return candidate


def _validate_binding(
    value: Any,
    path: str,
    repository_root: Path,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    binding = _exact_keys(value, {"repository_relative_path", "sha256"}, path)
    relative = binding["repository_relative_path"]
    digest = binding["sha256"]
    if not isinstance(relative, str) or not isinstance(digest, str):
        _fail(path, "binding path and sha256 must be strings")
    _relative_path(relative, f"{path}.repository_relative_path")
    if not SHA256.fullmatch(digest):
        _fail(f"{path}.sha256", "must be lower-case SHA-256")
    if expected_path is not None and relative != expected_path:
        _fail(f"{path}.repository_relative_path", f"must equal {expected_path}")
    if expected_sha256 is not None and digest != expected_sha256:
        _fail(f"{path}.sha256", "does not match frozen binding")
    actual = _sha256(repository_root / relative)
    if actual != digest:
        _fail(f"{path}.sha256", f"does not match {relative}")
    return {"repository_relative_path": relative, "sha256": digest}


def _validate_rational(value: Any, path: str) -> None:
    rational = _exact_keys(value, {"numerator", "denominator"}, path)
    numerator = rational["numerator"]
    denominator = rational["denominator"]
    if numerator == "-0":
        _fail(f"{path}.numerator", "negative zero is forbidden")
    if not isinstance(numerator, str) or not RATIONAL_NUMERATOR.fullmatch(numerator):
        _fail(f"{path}.numerator", "must be a canonical signed decimal string")
    if not isinstance(denominator, str) or not RATIONAL_DENOMINATOR.fullmatch(denominator):
        _fail(f"{path}.denominator", "must be a positive canonical decimal string")


def _validate_signature(value: Any, path: str) -> dict[str, Any]:
    signature = _exact_keys(
        value,
        {
            "schema",
            "profile",
            "status",
            "certificate_id",
            "evidence_sha256",
            "request",
            "obligations",
            "first_failed_obligation",
            "certified_segment_count",
            "failed_segment_index",
            "failed_segment_missing_obligations",
            "segment_profiles",
            "clock_ledger_count",
            "current_chart_id",
            "presence",
            "final_enclosure",
            "retained_region",
        },
        path,
    )
    if signature["schema"] != SEMANTIC_SCHEMA or signature["profile"] != SEMANTIC_PROFILE:
        _fail(path, "semantic schema/profile mismatch")
    if signature["status"] not in {"CERTIFIED_TO_T", "UNRESOLVED"}:
        _fail(f"{path}.status", "must be a proof-grade semantic status")
    if not isinstance(signature["certificate_id"], str) or not signature["certificate_id"]:
        _fail(f"{path}.certificate_id", "must be a nonempty string")
    if not isinstance(signature["evidence_sha256"], str) or not SHA256.fullmatch(
        signature["evidence_sha256"]
    ):
        _fail(f"{path}.evidence_sha256", "must be lower-case SHA-256")

    request = _exact_keys(
        signature["request"], {"target_physical_time", "maximum_component_width"}, f"{path}.request"
    )
    _validate_rational(request["target_physical_time"], f"{path}.request.target_physical_time")
    _validate_rational(
        request["maximum_component_width"], f"{path}.request.maximum_component_width"
    )

    obligations = signature["obligations"]
    if not isinstance(obligations, list) or len(obligations) != len(PROOF_OBLIGATION_IDS):
        _fail(f"{path}.obligations", "must contain exactly thirteen rows")
    obligation_ids: list[str] = []
    for index, obligation in enumerate(obligations):
        row = _exact_keys(obligation, {"id", "certified"}, f"{path}.obligations[{index}]")
        if not isinstance(row["id"], str) or not isinstance(row["certified"], bool):
            _fail(f"{path}.obligations[{index}]", "invalid obligation row types")
        obligation_ids.append(row["id"])
    if obligation_ids != PROOF_OBLIGATION_IDS:
        _fail(f"{path}.obligations", "proof obligation identifiers/order mismatch")

    first = signature["first_failed_obligation"]
    if first is not None and (not isinstance(first, str) or not first):
        _fail(f"{path}.first_failed_obligation", "must be null or a nonempty string")
    for name in ("certified_segment_count", "clock_ledger_count"):
        if not isinstance(signature[name], int) or isinstance(signature[name], bool) or signature[name] < 0:
            _fail(f"{path}.{name}", "must be a nonnegative integer")
    failed_index = signature["failed_segment_index"]
    if failed_index is not None and (
        not isinstance(failed_index, int) or isinstance(failed_index, bool) or failed_index < 0
    ):
        _fail(f"{path}.failed_segment_index", "must be null or a nonnegative integer")
    missing = signature["failed_segment_missing_obligations"]
    if not isinstance(missing, list) or not all(isinstance(item, str) and item for item in missing):
        _fail(f"{path}.failed_segment_missing_obligations", "must be strings")

    segments = signature["segment_profiles"]
    if segments != SEGMENT_PROFILES:
        _fail(f"{path}.segment_profiles", "must match the archived five-segment word")
    chart_id = signature["current_chart_id"]
    if chart_id is not None and (not isinstance(chart_id, str) or not chart_id):
        _fail(f"{path}.current_chart_id", "must be null or a nonempty string")
    presence = _exact_keys(
        signature["presence"],
        {
            "current_clock_origin",
            "covered_physical_time_interval",
            "target_parameter_preimage_interval",
            "maximum_final_component_width",
        },
        f"{path}.presence",
    )
    if not all(isinstance(item, bool) for item in presence.values()):
        _fail(f"{path}.presence", "all presence values must be Boolean")

    _validate_terminal_descriptor(signature["final_enclosure"], f"{path}.final_enclosure", "enclosure_type")
    _validate_terminal_descriptor(signature["retained_region"], f"{path}.retained_region", "region_type")
    return dict(signature)


def _validate_terminal_descriptor(value: Any, path: str, name: str) -> None:
    if value is None:
        return
    if name == "enclosure_type":
        descriptor = _exact_keys(value, {"enclosure_type", "coordinate_system", "component_count"}, path)
    else:
        descriptor = _exact_keys(
            value, {"region_type", "coordinate_system", "pair", "component_count"}, path
        )
        pair = descriptor["pair"]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item not in {0, 1, 2} for item in pair)
        ):
            _fail(f"{path}.pair", "must be a two-body-index pair")
    if not isinstance(descriptor[name], str) or not descriptor[name]:
        _fail(f"{path}.{name}", "must be a nonempty string")
    if not isinstance(descriptor["coordinate_system"], str) or not descriptor["coordinate_system"]:
        _fail(f"{path}.coordinate_system", "must be a nonempty string")
    count = descriptor["component_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        _fail(f"{path}.component_count", "must be a positive integer")


def _validate_cases(cases_value: dict[str, Any], repository_root: Path) -> list[dict[str, Any]]:
    cases = _exact_keys(
        cases_value,
        {"schema", "corpus_status", "frozen_specification", "rust_execution", "cases"},
        "cases",
    )
    if cases["schema"] != CASES_SCHEMA or cases["corpus_status"] != CORPUS_STATUS:
        _fail("cases", "schema or corpus status mismatch")
    _validate_binding(
        cases["frozen_specification"],
        "cases.frozen_specification",
        repository_root,
        FROZEN_SPEC_PATH,
        FROZEN_SPEC_SHA256,
    )
    rust_execution = _exact_keys(cases["rust_execution"], {"schema", "profile"}, "cases.rust_execution")
    if rust_execution != {"schema": EXECUTION_SCHEMA, "profile": EXECUTION_PROFILE}:
        _fail("cases.rust_execution", "proof-grade execution boundary mismatch")
    raw_cases = cases["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(CASE_ID_BY_INPUT):
        _fail("cases.cases", "must contain exactly the eighteen foundation cases")

    actual_inputs: set[str] = set()
    actual_case_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        path = f"cases.cases[{index}]"
        case = _exact_keys(raw_case, {"case_id", "input_path", "input_sha256", "expected"}, path)
        case_id = case["case_id"]
        input_path = case["input_path"]
        digest = case["input_sha256"]
        if not isinstance(case_id, str) or not isinstance(input_path, str) or not isinstance(digest, str):
            _fail(path, "case id, input path, and input sha256 must be strings")
        expected_case_id = CASE_ID_BY_INPUT.get(Path(input_path).name)
        if expected_case_id is None or input_path != f"conformance/raw-v1/inputs/{Path(input_path).name}":
            _fail(f"{path}.input_path", "is not a foundation input path")
        if case_id != expected_case_id:
            _fail(f"{path}.case_id", "does not match its input filename")
        if case_id in actual_case_ids or input_path in actual_inputs:
            _fail(path, "duplicate case id or input path")
        if not SHA256.fullmatch(digest):
            _fail(f"{path}.input_sha256", "must be lower-case SHA-256")
        actual_case_ids.add(case_id)
        actual_inputs.add(input_path)
        if _sha256(repository_root / input_path) != digest:
            _fail(f"{path}.input_sha256", "does not bind its input bytes")

        expected = _exact_keys(
            case["expected"],
            {"parse_outcome", "rejection_stage", "evaluation_outcome", "evaluation_error_stage", "semantic_signature"},
            f"{path}.expected",
        )
        rejection_stage = REJECTION_STAGES.get(Path(input_path).name)
        if rejection_stage is None:
            if expected["parse_outcome"] != "ACCEPT" or expected["rejection_stage"] is not None:
                _fail(f"{path}.expected", "admitted case must accept without a rejection stage")
            if expected["evaluation_outcome"] != "RESULT" or expected["evaluation_error_stage"] is not None:
                _fail(f"{path}.expected", "admitted case must be a Boolean semantic result")
            _validate_signature(expected["semantic_signature"], f"{path}.expected.semantic_signature")
        else:
            if expected != {
                "parse_outcome": "REJECT",
                "rejection_stage": rejection_stage,
                "evaluation_outcome": "NOT_RUN",
                "evaluation_error_stage": None,
                "semantic_signature": None,
            }:
                _fail(f"{path}.expected", "rejection classification differs from the fixed table")
        validated.append(dict(case))
    if actual_case_ids != set(CASE_ID_BY_INPUT.values()) or actual_inputs != {
        f"conformance/raw-v1/inputs/{name}" for name in CASE_ID_BY_INPUT
    }:
        _fail("cases.cases", "missing or extra foundation case")
    return validated


def validate(
    corpus_root: Path = CORPUS_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    """Strictly validate hashes, structures, and the fixed foundation table."""
    corpus_root = corpus_root.resolve()
    repository_root = repository_root.resolve()
    actual_local_files = {entry.name for entry in corpus_root.iterdir() if entry.is_file()}
    if actual_local_files != LOCAL_CORPUS_FILES:
        _fail("corpus", f"local files must be exactly {sorted(LOCAL_CORPUS_FILES)}")
    cases_path = corpus_root / "cases.json"
    schema_path = corpus_root / "cases.schema.json"
    readme_path = corpus_root / "README.md"
    manifest_path = corpus_root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest = _exact_keys(
        manifest,
        {
            "manifest_schema",
            "corpus_status",
            "hash_algorithm",
            "manifest_self_hash",
            "manifest_self_hash_exclusion",
            "frozen_specification",
            "cases_file",
            "schema_file",
            "readme_file",
            "runner_script",
            "referenced_inputs",
        },
        "manifest",
    )
    if manifest["manifest_schema"] != MANIFEST_SCHEMA or manifest["corpus_status"] != CORPUS_STATUS:
        _fail("manifest", "schema or corpus status mismatch")
    if manifest["hash_algorithm"] != "sha256":
        _fail("manifest.hash_algorithm", "must be sha256")
    if manifest["manifest_self_hash"] is not None:
        _fail("manifest.manifest_self_hash", "must be null by explicit self-hash exclusion")
    exclusion = manifest["manifest_self_hash_exclusion"]
    if not isinstance(exclusion, str) or "excluded" not in exclusion.lower():
        _fail("manifest.manifest_self_hash_exclusion", "must explicitly declare the exclusion")
    _validate_binding(
        manifest["frozen_specification"],
        "manifest.frozen_specification",
        repository_root,
        FROZEN_SPEC_PATH,
        FROZEN_SPEC_SHA256,
    )
    expected_files = {
        "cases_file": "conformance/raw-v1/proof-grade-v04/cases.json",
        "schema_file": "conformance/raw-v1/proof-grade-v04/cases.schema.json",
        "readme_file": "conformance/raw-v1/proof-grade-v04/README.md",
        "runner_script": "scripts/proof_grade_v04_conformance.py",
    }
    for field, expected_path in expected_files.items():
        _validate_binding(manifest[field], f"manifest.{field}", repository_root, expected_path)
    inputs = manifest["referenced_inputs"]
    if not isinstance(inputs, list) or len(inputs) != len(CASE_ID_BY_INPUT):
        _fail("manifest.referenced_inputs", "must bind exactly eighteen inputs")
    manifest_inputs = [_validate_binding(item, f"manifest.referenced_inputs[{index}]", repository_root) for index, item in enumerate(inputs)]
    if {item["repository_relative_path"] for item in manifest_inputs} != {
        f"conformance/raw-v1/inputs/{name}" for name in CASE_ID_BY_INPUT
    }:
        _fail("manifest.referenced_inputs", "missing or extra input binding")

    schema = _load_json(schema_path)
    if schema.get("$id") != "raw-v1-proof-grade-v04-cases-schema-v1":
        _fail("cases.schema.json", "schema id mismatch")
    if schema.get("additionalProperties") is not False:
        _fail("cases.schema.json", "root must reject extra fields")
    cases = _validate_cases(_load_json(cases_path), repository_root)
    case_input_bindings = {
        case["input_path"]: case["input_sha256"] for case in cases
    }
    manifest_input_bindings = {
        item["repository_relative_path"]: item["sha256"] for item in manifest_inputs
    }
    if case_input_bindings != manifest_input_bindings:
        _fail("manifest.referenced_inputs", "must exactly match cases.json input bindings")
    if _sha256(cases_path) != manifest["cases_file"]["sha256"]:
        _fail("manifest.cases_file", "cases file hash mismatch")
    if _sha256(schema_path) != manifest["schema_file"]["sha256"]:
        _fail("manifest.schema_file", "schema file hash mismatch")
    if _sha256(readme_path) != manifest["readme_file"]["sha256"]:
        _fail("manifest.readme_file", "README hash mismatch")
    if _sha256(SCRIPT_PATH) != manifest["runner_script"]["sha256"]:
        _fail("manifest.runner_script", "runner script hash mismatch")
    return cases


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _assert_rationals_are_canonical(value: Any, path: str = "semantic_result") -> None:
    if isinstance(value, dict):
        rational_keys = {"numerator", "denominator"}
        if rational_keys & set(value):
            if set(value) != rational_keys:
                _fail(path, "rational-looking object must contain exactly numerator and denominator")
            _validate_rational(value, path)
        for key, nested in value.items():
            _assert_rationals_are_canonical(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_rationals_are_canonical(nested, f"{path}[{index}]")


def _assert_no_private_semantic_keys(value: Any, path: str = "semantic_result") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_SEMANTIC_KEYS:
                _fail(path, f"forbidden private/diagnostic key {key!r}")
            _assert_no_private_semantic_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_private_semantic_keys(nested, f"{path}[{index}]")


def _semantic_signature(value: Any) -> dict[str, Any]:
    semantic = _exact_keys(
        value,
        {
            "schema",
            "profile",
            "status",
            "certificate_id",
            "evidence_sha256",
            "request",
            "top_level_obligations",
            "first_failed_obligation",
            "certified_segment_count",
            "failed_segment_index",
            "failed_segment_missing_obligations",
            "segment_profiles",
            "clock_ledger",
            "current_chart",
            "current_clock_origin",
            "covered_physical_time_interval",
            "target_parameter_preimage_interval",
            "maximum_final_component_width",
            "final_enclosure",
            "retained_region",
        },
        "semantic_result",
    )
    _assert_no_private_semantic_keys(semantic)
    _assert_rationals_are_canonical(semantic)
    chart = semantic["current_chart"]
    if chart is not None and not isinstance(chart, dict):
        _fail("semantic_result.current_chart", "must be null or an object")
    final = semantic["final_enclosure"]
    retained = semantic["retained_region"]
    return {
        "schema": semantic["schema"],
        "profile": semantic["profile"],
        "status": semantic["status"],
        "certificate_id": semantic["certificate_id"],
        "evidence_sha256": semantic["evidence_sha256"],
        "request": semantic["request"],
        "obligations": [
            {"id": row.get("obligation"), "certified": row.get("certified")}
            for row in semantic["top_level_obligations"]
            if isinstance(row, dict)
        ],
        "first_failed_obligation": semantic["first_failed_obligation"],
        "certified_segment_count": semantic["certified_segment_count"],
        "failed_segment_index": semantic["failed_segment_index"],
        "failed_segment_missing_obligations": semantic["failed_segment_missing_obligations"],
        "segment_profiles": semantic["segment_profiles"],
        "clock_ledger_count": len(semantic["clock_ledger"]) if isinstance(semantic["clock_ledger"], list) else -1,
        "current_chart_id": None if chart is None else chart.get("chart_id"),
        "presence": {
            "current_clock_origin": semantic["current_clock_origin"] is not None,
            "covered_physical_time_interval": semantic["covered_physical_time_interval"] is not None,
            "target_parameter_preimage_interval": semantic["target_parameter_preimage_interval"] is not None,
            "maximum_final_component_width": semantic["maximum_final_component_width"] is not None,
        },
        "final_enclosure": None
        if final is None
        else {
            "enclosure_type": final.get("enclosure_type"),
            "coordinate_system": final.get("coordinate_system"),
            "component_count": len(final.get("components", [])),
        },
        "retained_region": None
        if retained is None
        else {
            "region_type": retained.get("region_type"),
            "coordinate_system": retained.get("coordinate_system"),
            "pair": retained.get("pair"),
            "component_count": len(retained.get("components", [])),
        },
    }


def _validate_execution_bytes(payload: bytes, expected: Mapping[str, Any], case_id: str) -> None:
    if payload.endswith(b"\n"):
        _fail(case_id, "stdout must not end in a newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(case_id, f"stdout is not UTF-8: {error}")
    try:
        execution = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_forbid_nonfinite,
        )
    except (json.JSONDecodeError, ConformanceError) as error:
        _fail(case_id, f"stdout is not strict JSON: {error}")
    if _compact_json_bytes(execution) != payload:
        _fail(case_id, "stdout must be compact deterministic JSON")
    envelope = _exact_keys(
        execution,
        {
            "schema",
            "profile",
            "parse_outcome",
            "rejection_stage",
            "evaluation_outcome",
            "evaluation_error_stage",
            "semantic_result",
        },
        f"{case_id}.execution",
    )
    observed_classification = {
        key: envelope[key]
        for key in ("parse_outcome", "rejection_stage", "evaluation_outcome", "evaluation_error_stage")
    }
    expected_classification = {
        key: expected[key]
        for key in ("parse_outcome", "rejection_stage", "evaluation_outcome", "evaluation_error_stage")
    }
    if envelope["schema"] != EXECUTION_SCHEMA or envelope["profile"] != EXECUTION_PROFILE:
        _fail(case_id, "proof-grade execution schema/profile mismatch")
    if observed_classification != expected_classification:
        _fail(case_id, "execution classification mismatch")
    expected_signature = expected["semantic_signature"]
    if expected_signature is None:
        if envelope["semantic_result"] is not None:
            _fail(case_id, "rejection must have null semantic_result")
    else:
        if envelope["semantic_result"] is None:
            _fail(case_id, "admitted result must have a semantic_result")
        actual_signature = _semantic_signature(envelope["semantic_result"])
        if actual_signature != expected_signature:
            _fail(case_id, "semantic signature mismatch")


def run_rust(
    rust_executable: Path,
    repeat: int = 1,
    *,
    corpus_root: Path = CORPUS_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    case_ids: Sequence[str] | None = None,
    timeout_seconds: float = 60.0,
) -> None:
    """Run the opaque Rust proof-grade executable against the whole contract."""
    if repeat < 1:
        _fail("repeat", "must be at least one")
    cases = validate(corpus_root=corpus_root, repository_root=repository_root)
    rust_executable = rust_executable.resolve()
    if not rust_executable.is_file():
        _fail("rust_executable", "must name an executable file")
    selected = cases
    if case_ids is not None:
        wanted = set(case_ids)
        selected = [case for case in cases if case["case_id"] in wanted]
        if not wanted or {case["case_id"] for case in selected} != wanted:
            _fail("case_ids", "selects a missing or extra corpus case")
    for case in selected:
        outputs: list[bytes] = []
        input_path = repository_root / case["input_path"]
        for attempt in range(repeat):
            try:
                completed = subprocess.run(
                    [str(rust_executable), str(input_path)],
                    check=False,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except OSError as error:
                _fail(case["case_id"], f"cannot run rust executable: {error}")
            except subprocess.TimeoutExpired:
                _fail(case["case_id"], "rust executable exceeded bounded timeout")
            if completed.returncode != 0:
                _fail(case["case_id"], f"rust executable exit {completed.returncode}, expected 0")
            if completed.stderr:
                _fail(case["case_id"], "rust executable wrote stderr")
            _validate_execution_bytes(completed.stdout, case["expected"], case["case_id"])
            outputs.append(completed.stdout)
            if attempt and outputs[-1] != outputs[0]:
                _fail(case["case_id"], "repeated output bytes differ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate strict corpus metadata and hashes")
    run_parser = subparsers.add_parser("run-rust", help="run an opaque Rust proof-grade executable")
    run_parser.add_argument("--rust-executable", required=True, type=Path)
    run_parser.add_argument("--repeat", default=1, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            cases = validate()
            print(f"proof-grade-v04 foundation validated: {len(cases)} cases")
        else:
            run_rust(args.rust_executable, args.repeat)
            print(f"proof-grade-v04 rust replay passed: repeat={args.repeat}")
    except ConformanceError as error:
        print(f"proof-grade-v04 conformance failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
