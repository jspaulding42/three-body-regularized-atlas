#!/usr/bin/env python3
"""Raw-v1 cross-profile compatibility comparison and bounded live driver.

Both the pure offline API and the live corpus driver compare profile-labeled
artifacts only.  Neither is a theorem oracle, an independent proof, nor
evidence of mathematical agreement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from math import gcd
import os
from pathlib import Path, PurePosixPath
import selectors
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


MAX_DECIMAL_DIGITS = 100_000
MAX_JSON_INTEGER_DIGITS = 100_000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_RUST_STDOUT_BYTES = 64 * 1024 * 1024
MAX_RUST_STDERR_BYTES = 64 * 1024
RUST_TIMEOUT_SECONDS = 600.0
MAX_LIVE_JOBS = 2

EXPECTATION_SCHEMA = "raw-v1-seed-expectation-v1"
PYTHON_TRANSCRIPT_SCHEMA = "raw-planar-chain-replay-transcript-v1"
PYTHON_TRANSCRIPT_VERSION = "v0.3.0-review"
PYTHON_PROFILE = "frozen_python_raw_planar_chain_replay_v03"
PYTHON_SOURCE_MARKER = "comparator_assigned_to_unlabeled_v0.3_transcript"
RUST_EXECUTION_SCHEMA = "raw-v1-rust-execution-v1"
RUST_EXECUTION_PROFILE = "exact_rational_raw_v1_execution_v04"
RUST_SEMANTIC_SCHEMA = "raw-v1-rust-semantic-outcome-v1"
RUST_SEMANTIC_PROFILE = "exact_rational_admitted_raw_v1_outcome_v04"
REPORT_SCHEMA = "raw-v1-cross-profile-differential-report-v1"
LIVE_REPORT_SCHEMA = "raw-v1-cross-profile-differential-report-v2"
COMPARATOR_ID = "raw_v1_cross_profile_differential_gate_v1"
COMPARISON_KIND = "CROSS_PROFILE_COMPATIBILITY_OBSERVATION"
LIVE_CAPTURE_MODE = "LIVE_FRESH"
OFFLINE_CAPTURE_MODE = "OFFLINE_SUPPLIED"
PYTHON_REJECT_CAPTURE_SCHEMA = "raw-v1-python-parser-reject-observation-v1"
PYTHON_REJECT_PROFILE = "frozen_python_raw_v1_parser_v03"
PYTHON_LIVE_REPLAY_SOURCE_MARKER = "fresh_public_python_replay_adapter"
PYTHON_LIVE_REJECT_SOURCE_MARKER = "fresh_public_python_parser_adapter"
RUST_EXECUTION_FIELDS = {
    "evaluation_error_stage",
    "evaluation_outcome",
    "parse_outcome",
    "profile",
    "rejection_stage",
    "schema",
    "semantic_result",
}

FIXED_BLOCKERS = (
    "OPEN-V1-01",
    "OPEN-V1-06",
    "OPEN-V1-08",
    "CORPUS-INCOMPLETE",
    "SEMANTIC-EVALUATION-TOTALIZATION-INCOMPLETE",
    "ENVIRONMENTS-NOT-YET-REPRODUCED",
)
FIXED_NONCLAIMS = (
    "FROZEN_PARITY_NOT_CLAIMED",
    "MATHEMATICAL_AGREEMENT_NOT_CLAIMED",
    "IMPLEMENTATION_OR_VERIFIER_INDEPENDENCE_NOT_CLAIMED",
    "GENERAL_SOLUTION_OR_COMPLETENESS_NOT_CLAIMED",
    "RELEASE_READINESS_NOT_CLAIMED",
)

FINAL_COMPONENT_NAMES = (
    "q1x",
    "q1y",
    "q2x",
    "q2y",
    "q3x",
    "q3y",
    "v1x",
    "v1y",
    "v2x",
    "v2y",
    "v3x",
    "v3y",
)
LC_COMPONENT_NAMES = (
    "zx",
    "zy",
    "wx",
    "wy",
    "h",
    "Rx",
    "Ry",
    "Ux",
    "Uy",
    "yx",
    "yy",
    "Vx",
    "Vy",
    "t",
)


class LiveCorpusCase:
    """One strictly bound expectation/input pair, never serialized directly."""

    __slots__ = (
        "case_id",
        "expected_parse_outcome",
        "expectation_bytes",
        "expectation_path",
        "input_bytes",
        "input_path",
    )

    def __init__(
        self,
        *,
        case_id: str,
        expected_parse_outcome: str,
        expectation_bytes: bytes,
        expectation_path: Path,
        input_bytes: bytes,
        input_path: Path,
    ) -> None:
        self.case_id = case_id
        self.expected_parse_outcome = expected_parse_outcome
        self.expectation_bytes = expectation_bytes
        self.expectation_path = expectation_path
        self.input_bytes = input_bytes
        self.input_path = input_path


class PythonLiveCapture:
    """Canonical bytes produced by the live public-Python adapter."""

    __slots__ = ("kind", "payload", "profile_id")

    def __init__(self, *, kind: str, payload: bytes, profile_id: str) -> None:
        self.kind = kind
        self.payload = payload
        self.profile_id = profile_id


EQUAL = "EQUAL"
LEFT_CONTAINS_RIGHT = "LEFT_CONTAINS_RIGHT"
RIGHT_CONTAINS_LEFT = "RIGHT_CONTAINS_LEFT"
OVERLAP = "OVERLAP"
DISJOINT = "DISJOINT"


class DifferentialGateError(ValueError):
    """Stable failure raised for malformed differential-gate input."""

    def __init__(self, code: str, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code} at {path}")


def _fail(code: str, path: str) -> None:
    raise DifferentialGateError(code, path)


def _canonical_numerator(text: Any, path: str) -> str:
    if type(text) is not str:
        _fail("FRACTION_NUMERATOR_NOT_STRING", path)
    assert isinstance(text, str)
    digits = text[1:] if text.startswith("-") else text
    if not digits or not digits.isascii() or not digits.isdecimal():
        _fail("FRACTION_NUMERATOR_NOT_CANONICAL_DECIMAL", path)
    if (len(digits) > 1 and digits.startswith("0")) or text == "-0":
        _fail("FRACTION_NUMERATOR_NOT_CANONICAL_DECIMAL", path)
    if len(digits) > MAX_DECIMAL_DIGITS:
        _fail("FRACTION_DECIMAL_DIGIT_LIMIT_EXCEEDED", path)
    return text


def _canonical_denominator(text: Any, path: str) -> str:
    if type(text) is not str:
        _fail("FRACTION_DENOMINATOR_NOT_STRING", path)
    assert isinstance(text, str)
    if not text or not text.isascii() or not text.isdecimal() or text.startswith("0"):
        _fail("FRACTION_DENOMINATOR_NOT_CANONICAL_POSITIVE_DECIMAL", path)
    if len(text) > MAX_DECIMAL_DIGITS:
        _fail("FRACTION_DECIMAL_DIGIT_LIMIT_EXCEEDED", path)
    return text


def _decimal_to_int(text: str) -> int:
    """Parse canonical decimal without Python 3.11's int-string digit cap."""

    negative = text.startswith("-")
    digits = text[1:] if negative else text
    first_chunk_size = len(digits) % 9 or 9
    value = int(digits[:first_chunk_size])
    for offset in range(first_chunk_size, len(digits), 9):
        value = value * 1_000_000_000 + int(digits[offset : offset + 9])
    return -value if negative else value


def _json_integer(text: str) -> int:
    digits = text[1:] if text.startswith("-") else text
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        _fail("JSON_INTEGER_DIGIT_LIMIT_EXCEEDED", "json")
    return _decimal_to_int(text)


def _reject_json_float(_text: str) -> Any:
    _fail("JSON_FLOAT_NOT_PERMITTED", "json")


def _reject_json_constant(_text: str) -> Any:
    _fail("JSON_NONFINITE_CONSTANT_NOT_PERMITTED", "json")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON_DUPLICATE_KEY", "json")
        result[key] = value
    return result


def _load_json_object(payload: bytes, path: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        _fail("ARTIFACT_NOT_BYTES", path)
    if len(payload) > MAX_ARTIFACT_BYTES:
        _fail("JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED", path)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail("JSON_UTF8_INVALID", path)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except DifferentialGateError:
        raise
    except (json.JSONDecodeError, RecursionError):
        _fail("JSON_SYNTAX_INVALID", path)
    if type(value) is not dict:
        _fail("JSON_TOP_LEVEL_OBJECT_REQUIRED", path)
    return value


def _parse_fraction(value: Any, path: str) -> Fraction:
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        _fail("FRACTION_OBJECT_SHAPE_INVALID", path)
    numerator_text = _canonical_numerator(value["numerator"], f"{path}.numerator")
    denominator_text = _canonical_denominator(
        value["denominator"], f"{path}.denominator"
    )
    numerator = _decimal_to_int(numerator_text)
    denominator = _decimal_to_int(denominator_text)
    if gcd(abs(numerator), denominator) != 1:
        _fail("FRACTION_NOT_REDUCED", path)
    return Fraction(numerator, denominator)


def _parse_interval(value: Any, path: str) -> tuple[Fraction, Fraction]:
    if type(value) not in (list, tuple) or len(value) != 2:
        _fail("INTERVAL_SHAPE_INVALID", path)
    lower = _parse_fraction(value[0], f"{path}[0]")
    upper = _parse_fraction(value[1], f"{path}[1]")
    if lower > upper:
        _fail("INTERVAL_ENDPOINT_ORDER_INVALID", path)
    return lower, upper


def classify_interval(left: Any, right: Any) -> str:
    """Classify two canonical exact-rational closed intervals without floats."""

    return _classify_parsed_intervals(
        _parse_interval(left, "left"), _parse_interval(right, "right")
    )


def _classify_parsed_intervals(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> str:
    left_lower, left_upper = left
    right_lower, right_upper = right
    if left_lower == right_lower and left_upper == right_upper:
        return EQUAL
    if left_lower <= right_lower and right_upper <= left_upper:
        return LEFT_CONTAINS_RIGHT
    if right_lower <= left_lower and left_upper <= right_upper:
        return RIGHT_CONTAINS_LEFT
    if left_upper < right_lower or right_upper < left_lower:
        return DISJOINT
    return OVERLAP


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("OBJECT_REQUIRED", path)
    return value


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        _fail("ARRAY_REQUIRED", path)
    return value


def _field(value: dict[str, Any], key: str, path: str) -> Any:
    if key not in value:
        _fail("REQUIRED_FIELD_MISSING", f"{path}.{key}")
    return value[key]


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        _fail("NONEMPTY_STRING_REQUIRED", path)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _fail("STRING_NOT_UTF8_ENCODABLE", path)
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("BOUNDED_INTEGER_REQUIRED", path)
    return value


def _optional_integer(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail("BOOLEAN_REQUIRED", path)
    return value


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail("LOWERCASE_SHA256_REQUIRED", path)
    return text


def _declare(value: dict[str, Any], key: str, expected: str, path: str) -> None:
    observed = _string(_field(value, key, path), f"{path}.{key}")
    if observed != expected:
        _fail("DECLARED_SCHEMA_OR_PROFILE_INVALID", f"{path}.{key}")


def _request(value: Any, path: str) -> tuple[Fraction, Fraction]:
    request = _mapping(value, path)
    target = _parse_fraction(
        _field(request, "target_physical_time", path),
        f"{path}.target_physical_time",
    )
    width = _parse_fraction(
        _field(request, "maximum_component_width", path),
        f"{path}.maximum_component_width",
    )
    return target, width


def _obligations(value: Any, path: str) -> tuple[tuple[str, bool], ...]:
    rows = _array(value, path)
    if len(rows) != 13:
        _fail("EXACTLY_13_OBLIGATION_ROWS_REQUIRED", path)
    normalized: list[tuple[str, bool]] = []
    for index, raw_row in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw_row, row_path)
        normalized.append(
            (
                _string(_field(row, "obligation", row_path), f"{row_path}.obligation"),
                _boolean(_field(row, "certified", row_path), f"{row_path}.certified"),
            )
        )
    return tuple(normalized)


def _string_array(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _clock_identities(
    value: Any, path: str, *, python_shape: bool
) -> tuple[tuple[int, str], ...]:
    rows = _array(value, path)
    result: list[tuple[int, str]] = []
    for index, raw_row in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw_row, row_path)
        result.append(
            (
                _integer(
                    _field(row, "vertex_index", row_path), f"{row_path}.vertex_index"
                ),
                _string(_field(row, "chart_id", row_path), f"{row_path}.chart_id"),
            )
        )
        interval_key = "clock_origin_interval" if python_shape else "clock_origin"
        _field(row, interval_key, row_path)
    return tuple(result)


def _pair(value: Any, path: str) -> tuple[int, int]:
    values = _array(value, path)
    if len(values) != 2:
        _fail("PAIR_SHAPE_INVALID", path)
    return (_integer(values[0], f"{path}[0]"), _integer(values[1], f"{path}[1]"))


def _optional_pair(value: Any, path: str) -> tuple[int, int] | None:
    if value is None:
        return None
    return _pair(value, path)


def _shape_cartesian_components(enclosure: dict[str, Any], path: str) -> int:
    count = 0
    for field in ("position_intervals", "velocity_intervals"):
        bodies = _array(_field(enclosure, field, path), f"{path}.{field}")
        if len(bodies) != 3:
            _fail("PLANAR_BODY_SHAPE_INVALID", f"{path}.{field}")
        for body_index, raw_body in enumerate(bodies):
            body_path = f"{path}.{field}[{body_index}]"
            coordinates = _array(raw_body, body_path)
            if len(coordinates) != 2:
                _fail("PLANAR_COORDINATE_SHAPE_INVALID", body_path)
            for coordinate_index, raw_interval in enumerate(coordinates):
                interval_path = f"{body_path}[{coordinate_index}]"
                interval = _array(raw_interval, interval_path)
                if len(interval) != 2:
                    _fail("INTERVAL_SHAPE_INVALID", interval_path)
                count += 1
    return count


def _named_component_metadata(value: Any, path: str) -> tuple[str, ...]:
    components = _array(value, path)
    names: list[str] = []
    for index, raw_component in enumerate(components):
        component_path = f"{path}[{index}]"
        component = _mapping(raw_component, component_path)
        names.append(
            _string(
                _field(component, "component", component_path),
                f"{component_path}.component",
            )
        )
        _field(component, "interval", component_path)
    return tuple(names)


def _record_mismatch(mismatches: list[str], mismatch_id: str, matches: bool) -> None:
    if not matches and mismatch_id not in mismatches:
        mismatches.append(mismatch_id)


def _validate_utf8_tree(value: Any, path: str) -> None:
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            _fail("STRING_NOT_UTF8_ENCODABLE", path)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail("REPORT_OBJECT_KEY_NOT_STRING", path)
            _validate_utf8_tree(key, f"{path}.<key>")
            _validate_utf8_tree(child, f"{path}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_utf8_tree(child, f"{path}[{index}]")


def _canonical_json_bytes(value: Any, path: str) -> bytes:
    _validate_utf8_tree(value, path)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except UnicodeEncodeError:
        _fail("STRING_NOT_UTF8_ENCODABLE", path)
    except (TypeError, ValueError):
        _fail("JSON_SERIALIZATION_FAILED", path)


def _compare_committed_segments(
    python: dict[str, Any],
    rust: dict[str, Any],
    certified_count: int,
    mismatches: list[str],
) -> None:
    cocycles = _array(
        _field(python, "cocycle_ledger", "python_transcript"),
        "python_transcript.cocycle_ledger",
    )
    checkers = _string_array(
        _field(python, "segment_checker_ids", "python_transcript"),
        "python_transcript.segment_checker_ids",
    )
    profiles = _array(
        _field(rust, "segment_profiles", "rust_semantic"),
        "rust_semantic.segment_profiles",
    )
    _record_mismatch(
        mismatches,
        "PYTHON_COMMITTED_PREFIX_COUNT_MISMATCH",
        len(cocycles) == len(checkers) == certified_count,
    )
    _record_mismatch(
        mismatches,
        "RUST_COMMITTED_PREFIX_NOT_AVAILABLE",
        len(profiles) >= certified_count,
    )
    comparable_count = min(certified_count, len(cocycles), len(checkers), len(profiles))
    for index in range(comparable_count):
        python_path = f"python_transcript.cocycle_ledger[{index}]"
        rust_path = f"rust_semantic.segment_profiles[{index}]"
        cocycle = _mapping(cocycles[index], python_path)
        profile = _mapping(profiles[index], rust_path)
        python_index = _integer(
            _field(cocycle, "segment_index", python_path),
            f"{python_path}.segment_index",
        )
        rust_index = _integer(
            _field(profile, "segment_index", rust_path), f"{rust_path}.segment_index"
        )
        kind = _string(
            _field(cocycle, "cocycle_type", python_path), f"{python_path}.cocycle_type"
        )
        rust_kind = _string(_field(profile, "kind", rust_path), f"{rust_path}.kind")
        rust_pair = _optional_pair(
            _field(profile, "pair", rust_path), f"{rust_path}.pair"
        )
        _record_mismatch(
            mismatches,
            "COMMITTED_SEGMENT_INDEX_MISMATCH",
            python_index == rust_index == index,
        )
        if kind == "ordinary_bridge_v1":
            expected_kind = "ordinary_bridge"
            expected_checker = "carried_ordinary_bridge_checker_v1"
            python_pair = None
        elif kind == "planar_lc_passage_v1":
            expected_kind = "planar_lc_passage"
            expected_checker = "carried_planar_lc_exit_checker_v2"
            python_pair = _pair(
                _field(cocycle, "canonical_pair", python_path),
                f"{python_path}.canonical_pair",
            )
        else:
            _fail(
                "PYTHON_COMMITTED_SEGMENT_KIND_INVALID", f"{python_path}.cocycle_type"
            )
        _record_mismatch(
            mismatches,
            "PYTHON_COMMITTED_SEGMENT_CHECKER_MISMATCH",
            checkers[index] == expected_checker,
        )
        _record_mismatch(
            mismatches,
            "COMMITTED_SEGMENT_KIND_MISMATCH",
            rust_kind == expected_kind,
        )
        _record_mismatch(
            mismatches,
            "COMMITTED_SEGMENT_PAIR_MISMATCH",
            rust_pair == python_pair,
        )


def _compare_final_metadata(
    facts: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
    mismatches: list[str],
) -> None:
    python_final = _field(python, "final_enclosure", "python_transcript")
    rust_final = _field(rust, "final_enclosure", "rust_semantic")
    expected_summary = _field(facts, "final_enclosure_summary", "expectation.facts")
    _record_mismatch(
        mismatches,
        "FINAL_ENCLOSURE_PRESENCE_MISMATCH",
        (python_final is None) == (rust_final is None) == (expected_summary is None),
    )
    if python_final is None or rust_final is None or expected_summary is None:
        return
    python_final = _mapping(python_final, "python_transcript.final_enclosure")
    rust_final = _mapping(rust_final, "rust_semantic.final_enclosure")
    expected_summary = _mapping(
        expected_summary, "expectation.facts.final_enclosure_summary"
    )
    python_type = _string(
        _field(python_final, "enclosure_type", "python_transcript.final_enclosure"),
        "python_transcript.final_enclosure.enclosure_type",
    )
    rust_type = _string(
        _field(rust_final, "enclosure_type", "rust_semantic.final_enclosure"),
        "rust_semantic.final_enclosure.enclosure_type",
    )
    expected_type = _string(
        _field(
            expected_summary,
            "enclosure_type",
            "expectation.facts.final_enclosure_summary",
        ),
        "expectation.facts.final_enclosure_summary.enclosure_type",
    )
    python_count = _shape_cartesian_components(
        python_final, "python_transcript.final_enclosure"
    )
    rust_coordinate = _string(
        _field(rust_final, "coordinate_system", "rust_semantic.final_enclosure"),
        "rust_semantic.final_enclosure.coordinate_system",
    )
    rust_names = _named_component_metadata(
        _field(rust_final, "components", "rust_semantic.final_enclosure"),
        "rust_semantic.final_enclosure.components",
    )
    _record_mismatch(
        mismatches,
        "FINAL_ENCLOSURE_TYPE_MISMATCH",
        python_type == rust_type == expected_type,
    )
    _record_mismatch(
        mismatches,
        "FINAL_ENCLOSURE_COORDINATE_SYSTEM_MISMATCH",
        rust_coordinate == "planar_cartesian_12",
    )
    _record_mismatch(
        mismatches,
        "FINAL_ENCLOSURE_COMPONENT_COUNT_MISMATCH",
        python_count == len(rust_names) == 12,
    )
    _record_mismatch(
        mismatches,
        "FINAL_ENCLOSURE_COMPONENT_NAMES_MISMATCH",
        rust_names == FINAL_COMPONENT_NAMES,
    )
    _record_mismatch(
        mismatches,
        "EXPECTATION_FINAL_POSITION_SHAPE_MISMATCH",
        _field(
            expected_summary,
            "position_shape",
            "expectation.facts.final_enclosure_summary",
        )
        == [3, 2],
    )
    _record_mismatch(
        mismatches,
        "EXPECTATION_FINAL_VELOCITY_SHAPE_MISMATCH",
        _field(
            expected_summary,
            "velocity_shape",
            "expectation.facts.final_enclosure_summary",
        )
        == [3, 2],
    )


def _compare_retained_metadata(
    facts: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
    mismatches: list[str],
) -> None:
    python_regions = _array(
        _field(python, "retained_regions", "python_transcript"),
        "python_transcript.retained_regions",
    )
    expected_summaries = _array(
        _field(facts, "retained_region_summaries", "expectation.facts"),
        "expectation.facts.retained_region_summaries",
    )
    rust_region = _field(rust, "retained_region", "rust_semantic")
    rust_count = 0 if rust_region is None else 1
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_COUNT_MISMATCH",
        len(python_regions) == len(expected_summaries) == rust_count,
    )
    if len(python_regions) != 1 or len(expected_summaries) != 1 or rust_region is None:
        return
    python_region = _mapping(python_regions[0], "python_transcript.retained_regions[0]")
    expected = _mapping(
        expected_summaries[0], "expectation.facts.retained_region_summaries[0]"
    )
    rust_region = _mapping(rust_region, "rust_semantic.retained_region")
    python_type = _string(
        _field(python_region, "region_type", "python_transcript.retained_regions[0]"),
        "python_transcript.retained_regions[0].region_type",
    )
    rust_type = _string(
        _field(rust_region, "region_type", "rust_semantic.retained_region"),
        "rust_semantic.retained_region.region_type",
    )
    expected_type = _string(
        _field(
            expected, "region_type", "expectation.facts.retained_region_summaries[0]"
        ),
        "expectation.facts.retained_region_summaries[0].region_type",
    )
    python_coordinate = _string(
        _field(
            python_region,
            "coordinate_system",
            "python_transcript.retained_regions[0]",
        ),
        "python_transcript.retained_regions[0].coordinate_system",
    )
    rust_coordinate = _string(
        _field(rust_region, "coordinate_system", "rust_semantic.retained_region"),
        "rust_semantic.retained_region.coordinate_system",
    )
    expected_coordinate = _string(
        _field(
            expected,
            "coordinate_system",
            "expectation.facts.retained_region_summaries[0]",
        ),
        "expectation.facts.retained_region_summaries[0].coordinate_system",
    )
    python_count = len(
        _array(
            _field(
                python_region,
                "component_intervals",
                "python_transcript.retained_regions[0]",
            ),
            "python_transcript.retained_regions[0].component_intervals",
        )
    )
    rust_names = _named_component_metadata(
        _field(rust_region, "components", "rust_semantic.retained_region"),
        "rust_semantic.retained_region.components",
    )
    expected_count = _integer(
        _field(
            expected,
            "component_count",
            "expectation.facts.retained_region_summaries[0]",
        ),
        "expectation.facts.retained_region_summaries[0].component_count",
        minimum=1,
    )
    python_segments = _integer(
        _field(
            python_region,
            "certified_segment_count",
            "python_transcript.retained_regions[0]",
        ),
        "python_transcript.retained_regions[0].certified_segment_count",
    )
    rust_segments = _integer(
        _field(rust_region, "certified_segment_count", "rust_semantic.retained_region"),
        "rust_semantic.retained_region.certified_segment_count",
    )
    expected_segments = _integer(
        _field(
            expected,
            "certified_segment_count",
            "expectation.facts.retained_region_summaries[0]",
        ),
        "expectation.facts.retained_region_summaries[0].certified_segment_count",
    )
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_TYPE_MISMATCH",
        python_type == rust_type == expected_type,
    )
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_COORDINATE_SYSTEM_MISMATCH",
        python_coordinate == rust_coordinate == expected_coordinate,
    )
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_COMPONENT_COUNT_MISMATCH",
        python_count == len(rust_names) == expected_count,
    )
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_CERTIFIED_COUNT_MISMATCH",
        python_segments == rust_segments == expected_segments,
    )
    expected_names = (
        LC_COMPONENT_NAMES
        if rust_coordinate == "planar_lc_lifted_14"
        else FINAL_COMPONENT_NAMES
    )
    _record_mismatch(
        mismatches,
        "RETAINED_REGION_COMPONENT_NAMES_MISMATCH",
        rust_names == expected_names,
    )


def _append_interval_relation(
    records: list[dict[str, str]],
    mismatches: list[str],
    *,
    relation_id: str,
    left_value: Any,
    left_path: str,
    right_value: Any,
    right_path: str,
    exact_required: bool,
) -> None:
    relation = _classify_parsed_intervals(
        _parse_interval(left_value, left_path),
        _parse_interval(right_value, right_path),
    )
    records.append(
        {
            "id": relation_id,
            "left_path": left_path,
            "relation": relation,
            "right_path": right_path,
            "scope": "EXPECTATION_PYTHON" if exact_required else "CROSS_PROFILE",
        }
    )
    if exact_required:
        _record_mismatch(
            mismatches,
            "EXPECTATION_PYTHON_INTERVAL_MISMATCH",
            relation == EQUAL,
        )
    elif relation == DISJOINT:
        _record_mismatch(mismatches, "CROSS_PROFILE_INTERVAL_DISJOINT", False)


def _append_optional_interval_relation(
    records: list[dict[str, str]],
    mismatches: list[str],
    *,
    relation_id: str,
    left_value: Any,
    left_path: str,
    right_value: Any,
    right_path: str,
    exact_required: bool,
) -> None:
    if (left_value is None) != (right_value is None):
        mismatch_id = (
            "EXPECTATION_PYTHON_INTERVAL_PRESENCE_MISMATCH"
            if exact_required
            else "CROSS_PROFILE_INTERVAL_PRESENCE_MISMATCH"
        )
        _record_mismatch(mismatches, mismatch_id, False)
        return
    if left_value is None:
        return
    _append_interval_relation(
        records,
        mismatches,
        relation_id=relation_id,
        left_value=left_value,
        left_path=left_path,
        right_value=right_value,
        right_path=right_path,
        exact_required=exact_required,
    )


def _compare_interval_surfaces(
    facts: dict[str, Any],
    python: dict[str, Any],
    rust: dict[str, Any],
    mismatches: list[str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    expectation_python_fields = (
        (
            "expectation_python.current_clock_origin",
            "current_clock_origin_interval",
            "current_clock_origin_interval",
        ),
        (
            "expectation_python.covered_physical_time",
            "covered_physical_time_interval",
            "covered_physical_time_interval",
        ),
    )
    for relation_id, fact_key, python_key in expectation_python_fields:
        _append_interval_relation(
            records,
            mismatches,
            relation_id=relation_id,
            left_value=_field(facts, fact_key, "expectation.facts"),
            left_path=f"expectation.facts.{fact_key}",
            right_value=_field(python, python_key, "python_transcript"),
            right_path=f"python_transcript.{python_key}",
            exact_required=True,
        )
    _append_optional_interval_relation(
        records,
        mismatches,
        relation_id="expectation_python.target_parameter_preimage",
        left_value=_field(
            facts, "target_parameter_preimage_interval", "expectation.facts"
        ),
        left_path="expectation.facts.target_parameter_preimage_interval",
        right_value=_field(
            python, "target_parameter_preimage_interval", "python_transcript"
        ),
        right_path="python_transcript.target_parameter_preimage_interval",
        exact_required=True,
    )

    expected_final = _field(facts, "final_enclosure_summary", "expectation.facts")
    python_final = _field(python, "final_enclosure", "python_transcript")
    if expected_final is not None and python_final is not None:
        expected_final = _mapping(
            expected_final, "expectation.facts.final_enclosure_summary"
        )
        python_final = _mapping(python_final, "python_transcript.final_enclosure")
        for relation_id, expected_key, python_key in (
            (
                "expectation_python.final_physical_time",
                "physical_time_interval",
                "physical_time_interval",
            ),
            (
                "expectation_python.final_parameter",
                "parameter_preimage_interval",
                "parameter_preimage_interval",
            ),
        ):
            _append_interval_relation(
                records,
                mismatches,
                relation_id=relation_id,
                left_value=_field(
                    expected_final,
                    expected_key,
                    "expectation.facts.final_enclosure_summary",
                ),
                left_path=f"expectation.facts.final_enclosure_summary.{expected_key}",
                right_value=_field(
                    python_final, python_key, "python_transcript.final_enclosure"
                ),
                right_path=f"python_transcript.final_enclosure.{python_key}",
                exact_required=True,
            )

    expected_regions = _array(
        _field(facts, "retained_region_summaries", "expectation.facts"),
        "expectation.facts.retained_region_summaries",
    )
    python_regions = _array(
        _field(python, "retained_regions", "python_transcript"),
        "python_transcript.retained_regions",
    )
    if len(expected_regions) == len(python_regions) == 1:
        expected_region = _mapping(
            expected_regions[0], "expectation.facts.retained_region_summaries[0]"
        )
        python_region = _mapping(
            python_regions[0], "python_transcript.retained_regions[0]"
        )
        for relation_id, key in (
            ("expectation_python.retained_physical_time", "physical_time_interval"),
            ("expectation_python.retained_parameter", "parameter_interval"),
        ):
            _append_interval_relation(
                records,
                mismatches,
                relation_id=relation_id,
                left_value=_field(
                    expected_region,
                    key,
                    "expectation.facts.retained_region_summaries[0]",
                ),
                left_path=f"expectation.facts.retained_region_summaries[0].{key}",
                right_value=_field(
                    python_region, key, "python_transcript.retained_regions[0]"
                ),
                right_path=f"python_transcript.retained_regions[0].{key}",
                exact_required=True,
            )

    python_clock_rows = _array(
        _field(python, "clock_origin_ledger", "python_transcript"),
        "python_transcript.clock_origin_ledger",
    )
    rust_clock_rows = _array(
        _field(rust, "clock_ledger", "rust_semantic"), "rust_semantic.clock_ledger"
    )
    for index in range(min(len(python_clock_rows), len(rust_clock_rows))):
        python_clock = _mapping(
            python_clock_rows[index], f"python_transcript.clock_origin_ledger[{index}]"
        )
        rust_clock = _mapping(
            rust_clock_rows[index], f"rust_semantic.clock_ledger[{index}]"
        )
        _append_interval_relation(
            records,
            mismatches,
            relation_id=f"cross_profile.clock_origin[{index}]",
            left_value=_field(
                python_clock,
                "clock_origin_interval",
                f"python_transcript.clock_origin_ledger[{index}]",
            ),
            left_path=f"python_transcript.clock_origin_ledger[{index}].clock_origin_interval",
            right_value=_field(
                rust_clock, "clock_origin", f"rust_semantic.clock_ledger[{index}]"
            ),
            right_path=f"rust_semantic.clock_ledger[{index}].clock_origin",
            exact_required=False,
        )

    for relation_id, python_key, rust_key in (
        (
            "cross_profile.current_clock_origin",
            "current_clock_origin_interval",
            "current_clock_origin",
        ),
        (
            "cross_profile.covered_physical_time",
            "covered_physical_time_interval",
            "covered_physical_time_interval",
        ),
    ):
        _append_optional_interval_relation(
            records,
            mismatches,
            relation_id=relation_id,
            left_value=_field(python, python_key, "python_transcript"),
            left_path=f"python_transcript.{python_key}",
            right_value=_field(rust, rust_key, "rust_semantic"),
            right_path=f"rust_semantic.{rust_key}",
            exact_required=False,
        )
    _append_optional_interval_relation(
        records,
        mismatches,
        relation_id="cross_profile.target_parameter_preimage",
        left_value=_field(
            python, "target_parameter_preimage_interval", "python_transcript"
        ),
        left_path="python_transcript.target_parameter_preimage_interval",
        right_value=_field(rust, "target_parameter_preimage_interval", "rust_semantic"),
        right_path="rust_semantic.target_parameter_preimage_interval",
        exact_required=False,
    )

    rust_final = _field(rust, "final_enclosure", "rust_semantic")
    if python_final is not None and rust_final is not None:
        python_final = _mapping(python_final, "python_transcript.final_enclosure")
        rust_final = _mapping(rust_final, "rust_semantic.final_enclosure")
        for relation_id, python_key, rust_key in (
            (
                "cross_profile.final_physical_time",
                "physical_time_interval",
                "physical_time_interval",
            ),
            (
                "cross_profile.final_parameter",
                "parameter_preimage_interval",
                "parameter_interval",
            ),
        ):
            _append_interval_relation(
                records,
                mismatches,
                relation_id=relation_id,
                left_value=_field(
                    python_final, python_key, "python_transcript.final_enclosure"
                ),
                left_path=f"python_transcript.final_enclosure.{python_key}",
                right_value=_field(
                    rust_final, rust_key, "rust_semantic.final_enclosure"
                ),
                right_path=f"rust_semantic.final_enclosure.{rust_key}",
                exact_required=False,
            )
        python_component_values: list[Any] = []
        python_component_paths: list[str] = []
        for field in ("position_intervals", "velocity_intervals"):
            bodies = _array(
                _field(python_final, field, "python_transcript.final_enclosure"),
                f"python_transcript.final_enclosure.{field}",
            )
            for body_index, body in enumerate(bodies):
                coordinates = _array(
                    body,
                    f"python_transcript.final_enclosure.{field}[{body_index}]",
                )
                for coordinate_index, component_interval in enumerate(coordinates):
                    python_component_values.append(component_interval)
                    python_component_paths.append(
                        f"python_transcript.final_enclosure.{field}"
                        f"[{body_index}][{coordinate_index}]"
                    )
        rust_components = _array(
            _field(rust_final, "components", "rust_semantic.final_enclosure"),
            "rust_semantic.final_enclosure.components",
        )
        for index in range(
            min(
                len(python_component_values),
                len(rust_components),
                len(FINAL_COMPONENT_NAMES),
            )
        ):
            rust_component = _mapping(
                rust_components[index],
                f"rust_semantic.final_enclosure.components[{index}]",
            )
            _append_interval_relation(
                records,
                mismatches,
                relation_id=f"cross_profile.final_component.{FINAL_COMPONENT_NAMES[index]}",
                left_value=python_component_values[index],
                left_path=python_component_paths[index],
                right_value=_field(
                    rust_component,
                    "interval",
                    f"rust_semantic.final_enclosure.components[{index}]",
                ),
                right_path=f"rust_semantic.final_enclosure.components[{index}].interval",
                exact_required=False,
            )

    rust_region = _field(rust, "retained_region", "rust_semantic")
    if len(python_regions) == 1 and rust_region is not None:
        python_region = _mapping(
            python_regions[0], "python_transcript.retained_regions[0]"
        )
        rust_region = _mapping(rust_region, "rust_semantic.retained_region")
        for relation_id, key in (
            ("cross_profile.retained_physical_time", "physical_time_interval"),
            ("cross_profile.retained_parameter", "parameter_interval"),
        ):
            _append_interval_relation(
                records,
                mismatches,
                relation_id=relation_id,
                left_value=_field(
                    python_region, key, "python_transcript.retained_regions[0]"
                ),
                left_path=f"python_transcript.retained_regions[0].{key}",
                right_value=_field(rust_region, key, "rust_semantic.retained_region"),
                right_path=f"rust_semantic.retained_region.{key}",
                exact_required=False,
            )
        python_components = _array(
            _field(
                python_region,
                "component_intervals",
                "python_transcript.retained_regions[0]",
            ),
            "python_transcript.retained_regions[0].component_intervals",
        )
        rust_components = _array(
            _field(rust_region, "components", "rust_semantic.retained_region"),
            "rust_semantic.retained_region.components",
        )
        for index in range(
            min(len(python_components), len(rust_components), len(LC_COMPONENT_NAMES))
        ):
            rust_component = _mapping(
                rust_components[index],
                f"rust_semantic.retained_region.components[{index}]",
            )
            _append_interval_relation(
                records,
                mismatches,
                relation_id=f"cross_profile.retained_component.{LC_COMPONENT_NAMES[index]}",
                left_value=python_components[index],
                left_path=f"python_transcript.retained_regions[0].component_intervals[{index}]",
                right_value=_field(
                    rust_component,
                    "interval",
                    f"rust_semantic.retained_region.components[{index}]",
                ),
                right_path=f"rust_semantic.retained_region.components[{index}].interval",
                exact_required=False,
            )
    return records


def _comparison_result(
    *,
    case_id: str,
    expected_parse_outcome: str,
    theorem_status: str | None,
    input_sha256: str,
    expectation_sha256: str,
    python_sha256: str | None,
    rust_sha256: str,
    theorem_evidence_sha256: str | None,
    rust_semantic_present: bool,
    mismatches: list[str],
    interval_relations: list[dict[str, str]],
) -> dict[str, Any]:
    if mismatches:
        comparison_status = "DIFFERENTIAL_DISAGREEMENT"
    elif expected_parse_outcome == "ACCEPT":
        comparison_status = "PROFILE_COMPATIBILITY_OBSERVED"
    else:
        comparison_status = "PARSER_EXPECTATION_COMPATIBILITY_OBSERVED"
    return {
        "case_id": case_id,
        "comparison_status": comparison_status,
        "differential_gate_passed": not mismatches,
        "expected_parse_outcome": expected_parse_outcome,
        "interval_relations": interval_relations,
        "mismatch_ids": mismatches,
        "profiles": {
            "python": {
                "profile_id": PYTHON_PROFILE,
                "source_marker": PYTHON_SOURCE_MARKER,
                "transcript_present": python_sha256 is not None,
            },
            "rust_execution": {"profile_id": RUST_EXECUTION_PROFILE},
            "rust_semantic": (
                {"profile_id": RUST_SEMANTIC_PROFILE} if rust_semantic_present else None
            ),
        },
        "sha256_bindings": {
            "expectation_sha256": expectation_sha256,
            "input_sha256": input_sha256,
            "python_transcript_sha256": python_sha256,
            "rust_execution_sha256": rust_sha256,
            "theorem_evidence_sha256": theorem_evidence_sha256,
        },
        "theorem_status": theorem_status,
    }


def compare_case(
    case_id: str,
    input_bytes: bytes,
    expectation_bytes: bytes,
    python_transcript_bytes_or_none: bytes | None,
    rust_execution_bytes: bytes,
) -> dict[str, Any]:
    """Compare the discrete surface of one supplied offline raw-v1 case."""

    case_id = _string(case_id, "case_id")
    if type(input_bytes) is not bytes:
        _fail("ARTIFACT_NOT_BYTES", "input")
    expectation = _load_json_object(expectation_bytes, "expectation")
    rust_execution = _load_json_object(rust_execution_bytes, "rust_execution")
    _declare(expectation, "expectation_schema", EXPECTATION_SCHEMA, "expectation")
    if set(rust_execution) != RUST_EXECUTION_FIELDS:
        _fail("RUST_EXECUTION_FIELDS_INVALID", "rust_execution")
    _declare(rust_execution, "schema", RUST_EXECUTION_SCHEMA, "rust_execution")
    _declare(rust_execution, "profile", RUST_EXECUTION_PROFILE, "rust_execution")

    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    expectation_sha256 = hashlib.sha256(expectation_bytes).hexdigest()
    rust_sha256 = hashlib.sha256(rust_execution_bytes).hexdigest()
    expected_case_id = _string(
        _field(expectation, "case_id", "expectation"), "expectation.case_id"
    )
    expected_input_sha256 = _sha256(
        _field(expectation, "input_sha256", "expectation"),
        "expectation.input_sha256",
    )
    expected_parse = _string(
        _field(expectation, "expected_parse_outcome", "expectation"),
        "expectation.expected_parse_outcome",
    )
    if expected_parse not in ("ACCEPT", "REJECT"):
        _fail("EXPECTED_PARSE_OUTCOME_INVALID", "expectation.expected_parse_outcome")

    rust_parse = _string(
        _field(rust_execution, "parse_outcome", "rust_execution"),
        "rust_execution.parse_outcome",
    )
    if rust_parse not in ("ACCEPT", "REJECT"):
        _fail("RUST_PARSE_OUTCOME_INVALID", "rust_execution.parse_outcome")
    rust_evaluation = _string(
        _field(rust_execution, "evaluation_outcome", "rust_execution"),
        "rust_execution.evaluation_outcome",
    )
    if rust_evaluation not in ("NOT_RUN", "RESULT", "ERROR"):
        _fail("RUST_EVALUATION_OUTCOME_INVALID", "rust_execution.evaluation_outcome")
    rejection_stage = _field(rust_execution, "rejection_stage", "rust_execution")
    if rejection_stage is not None and rejection_stage not in (
        "CANONICAL_WIRE",
        "SCHEMA",
    ):
        _fail("RUST_REJECTION_STAGE_INVALID", "rust_execution.rejection_stage")
    evaluation_error_stage = _field(
        rust_execution, "evaluation_error_stage", "rust_execution"
    )
    if evaluation_error_stage is not None and evaluation_error_stage not in (
        "NAMESPACE",
        "MIXED_REPLAY",
    ):
        _fail(
            "RUST_EVALUATION_ERROR_STAGE_INVALID",
            "rust_execution.evaluation_error_stage",
        )
    semantic_value = _field(rust_execution, "semantic_result", "rust_execution")
    if semantic_value is not None and type(semantic_value) is not dict:
        _fail(
            "RUST_SEMANTIC_RESULT_OBJECT_OR_NULL_REQUIRED",
            "rust_execution.semantic_result",
        )
    if rust_parse == "REJECT":
        if rejection_stage not in ("CANONICAL_WIRE", "SCHEMA"):
            _fail("RUST_REJECTION_STAGE_INVALID", "rust_execution.rejection_stage")
        if (
            rust_evaluation != "NOT_RUN"
            or evaluation_error_stage is not None
            or semantic_value is not None
        ):
            _fail("RUST_REJECT_ENVELOPE_INCONSISTENT", "rust_execution")
    else:
        if rejection_stage is not None:
            _fail(
                "RUST_ACCEPT_REJECTION_STAGE_NOT_NULL",
                "rust_execution.rejection_stage",
            )
        if rust_evaluation == "NOT_RUN":
            _fail("RUST_ACCEPT_ENVELOPE_INCONSISTENT", "rust_execution")
        if rust_evaluation == "RESULT" and (
            evaluation_error_stage is not None or semantic_value is None
        ):
            _fail("RUST_RESULT_ENVELOPE_INCONSISTENT", "rust_execution")
        if rust_evaluation == "ERROR" and (
            evaluation_error_stage not in ("NAMESPACE", "MIXED_REPLAY")
            or semantic_value is not None
        ):
            _fail("RUST_ERROR_ENVELOPE_INCONSISTENT", "rust_execution")

    mismatches: list[str] = []
    _record_mismatch(mismatches, "CASE_ID_MISMATCH", case_id == expected_case_id)
    _record_mismatch(
        mismatches,
        "EXPECTATION_INPUT_SHA256_MISMATCH",
        input_sha256 == expected_input_sha256,
    )
    _record_mismatch(
        mismatches, "RUST_PARSE_OUTCOME_MISMATCH", rust_parse == expected_parse
    )

    if expected_parse == "REJECT":
        for field in (
            "expected_replay_outcome",
            "source_replay_transcript",
            "source_replay_sha256",
            "facts",
        ):
            if _field(expectation, field, "expectation") is not None:
                _fail("REJECT_EXPECTATION_FIELD_MUST_BE_NULL", f"expectation.{field}")
        if python_transcript_bytes_or_none is not None:
            _fail("REJECT_CASE_PYTHON_TRANSCRIPT_MUST_BE_ABSENT", "python_transcript")
        _record_mismatch(
            mismatches,
            "RUST_REJECT_EVALUATION_OUTCOME_MISMATCH",
            rust_evaluation == "NOT_RUN",
        )
        _record_mismatch(
            mismatches,
            "RUST_REJECT_SEMANTIC_RESULT_PRESENT",
            semantic_value is None,
        )
        _record_mismatch(
            mismatches,
            "RUST_REJECT_EVALUATION_ERROR_STAGE_PRESENT",
            _field(rust_execution, "evaluation_error_stage", "rust_execution") is None,
        )
        return _comparison_result(
            case_id=case_id,
            expected_parse_outcome=expected_parse,
            theorem_status=None,
            input_sha256=input_sha256,
            expectation_sha256=expectation_sha256,
            python_sha256=None,
            rust_sha256=rust_sha256,
            theorem_evidence_sha256=None,
            rust_semantic_present=False,
            mismatches=mismatches,
            interval_relations=[],
        )

    if python_transcript_bytes_or_none is None:
        _fail("ACCEPT_CASE_PYTHON_TRANSCRIPT_REQUIRED", "python_transcript")
    python_sha256 = hashlib.sha256(python_transcript_bytes_or_none).hexdigest()
    python = _load_json_object(python_transcript_bytes_or_none, "python_transcript")
    _declare(python, "artifact_schema", PYTHON_TRANSCRIPT_SCHEMA, "python_transcript")
    _declare(python, "artifact_version", PYTHON_TRANSCRIPT_VERSION, "python_transcript")

    expected_status = _string(
        _field(expectation, "expected_replay_outcome", "expectation"),
        "expectation.expected_replay_outcome",
    )
    if expected_status not in ("CERTIFIED_TO_T", "UNRESOLVED"):
        _fail("EXPECTED_REPLAY_OUTCOME_INVALID", "expectation.expected_replay_outcome")
    expected_source_sha256 = _sha256(
        _field(expectation, "source_replay_sha256", "expectation"),
        "expectation.source_replay_sha256",
    )
    _string(
        _field(expectation, "source_replay_transcript", "expectation"),
        "expectation.source_replay_transcript",
    )
    facts = _mapping(_field(expectation, "facts", "expectation"), "expectation.facts")

    _record_mismatch(
        mismatches,
        "EXPECTATION_SOURCE_REPLAY_SHA256_MISMATCH",
        expected_source_sha256 == python_sha256,
    )
    _record_mismatch(
        mismatches,
        "RUST_ACCEPT_EVALUATION_OUTCOME_MISMATCH",
        rust_evaluation == "RESULT",
    )
    _record_mismatch(
        mismatches,
        "RUST_ACCEPT_SEMANTIC_RESULT_ABSENT",
        semantic_value is not None,
    )
    _record_mismatch(
        mismatches,
        "RUST_ACCEPT_REJECTION_STAGE_PRESENT",
        _field(rust_execution, "rejection_stage", "rust_execution") is None,
    )
    _record_mismatch(
        mismatches,
        "RUST_ACCEPT_EVALUATION_ERROR_STAGE_PRESENT",
        _field(rust_execution, "evaluation_error_stage", "rust_execution") is None,
    )

    if semantic_value is None:
        return _comparison_result(
            case_id=case_id,
            expected_parse_outcome=expected_parse,
            theorem_status=expected_status,
            input_sha256=input_sha256,
            expectation_sha256=expectation_sha256,
            python_sha256=python_sha256,
            rust_sha256=rust_sha256,
            theorem_evidence_sha256=None,
            rust_semantic_present=False,
            mismatches=mismatches,
            interval_relations=[],
        )
    rust = _mapping(semantic_value, "rust_semantic")
    _declare(rust, "schema", RUST_SEMANTIC_SCHEMA, "rust_semantic")
    _declare(rust, "profile", RUST_SEMANTIC_PROFILE, "rust_semantic")

    python_status = _string(
        _field(python, "status", "python_transcript"), "python_transcript.status"
    )
    rust_status = _string(
        _field(rust, "status", "rust_semantic"), "rust_semantic.status"
    )
    if python_status not in ("CERTIFIED_TO_T", "UNRESOLVED"):
        _fail("PYTHON_REPLAY_STATUS_INVALID", "python_transcript.status")
    if rust_status not in ("CERTIFIED_TO_T", "UNRESOLVED"):
        _fail("RUST_SEMANTIC_STATUS_INVALID", "rust_semantic.status")
    _record_mismatch(
        mismatches,
        "THEOREM_STATUS_MISMATCH",
        expected_status == python_status == rust_status,
    )

    expected_certificate = _string(
        _field(facts, "certificate_id", "expectation.facts"),
        "expectation.facts.certificate_id",
    )
    python_certificate = _string(
        _field(python, "certificate_id", "python_transcript"),
        "python_transcript.certificate_id",
    )
    rust_certificate = _string(
        _field(rust, "certificate_id", "rust_semantic"),
        "rust_semantic.certificate_id",
    )
    _record_mismatch(
        mismatches,
        "CERTIFICATE_ID_MISMATCH",
        expected_certificate == python_certificate == rust_certificate,
    )

    expected_evidence = _sha256(
        _field(facts, "theorem_evidence_sha256", "expectation.facts"),
        "expectation.facts.theorem_evidence_sha256",
    )
    python_evidence = _sha256(
        _field(python, "theorem_evidence_sha256", "python_transcript"),
        "python_transcript.theorem_evidence_sha256",
    )
    rust_evidence = _sha256(
        _field(rust, "evidence_sha256", "rust_semantic"),
        "rust_semantic.evidence_sha256",
    )
    _record_mismatch(
        mismatches,
        "THEOREM_EVIDENCE_SHA256_MISMATCH",
        expected_evidence == python_evidence == rust_evidence == input_sha256,
    )

    expected_request = _request(
        _field(facts, "request", "expectation.facts"), "expectation.facts.request"
    )
    python_request = _request(
        _field(python, "request", "python_transcript"), "python_transcript.request"
    )
    rust_request = _request(
        _field(rust, "request", "rust_semantic"), "rust_semantic.request"
    )
    _record_mismatch(
        mismatches,
        "REQUEST_MISMATCH",
        expected_request == python_request == rust_request,
    )

    expected_obligations = _obligations(
        _field(facts, "top_level_obligations", "expectation.facts"),
        "expectation.facts.top_level_obligations",
    )
    python_obligations = _obligations(
        _field(python, "obligations", "python_transcript"),
        "python_transcript.obligations",
    )
    rust_obligations = _obligations(
        _field(rust, "top_level_obligations", "rust_semantic"),
        "rust_semantic.top_level_obligations",
    )
    _record_mismatch(
        mismatches,
        "TOP_LEVEL_OBLIGATIONS_MISMATCH",
        expected_obligations == python_obligations == rust_obligations,
    )

    expected_first_failure = _optional_string(
        _field(facts, "first_failed_obligation", "expectation.facts"),
        "expectation.facts.first_failed_obligation",
    )
    python_first_failure = _optional_string(
        _field(python, "first_failed_obligation", "python_transcript"),
        "python_transcript.first_failed_obligation",
    )
    rust_first_failure = _optional_string(
        _field(rust, "first_failed_obligation", "rust_semantic"),
        "rust_semantic.first_failed_obligation",
    )
    _record_mismatch(
        mismatches,
        "FIRST_FAILED_OBLIGATION_MISMATCH",
        expected_first_failure == python_first_failure == rust_first_failure,
    )

    expected_count = _integer(
        _field(facts, "certified_segment_count", "expectation.facts"),
        "expectation.facts.certified_segment_count",
    )
    python_count = _integer(
        _field(python, "certified_segment_count", "python_transcript"),
        "python_transcript.certified_segment_count",
    )
    rust_count = _integer(
        _field(rust, "certified_segment_count", "rust_semantic"),
        "rust_semantic.certified_segment_count",
    )
    _record_mismatch(
        mismatches,
        "CERTIFIED_SEGMENT_COUNT_MISMATCH",
        expected_count == python_count == rust_count,
    )

    expected_failed_index = _optional_integer(
        _field(facts, "failed_segment_index", "expectation.facts"),
        "expectation.facts.failed_segment_index",
    )
    python_failed_index = _optional_integer(
        _field(python, "failed_segment_index", "python_transcript"),
        "python_transcript.failed_segment_index",
    )
    rust_failed_index = _optional_integer(
        _field(rust, "failed_segment_index", "rust_semantic"),
        "rust_semantic.failed_segment_index",
    )
    _record_mismatch(
        mismatches,
        "FAILED_SEGMENT_INDEX_MISMATCH",
        expected_failed_index == python_failed_index == rust_failed_index,
    )

    expected_missing = _string_array(
        _field(facts, "failed_segment_missing_obligations", "expectation.facts"),
        "expectation.facts.failed_segment_missing_obligations",
    )
    python_missing = _string_array(
        _field(python, "failed_segment_missing_obligations", "python_transcript"),
        "python_transcript.failed_segment_missing_obligations",
    )
    rust_missing = _string_array(
        _field(rust, "failed_segment_missing_obligations", "rust_semantic"),
        "rust_semantic.failed_segment_missing_obligations",
    )
    _record_mismatch(
        mismatches,
        "FAILED_SEGMENT_MISSING_OBLIGATIONS_MISMATCH",
        expected_missing == python_missing == rust_missing,
    )

    expected_current = _optional_string(
        _field(facts, "current_chart_id", "expectation.facts"),
        "expectation.facts.current_chart_id",
    )
    python_current = _optional_string(
        _field(python, "current_chart_id", "python_transcript"),
        "python_transcript.current_chart_id",
    )
    raw_rust_current = _field(rust, "current_chart", "rust_semantic")
    if raw_rust_current is None:
        rust_current = None
    else:
        rust_current_object = _mapping(raw_rust_current, "rust_semantic.current_chart")
        rust_current = _string(
            _field(rust_current_object, "chart_id", "rust_semantic.current_chart"),
            "rust_semantic.current_chart.chart_id",
        )
    _record_mismatch(
        mismatches,
        "CURRENT_CHART_ID_MISMATCH",
        expected_current == python_current == rust_current,
    )

    python_clocks = _clock_identities(
        _field(python, "clock_origin_ledger", "python_transcript"),
        "python_transcript.clock_origin_ledger",
        python_shape=True,
    )
    rust_clocks = _clock_identities(
        _field(rust, "clock_ledger", "rust_semantic"),
        "rust_semantic.clock_ledger",
        python_shape=False,
    )
    expected_clock_count = _integer(
        _field(facts, "clock_vertex_count", "expectation.facts"),
        "expectation.facts.clock_vertex_count",
    )
    _record_mismatch(
        mismatches,
        "CLOCK_LEDGER_LENGTH_MISMATCH",
        expected_clock_count == len(python_clocks) == len(rust_clocks),
    )
    _record_mismatch(
        mismatches,
        "CLOCK_LEDGER_VERTEX_OR_CHART_ID_MISMATCH",
        python_clocks == rust_clocks,
    )

    _compare_committed_segments(python, rust, python_count, mismatches)
    python_cocycles = _array(
        _field(python, "cocycle_ledger", "python_transcript"),
        "python_transcript.cocycle_ledger",
    )
    python_kinds = tuple(
        _string(
            _field(
                _mapping(row, f"python_transcript.cocycle_ledger[{index}]"),
                "cocycle_type",
                f"python_transcript.cocycle_ledger[{index}]",
            ),
            f"python_transcript.cocycle_ledger[{index}].cocycle_type",
        )
        for index, row in enumerate(python_cocycles)
    )
    python_pairs = tuple(
        _pair(
            _field(
                _mapping(row, f"python_transcript.cocycle_ledger[{index}]"),
                "canonical_pair",
                f"python_transcript.cocycle_ledger[{index}]",
            ),
            f"python_transcript.cocycle_ledger[{index}].canonical_pair",
        )
        for index, row in enumerate(python_cocycles)
        if python_kinds[index] == "planar_lc_passage_v1"
    )
    expected_kinds = _string_array(
        _field(facts, "cocycle_kinds", "expectation.facts"),
        "expectation.facts.cocycle_kinds",
    )
    expected_pairs = tuple(
        _pair(value, f"expectation.facts.cocycle_pairs[{index}]")
        for index, value in enumerate(
            _array(
                _field(facts, "cocycle_pairs", "expectation.facts"),
                "expectation.facts.cocycle_pairs",
            )
        )
    )
    expected_checkers = _string_array(
        _field(facts, "segment_checker_ids", "expectation.facts"),
        "expectation.facts.segment_checker_ids",
    )
    python_checkers = _string_array(
        _field(python, "segment_checker_ids", "python_transcript"),
        "python_transcript.segment_checker_ids",
    )
    _record_mismatch(
        mismatches,
        "EXPECTATION_COMMITTED_SEGMENT_METADATA_MISMATCH",
        expected_kinds == python_kinds
        and expected_pairs == python_pairs
        and expected_checkers == python_checkers,
    )

    expected_width_present = _boolean(
        _field(facts, "maximum_final_component_width_present", "expectation.facts"),
        "expectation.facts.maximum_final_component_width_present",
    )
    python_width = _field(python, "maximum_final_component_width", "python_transcript")
    rust_width = _field(rust, "maximum_final_component_width", "rust_semantic")
    python_width_value = None
    if python_width is not None:
        python_width_value = _parse_fraction(
            python_width, "python_transcript.maximum_final_component_width"
        )
    rust_width_value = None
    if rust_width is not None:
        rust_width_value = _parse_fraction(
            rust_width, "rust_semantic.maximum_final_component_width"
        )
    _record_mismatch(
        mismatches,
        "MAXIMUM_FINAL_COMPONENT_WIDTH_PRESENCE_MISMATCH",
        expected_width_present
        == (python_width is not None)
        == (rust_width is not None),
    )
    _record_mismatch(
        mismatches,
        "PYTHON_MAXIMUM_FINAL_WIDTH_EXCEEDS_REQUEST",
        python_width_value is None or python_width_value <= python_request[1],
    )
    _record_mismatch(
        mismatches,
        "PYTHON_MAXIMUM_FINAL_WIDTH_NEGATIVE",
        python_width_value is None or python_width_value >= 0,
    )
    _record_mismatch(
        mismatches,
        "RUST_MAXIMUM_FINAL_WIDTH_EXCEEDS_REQUEST",
        rust_width_value is None or rust_width_value <= rust_request[1],
    )
    _record_mismatch(
        mismatches,
        "RUST_MAXIMUM_FINAL_WIDTH_NEGATIVE",
        rust_width_value is None or rust_width_value >= 0,
    )

    _compare_final_metadata(facts, python, rust, mismatches)
    _compare_retained_metadata(facts, python, rust, mismatches)
    interval_relations = _compare_interval_surfaces(facts, python, rust, mismatches)

    return _comparison_result(
        case_id=case_id,
        expected_parse_outcome=expected_parse,
        theorem_status=expected_status,
        input_sha256=input_sha256,
        expectation_sha256=expectation_sha256,
        python_sha256=python_sha256,
        rust_sha256=rust_sha256,
        theorem_evidence_sha256=python_evidence,
        rust_semantic_present=True,
        mismatches=mismatches,
        interval_relations=interval_relations,
    )


def build_report(cases: Any, *, capture_mode: str = OFFLINE_CAPTURE_MODE) -> bytes:
    """Serialize ordered comparison results as canonical compact JSON bytes."""

    if capture_mode not in (OFFLINE_CAPTURE_MODE, LIVE_CAPTURE_MODE):
        _fail("CAPTURE_MODE_INVALID", "capture_mode")
    if isinstance(cases, (str, bytes, bytearray, dict)):
        _fail("ORDERED_CASE_ITERABLE_REQUIRED", "cases")
    try:
        ordered_cases = list(cases)
    except TypeError:
        _fail("ORDERED_CASE_ITERABLE_REQUIRED", "cases")

    seen_case_ids: set[str] = set()
    mismatch_count = 0
    mismatch_case_count = 0
    passed_case_count = 0
    disjoint_interval_count = 0
    all_relation_counts = {
        EQUAL: 0,
        LEFT_CONTAINS_RIGHT: 0,
        RIGHT_CONTAINS_LEFT: 0,
        OVERLAP: 0,
        DISJOINT: 0,
    }
    expectation_python_relation_counts = dict.fromkeys(all_relation_counts, 0)
    cross_profile_relation_counts = dict.fromkeys(all_relation_counts, 0)
    accepted_case_count = 0
    rejected_case_count = 0
    for index, raw_case in enumerate(ordered_cases):
        path = f"cases[{index}]"
        case = _mapping(raw_case, path)
        case_id = _string(_field(case, "case_id", path), f"{path}.case_id")
        if case_id in seen_case_ids:
            _fail("DUPLICATE_CASE_ID", f"{path}.case_id")
        seen_case_ids.add(case_id)
        passed = _boolean(
            _field(case, "differential_gate_passed", path),
            f"{path}.differential_gate_passed",
        )
        mismatch_ids = _string_array(
            _field(case, "mismatch_ids", path), f"{path}.mismatch_ids"
        )
        if passed != (not mismatch_ids):
            _fail("CASE_PASS_FLAG_INCONSISTENT", path)
        mismatch_count += len(mismatch_ids)
        mismatch_case_count += int(not passed)
        passed_case_count += int(passed)
        parse_outcome = _string(
            _field(case, "expected_parse_outcome", path),
            f"{path}.expected_parse_outcome",
        )
        if parse_outcome == "ACCEPT":
            accepted_case_count += 1
        elif parse_outcome == "REJECT":
            rejected_case_count += 1
        else:
            _fail("EXPECTED_PARSE_OUTCOME_INVALID", f"{path}.expected_parse_outcome")
        relations = _array(
            _field(case, "interval_relations", path), f"{path}.interval_relations"
        )
        seen_relation_ids: set[str] = set()
        for relation_index, raw_relation in enumerate(relations):
            relation_path = f"{path}.interval_relations[{relation_index}]"
            relation_record = _mapping(raw_relation, relation_path)
            if set(relation_record) != {
                "id",
                "left_path",
                "relation",
                "right_path",
                "scope",
            }:
                _fail("INTERVAL_RELATION_RECORD_FIELDS_INVALID", relation_path)
            relation_id = _string(
                _field(relation_record, "id", relation_path),
                f"{relation_path}.id",
            )
            if relation_id in seen_relation_ids:
                _fail("DUPLICATE_INTERVAL_RELATION_ID", f"{relation_path}.id")
            seen_relation_ids.add(relation_id)
            _string(
                _field(relation_record, "left_path", relation_path),
                f"{relation_path}.left_path",
            )
            _string(
                _field(relation_record, "right_path", relation_path),
                f"{relation_path}.right_path",
            )
            relation = _string(
                _field(relation_record, "relation", relation_path),
                f"{relation_path}.relation",
            )
            if relation not in all_relation_counts:
                _fail("INTERVAL_RELATION_INVALID", f"{relation_path}.relation")
            scope = _string(
                _field(relation_record, "scope", relation_path),
                f"{relation_path}.scope",
            )
            if scope == "EXPECTATION_PYTHON":
                scoped_counts = expectation_python_relation_counts
            elif scope == "CROSS_PROFILE":
                scoped_counts = cross_profile_relation_counts
            else:
                _fail("INTERVAL_RELATION_SCOPE_INVALID", f"{relation_path}.scope")
            all_relation_counts[relation] += 1
            scoped_counts[relation] += 1
            disjoint_interval_count += int(relation == DISJOINT)

    differential_gate_passed = mismatch_case_count == 0
    blockers = list(FIXED_BLOCKERS)
    if not differential_gate_passed:
        blockers.append("DIFFERENTIAL-DISAGREEMENT")
    document: dict[str, Any] = {
        "aggregate": {
            "accepted_case_count": accepted_case_count,
            "blockers": blockers,
            "case_count": len(ordered_cases),
            "differential_gate_passed": differential_gate_passed,
            "disjoint_interval_count": disjoint_interval_count,
            "interval_relation_counts": {
                "all": all_relation_counts,
                "cross_profile": cross_profile_relation_counts,
                "expectation_python": expectation_python_relation_counts,
            },
            "mathematical_outcomes_comparable": False,
            "mismatch_case_count": mismatch_case_count,
            "mismatch_count": mismatch_count,
            "nonclaims": list(FIXED_NONCLAIMS),
            "passed_case_count": passed_case_count,
            "rejected_case_count": rejected_case_count,
            "release_gate_status": "BLOCKED",
        },
        "cases": ordered_cases,
        "comparator_id": COMPARATOR_ID,
        "comparison_kind": COMPARISON_KIND,
        "report_schema": (
            LIVE_REPORT_SCHEMA if capture_mode == LIVE_CAPTURE_MODE else REPORT_SCHEMA
        ),
    }
    if capture_mode == LIVE_CAPTURE_MODE:
        document["capture_mode"] = LIVE_CAPTURE_MODE
    return _canonical_json_bytes(document, "report")


def _path_argument(value: str | os.PathLike[str], path: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        _fail("FILESYSTEM_PATH_REQUIRED", path)
    candidate = Path(value)
    if ".." in candidate.parts:
        _fail("FILESYSTEM_PATH_TRAVERSAL_REJECTED", path)
    return candidate


def _require_real_directory(path: Path, label: str) -> Path:
    try:
        if path.is_symlink():
            _fail("CORPUS_SYMLINK_REJECTED", label)
        information = path.stat()
    except OSError:
        _fail("CORPUS_DIRECTORY_UNAVAILABLE", label)
    if not stat.S_ISDIR(information.st_mode):
        _fail("CORPUS_DIRECTORY_REQUIRED", label)
    try:
        return path.resolve(strict=True)
    except OSError:
        _fail("CORPUS_DIRECTORY_UNAVAILABLE", label)


def _read_regular_file_bounded(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _fail("CORPUS_FILE_OPEN_FAILED", label)
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            _fail("CORPUS_REGULAR_FILE_REQUIRED", label)
        if information.st_size > MAX_ARTIFACT_BYTES:
            _fail("JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED", label)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, MAX_ARTIFACT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ARTIFACT_BYTES:
                _fail("JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED", label)
        return b"".join(chunks)
    except OSError:
        _fail("CORPUS_FILE_READ_FAILED", label)
    finally:
        os.close(descriptor)


def enumerate_live_corpus(
    corpus_root: str | os.PathLike[str],
) -> list[LiveCorpusCase]:
    """Mechanically enumerate the flat raw-v1 expectation/input corpus."""

    root_argument = _path_argument(corpus_root, "corpus_root")
    root = _require_real_directory(root_argument, "corpus_root")
    expectation_root = _require_real_directory(
        root / "expectations", "corpus_root.expectations"
    )
    input_root = _require_real_directory(root / "inputs", "corpus_root.inputs")

    try:
        expectation_entries = sorted(
            expectation_root.iterdir(), key=lambda item: item.name
        )
        input_entries = sorted(input_root.iterdir(), key=lambda item: item.name)
    except OSError:
        _fail("CORPUS_DIRECTORY_ENUMERATION_FAILED", "corpus_root")
    if not expectation_entries:
        _fail("CORPUS_EXPECTATIONS_EMPTY", "corpus_root.expectations")

    cases: list[LiveCorpusCase] = []
    seen_case_ids: set[str] = set()
    bound_input_names: set[str] = set()
    for index, expectation_path in enumerate(expectation_entries):
        expectation_label = f"expectations[{index}]"
        if expectation_path.is_symlink():
            _fail("CORPUS_SYMLINK_REJECTED", expectation_label)
        if not expectation_path.name.endswith(".expected.json"):
            _fail("CORPUS_EXPECTATION_FILE_NAME_INVALID", expectation_label)
        expectation_bytes = _read_regular_file_bounded(
            expectation_path, expectation_label
        )
        expectation = _load_json_object(expectation_bytes, expectation_label)
        _declare(
            expectation,
            "expectation_schema",
            EXPECTATION_SCHEMA,
            expectation_label,
        )
        case_id = _string(
            _field(expectation, "case_id", expectation_label),
            f"{expectation_label}.case_id",
        )
        if expectation_path.name != f"{case_id}.expected.json":
            _fail("CORPUS_EXPECTATION_CASE_FILE_MISMATCH", expectation_label)
        if case_id in seen_case_ids:
            _fail("DUPLICATE_CASE_ID", f"{expectation_label}.case_id")
        seen_case_ids.add(case_id)
        expected_parse = _string(
            _field(expectation, "expected_parse_outcome", expectation_label),
            f"{expectation_label}.expected_parse_outcome",
        )
        if expected_parse not in ("ACCEPT", "REJECT"):
            _fail(
                "EXPECTED_PARSE_OUTCOME_INVALID",
                f"{expectation_label}.expected_parse_outcome",
            )
        input_name = _string(
            _field(expectation, "input_file", expectation_label),
            f"{expectation_label}.input_file",
        )
        relative_input = PurePosixPath(input_name)
        if (
            "\\" in input_name
            or relative_input.is_absolute()
            or relative_input.as_posix() != input_name
            or len(relative_input.parts) != 2
            or relative_input.parts[0] != "inputs"
            or any(part in ("", ".", "..") for part in relative_input.parts)
        ):
            _fail(
                "CORPUS_INPUT_BINDING_PATH_INVALID", f"{expectation_label}.input_file"
            )
        bound_name = relative_input.parts[1]
        if bound_name in bound_input_names:
            _fail("CORPUS_INPUT_BINDING_AMBIGUOUS", f"{expectation_label}.input_file")
        bound_input_names.add(bound_name)
        input_path = input_root / bound_name
        if input_path.is_symlink():
            _fail("CORPUS_SYMLINK_REJECTED", f"{expectation_label}.input_file")
        input_bytes = _read_regular_file_bounded(
            input_path, f"{expectation_label}.input_file"
        )
        cases.append(
            LiveCorpusCase(
                case_id=case_id,
                expected_parse_outcome=expected_parse,
                expectation_bytes=expectation_bytes,
                expectation_path=expectation_path,
                input_bytes=input_bytes,
                input_path=input_path,
            )
        )

    actual_input_names: set[str] = set()
    for index, input_path in enumerate(input_entries):
        input_label = f"inputs[{index}]"
        if input_path.is_symlink():
            _fail("CORPUS_SYMLINK_REJECTED", input_label)
        _read_regular_file_bounded(input_path, input_label)
        actual_input_names.add(input_path.name)
    if actual_input_names != bound_input_names:
        _fail("CORPUS_INPUT_SET_NOT_EXACTLY_BOUND", "corpus_root.inputs")
    return cases


def capture_python_raw_v1(
    case_id: str, input_bytes: bytes, expected_parse_outcome: str
) -> PythonLiveCapture:
    """Capture one fresh public-Python parser/replay observation lazily."""

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from three_body_symmetry.planar_chain_review_artifact import (
        ReviewArtifactError,
        canonical_replay_transcript_json,
        replay_transcript,
        strict_load_raw_planar_chain_bytes,
    )

    case_id = _string(case_id, "python_capture.case_id")
    if type(input_bytes) is not bytes:
        _fail("ARTIFACT_NOT_BYTES", "python_capture.input")
    if expected_parse_outcome == "ACCEPT":
        try:
            certificate = strict_load_raw_planar_chain_bytes(input_bytes)
        except ReviewArtifactError:
            _fail("PYTHON_EXPECTED_ACCEPT_WAS_REJECTED", "python_capture")
        transcript = replay_transcript(certificate)
        payload = canonical_replay_transcript_json(transcript).encode("utf-8")
        if len(payload) > MAX_ARTIFACT_BYTES:
            _fail("JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED", "python_capture")
        return PythonLiveCapture(
            kind="REPLAY_TRANSCRIPT", payload=payload, profile_id=PYTHON_PROFILE
        )
    if expected_parse_outcome != "REJECT":
        _fail("EXPECTED_PARSE_OUTCOME_INVALID", "python_capture.expected_parse_outcome")
    try:
        strict_load_raw_planar_chain_bytes(input_bytes)
    except ReviewArtifactError:
        observation = {
            "capture_kind": "PARSER_REJECT_OBSERVATION",
            "capture_schema": PYTHON_REJECT_CAPTURE_SCHEMA,
            "case_id": case_id,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "parse_outcome": "REJECT",
            "profile": PYTHON_REJECT_PROFILE,
        }
        return PythonLiveCapture(
            kind="PARSER_REJECT_OBSERVATION",
            payload=_canonical_json_bytes(observation, "python_capture"),
            profile_id=PYTHON_REJECT_PROFILE,
        )
    _fail("PYTHON_EXPECTED_REJECT_WAS_ACCEPTED", "python_capture")


def _validate_python_live_capture(
    case: LiveCorpusCase, capture: PythonLiveCapture
) -> None:
    if type(capture) is not PythonLiveCapture:
        _fail("PYTHON_CAPTURE_TYPE_INVALID", "python_capture")
    if type(capture.payload) is not bytes:
        _fail("ARTIFACT_NOT_BYTES", "python_capture")
    value = _load_json_object(capture.payload, "python_capture")
    if capture.payload != _canonical_json_bytes(value, "python_capture"):
        _fail("PYTHON_CAPTURE_NOT_CANONICAL", "python_capture")
    if case.expected_parse_outcome == "ACCEPT":
        if capture.kind != "REPLAY_TRANSCRIPT" or capture.profile_id != PYTHON_PROFILE:
            _fail("PYTHON_ACCEPT_CAPTURE_PROFILE_INVALID", "python_capture")
        _declare(value, "artifact_schema", PYTHON_TRANSCRIPT_SCHEMA, "python_capture")
        _declare(value, "artifact_version", PYTHON_TRANSCRIPT_VERSION, "python_capture")
        return
    if (
        capture.kind != "PARSER_REJECT_OBSERVATION"
        or capture.profile_id != PYTHON_REJECT_PROFILE
    ):
        _fail("PYTHON_REJECT_CAPTURE_PROFILE_INVALID", "python_capture")
    if set(value) != {
        "capture_kind",
        "capture_schema",
        "case_id",
        "input_sha256",
        "parse_outcome",
        "profile",
    }:
        _fail("PYTHON_REJECT_CAPTURE_FIELDS_INVALID", "python_capture")
    _declare(value, "capture_schema", PYTHON_REJECT_CAPTURE_SCHEMA, "python_capture")
    _declare(value, "profile", PYTHON_REJECT_PROFILE, "python_capture")
    if _field(value, "capture_kind", "python_capture") != "PARSER_REJECT_OBSERVATION":
        _fail("PYTHON_REJECT_CAPTURE_KIND_INVALID", "python_capture.capture_kind")
    if _field(value, "parse_outcome", "python_capture") != "REJECT":
        _fail("PYTHON_REJECT_CAPTURE_OUTCOME_INVALID", "python_capture.parse_outcome")
    if _field(value, "case_id", "python_capture") != case.case_id:
        _fail("PYTHON_REJECT_CAPTURE_CASE_MISMATCH", "python_capture.case_id")
    expected_input_sha = hashlib.sha256(case.input_bytes).hexdigest()
    if _field(value, "input_sha256", "python_capture") != expected_input_sha:
        _fail("PYTHON_REJECT_CAPTURE_INPUT_MISMATCH", "python_capture.input_sha256")


def _validate_rust_executable(value: str | os.PathLike[str]) -> Path:
    executable = _path_argument(value, "rust_verifier")
    try:
        if executable.is_symlink():
            _fail("RUST_EXECUTABLE_SYMLINK_REJECTED", "rust_verifier")
        information = executable.stat()
    except OSError:
        _fail("RUST_EXECUTABLE_UNAVAILABLE", "rust_verifier")
    if not stat.S_ISREG(information.st_mode):
        _fail("RUST_EXECUTABLE_REGULAR_FILE_REQUIRED", "rust_verifier")
    if not os.access(executable, os.X_OK):
        _fail("RUST_EXECUTABLE_NOT_EXECUTABLE", "rust_verifier")
    try:
        return executable.resolve(strict=True)
    except OSError:
        _fail("RUST_EXECUTABLE_UNAVAILABLE", "rust_verifier")


def run_rust_raw_v1(
    executable: str | os.PathLike[str], input_path: str | os.PathLike[str]
) -> bytes:
    """Run the documented Rust CLI with bounded output and elapsed time."""

    verifier = _validate_rust_executable(executable)
    input_file = _path_argument(input_path, "rust_input")
    try:
        process = subprocess.Popen(
            [os.fspath(verifier), os.fspath(input_file)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        _fail("RUST_PROCESS_START_FAILED", "rust_execution")
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        process.wait()
        _fail("RUST_PROCESS_PIPE_SETUP_FAILED", "rust_execution")

    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", MAX_RUST_STDOUT_BYTES),
        process.stderr.fileno(): ("stderr", MAX_RUST_STDERR_BYTES),
    }
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    for stream in (process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + RUST_TIMEOUT_SECONDS
    failure_code: str | None = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_code = "RUST_PROCESS_TIMEOUT"
                break
            events = selector.select(min(remaining, 0.25))
            for key, _mask in events:
                descriptor = key.fileobj.fileno()
                name, limit = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[name]
                if len(buffer) + len(chunk) > limit:
                    failure_code = f"RUST_{name.upper()}_LIMIT_EXCEEDED"
                    break
                buffer.extend(chunk)
            if failure_code is not None:
                break
        if failure_code is not None:
            process.kill()
            process.wait()
            _fail(failure_code, "rust_execution")
        remaining = deadline - time.monotonic()
        try:
            return_code = process.wait(timeout=max(remaining, 0.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            _fail("RUST_PROCESS_TIMEOUT", "rust_execution")
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()

    stdout = bytes(buffers["stdout"])
    stderr = bytes(buffers["stderr"])
    if return_code != 0:
        _fail("RUST_PROCESS_EXIT_NONZERO", "rust_execution")
    if stderr:
        _fail("RUST_PROCESS_STDERR_NOT_EMPTY", "rust_execution")
    if not stdout:
        _fail("RUST_PROCESS_STDOUT_EMPTY", "rust_execution")
    if stdout.endswith(b"\n"):
        _fail("RUST_PROCESS_STDOUT_TRAILING_NEWLINE", "rust_execution")
    _load_json_object(stdout, "rust_execution")
    return stdout


PythonCaptureAdapter = Callable[[str, bytes, str], PythonLiveCapture]
RustExecutionRunner = Callable[[str | os.PathLike[str], Path], bytes]


def _compare_live_case(
    case: LiveCorpusCase,
    rust_executable: str | os.PathLike[str],
    python_capture_adapter: PythonCaptureAdapter,
    rust_execution_runner: RustExecutionRunner,
) -> dict[str, Any]:
    capture = python_capture_adapter(
        case.case_id, case.input_bytes, case.expected_parse_outcome
    )
    _validate_python_live_capture(case, capture)
    with tempfile.TemporaryDirectory(prefix="raw-v1-differential-") as temporary:
        temporary_root = Path(temporary)
        os.chmod(temporary_root, 0o700)
        private_input = temporary_root / "captured-input.raw"
        descriptor = os.open(
            private_input,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        write_failed = False
        try:
            offset = 0
            while offset < len(case.input_bytes):
                try:
                    written = os.write(descriptor, case.input_bytes[offset:])
                except OSError:
                    write_failed = True
                    break
                if written <= 0:
                    write_failed = True
                    break
                offset += written
        finally:
            try:
                os.close(descriptor)
            except OSError:
                write_failed = True
        if write_failed:
            _fail("RUST_PRIVATE_INPUT_WRITE_FAILED", "rust_private_input")
        rust_execution = rust_execution_runner(rust_executable, private_input)
        retained_private_input = _read_regular_file_bounded(
            private_input, "rust_private_input"
        )
        if retained_private_input != case.input_bytes:
            _fail("RUST_PRIVATE_INPUT_MUTATED", "rust_private_input")
    if type(rust_execution) is not bytes:
        _fail("ARTIFACT_NOT_BYTES", "rust_execution")
    if len(rust_execution) > MAX_ARTIFACT_BYTES:
        _fail("JSON_ARTIFACT_SIZE_LIMIT_EXCEEDED", "rust_execution")

    python_transcript = capture.payload if capture.kind == "REPLAY_TRANSCRIPT" else None
    result = compare_case(
        case.case_id,
        case.input_bytes,
        case.expectation_bytes,
        python_transcript,
        rust_execution,
    )
    capture_sha256 = hashlib.sha256(capture.payload).hexdigest()
    result["python_capture_kind"] = capture.kind
    result["sha256_bindings"]["python_capture_sha256"] = capture_sha256
    if capture.kind == "REPLAY_TRANSCRIPT":
        result["profiles"]["python"] = {
            "capture_kind": capture.kind,
            "capture_schema": PYTHON_TRANSCRIPT_SCHEMA,
            "profile_id": PYTHON_PROFILE,
            "source_marker": PYTHON_LIVE_REPLAY_SOURCE_MARKER,
            "transcript_present": True,
        }
    else:
        result["profiles"]["python"] = {
            "capture_kind": capture.kind,
            "capture_schema": PYTHON_REJECT_CAPTURE_SCHEMA,
            "profile_id": PYTHON_REJECT_PROFILE,
            "source_marker": PYTHON_LIVE_REJECT_SOURCE_MARKER,
            "transcript_present": False,
        }
        if result["differential_gate_passed"]:
            result["comparison_status"] = "PARSER_PROFILE_COMPATIBILITY_OBSERVED"
    return result


def run_live_corpus(
    corpus_root: str | os.PathLike[str],
    rust_verifier_executable: str | os.PathLike[str],
    *,
    jobs: int = MAX_LIVE_JOBS,
    python_capture_adapter: PythonCaptureAdapter | None = None,
    rust_execution_runner: RustExecutionRunner | None = None,
) -> bytes:
    """Freshly capture and compare every mechanically bound corpus case."""

    if type(jobs) is not int or not 1 <= jobs <= MAX_LIVE_JOBS:
        _fail("LIVE_JOBS_OUT_OF_RANGE", "jobs")
    cases = enumerate_live_corpus(corpus_root)
    python_adapter = python_capture_adapter or capture_python_raw_v1
    if rust_execution_runner is None:
        rust_executable = _validate_rust_executable(rust_verifier_executable)
        rust_runner = run_rust_raw_v1
    else:
        rust_executable = _path_argument(rust_verifier_executable, "rust_verifier")
        rust_runner = rust_execution_runner

    def capture(case: LiveCorpusCase) -> dict[str, Any]:
        return _compare_live_case(
            case,
            rust_executable,
            python_adapter,
            rust_runner,
        )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(capture, cases))
    return build_report(results, capture_mode=LIVE_CAPTURE_MODE)


def _write_report_file(path_value: str | os.PathLike[str], payload: bytes) -> None:
    output_path = _path_argument(path_value, "output")
    parent = output_path.parent
    try:
        if parent.is_symlink() or not parent.is_dir():
            _fail("OUTPUT_PARENT_DIRECTORY_INVALID", "output")
        if output_path.is_symlink():
            _fail("OUTPUT_SYMLINK_REJECTED", "output")
    except OSError:
        _fail("OUTPUT_PATH_UNAVAILABLE", "output")
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=parent, prefix=f".{output_path.name}.", suffix=".tmp"
        )
    except OSError:
        _fail("OUTPUT_OPEN_FAILED", "output")
    write_failed = False
    try:
        os.fchmod(descriptor, 0o644)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                write_failed = True
                break
            offset += written
        os.fsync(descriptor)
    except OSError:
        write_failed = True
    finally:
        os.close(descriptor)
    if write_failed:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        _fail("OUTPUT_WRITE_FAILED", "output")
    try:
        os.replace(temporary_path, output_path)
    except OSError:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        _fail("OUTPUT_REPLACE_FAILED", "output")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded raw-v1 cross-profile compatibility comparison"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser(
        "live", help="freshly capture and compare the complete raw-v1 corpus"
    )
    live.add_argument("--corpus-root", required=True, type=Path)
    live.add_argument("--rust-verifier", required=True, type=Path)
    live.add_argument("--output", required=True, type=Path)
    live.add_argument(
        "--jobs", type=int, choices=range(1, MAX_LIVE_JOBS + 1), default=MAX_LIVE_JOBS
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    live_runner: Callable[..., bytes] = run_live_corpus,
) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.command != "live":  # pragma: no cover - argparse enforces this
            _fail("CLI_COMMAND_INVALID", "command")
        report_bytes = live_runner(
            args.corpus_root,
            args.rust_verifier,
            jobs=args.jobs,
        )
        report = _load_json_object(report_bytes, "report")
        aggregate = _mapping(_field(report, "aggregate", "report"), "report.aggregate")
        passed = _boolean(
            _field(aggregate, "differential_gate_passed", "report.aggregate"),
            "report.aggregate.differential_gate_passed",
        )
        _write_report_file(args.output, report_bytes)
        if passed:
            return 0
        print("raw-v1-live:DIFFERENTIAL_DISAGREEMENT", file=sys.stderr)
        return 1
    except DifferentialGateError as error:
        print(f"raw-v1-live:{error.code}", file=sys.stderr)
        return 2
    except Exception:
        print("raw-v1-live:INTERNAL_CAPTURE_FAILURE", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
