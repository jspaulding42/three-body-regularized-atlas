"""Byte-level canonical admission for the proof-grade raw-v1 envelope.

This is intentionally a parser boundary, not a semantic replay boundary.  It
owns one exact canonical JSON byte string and binds it to the exact
``RawPlanarChainCertificate`` decoded from those bytes.  Wire failures always
precede typed-schema failures, matching the raw-v1 proof-grade envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, NoReturn

from .proof_carrying_planar_chain import RawPlanarChainCertificate


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_NESTING_DEPTH = 128
MAX_TOTAL_NODES = 1_000_000
MAX_DECODED_STRING_UTF8_BYTES = 8 * 1024 * 1024
MAX_ARRAY_MEMBERS = 250_000
MAX_OBJECT_MEMBERS = 250_000
MAX_NUMBER_SIGNIFICAND_DIGITS = 4096
MAX_NUMBER_EXPONENT_DIGITS = 6
MAX_ABSOLUTE_DECIMAL_EXPONENT = 10_000

RAW_V1_ADMISSION_PROFILE_ID = "proof_grade_raw_v1_canonical_admission_v04"


class RawV1AdmissionStage(str, Enum):
    """The stable stage at which raw-v1 byte admission stopped."""

    CANONICAL_WIRE = "CANONICAL_WIRE"
    SCHEMA = "SCHEMA"


class RawV1AdmissionError(ValueError):
    """Base class for a total, stage-classified admission rejection."""

    stage: RawV1AdmissionStage


class CanonicalWireAdmissionError(RawV1AdmissionError):
    """JSON syntax, resource, scalar, or canonical-byte rejection."""

    stage = RawV1AdmissionStage.CANONICAL_WIRE


class SchemaAdmissionError(RawV1AdmissionError):
    """Typed raw-v1 wire-schema rejection after canonical byte admission."""

    stage = RawV1AdmissionStage.SCHEMA


@dataclass(frozen=True)
class ProofGradeRawV1Admission:
    """Immutable canonical bytes bound to their exact raw-v1 certificate."""

    canonical_bytes: bytes
    raw_certificate: RawPlanarChainCertificate
    evidence_sha256: str

    @property
    def profile_id(self) -> str:
        return RAW_V1_ADMISSION_PROFILE_ID

    @property
    def certificate(self) -> RawPlanarChainCertificate:
        """The exact certificate decoded from ``canonical_bytes``."""

        return self.raw_certificate


# The shorter name is an alias, not a second result type.
CanonicalRawV1Admission = ProofGradeRawV1Admission


def admit_proof_grade_raw_v1(payload: bytes) -> ProofGradeRawV1Admission:
    """Admit exact canonical raw-v1 bytes without starting semantic replay.

    ``payload`` must be exactly ``bytes``.  The wire phase parses an RFC 8259
    JSON value under the frozen resource limits, preserves integer versus
    real scalar classes, and compares the CPython canonical serializer bytes
    before any raw-v1 schema code executes.
    """

    if type(payload) is not bytes:
        raise TypeError("proof-grade raw-v1 admission requires exact bytes")
    value = _parse_canonical_wire(payload)
    _check_raw_v1_scalar_schema(value, "$")
    try:
        certificate = RawPlanarChainCertificate.from_dict(value)
    except Exception as error:
        raise SchemaAdmissionError(f"raw-v1 schema rejected: {error}") from error
    if type(certificate) is not RawPlanarChainCertificate:
        raise SchemaAdmissionError("raw-v1 schema returned a non-exact certificate")
    # ``bytes(bytes_value)`` may return its argument; copying through a mutable
    # buffer makes the ownership claim explicit even though bytes are immutable.
    owned_bytes = bytes(bytearray(payload))
    return ProofGradeRawV1Admission(
        canonical_bytes=owned_bytes,
        raw_certificate=certificate,
        evidence_sha256=hashlib.sha256(owned_bytes).hexdigest(),
    )


admit_canonical_raw_v1 = admit_proof_grade_raw_v1


def _parse_canonical_wire(payload: bytes) -> Any:
    if len(payload) > MAX_INPUT_BYTES:
        raise CanonicalWireAdmissionError("JSON input byte limit exceeded")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CanonicalWireAdmissionError("a UTF-8 byte-order mark is forbidden")
    try:
        text = payload.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise CanonicalWireAdmissionError("JSON input is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_int=_parse_integer_token,
            parse_float=_parse_real_token,
            parse_constant=_reject_nonfinite_token,
            object_pairs_hook=_object_without_duplicates,
        )
    except CanonicalWireAdmissionError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise CanonicalWireAdmissionError("invalid RFC 8259 JSON value") from error
    _check_wire_resources_and_strings(value)
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise CanonicalWireAdmissionError(
            "JSON value cannot be CPython-canonically encoded"
        ) from error
    if canonical != payload:
        raise CanonicalWireAdmissionError("JSON bytes are not canonical")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CanonicalWireAdmissionError(
                "duplicate decoded JSON object key"
            )
        value[key] = item
    return value


def _reject_nonfinite_token(token: str) -> NoReturn:
    raise CanonicalWireAdmissionError(f"non-finite JSON token {token!r}")


_EXPONENT_RE = re.compile(r"[eE]([+-]?)([0-9]+)$")


def _number_components(token: str) -> tuple[int, int, int, int]:
    """Return significand/exponent digit counts and both decimal exponents."""

    exponent_match = _EXPONENT_RE.search(token)
    significand = token if exponent_match is None else token[: exponent_match.start()]
    exponent_digits = 0
    written_exponent = 0
    if exponent_match is not None:
        sign, digits = exponent_match.groups()
        exponent_digits = len(digits)
        written_exponent = int(digits)
        if sign == "-":
            written_exponent = -written_exponent
    unsigned = significand[1:] if significand.startswith("-") else significand
    fraction_digits = len(unsigned.partition(".")[2])
    significand_digits = sum(character.isdigit() for character in unsigned)
    return (
        significand_digits,
        exponent_digits,
        written_exponent,
        written_exponent - fraction_digits,
    )


def _check_number_limits(token: str) -> None:
    (
        significand_digits,
        exponent_digits,
        written_exponent,
        effective_exponent,
    ) = _number_components(token)
    if significand_digits > MAX_NUMBER_SIGNIFICAND_DIGITS:
        raise CanonicalWireAdmissionError("JSON number significand digit limit exceeded")
    if exponent_digits > MAX_NUMBER_EXPONENT_DIGITS:
        raise CanonicalWireAdmissionError("JSON number exponent digit limit exceeded")
    if (
        abs(written_exponent) > MAX_ABSOLUTE_DECIMAL_EXPONENT
        or abs(effective_exponent) > MAX_ABSOLUTE_DECIMAL_EXPONENT
    ):
        raise CanonicalWireAdmissionError("JSON number effective exponent limit exceeded")


def _parse_integer_token(token: str) -> int:
    _check_number_limits(token)
    try:
        return int(token)
    except ValueError as error:
        raise CanonicalWireAdmissionError("invalid JSON integer token") from error


def _parse_real_token(token: str) -> float:
    _check_number_limits(token)
    try:
        value = float(token)
    except ValueError as error:
        raise CanonicalWireAdmissionError("invalid JSON real token") from error
    if not math.isfinite(value):
        raise CanonicalWireAdmissionError("JSON real overflows binary64")
    return value


def _check_wire_resources_and_strings(value: Any) -> None:
    """Check Rust-parity resource caps after strict decoded JSON parsing."""

    nodes = 0
    decoded_string_bytes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, parent_depth = stack.pop()
        nodes += 1
        if nodes > MAX_TOTAL_NODES:
            raise CanonicalWireAdmissionError("JSON total node limit exceeded")
        if type(item) is str:
            decoded_string_bytes = _add_string_bytes(
                decoded_string_bytes,
                item,
            )
            continue
        if type(item) is list:
            depth = parent_depth + 1
            if depth > MAX_NESTING_DEPTH:
                raise CanonicalWireAdmissionError("JSON nesting depth limit exceeded")
            if len(item) > MAX_ARRAY_MEMBERS:
                raise CanonicalWireAdmissionError("JSON array member limit exceeded")
            stack.extend((child, depth) for child in reversed(item))
            continue
        if type(item) is dict:
            depth = parent_depth + 1
            if depth > MAX_NESTING_DEPTH:
                raise CanonicalWireAdmissionError("JSON nesting depth limit exceeded")
            if len(item) > MAX_OBJECT_MEMBERS:
                raise CanonicalWireAdmissionError("JSON object member limit exceeded")
            for key, child in reversed(tuple(item.items())):
                decoded_string_bytes = _add_string_bytes(decoded_string_bytes, key)
                stack.append((child, depth))
            continue
        if type(item) not in {type(None), bool, int, float}:
            raise CanonicalWireAdmissionError("JSON syntax tree scalar class mismatch")


def _add_string_bytes(current: int, value: str) -> int:
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise CanonicalWireAdmissionError("JSON contains a lone Unicode surrogate")
    total = current + len(value.encode("utf-8"))
    if total > MAX_DECODED_STRING_UTF8_BYTES:
        raise CanonicalWireAdmissionError("JSON decoded string byte limit exceeded")
    return total


def _schema_error(path: str, detail: str) -> NoReturn:
    raise SchemaAdmissionError(f"raw-v1 schema error at {path}: {detail}")


def _object(value: Any, path: str, fields: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict:
        _schema_error(path, "expected object")
    unknown = next((key for key in value if key not in fields), None)
    if unknown is not None:
        _schema_error(f"{path}.{unknown}", "unknown field")
    missing = next((field for field in fields if field not in value), None)
    if missing is not None:
        _schema_error(f"{path}.{missing}", "missing field")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        _schema_error(path, "expected array")
    return value


def _string(value: Any, path: str) -> None:
    if type(value) is not str:
        _schema_error(path, "expected string")


def _real(value: Any, path: str) -> None:
    if type(value) is not float:
        _schema_error(path, "expected real-number token")


def _integer(value: Any, path: str) -> None:
    if type(value) is not int:
        _schema_error(path, "expected integer token")


def _boolean(value: Any, path: str) -> None:
    if type(value) is not bool:
        _schema_error(path, "expected Boolean")


def _field_string(value: dict[str, Any], path: str, field: str) -> None:
    _string(value[field], f"{path}.{field}")


def _field_real(value: dict[str, Any], path: str, field: str) -> None:
    _real(value[field], f"{path}.{field}")


def _field_integer(value: dict[str, Any], path: str, field: str) -> None:
    _integer(value[field], f"{path}.{field}")


def _real_vector(value: Any, path: str) -> None:
    for index, item in enumerate(_array(value, path)):
        _real(item, f"{path}[{index}]")


def _integer_pair(value: Any, path: str) -> None:
    items = _array(value, path)
    if len(items) != 2:
        _schema_error(path, f"expected array length 2, found length {len(items)}")
    for index, item in enumerate(items):
        _integer(item, f"{path}[{index}]")


def _real_pair(value: Any, path: str) -> None:
    items = _array(value, path)
    if len(items) != 2:
        _schema_error(path, f"expected array length 2, found length {len(items)}")
    for index, item in enumerate(items):
        _real(item, f"{path}[{index}]")


def _real_matrix(value: Any, path: str) -> None:
    for index, item in enumerate(_array(value, path)):
        _real_vector(item, f"{path}[{index}]")


def _real_tensor3(value: Any, path: str) -> None:
    for index, item in enumerate(_array(value, path)):
        _real_matrix(item, f"{path}[{index}]")


def _check_root_binding(value: Any, path: str) -> None:
    fields = (
        "binding_id", "chart_id", "masses", "initial_time", "chart_parameter",
        "positions", "velocities", "time_tolerance", "position_tolerance",
        "velocity_tolerance", "source",
    )
    item = _object(value, path, fields)
    for field in ("binding_id", "chart_id", "source"):
        _field_string(item, path, field)
    _real_vector(item["masses"], f"{path}.masses")
    for field in (
        "initial_time", "chart_parameter", "time_tolerance",
        "position_tolerance", "velocity_tolerance",
    ):
        _field_real(item, path, field)
    _real_matrix(item["positions"], f"{path}.positions")
    _real_matrix(item["velocities"], f"{path}.velocities")


def _check_ordinary_chart(value: Any, path: str) -> None:
    fields = (
        "certificate_id", "chart_id", "chart_type", "masses",
        "position_coefficients", "velocity_coefficients", "parameter_interval",
        "physical_time_interval", "coefficient_tolerance", "residual_tolerance",
        "tail_bound", "sample_count", "source",
    )
    item = _object(value, path, fields)
    for field in ("certificate_id", "chart_id", "chart_type", "source"):
        _field_string(item, path, field)
    _real_vector(item["masses"], f"{path}.masses")
    _real_tensor3(item["position_coefficients"], f"{path}.position_coefficients")
    _real_tensor3(item["velocity_coefficients"], f"{path}.velocity_coefficients")
    _real_pair(item["parameter_interval"], f"{path}.parameter_interval")
    _real_pair(item["physical_time_interval"], f"{path}.physical_time_interval")
    for field in ("coefficient_tolerance", "residual_tolerance", "tail_bound"):
        _field_real(item, path, field)
    _field_integer(item, path, "sample_count")


def _check_ordinary_tube(value: Any, path: str) -> None:
    fields = (
        "tube_id", "chart_id", "anchor_parameter", "initial_error_bound",
        "tube_radius", "max_defect_bound", "max_lipschitz_bound", "source",
    )
    item = _object(value, path, fields)
    for field in ("tube_id", "chart_id", "source"):
        _field_string(item, path, field)
    for field in (
        "anchor_parameter", "initial_error_bound", "tube_radius",
        "max_defect_bound", "max_lipschitz_bound",
    ):
        _field_real(item, path, field)


def _check_ordinary_bridge_transition(value: Any, path: str) -> None:
    fields = (
        "transition_id", "source_chart_id", "source_tube_id", "target_chart_id",
        "target_tube_id", "source_parameter", "target_parameter", "schema_version",
        "record_type", "source",
    )
    item = _object(value, path, fields)
    for field in (
        "transition_id", "source_chart_id", "source_tube_id", "target_chart_id",
        "target_tube_id", "record_type", "source",
    ):
        _field_string(item, path, field)
    _field_real(item, path, "source_parameter")
    _field_real(item, path, "target_parameter")
    _field_integer(item, path, "schema_version")


def _check_lc_entry_transition(value: Any, path: str) -> None:
    fields = (
        "transition_id", "source_chart_id", "source_tube_id", "target_chart_id",
        "target_tube_id", "source_right_parameter", "target_left_parameter",
        "schema_version", "record_type", "source",
    )
    item = _object(value, path, fields)
    for field in (
        "transition_id", "source_chart_id", "source_tube_id", "target_chart_id",
        "target_tube_id", "record_type", "source",
    ):
        _field_string(item, path, field)
    _field_real(item, path, "source_right_parameter")
    _field_real(item, path, "target_left_parameter")
    _field_integer(item, path, "schema_version")


def _check_lc_chart(value: Any, path: str) -> None:
    fields = (
        "certificate_id", "chart_id", "chart_type", "masses", "pair",
        "z_coefficients", "z_velocity_coefficients", "pair_energy_coefficients",
        "binary_center_coefficients", "binary_center_velocity_coefficients",
        "third_offset_coefficients", "third_offset_velocity_coefficients",
        "physical_time_coefficients", "parameter_interval", "physical_time_interval",
        "coefficient_tolerance", "regularized_residual_tolerance",
        "projected_residual_tolerance", "tail_bound", "sample_count",
        "projection_rho_lower_bound", "source",
    )
    item = _object(value, path, fields)
    for field in ("certificate_id", "chart_id", "chart_type", "source"):
        _field_string(item, path, field)
    _real_vector(item["masses"], f"{path}.masses")
    _integer_pair(item["pair"], f"{path}.pair")
    for field in (
        "z_coefficients", "z_velocity_coefficients", "binary_center_coefficients",
        "binary_center_velocity_coefficients", "third_offset_coefficients",
        "third_offset_velocity_coefficients",
    ):
        _real_matrix(item[field], f"{path}.{field}")
    for field in ("pair_energy_coefficients", "physical_time_coefficients"):
        _real_vector(item[field], f"{path}.{field}")
    _real_pair(item["parameter_interval"], f"{path}.parameter_interval")
    _real_pair(item["physical_time_interval"], f"{path}.physical_time_interval")
    for field in (
        "coefficient_tolerance", "regularized_residual_tolerance",
        "projected_residual_tolerance", "tail_bound", "projection_rho_lower_bound",
    ):
        _field_real(item, path, field)
    _field_integer(item, path, "sample_count")


def _check_lc_tube(value: Any, path: str) -> None:
    fields = (
        "tube_id", "chart_id", "anchor_parameter", "initial_error_bound",
        "tube_radius", "max_defect_bound", "max_lipschitz_bound",
        "require_pair_energy_constraint", "source",
    )
    item = _object(value, path, fields)
    for field in ("tube_id", "chart_id", "source"):
        _field_string(item, path, field)
    for field in (
        "anchor_parameter", "initial_error_bound", "tube_radius",
        "max_defect_bound", "max_lipschitz_bound",
    ):
        _field_real(item, path, field)
    _boolean(item["require_pair_energy_constraint"], f"{path}.require_pair_energy_constraint")


def _check_lc_exit_transition(value: Any, path: str) -> None:
    fields = (
        "transition_id", "source_chart_id", "target_chart_id", "source_parameter",
        "target_parameter", "source",
    )
    item = _object(value, path, fields)
    for field in ("transition_id", "source_chart_id", "target_chart_id", "source"):
        _field_string(item, path, field)
    _field_real(item, path, "source_parameter")
    _field_real(item, path, "target_parameter")


def _check_segment(value: Any, path: str) -> None:
    item = _object(value, path, tuple(value) if type(value) is dict else ())
    tag = item.get("segment_type")
    if type(tag) is not str:
        _schema_error(f"{path}.segment_type", "expected string")
    if tag == "ordinary_bridge_v1":
        _object(item, path, ("transition", "target_chart", "target_tube", "segment_type"))
        _check_ordinary_bridge_transition(item["transition"], f"{path}.transition")
        _check_ordinary_chart(item["target_chart"], f"{path}.target_chart")
        _check_ordinary_tube(item["target_tube"], f"{path}.target_tube")
        return
    if tag == "planar_lc_passage_v1":
        _object(
            item,
            path,
            (
                "entry_transition", "lc_chart", "lc_tube", "exit_transition",
                "target_chart", "target_tube", "segment_type",
            ),
        )
        _check_lc_entry_transition(item["entry_transition"], f"{path}.entry_transition")
        _check_lc_chart(item["lc_chart"], f"{path}.lc_chart")
        _check_lc_tube(item["lc_tube"], f"{path}.lc_tube")
        _check_lc_exit_transition(item["exit_transition"], f"{path}.exit_transition")
        _check_ordinary_chart(item["target_chart"], f"{path}.target_chart")
        _check_ordinary_tube(item["target_tube"], f"{path}.target_tube")
        return
    _schema_error(f"{path}.segment_type", f"unknown segment_type {tag!r}")


def _check_raw_v1_scalar_schema(value: Any, path: str) -> None:
    fields = (
        "certificate_id", "root_binding", "initial_chart", "initial_tube", "segments",
        "requested_target_time", "requested_maximum_component_width", "schema_version",
        "certificate_type", "source",
    )
    item = _object(value, path, fields)
    for field in ("certificate_id", "certificate_type", "source"):
        _field_string(item, path, field)
    _check_root_binding(item["root_binding"], f"{path}.root_binding")
    _check_ordinary_chart(item["initial_chart"], f"{path}.initial_chart")
    _check_ordinary_tube(item["initial_tube"], f"{path}.initial_tube")
    segments = _array(item["segments"], f"{path}.segments")
    for index, segment in enumerate(segments):
        _check_segment(segment, f"{path}.segments[{index}]")
    _field_real(item, path, "requested_target_time")
    _field_real(item, path, "requested_maximum_component_width")
    _field_integer(item, path, "schema_version")
