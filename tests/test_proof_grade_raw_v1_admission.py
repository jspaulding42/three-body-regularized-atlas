from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from three_body_symmetry.proof_grade_raw_v1_admission import (
    MAX_ARRAY_MEMBERS,
    MAX_DECODED_STRING_UTF8_BYTES,
    MAX_INPUT_BYTES,
    CanonicalWireAdmissionError,
    ProofGradeRawV1Admission,
    RawPlanarChainCertificate,
    RawV1AdmissionStage,
    SchemaAdmissionError,
    admit_proof_grade_raw_v1,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "conformance" / "raw-v1" / "proof-grade-v04" / "cases.json"


def _cases() -> tuple[dict[str, object], ...]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return tuple(data["cases"])


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
def test_current_proof_grade_conformance_inputs_pin_admission_stages(
    case: dict[str, object],
) -> None:
    payload = (ROOT / str(case["input_path"])).read_bytes()
    expected = case["expected"]
    assert hashlib.sha256(payload).hexdigest() == case["input_sha256"]

    if expected["parse_outcome"] == "ACCEPT":
        admission = admit_proof_grade_raw_v1(payload)
        assert type(admission) is ProofGradeRawV1Admission
        assert type(admission.raw_certificate) is RawPlanarChainCertificate
        assert admission.canonical_bytes == payload
        assert admission.evidence_sha256 == case["input_sha256"]
        return

    exception_type = (
        CanonicalWireAdmissionError
        if expected["rejection_stage"] == "CANONICAL_WIRE"
        else SchemaAdmissionError
    )
    with pytest.raises(exception_type) as error_info:
        admit_proof_grade_raw_v1(payload)
    assert error_info.value.stage.value == expected["rejection_stage"]


def test_admission_owns_bytes_and_binds_the_exact_decoded_certificate() -> None:
    case = next(case for case in _cases() if case["case_id"] == "success")
    source = bytearray((ROOT / str(case["input_path"])).read_bytes())
    payload = bytes(source)
    admission = admit_proof_grade_raw_v1(payload)
    source[0] = ord("[")

    assert admission.canonical_bytes == payload
    assert admission.raw_certificate.certificate_id == "review-v03:chain:success"
    assert admission.evidence_sha256 == hashlib.sha256(payload).hexdigest()


def test_exact_bytes_input_and_integer_real_schema_classes_are_distinct() -> None:
    payload = (ROOT / "conformance/raw-v1/inputs/success.raw.json").read_bytes()
    with pytest.raises(TypeError, match="exact bytes"):
        admit_proof_grade_raw_v1(bytearray(payload))

    mutated = payload.replace(b'"schema_version":1', b'"schema_version":1.0', 1)
    with pytest.raises(SchemaAdmissionError) as error_info:
        admit_proof_grade_raw_v1(mutated)
    assert error_info.value.stage is RawV1AdmissionStage.SCHEMA


@pytest.mark.parametrize(
    "payload, message",
    (
        (b"\xef\xbb\xbf{}", "byte-order mark"),
        (b'{"a":0,"a":0}', "duplicate decoded"),
        (b"[" * 129 + b"0" + b"]" * 129, "nesting depth"),
        (b"[" + b"0," * MAX_ARRAY_MEMBERS + b"0]", "array member"),
    ),
)
def test_focused_canonical_wire_resource_failures(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(CanonicalWireAdmissionError, match=message):
        admit_proof_grade_raw_v1(payload)


def test_input_and_decoded_string_resource_limits_fail_in_wire_stage() -> None:
    with pytest.raises(CanonicalWireAdmissionError, match="input byte limit"):
        admit_proof_grade_raw_v1(b" " * (MAX_INPUT_BYTES + 1))

    oversized_string = b'{"x":"' + b"a" * MAX_DECODED_STRING_UTF8_BYTES + b'"}'
    with pytest.raises(CanonicalWireAdmissionError, match="decoded string"):
        admit_proof_grade_raw_v1(oversized_string)


def test_total_node_limit_is_checked_before_canonical_or_schema_work() -> None:
    branch = b"[" + b"0," * (MAX_ARRAY_MEMBERS - 1) + b"0]"
    oversized_tree = b"[" + b",".join((branch,) * 4) + b"]"

    with pytest.raises(CanonicalWireAdmissionError, match="total node"):
        admit_proof_grade_raw_v1(oversized_tree)
