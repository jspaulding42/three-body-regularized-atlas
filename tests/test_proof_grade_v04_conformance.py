from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "conformance" / "raw-v1" / "proof-grade-v04"
SCRIPT_PATH = ROOT / "scripts" / "proof_grade_v04_conformance.py"

SPEC = importlib.util.spec_from_file_location("proof_grade_v04_conformance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
conformance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conformance)


def test_foundation_validate_binds_exactly_eighteen_cases() -> None:
    cases = conformance.validate()

    assert len(cases) == 18
    assert [case["case_id"] for case in cases[:2]] == ["success", "failed-revisit"]
    assert cases[0]["expected"]["semantic_signature"]["status"] == "CERTIFIED_TO_T"
    assert cases[1]["expected"]["semantic_signature"]["status"] == "UNRESOLVED"


def test_cases_schema_is_explicitly_strict_and_foundation_labeled() -> None:
    schema = json.loads((CORPUS_ROOT / "cases.schema.json").read_text(encoding="utf-8"))
    cases = json.loads((CORPUS_ROOT / "cases.json").read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["case"]["additionalProperties"] is False
    assert schema["$defs"]["semantic_signature"]["additionalProperties"] is False
    numerator_pattern = schema["$defs"]["rational"]["properties"]["numerator"]["pattern"]
    assert numerator_pattern == "^(0|[1-9][0-9]*|-[1-9][0-9]*)$"
    assert re.fullmatch(numerator_pattern, "-0") is None
    assert cases["corpus_status"] == "foundation_incomplete"


def test_tampered_cases_payload_is_detected_before_replay(tmp_path: Path) -> None:
    copied = tmp_path / "proof-grade-v04"
    shutil.copytree(CORPUS_ROOT, copied)
    cases_path = copied / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases["cases"][0]["expected"]["semantic_signature"]["status"] = "UNRESOLVED"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(conformance.ConformanceError, match="cases_file"):
        conformance.validate(corpus_root=copied, repository_root=ROOT)


def test_tampered_manifest_tool_binding_is_detected(tmp_path: Path) -> None:
    copied = tmp_path / "proof-grade-v04"
    shutil.copytree(CORPUS_ROOT, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runner_script"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(conformance.ConformanceError, match="runner_script"):
        conformance.validate(corpus_root=copied, repository_root=ROOT)


def test_negative_zero_rational_is_rejected_by_the_stdlib_validator() -> None:
    with pytest.raises(conformance.ConformanceError, match="negative zero"):
        conformance._validate_rational(
            {"numerator": "-0", "denominator": "1"}, "test.rational"
        )


def test_rational_looking_object_with_extra_field_is_rejected() -> None:
    with pytest.raises(conformance.ConformanceError, match="exactly numerator and denominator"):
        conformance._assert_rationals_are_canonical(
            {"nested": {"numerator": "1", "denominator": "2", "extra": "forbidden"}}
        )


def test_built_release_binary_can_run_one_fast_rejection_when_available() -> None:
    executable = ROOT / "verifiers" / "rust-v1" / "target" / "release" / "raw_v1_proof_grade_verify"
    if not executable.is_file():
        pytest.skip("release proof-grade Rust executable is not built")

    conformance.run_rust(
        executable,
        corpus_root=CORPUS_ROOT,
        repository_root=ROOT,
        case_ids=["reject-trailing-newline"],
        timeout_seconds=30.0,
    )
