from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from three_body_symmetry.planar_chain_review_artifact import (
    ReviewArtifactError,
    strict_load_raw_planar_chain_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "conformance" / "raw-v1"
INPUT_ROOT = CORPUS_ROOT / "inputs"
EXPECTATION_ROOT = CORPUS_ROOT / "expectations"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
SOURCE_ARTIFACT_ROOT = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain"

ACCEPT = "ACCEPT"
REJECT = "REJECT"

# This admission table is deliberately independent of the expectation JSON.
RAW_V1_CASES: tuple[tuple[str, str], ...] = (
    ("failed-revisit.raw.json", ACCEPT),
    ("reject-alternate-real-spelling.json", REJECT),
    ("reject-decoded-duplicate-unicode-key.json", REJECT),
    ("reject-duplicate-outer-key.json", REJECT),
    ("reject-invalid-utf8.bin", REJECT),
    ("reject-leading-whitespace.json", REJECT),
    ("reject-lone-surrogate.json", REJECT),
    ("reject-missing-outer-field.json", REJECT),
    ("reject-nonfinite-token.json", REJECT),
    ("reject-overflow-real.json", REJECT),
    ("reject-trailing-newline.json", REJECT),
    ("reject-unknown-outer-field.json", REJECT),
    ("reject-unknown-segment-tag.json", REJECT),
    ("reject-unsorted-outer-keys.json", REJECT),
    ("reject-wrong-interval-length.json", REJECT),
    ("reject-wrong-outer-scalar-class.json", REJECT),
    ("reject-wrong-pair-length.json", REJECT),
    ("success.raw.json", ACCEPT),
)

ACCEPTED_EXPECTATIONS = (
    "failed-revisit.expected.json",
    "success.expected.json",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outer_matrix_shape(value: list[Any]) -> list[int]:
    assert value
    assert all(type(row) is list for row in value)
    widths = {len(row) for row in value}
    assert len(widths) == 1
    return [len(value), widths.pop()]


def _final_enclosure_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "enclosure_type": value["enclosure_type"],
        "physical_time_interval": value["physical_time_interval"],
        "parameter_preimage_interval": value["parameter_preimage_interval"],
        "position_shape": _outer_matrix_shape(value["position_intervals"]),
        "velocity_shape": _outer_matrix_shape(value["velocity_intervals"]),
    }


def _retained_region_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_type": value["region_type"],
        "coordinate_system": value["coordinate_system"],
        "physical_time_interval": value["physical_time_interval"],
        "parameter_interval": value["parameter_interval"],
        "component_count": len(value["component_intervals"]),
        "certified_segment_count": value["certified_segment_count"],
        "provenance_checker_id": value["provenance_checker_id"],
    }


def _expected_fact_projection(transcript: dict[str, Any]) -> dict[str, Any]:
    cocycles = transcript["cocycle_ledger"]
    return {
        "certificate_id": transcript["certificate_id"],
        "theorem_evidence_sha256": transcript["theorem_evidence_sha256"],
        "request": transcript["request"],
        "certified_segment_count": transcript["certified_segment_count"],
        "failed_segment_index": transcript["failed_segment_index"],
        "failed_segment_missing_obligations": transcript[
            "failed_segment_missing_obligations"
        ],
        "first_failed_obligation": transcript["first_failed_obligation"],
        "top_level_obligations": transcript["obligations"],
        "segment_checker_ids": transcript["segment_checker_ids"],
        "clock_vertex_count": len(transcript["clock_origin_ledger"]),
        "cocycle_kinds": [entry["cocycle_type"] for entry in cocycles],
        "cocycle_pairs": [
            entry["canonical_pair"] for entry in cocycles if "canonical_pair" in entry
        ],
        "current_chart_id": transcript["current_chart_id"],
        "current_clock_origin_interval": transcript["current_clock_origin_interval"],
        "covered_physical_time_interval": transcript[
            "covered_physical_time_interval"
        ],
        "target_parameter_preimage_interval": transcript[
            "target_parameter_preimage_interval"
        ],
        "maximum_final_component_width_present": (
            transcript["maximum_final_component_width"] is not None
        ),
        "final_enclosure_summary": _final_enclosure_summary(
            transcript["final_enclosure"]
        ),
        "retained_region_summaries": [
            _retained_region_summary(region)
            for region in transcript["retained_regions"]
        ],
    }


def test_raw_v1_admission_table_covers_the_current_input_directory() -> None:
    table_names = [file_name for file_name, _outcome in RAW_V1_CASES]
    assert len(table_names) == len(set(table_names)) == 18

    input_names = sorted(path.name for path in INPUT_ROOT.iterdir() if path.is_file())
    assert sorted(table_names) == input_names

    outcomes = [outcome for _file_name, outcome in RAW_V1_CASES]
    assert outcomes.count(ACCEPT) == 2
    assert outcomes.count(REJECT) == 16
    assert set(outcomes) == {ACCEPT, REJECT}


@pytest.mark.parametrize(("file_name", "expected_outcome"), RAW_V1_CASES)
def test_public_python_admission_matches_raw_v1_corpus(
    file_name: str,
    expected_outcome: str,
) -> None:
    payload = (INPUT_ROOT / file_name).read_bytes()
    if expected_outcome == ACCEPT:
        strict_load_raw_planar_chain_bytes(payload)
        return

    # ReviewArtifactError is the public fail-closed boundary.  Built-in UTF-8,
    # JSON, or encoding exceptions escaping here are verifier API defects, not
    # successful parser rejection.
    with pytest.raises(ReviewArtifactError) as error_info:
        strict_load_raw_planar_chain_bytes(payload)
    if file_name == "reject-lone-surrogate.json":
        assert str(error_info.value) == "raw evidence contains a lone Unicode surrogate"


def test_raw_v1_manifest_binds_every_corpus_file_and_provenance_source() -> None:
    manifest = _load_json(MANIFEST_PATH)
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["manifest_self_hash"] is None

    entries = manifest["files"]
    listed_paths = [entry["path"] for entry in entries]
    assert len(listed_paths) == len(set(listed_paths))

    actual_paths = {
        path.relative_to(CORPUS_ROOT).as_posix()
        for path in CORPUS_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST_PATH
    }
    assert set(listed_paths) == actual_paths

    entries_by_path = {entry["path"]: entry for entry in entries}
    for relative_name, entry in entries_by_path.items():
        relative_path = Path(relative_name)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        payload_path = CORPUS_ROOT / relative_path
        payload = payload_path.read_bytes()
        assert entry["bytes"] == len(payload), relative_name
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest(), relative_name

    expected_input_paths: list[str] = []
    for expectation_path in sorted(EXPECTATION_ROOT.glob("*.expected.json")):
        expectation = _load_json(expectation_path)
        input_name = expectation["input_file"]
        expected_input_paths.append(input_name)
        assert expectation["input_sha256"] == _sha256(CORPUS_ROOT / input_name)

    assert len(expected_input_paths) == len(set(expected_input_paths)) == 18
    assert set(expected_input_paths) == {
        f"inputs/{file_name}" for file_name, _outcome in RAW_V1_CASES
    }

    for file_name in ("failed-revisit.raw.json", "success.raw.json"):
        corpus_path = INPUT_ROOT / file_name
        artifact_path = SOURCE_ARTIFACT_ROOT / file_name
        assert corpus_path.read_bytes() == artifact_path.read_bytes()
        manifest_entry = entries_by_path[f"inputs/{file_name}"]
        copied_from = (CORPUS_ROOT / manifest_entry["copied_from"]).resolve()
        assert copied_from == artifact_path.resolve()

    provenance = manifest["provenance"]
    for path_key, hash_key in (
        ("source_artifact_manifest", "source_artifact_manifest_sha256"),
        ("normative_specification", "normative_specification_sha256"),
    ):
        source_path = (CORPUS_ROOT / provenance[path_key]).resolve()
        assert provenance[hash_key] == _sha256(source_path)


@pytest.mark.parametrize("expectation_name", ACCEPTED_EXPECTATIONS)
def test_accepted_expectations_are_direct_archived_transcript_projections(
    expectation_name: str,
) -> None:
    expectation = _load_json(EXPECTATION_ROOT / expectation_name)
    transcript_path = (
        CORPUS_ROOT / expectation["source_replay_transcript"]
    ).resolve()
    transcript = _load_json(transcript_path)

    assert expectation["expected_parse_outcome"] == ACCEPT
    assert expectation["expected_replay_outcome"] == transcript["status"]
    assert expectation["source_replay_sha256"] == _sha256(transcript_path)
    assert expectation["facts"] == _expected_fact_projection(transcript)
