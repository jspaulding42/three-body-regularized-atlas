from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterator

import pytest

from three_body_symmetry.planar_chain_review_artifact import (
    FAILED_REVISIT_RAW_FILENAME,
    FAILED_REVISIT_REPLAY_FILENAME,
    MANIFEST_FILENAME,
    ReviewArtifactError,
    SUCCESS_RAW_FILENAME,
    SUCCESS_REPLAY_FILENAME,
    build_untrusted_review_candidates,
    canonical_replay_transcript_json,
    environment_version_report,
    export_review_bundle,
    fraction_to_json,
    replay_transcript,
    strict_load_raw_planar_chain,
    strict_load_raw_planar_chain_bytes,
    verify_review_bundle,
)
from three_body_symmetry.proof_carrying_continuation import (
    CERTIFIED_TO_T,
    UNRESOLVED,
)
from three_body_symmetry.proof_carrying_planar_chain import (
    OrdinaryBridgeV1Segment,
    PlanarChainRetainedRegion,
    PlanarLCPassageV1Segment,
    canonical_planar_chain_evidence_json,
    check_raw_planar_chain,
    raw_planar_chain_evidence_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_BUNDLE = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain"
CLI_PATH = ROOT / "scripts" / "certify_repeated_planar_chain.py"

# This digest deliberately pins the theorem-facing raw success evidence.  If a
# mathematically relevant candidate field changes, update it only after fresh
# replay and review of the canonical raw JSON diff.
GOLDEN_SUCCESS_THEOREM_SHA256 = (
    "ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _copy_bundle(source: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "certify_repeated_planar_chain_test_cli",
        CLI_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load planar-chain review CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fraction_objects(value: Any) -> Iterator[dict[str, str]]:
    if type(value) is dict:
        if set(value) == {"numerator", "denominator"}:
            yield value
            return
        for child in value.values():
            yield from _fraction_objects(child)
    elif type(value) is list:
        for child in value:
            yield from _fraction_objects(child)


def _assert_canonical_fraction_object(value: dict[str, str]) -> None:
    assert type(value) is dict
    assert set(value) == {"numerator", "denominator"}
    numerator = value["numerator"]
    denominator = value["denominator"]
    assert type(numerator) is str
    assert type(denominator) is str
    assert str(int(numerator)) == numerator
    assert str(int(denominator)) == denominator
    assert int(denominator) > 0
    reduced = Fraction(int(numerator), int(denominator))
    assert str(reduced.numerator) == numerator
    assert str(reduced.denominator) == denominator


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("planar-chain-review-bundle")
    export_review_bundle(destination)
    return destination


def test_builder_and_export_are_deterministic_and_pin_success_theorem_sha(
    tmp_path: Path,
):
    success_first, failure_first = build_untrusted_review_candidates()
    success_first_raw = canonical_planar_chain_evidence_json(success_first)
    failure_first_raw = canonical_planar_chain_evidence_json(failure_first)

    # Exercise a genuinely fresh producer call instead of observing the lru cache.
    build_untrusted_review_candidates.cache_clear()
    success_second, failure_second = build_untrusted_review_candidates()
    assert canonical_planar_chain_evidence_json(success_second) == success_first_raw
    assert canonical_planar_chain_evidence_json(failure_second) == failure_first_raw
    assert raw_planar_chain_evidence_sha256(success_second) == (
        GOLDEN_SUCCESS_THEOREM_SHA256
    )

    # Each export starts a new interpreter, so no producer cache or in-memory
    # certificate object can account for byte equality.
    first_directory = tmp_path / "first-process"
    second_directory = tmp_path / "second-process"
    for directory in (first_directory, second_directory):
        subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "export",
                "--directory",
                str(directory),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    for filename in (
        SUCCESS_RAW_FILENAME,
        SUCCESS_REPLAY_FILENAME,
        FAILED_REVISIT_RAW_FILENAME,
        FAILED_REVISIT_REPLAY_FILENAME,
    ):
        assert (first_directory / filename).read_bytes() == (
            second_directory / filename
        ).read_bytes()
    first_manifest = verify_review_bundle(first_directory)
    second_manifest = verify_review_bundle(second_directory)
    assert first_manifest["cases"]["success"]["theorem_evidence_sha256"] == (
        GOLDEN_SUCCESS_THEOREM_SHA256
    )
    assert second_manifest["cases"]["success"]["theorem_evidence_sha256"] == (
        GOLDEN_SUCCESS_THEOREM_SHA256
    )
    assert verify_review_bundle(first_directory) == first_manifest


def test_success_has_exactly_the_ordinary_and_four_requested_lc_segments():
    success, _failure = build_untrusted_review_candidates()
    transcript = replay_transcript(success)

    assert len(success.segments) == 5
    assert type(success.segments[0]) is OrdinaryBridgeV1Segment
    assert all(
        type(segment) is PlanarLCPassageV1Segment for segment in success.segments[1:]
    )
    assert tuple(segment.lc_chart.pair for segment in success.segments[1:]) == (
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1),
    )
    assert transcript["status"] == CERTIFIED_TO_T
    assert transcript["certified_segment_count"] == 5
    assert transcript["failed_segment_index"] is None
    assert transcript["first_failed_obligation"] is None
    assert tuple(
        record.get("canonical_pair") for record in transcript["cocycle_ledger"]
    ) == (None, [0, 1], [0, 2], [1, 2], [0, 1])
    first_revisit = transcript["cocycle_ledger"][1]
    second_revisit = transcript["cocycle_ledger"][4]
    assert first_revisit["entry_transition_id"] != second_revisit["entry_transition_id"]
    assert first_revisit["lc_chart_id"] != second_revisit["lc_chart_id"]
    first_revisit_assignment = first_revisit["gauge_assignment"]
    second_revisit_assignment = second_revisit["gauge_assignment"]
    assert first_revisit_assignment
    assert second_revisit_assignment
    first_patch_ids = {item["patch_id"] for item in first_revisit_assignment}
    second_patch_ids = {item["patch_id"] for item in second_revisit_assignment}
    assert first_patch_ids.isdisjoint(second_patch_ids)
    assert all(
        patch_id.startswith(f"{first_revisit['entry_transition_id']}:patch:")
        for patch_id in first_patch_ids
    )
    assert all(
        patch_id.startswith(f"{second_revisit['entry_transition_id']}:patch:")
        for patch_id in second_patch_ids
    )


def test_failed_final_revisit_retains_four_segment_typed_lc_frontier():
    _success, failure = build_untrusted_review_candidates()
    assert type(failure.segments[-1]) is PlanarLCPassageV1Segment
    assert failure.segments[-1].lc_chart.pair == (0, 1)

    result = check_raw_planar_chain(failure)
    transcript = replay_transcript(failure)
    assert result.status == UNRESOLVED
    assert transcript["status"] == UNRESOLVED
    assert result.certified_segment_count == 4
    assert transcript["certified_segment_count"] == 4
    assert result.failed_segment_index == 4
    assert transcript["failed_segment_index"] == 4
    assert result.first_failed_obligation == (
        "segment[4]:carried_lc_exit_target_initial_ball_contains_complete_projection"
    )
    assert len(result.retained_regions) == 1
    frontier = result.retained_regions[0]
    assert type(frontier) is PlanarChainRetainedRegion
    assert frontier.region_type == "certified_lifted_lc_right_frontier"
    assert frontier.coordinate_system == "planar_lc_lifted_14"
    assert len(frontier.component_intervals) == 14
    assert frontier.physical_time_interval == frontier.component_intervals[13]
    assert frontier.certified_segment_count == 4
    assert transcript["retained_regions"][0]["coordinate_system"] == (
        "planar_lc_lifted_14"
    )
    assert len(transcript["retained_regions"][0]["component_intervals"]) == 14


def test_transcripts_encode_rationals_as_reduced_exact_fraction_objects():
    success, failure = build_untrusted_review_candidates()
    assert fraction_to_json(Fraction(-6, 8)) == {
        "numerator": "-3",
        "denominator": "4",
    }

    for certificate in (success, failure):
        transcript = replay_transcript(certificate)
        objects = tuple(_fraction_objects(transcript))
        assert objects
        for value in objects:
            _assert_canonical_fraction_object(value)
        assert transcript["request"]["target_physical_time"] == fraction_to_json(
            Fraction.from_float(certificate.requested_target_time)
        )
        assert transcript["request"]["maximum_component_width"] == (
            fraction_to_json(
                Fraction.from_float(certificate.requested_maximum_component_width)
            )
        )
        assert canonical_replay_transcript_json(transcript).encode("utf-8") == (
            _canonical_json_bytes(transcript)
        )


def test_strict_raw_loader_rejects_duplicate_noncanonical_nonfinite_and_unknown():
    success, _failure = build_untrusted_review_candidates()
    raw = canonical_planar_chain_evidence_json(success).encode("utf-8")
    assert strict_load_raw_planar_chain_bytes(raw) == success

    duplicate_key = b'{"certificate_id":"forged",' + raw[1:]
    with pytest.raises(ReviewArtifactError, match="duplicate"):
        strict_load_raw_planar_chain_bytes(duplicate_key)
    with pytest.raises(ReviewArtifactError, match="noncanonical"):
        strict_load_raw_planar_chain_bytes(b" " + raw)
    with pytest.raises(ReviewArtifactError, match="newline"):
        strict_load_raw_planar_chain_bytes(raw + b"\n")

    nonfinite = success.to_dict()
    nonfinite["requested_target_time"] = math.nan
    nonfinite_payload = json.dumps(
        nonfinite,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    with pytest.raises(ReviewArtifactError, match="nonfinite"):
        strict_load_raw_planar_chain_bytes(nonfinite_payload)

    unknown = success.to_dict()
    unknown["unknown_theorem_claim"] = True
    with pytest.raises((ReviewArtifactError, ValueError), match="unknown"):
        strict_load_raw_planar_chain_bytes(_canonical_json_bytes(unknown))


def test_bundle_rejects_payload_tamper_even_with_rewritten_transport_hash(
    exported_bundle: Path,
    tmp_path: Path,
):
    tampered = _copy_bundle(exported_bundle, tmp_path, "payload-tamper")
    replay_path = tampered / SUCCESS_REPLAY_FILENAME
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["status"] = "FORGED_CERTIFIED_STATUS"
    replay_payload = _canonical_json_bytes(replay)
    replay_path.write_bytes(replay_payload)

    manifest_path = tampered / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_transport_sha256"][SUCCESS_REPLAY_FILENAME] = hashlib.sha256(
        replay_payload
    ).hexdigest()
    manifest_path.write_bytes(_canonical_json_bytes(manifest))

    with pytest.raises(ReviewArtifactError, match="tracked replay transcript"):
        verify_review_bundle(tampered)


def test_bundle_rejects_canonical_manifest_tamper(
    exported_bundle: Path,
    tmp_path: Path,
):
    tampered = _copy_bundle(exported_bundle, tmp_path, "manifest-tamper")
    manifest_path = tampered / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"]["success"]["theorem_evidence_sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical_json_bytes(manifest))

    with pytest.raises(ReviewArtifactError, match="theorem evidence digest"):
        verify_review_bundle(tampered)


def test_build_environment_is_informational_not_an_acceptance_gate(
    exported_bundle: Path,
    tmp_path: Path,
):
    modified = _copy_bundle(exported_bundle, tmp_path, "different-environment")
    manifest_path = modified / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_environment"] = {
        key: f"deliberately-unrelated-{key}" for key in manifest["build_environment"]
    }
    manifest_path.write_bytes(_canonical_json_bytes(manifest))

    verified = verify_review_bundle(modified)
    report = environment_version_report(verified)
    assert verified == manifest
    assert report["recorded"] == manifest["build_environment"]
    assert not report["exact_match"]


def test_checked_in_bundle_freshly_verifies_when_present():
    if not (CHECKED_IN_BUNDLE / MANIFEST_FILENAME).is_file():
        pytest.skip("checked-in v0.3 planar-chain review bundle is not present")
    manifest = verify_review_bundle(CHECKED_IN_BUNDLE)
    assert manifest["cases"]["success"]["theorem_evidence_sha256"] == (
        GOLDEN_SUCCESS_THEOREM_SHA256
    )


def test_cli_main_exports_replays_and_verifies_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    cli = _load_cli_module()
    bundle = tmp_path / "cli-bundle"
    replay_output = tmp_path / "fresh-success.replay.json"

    assert cli.main(["export", "--directory", str(bundle)]) == 0
    assert (
        cli.main(
            [
                "replay",
                str(bundle / SUCCESS_RAW_FILENAME),
                "--output",
                str(replay_output),
            ]
        )
        == 0
    )
    assert replay_output.read_bytes() == (bundle / SUCCESS_REPLAY_FILENAME).read_bytes()
    assert cli.main(["verify-bundle", "--directory", str(bundle)]) == 0
    output = capsys.readouterr().out
    assert "exported and verified" in output
    assert "wrote" in output
    assert "verified" in output

    loaded = strict_load_raw_planar_chain(bundle / SUCCESS_RAW_FILENAME)
    assert raw_planar_chain_evidence_sha256(loaded) == (GOLDEN_SUCCESS_THEOREM_SHA256)
