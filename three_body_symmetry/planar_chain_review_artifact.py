"""Deterministic review artifact for the finite planar-chain checker.

The candidate builder in this module is explicitly *untrusted*: its output is
only evidence after serialization and a fresh call to
``check_raw_planar_chain``.  The checked-in raw JSON files are the primary
reproducibility inputs.  They describe a supplied finite chain of regularized
chart passages, not detected physical collisions, a complete producer, or a
general solution of the three-body problem.

Replay uses the same checker implementation and is not an independent
verification.  The artifact makes no completeness or termination claim.

The masses ``(0.1, 0.2, 0.3)`` below are binary64 literals and the checker
therefore treats their exact dyadic values, not mathematical decimal tenths.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np

from .binary_chart import (
    planar_to_regularized_binary_collision_chart,
    regularized_binary_collision_chart_to_planar,
)
from .binary_series import construct_regularized_binary_taylor_solution
from .certificate_language import (
    InitialValueProblemBindingCertificate,
    OrdinaryAposterioriTubeCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    ordinary_taylor_chart_certificate_from_solution,
    planar_levi_civita_binary_chart_certificate_from_solution,
)
from .proof_carrying_carried_planar_lc_entry import (
    CarriedPlanarLCEntryTransitionRecord,
)
from .proof_carrying_carried_planar_lc_exit import (
    check_carried_planar_lc_exit,
)
from .proof_carrying_continuation import CERTIFIED_TO_T, UNRESOLVED
from .proof_carrying_ordinary_bridge import (
    OrdinaryBridgeTransitionRecord,
    check_carried_ordinary_bridge,
)
from .proof_carrying_planar_chain import (
    ClockOriginLedgerEntry,
    OrdinaryBridgeCocycleRecord,
    OrdinaryBridgeV1Segment,
    PlanarChainOrdinaryStateEnclosure,
    PlanarChainRetainedRegion,
    PlanarLCPassageCocycleRecord,
    PlanarLCPassageV1Segment,
    RawPlanarChainCertificate,
    RawPlanarChainReplayResult,
    canonical_planar_chain_evidence_json,
    check_raw_planar_chain,
    raw_planar_chain_evidence_sha256,
)
from .series import construct_taylor_solution


ARTIFACT_SCHEMA = "planar-chain-review-bundle-v1"
TRANSCRIPT_SCHEMA = "raw-planar-chain-replay-transcript-v1"
ARTIFACT_VERSION = "v0.3.0-review"
BASE_SOURCE_COMMIT = "ddb3b07554688dd9ba247251deaef202b98e6a45"
ARTIFACT_SOURCE_RELATION = (
    "review artifact added after the base commit; the base commit does not "
    "claim to contain these payloads"
)
SUCCESS_RAW_FILENAME = "success.raw.json"
SUCCESS_REPLAY_FILENAME = "success.replay.json"
FAILED_REVISIT_RAW_FILENAME = "failed-revisit.raw.json"
FAILED_REVISIT_REPLAY_FILENAME = "failed-revisit.replay.json"
MANIFEST_FILENAME = "manifest.json"

_ROOT_CHECKER_ID = "validated_ordinary_ivp_chart_checker_v1"
_ORDINARY_CHECKER_ID = "carried_ordinary_bridge_checker_v1"
_LC_CHECKER_ID = "carried_planar_lc_exit_checker_v2"
_TRUSTED_NESTED_IDENTIFIERS = {
    "ordinary_bridge_analytic_kernel_id": (
        "ordinary_autonomous_uniqueness_bridge_kernel_v1"
    ),
    "ordinary_tube_checker_id": ("independent_ordinary_aposteriori_tube_checker_v1"),
    "lc_entry_checker_id": "carried_planar_lc_entry_checker_v2",
    "lc_entry_analytic_kernel_id": ("planar_lc_constrained_lift_deck_gauge_kernel_v1"),
    "lc_exit_analytic_kernel_id": "planar_lc_analytic_kernel_v1",
    "lc_parent_invariant_id": "parent_carried_ordinary_solution_invariant_v1",
    "lc_tube_checker_id": "independent_planar_lc_aposteriori_tube_checker_v2",
    "lc_gauge_checker_id": "planar_lc_z2_gauge_gluing_checker_v1",
}


class ReviewArtifactError(ValueError):
    """Raised when a review artifact is malformed or fails replay."""


def current_environment_versions() -> dict[str, str]:
    """Return informational runtime versions; they are not proof obligations."""

    packages: dict[str, str] = {}
    for distribution in ("numpy", "scipy", "sympy"):
        try:
            packages[distribution] = package_version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        **packages,
    }


def environment_version_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare informational versions without affecting bundle validity."""

    if (
        type(manifest) is not dict
        or type(manifest.get("build_environment")) is not dict
    ):
        raise ReviewArtifactError("manifest build environment is malformed")
    recorded = manifest["build_environment"]
    current = current_environment_versions()
    return {
        "recorded": recorded,
        "current": current,
        "exact_match": recorded == current,
    }


@lru_cache(maxsize=1)
def build_untrusted_review_candidates() -> tuple[
    RawPlanarChainCertificate,
    RawPlanarChainCertificate,
]:
    """Build deterministic success/failure candidates for review.

    This numerical construction is not part of the proof boundary.  In
    particular, callers must serialize and freshly replay the raw certificate
    before relying on a status or enclosure.
    """

    masses = np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
    ordinary_step = 2.0**-40
    lc_step = 2.0**-20
    positions = np.asarray(
        ((0.0, 0.0), (1.0, 0.5), (-0.5, 1.5)),
        dtype=np.float64,
    )
    velocities = np.asarray(
        ((0.01, 0.0), (-0.005, 0.01), (0.0, -0.01)),
        dtype=np.float64,
    )

    root_solution = construct_taylor_solution(
        positions,
        velocities,
        masses,
        order=12,
    )
    root_chart = ordinary_taylor_chart_certificate_from_solution(
        root_solution,
        certificate_id="review-v03:n:certificate:0",
        chart_id="review-v03:n:chart:0",
        parameter_interval=(0.0, ordinary_step),
        physical_time_interval=(0.0, ordinary_step),
        coefficient_tolerance=1.0e-9,
        residual_tolerance=1.0,
        tail_bound=1.0e-11,
        sample_count=5,
    )
    root_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="review-v03:n:tube:0",
        chart_id=root_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=0.0,
        tube_radius=1.0e-4,
        max_defect_bound=1.0,
        max_lipschitz_bound=1.0e5,
    )

    ordinary_target_solution = construct_taylor_solution(
        root_solution.positions_at(ordinary_step),
        root_solution.velocities_at(ordinary_step),
        masses,
        order=12,
    )
    current_chart = ordinary_taylor_chart_certificate_from_solution(
        ordinary_target_solution,
        certificate_id="review-v03:n:certificate:1",
        chart_id="review-v03:n:chart:1",
        parameter_interval=(0.0, lc_step),
        physical_time_interval=(ordinary_step, ordinary_step + lc_step),
        coefficient_tolerance=1.0e-9,
        residual_tolerance=1.0,
        tail_bound=1.0e-11,
        sample_count=5,
    )
    current_tube = OrdinaryAposterioriTubeCertificate(
        tube_id="review-v03:n:tube:1",
        chart_id=current_chart.chart_id,
        anchor_parameter=0.0,
        initial_error_bound=1.0e-10,
        tube_radius=1.0e-4,
        max_defect_bound=1.0,
        max_lipschitz_bound=1.0e5,
    )
    ordinary_transition = OrdinaryBridgeTransitionRecord(
        transition_id="review-v03:ordinary:transition:0",
        source_chart_id=root_chart.chart_id,
        source_tube_id=root_tube.tube_id,
        target_chart_id=current_chart.chart_id,
        target_tube_id=current_tube.tube_id,
        source_parameter=ordinary_step,
        target_parameter=0.0,
    )
    ordinary_result = check_carried_ordinary_bridge(
        ordinary_transition,
        root_chart,
        root_tube,
        current_chart,
        current_tube,
        (Fraction(0), Fraction(0)),
    )
    if not ordinary_result.certified:
        raise ReviewArtifactError(
            "untrusted ordinary candidate did not certify: "
            f"{ordinary_result.missing_obligations!r}; "
            f"maximum-gap={ordinary_result.maximum_target_anchor_gap!r}"
        )

    segments: list[OrdinaryBridgeV1Segment | PlanarLCPassageV1Segment] = [
        OrdinaryBridgeV1Segment(
            transition=ordinary_transition,
            target_chart=current_chart,
            target_tube=current_tube,
        )
    ]
    current_solution = ordinary_target_solution
    clock = ordinary_result.target_clock_origin_interval
    settings = (
        ((0, 1), 1.0e-7, 1.0e-6, lc_step),
        ((0, 2), 3.0e-6, 2.0e-5, lc_step),
        ((1, 2), 8.0e-5, 3.0e-4, 1.0e-3),
        ((0, 1), 2.0e-3, 1.0e-2, 1.0e-2),
    )
    for passage_index, (
        pair,
        lc_initial_error,
        target_initial_error,
        target_right,
    ) in enumerate(settings):
        source_right = float(current_chart.parameter_interval[1])
        lifted = planar_to_regularized_binary_collision_chart(
            current_solution.positions_at(source_right),
            current_solution.velocities_at(source_right),
            masses,
            pair=pair,
        )
        lc_solution = construct_regularized_binary_taylor_solution(
            lifted,
            order=12,
        )
        physical_shift = 0.5 * float(clock[0] + clock[1]) + source_right
        lc_chart = planar_levi_civita_binary_chart_certificate_from_solution(
            lc_solution,
            certificate_id=(f"review-v03:lc:certificate:{passage_index}"),
            chart_id=f"review-v03:lc:chart:{passage_index}",
            parameter_interval=(0.0, lc_step),
            coefficient_tolerance=1.0e-9,
            regularized_residual_tolerance=1.0e-6,
            projected_residual_tolerance=0.1,
            tail_bound=1.0e-11,
            sample_count=5,
            projection_rho_lower_bound=1.0e-8,
            physical_time_shift=physical_shift,
        )
        lc_tube = PlanarLCAposterioriTubeCertificate(
            tube_id=f"review-v03:lc:tube:{passage_index}",
            chart_id=lc_chart.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=lc_initial_error,
            tube_radius=max(1.5 * lc_initial_error, 1.0e-4),
            max_defect_bound=1.0,
            max_lipschitz_bound=1.0e5,
        )
        entry_transition = CarriedPlanarLCEntryTransitionRecord(
            transition_id=f"review-v03:lc:entry:{passage_index}",
            source_chart_id=current_chart.chart_id,
            source_tube_id=current_tube.tube_id,
            target_chart_id=lc_chart.chart_id,
            target_tube_id=lc_tube.tube_id,
            source_right_parameter=source_right,
            target_left_parameter=0.0,
        )

        target_positions, target_velocities = (
            regularized_binary_collision_chart_to_planar(lc_solution.state_at(lc_step))
        )
        target_solution = construct_taylor_solution(
            target_positions,
            target_velocities,
            masses,
            order=12,
        )
        target_index = passage_index + 2
        target_physical_time = physical_shift + lc_solution.physical_time_at(lc_step)
        target_chart = ordinary_taylor_chart_certificate_from_solution(
            target_solution,
            certificate_id=f"review-v03:n:certificate:{target_index}",
            chart_id=f"review-v03:n:chart:{target_index}",
            parameter_interval=(0.0, target_right),
            physical_time_interval=(
                target_physical_time,
                target_physical_time + target_right,
            ),
            coefficient_tolerance=1.0e-9,
            residual_tolerance=1.0,
            tail_bound=1.0e-11,
            sample_count=5,
        )
        target_tube = OrdinaryAposterioriTubeCertificate(
            tube_id=f"review-v03:n:tube:{target_index}",
            chart_id=target_chart.chart_id,
            anchor_parameter=0.0,
            initial_error_bound=target_initial_error,
            tube_radius=max(1.5 * target_initial_error, 1.0e-4),
            max_defect_bound=1.0,
            max_lipschitz_bound=1.0e5,
        )
        exit_transition = PlanarLCToOrdinaryEnclosureTransitionCertificate(
            transition_id=f"review-v03:lc:exit:{passage_index}",
            source_chart_id=lc_chart.chart_id,
            target_chart_id=target_chart.chart_id,
            source_parameter=lc_step,
            target_parameter=0.0,
        )
        passage_result = check_carried_planar_lc_exit(
            entry_transition,
            current_chart,
            current_tube,
            lc_chart,
            lc_tube,
            exit_transition,
            target_chart,
            target_tube,
            clock,
        )
        if not passage_result.certified:
            entry_missing = (
                passage_result.entry_result.missing_obligations
                if passage_result.entry_result is not None
                else ()
            )
            raise ReviewArtifactError(
                "untrusted LC candidate did not certify at passage "
                f"{passage_index}: exit={passage_result.missing_obligations!r}; "
                f"entry={entry_missing!r}; "
                "lift-gaps="
                f"{getattr(passage_result.entry_result, 'tested_lift_max_gaps', ())!r}; "
                "source-chart-missing="
                f"{getattr(getattr(passage_result.entry_result, 'source_chart_result', None), 'missing_obligations', ())!r}; "
                "target-chart-missing="
                f"{getattr(getattr(passage_result.entry_result, 'target_chart_result', None), 'missing_obligations', ())!r}; "
                f"projected-gap={passage_result.maximum_projected_anchor_gap!r}"
            )
        segments.append(
            PlanarLCPassageV1Segment(
                entry_transition=entry_transition,
                lc_chart=lc_chart,
                lc_tube=lc_tube,
                exit_transition=exit_transition,
                target_chart=target_chart,
                target_tube=target_tube,
            )
        )
        current_chart = target_chart
        current_tube = target_tube
        current_solution = target_solution
        clock = passage_result.target_clock_origin_interval

    root_binding = InitialValueProblemBindingCertificate(
        binding_id="review-v03:root:binding",
        chart_id=root_chart.chart_id,
        masses=root_chart.masses,
        initial_time=0.0,
        chart_parameter=0.0,
        positions=root_chart.position_coefficients[0],
        velocities=root_chart.velocity_coefficients[0],
        time_tolerance=0.0,
        position_tolerance=0.0,
        velocity_tolerance=0.0,
    )
    success = RawPlanarChainCertificate(
        certificate_id="review-v03:chain:success",
        root_binding=root_binding,
        initial_chart=root_chart,
        initial_tube=root_tube,
        segments=tuple(segments),
        requested_target_time=float(
            clock[1] + Fraction.from_float(current_chart.parameter_interval[1]) / 2
        ),
        requested_maximum_component_width=1.0e-1,
    )
    final_segment = success.segments[-1]
    if type(final_segment) is not PlanarLCPassageV1Segment:
        raise ReviewArtifactError("untrusted candidate has wrong final segment")
    broken_final = replace(
        final_segment,
        target_tube=replace(
            final_segment.target_tube,
            initial_error_bound=0.0,
        ),
    )
    failed_revisit = replace(
        success,
        certificate_id="review-v03:chain:failed-revisit",
        segments=success.segments[:-1] + (broken_final,),
    )
    return success, failed_revisit


def fraction_to_json(value: Fraction) -> dict[str, str]:
    """Encode an exact rational using decimal numerator/denominator strings."""

    if type(value) is not Fraction:
        raise TypeError("value must have exact Fraction type")
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def replay_transcript(
    certificate: RawPlanarChainCertificate,
) -> dict[str, Any]:
    """Freshly replay raw evidence and return its deterministic transcript."""

    if type(certificate) is not RawPlanarChainCertificate:
        raise TypeError("certificate must have exact RawPlanarChainCertificate type")
    result = check_raw_planar_chain(certificate)
    if type(result) is not RawPlanarChainReplayResult:
        raise ReviewArtifactError("checker returned a noncanonical result type")
    return _transcript_from_fresh_result(result)


def canonical_replay_transcript_json(transcript: dict[str, Any]) -> str:
    """Serialize a transcript with one stable, finite JSON encoding."""

    if type(transcript) is not dict:
        raise TypeError("transcript must be a dict")
    return _canonical_json(transcript)


def strict_load_raw_planar_chain_bytes(
    payload: bytes,
) -> RawPlanarChainCertificate:
    """Load canonical raw JSON, rejecting duplicate keys and nonfinite values."""

    value, canonical_payload = _strict_json_value(payload)
    if type(value) is not dict:
        raise ReviewArtifactError("raw evidence root must be an object")
    try:
        certificate = RawPlanarChainCertificate.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ReviewArtifactError(f"raw evidence schema rejected: {error}") from error
    expected = canonical_planar_chain_evidence_json(certificate).encode("utf-8")
    if canonical_payload != expected:
        raise ReviewArtifactError("raw evidence bytes are noncanonical")
    return certificate


def strict_load_raw_planar_chain(path: str | Path) -> RawPlanarChainCertificate:
    return strict_load_raw_planar_chain_bytes(Path(path).read_bytes())


def export_review_bundle(directory: str | Path) -> dict[str, Any]:
    """Build, replay, and export the deterministic four-payload bundle."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    success, failed_revisit = build_untrusted_review_candidates()
    cases = {
        "success": (
            success,
            CERTIFIED_TO_T,
            SUCCESS_RAW_FILENAME,
            SUCCESS_REPLAY_FILENAME,
        ),
        "failed_revisit": (
            failed_revisit,
            UNRESOLVED,
            FAILED_REVISIT_RAW_FILENAME,
            FAILED_REVISIT_REPLAY_FILENAME,
        ),
    }
    payloads: dict[str, bytes] = {}
    case_manifest: dict[str, dict[str, str]] = {}
    transcripts: dict[str, dict[str, Any]] = {}
    for name, (certificate, expected_status, raw_name, replay_name) in cases.items():
        raw_bytes = canonical_planar_chain_evidence_json(certificate).encode("utf-8")
        parsed = strict_load_raw_planar_chain_bytes(raw_bytes)
        transcript = replay_transcript(parsed)
        _require_expected_case_transcript(name, transcript)
        replay_bytes = canonical_replay_transcript_json(transcript).encode("utf-8")
        payloads[raw_name] = raw_bytes
        payloads[replay_name] = replay_bytes
        transcripts[name] = transcript
        case_manifest[name] = {
            "raw_file": raw_name,
            "replay_file": replay_name,
            "expected_status": expected_status,
            "theorem_evidence_sha256": raw_planar_chain_evidence_sha256(parsed),
        }

    for filename, data in payloads.items():
        (destination / filename).write_bytes(data)
    checker_ids = transcripts["success"]["checker_and_kernel_ids"]
    manifest: dict[str, Any] = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "artifact_source_relation": ARTIFACT_SOURCE_RELATION,
        "build_environment": current_environment_versions(),
        "cases": case_manifest,
        "checker_and_kernel_ids": checker_ids,
        "schema_versions": transcripts["success"]["schema_versions"],
        "payload_transport_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(payloads.items())
        },
    }
    (destination / MANIFEST_FILENAME).write_bytes(
        _canonical_json(manifest).encode("utf-8")
    )
    verify_review_bundle(destination)
    return manifest


def verify_review_bundle(directory: str | Path) -> dict[str, Any]:
    """Strictly reload and freshly replay every checked-in raw payload."""

    source = Path(directory)
    manifest_payload = (source / MANIFEST_FILENAME).read_bytes()
    manifest_value, canonical_manifest = _strict_json_value(manifest_payload)
    if type(manifest_value) is not dict:
        raise ReviewArtifactError("manifest root must be an object")
    if canonical_manifest != _canonical_json(manifest_value).encode("utf-8"):
        raise ReviewArtifactError("manifest bytes are noncanonical")
    _require_manifest_shape(manifest_value)

    payload_hashes = manifest_value["payload_transport_sha256"]
    expected_filenames = {
        SUCCESS_RAW_FILENAME,
        SUCCESS_REPLAY_FILENAME,
        FAILED_REVISIT_RAW_FILENAME,
        FAILED_REVISIT_REPLAY_FILENAME,
    }
    if set(payload_hashes) != expected_filenames:
        raise ReviewArtifactError("manifest payload set is not exact")
    for filename in sorted(expected_filenames):
        payload = (source / filename).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if payload_hashes[filename] != actual_hash:
            raise ReviewArtifactError(f"transport hash mismatch for {filename}")

    fresh_transcripts: dict[str, dict[str, Any]] = {}
    for case_name in ("success", "failed_revisit"):
        case = manifest_value["cases"][case_name]
        raw_filename = case["raw_file"]
        replay_filename = case["replay_file"]
        certificate = strict_load_raw_planar_chain(source / raw_filename)
        evidence_sha = raw_planar_chain_evidence_sha256(certificate)
        if evidence_sha != case["theorem_evidence_sha256"]:
            raise ReviewArtifactError(
                f"theorem evidence digest mismatch for {case_name}"
            )
        fresh = replay_transcript(certificate)
        if fresh["status"] != case["expected_status"]:
            raise ReviewArtifactError(f"unexpected replay status for {case_name}")
        _require_expected_case_transcript(case_name, fresh)
        tracked_payload = (source / replay_filename).read_bytes()
        tracked, canonical_tracked = _strict_json_value(tracked_payload)
        fresh_payload = canonical_replay_transcript_json(fresh).encode("utf-8")
        if canonical_tracked != fresh_payload or tracked != fresh:
            raise ReviewArtifactError(
                f"tracked replay transcript mismatch for {case_name}"
            )
        fresh_transcripts[case_name] = fresh

    if (
        manifest_value["checker_and_kernel_ids"]
        != fresh_transcripts["success"]["checker_and_kernel_ids"]
    ):
        raise ReviewArtifactError("manifest checker/kernel identifiers mismatch")
    if (
        manifest_value["schema_versions"]
        != fresh_transcripts["success"]["schema_versions"]
    ):
        raise ReviewArtifactError("manifest schema versions mismatch")
    return manifest_value


def _require_expected_case_transcript(
    case_name: str,
    transcript: dict[str, Any],
) -> None:
    expected_segment_checkers = [
        _ORDINARY_CHECKER_ID,
        _LC_CHECKER_ID,
        _LC_CHECKER_ID,
        _LC_CHECKER_ID,
        _LC_CHECKER_ID,
    ]
    cocycles = transcript.get("cocycle_ledger")
    if not (
        type(cocycles) is list
        and len(cocycles) >= 4
        and [item.get("canonical_pair") for item in cocycles[1:4]]
        == [[0, 1], [0, 2], [1, 2]]
    ):
        raise ReviewArtifactError(f"unexpected chart-visit ledger for {case_name}")
    if case_name == "success":
        if not (
            transcript.get("status") == CERTIFIED_TO_T
            and transcript.get("certified_segment_count") == 5
            and transcript.get("failed_segment_index") is None
            and transcript.get("first_failed_obligation") is None
            and transcript.get("segment_checker_ids") == expected_segment_checkers
            and len(cocycles) == 5
            and cocycles[4].get("canonical_pair") == [0, 1]
            and type(transcript.get("final_enclosure")) is dict
            and type(transcript.get("maximum_final_component_width")) is dict
            and transcript.get("retained_regions") == []
        ):
            raise ReviewArtifactError("success transcript violates review contract")
        return
    if case_name == "failed_revisit":
        retained = transcript.get("retained_regions")
        if not (
            transcript.get("status") == UNRESOLVED
            and transcript.get("certified_segment_count") == 4
            and transcript.get("failed_segment_index") == 4
            and transcript.get("first_failed_obligation")
            == (
                "segment[4]:"
                "carried_lc_exit_target_initial_ball_contains_complete_projection"
            )
            and transcript.get("segment_checker_ids") == expected_segment_checkers[:4]
            and len(cocycles) == 4
            and transcript.get("final_enclosure") is None
            and type(retained) is list
            and len(retained) == 1
            and retained[0].get("region_type") == "certified_lifted_lc_right_frontier"
            and retained[0].get("coordinate_system") == "planar_lc_lifted_14"
            and retained[0].get("certified_segment_count") == 4
            and len(retained[0].get("component_intervals", [])) == 14
        ):
            raise ReviewArtifactError(
                "failed-revisit transcript violates review contract"
            )
        return
    raise ReviewArtifactError(f"unknown review case {case_name!r}")


def _transcript_from_fresh_result(result: RawPlanarChainReplayResult) -> dict[str, Any]:
    cocycles: list[dict[str, Any]] = []
    for record in result.cocycle_records:
        if type(record) is OrdinaryBridgeCocycleRecord:
            cocycles.append(
                {
                    "cocycle_type": record.cocycle_type,
                    "segment_index": record.segment_index,
                    "transition_id": record.transition_id,
                    "source_chart_id": record.source_chart_id,
                    "target_chart_id": record.target_chart_id,
                    "source_clock_origin_interval": _fraction_interval_json(
                        record.source_clock_origin_interval
                    ),
                    "target_clock_origin_interval": _fraction_interval_json(
                        record.target_clock_origin_interval
                    ),
                    "exact_parameter_translation": fraction_to_json(
                        record.exact_parameter_translation
                    ),
                }
            )
        elif type(record) is PlanarLCPassageCocycleRecord:
            cocycles.append(
                {
                    "cocycle_type": record.cocycle_type,
                    "segment_index": record.segment_index,
                    "entry_transition_id": record.entry_transition_id,
                    "exit_transition_id": record.exit_transition_id,
                    "source_chart_id": record.source_chart_id,
                    "lc_chart_id": record.lc_chart_id,
                    "target_chart_id": record.target_chart_id,
                    "source_clock_origin_interval": _fraction_interval_json(
                        record.source_clock_origin_interval
                    ),
                    "entry_time_interval": _fraction_interval_json(
                        record.entry_time_interval
                    ),
                    "exit_time_interval": _fraction_interval_json(
                        record.exit_time_interval
                    ),
                    "target_clock_origin_interval": _fraction_interval_json(
                        record.target_clock_origin_interval
                    ),
                    "canonical_pair": list(record.canonical_pair),
                    "gauge_assignment": [
                        {"patch_id": patch_id, "bit": bit}
                        for patch_id, bit in record.gauge_assignment
                    ],
                    "gauge_edges": [
                        {
                            "overlap_id": edge.overlap_id,
                            "source_chart_id": edge.source_chart_id,
                            "target_chart_id": edge.target_chart_id,
                            "parity": edge.parity,
                        }
                        for edge in record.gauge_edges
                    ],
                }
            )
        else:
            raise ReviewArtifactError("fresh replay contains unknown cocycle type")

    request = {
        "target_physical_time": _optional_float_fraction_json(
            result.requested_target_time
        ),
        "maximum_component_width": _optional_float_fraction_json(
            result.requested_maximum_component_width
        ),
    }
    transcript: dict[str, Any] = {
        "artifact_schema": TRANSCRIPT_SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "certificate_id": result.certificate_id,
        "theorem_evidence_sha256": result.evidence_sha256,
        "status": result.status,
        "schema_versions": {
            "raw_planar_chain_certificate": result.raw_certificate.schema_version,
            "raw_planar_chain_replay_result": result.schema_version,
        },
        "checker_and_kernel_ids": {
            "replay_checker_id": result.checker_id,
            "root_checker_id": _ROOT_CHECKER_ID,
            "ordinary_segment_checker_id": _ORDINARY_CHECKER_ID,
            "lc_segment_checker_id": _LC_CHECKER_ID,
            "induction_kernel_id": result.induction_kernel_id,
            "fixed_time_kernel_id": result.fixed_time_kernel_id,
            "clock_ledger_id": result.clock_ledger_id,
            "mass_arithmetic_kernel_id": result.mass_arithmetic_kernel_id,
            **_TRUSTED_NESTED_IDENTIFIERS,
        },
        "request": request,
        "obligations": [
            {
                "obligation": item.obligation,
                "certified": item.certified,
            }
            for item in result.obligations
        ],
        "first_failed_obligation": result.first_failed_obligation,
        "certified_segment_count": result.certified_segment_count,
        "failed_segment_index": result.failed_segment_index,
        "failed_segment_missing_obligations": list(
            result.failed_segment_missing_obligations
        ),
        "segment_checker_ids": list(result.certified_transition_checker_ids),
        "clock_origin_ledger": [
            _clock_entry_json(item) for item in result.clock_origin_ledger
        ],
        "cocycle_ledger": cocycles,
        "current_chart_id": result.current_chart_id,
        "current_clock_origin_interval": _optional_fraction_interval_json(
            result.current_clock_origin_interval
        ),
        "covered_physical_time_interval": _optional_fraction_interval_json(
            result.covered_physical_time_interval
        ),
        "target_parameter_preimage_interval": _optional_fraction_interval_json(
            result.target_parameter_preimage_interval
        ),
        "maximum_final_component_width": (
            fraction_to_json(result.maximum_final_component_width)
            if result.maximum_final_component_width is not None
            else None
        ),
        "final_enclosure": _final_enclosure_json(result.final_enclosure),
        "retained_regions": [
            _retained_region_json(region) for region in result.retained_regions
        ],
    }
    return transcript


def _clock_entry_json(entry: ClockOriginLedgerEntry) -> dict[str, Any]:
    return {
        "vertex_index": entry.vertex_index,
        "chart_id": entry.chart_id,
        "clock_origin_interval": _fraction_interval_json(entry.clock_origin_interval),
    }


def _final_enclosure_json(
    enclosure: PlanarChainOrdinaryStateEnclosure | None,
) -> dict[str, Any] | None:
    if enclosure is None:
        return None
    return {
        "enclosure_type": enclosure.enclosure_type,
        "physical_time_interval": _fraction_interval_json(
            enclosure.physical_time_interval
        ),
        "parameter_preimage_interval": _fraction_interval_json(
            enclosure.parameter_preimage_interval
        ),
        "position_intervals": _fraction_matrix_json(enclosure.position_intervals),
        "velocity_intervals": _fraction_matrix_json(enclosure.velocity_intervals),
    }


def _retained_region_json(region: PlanarChainRetainedRegion) -> dict[str, Any]:
    return {
        "region_type": region.region_type,
        "coordinate_system": region.coordinate_system,
        "physical_time_interval": _fraction_interval_json(
            region.physical_time_interval
        ),
        "parameter_interval": _fraction_interval_json(region.parameter_interval),
        "component_intervals": [
            _fraction_interval_json(interval) for interval in region.component_intervals
        ],
        "certified_segment_count": region.certified_segment_count,
        "provenance_checker_id": region.provenance_checker_id,
    }


def _fraction_matrix_json(
    matrix: tuple[tuple[tuple[Fraction, Fraction], ...], ...],
) -> list[list[list[dict[str, str]]]]:
    return [[_fraction_interval_json(interval) for interval in row] for row in matrix]


def _fraction_interval_json(
    interval: tuple[Fraction, Fraction],
) -> list[dict[str, str]]:
    if not (
        type(interval) is tuple
        and len(interval) == 2
        and all(type(value) is Fraction for value in interval)
    ):
        raise ReviewArtifactError("expected an exact Fraction interval")
    return [fraction_to_json(interval[0]), fraction_to_json(interval[1])]


def _optional_fraction_interval_json(
    interval: tuple[Fraction, Fraction] | tuple[()],
) -> list[dict[str, str]] | None:
    if interval == ():
        return None
    return _fraction_interval_json(interval)


def _optional_float_fraction_json(value: float | None) -> dict[str, str] | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ReviewArtifactError("request value is not a finite binary64 float")
    return fraction_to_json(Fraction.from_float(value))


def _strict_json_value(payload: bytes) -> tuple[Any, bytes]:
    if type(payload) is not bytes:
        raise TypeError("JSON payload must have exact bytes type")
    if payload.endswith(b"\n"):
        raise ReviewArtifactError("noncanonical JSON: final newline is forbidden")
    canonical_payload = payload
    try:
        text = canonical_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReviewArtifactError("JSON payload is not valid UTF-8") from error

    def reject_constant(token: str) -> None:
        raise ReviewArtifactError(f"nonfinite JSON constant {token!r}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except ReviewArtifactError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ReviewArtifactError("malformed JSON payload") from error
    _reject_nonfinite_json_numbers(value)
    return value, canonical_payload


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewArtifactError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_numbers(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ReviewArtifactError("JSON payload contains a nonfinite number")
    if type(value) is list:
        for item in value:
            _reject_nonfinite_json_numbers(item)
    elif type(value) is dict:
        for item in value.values():
            _reject_nonfinite_json_numbers(item)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_manifest_shape(manifest: dict[str, Any]) -> None:
    required = {
        "artifact_schema",
        "artifact_version",
        "base_source_commit",
        "artifact_source_relation",
        "build_environment",
        "cases",
        "checker_and_kernel_ids",
        "schema_versions",
        "payload_transport_sha256",
    }
    if set(manifest) != required:
        raise ReviewArtifactError("manifest fields are not exact")
    if not (
        manifest["artifact_schema"] == ARTIFACT_SCHEMA
        and manifest["artifact_version"] == ARTIFACT_VERSION
        and manifest["base_source_commit"] == BASE_SOURCE_COMMIT
        and manifest["artifact_source_relation"] == ARTIFACT_SOURCE_RELATION
    ):
        raise ReviewArtifactError("manifest identity is not canonical")
    environment = manifest["build_environment"]
    if not (
        type(environment) is dict
        and set(environment)
        == {"python", "python_implementation", "numpy", "scipy", "sympy"}
        and all(type(value) is str and bool(value) for value in environment.values())
    ):
        raise ReviewArtifactError("manifest build environment is malformed")
    cases = manifest["cases"]
    if type(cases) is not dict or set(cases) != {"success", "failed_revisit"}:
        raise ReviewArtifactError("manifest cases are not exact")
    expected = {
        "success": (SUCCESS_RAW_FILENAME, SUCCESS_REPLAY_FILENAME, CERTIFIED_TO_T),
        "failed_revisit": (
            FAILED_REVISIT_RAW_FILENAME,
            FAILED_REVISIT_REPLAY_FILENAME,
            UNRESOLVED,
        ),
    }
    for name, (raw_file, replay_file, status) in expected.items():
        case = cases[name]
        if not (
            type(case) is dict
            and set(case)
            == {
                "raw_file",
                "replay_file",
                "expected_status",
                "theorem_evidence_sha256",
            }
            and case["raw_file"] == raw_file
            and case["replay_file"] == replay_file
            and case["expected_status"] == status
            and _sha256_hex(case["theorem_evidence_sha256"])
        ):
            raise ReviewArtifactError(f"manifest case {name!r} is malformed")
    hashes = manifest["payload_transport_sha256"]
    if not (
        type(hashes) is dict
        and all(
            type(key) is str and _sha256_hex(value) for key, value in hashes.items()
        )
    ):
        raise ReviewArtifactError("manifest transport hashes are malformed")
    checker_ids = manifest["checker_and_kernel_ids"]
    expected_checker_keys = {
        "replay_checker_id",
        "root_checker_id",
        "ordinary_segment_checker_id",
        "lc_segment_checker_id",
        "induction_kernel_id",
        "fixed_time_kernel_id",
        "clock_ledger_id",
        "mass_arithmetic_kernel_id",
        *_TRUSTED_NESTED_IDENTIFIERS,
    }
    if not (
        type(checker_ids) is dict
        and set(checker_ids) == expected_checker_keys
        and all(type(value) is str and bool(value) for value in checker_ids.values())
    ):
        raise ReviewArtifactError("manifest checker/kernel identifiers are malformed")
    schema_versions = manifest["schema_versions"]
    if not (
        type(schema_versions) is dict
        and set(schema_versions)
        == {
            "raw_planar_chain_certificate",
            "raw_planar_chain_replay_result",
        }
        and all(type(value) is int and value >= 1 for value in schema_versions.values())
    ):
        raise ReviewArtifactError("manifest schema versions are malformed")


def _sha256_hex(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SOURCE_RELATION",
    "ARTIFACT_VERSION",
    "BASE_SOURCE_COMMIT",
    "FAILED_REVISIT_RAW_FILENAME",
    "FAILED_REVISIT_REPLAY_FILENAME",
    "MANIFEST_FILENAME",
    "ReviewArtifactError",
    "SUCCESS_RAW_FILENAME",
    "SUCCESS_REPLAY_FILENAME",
    "TRANSCRIPT_SCHEMA",
    "build_untrusted_review_candidates",
    "canonical_replay_transcript_json",
    "current_environment_versions",
    "environment_version_report",
    "export_review_bundle",
    "fraction_to_json",
    "replay_transcript",
    "strict_load_raw_planar_chain",
    "strict_load_raw_planar_chain_bytes",
    "verify_review_bundle",
]
