from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path

import pytest

from three_body_symmetry.planar_chain_review_artifact import (
    strict_load_raw_planar_chain,
)
from three_body_symmetry.proof_carrying_carried_planar_lc_entry import (
    ProofGradeCarriedPlanarLCEntryResult,
    check_carried_planar_lc_entry,
    check_proof_grade_carried_planar_lc_entry,
)
from three_body_symmetry.proof_carrying_planar_chain import (
    OrdinaryBridgeV1Segment,
    PlanarLCPassageCocycleRecord,
    PlanarLCPassageV1Segment,
    check_raw_planar_chain,
)


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_RAW = ROOT / "artifacts" / "v0.3.0-review" / "planar-chain" / "success.raw.json"

_PROOF_GRADE_OBLIGATION_IDS = (
    "proof_grade_carried_lc_entry_exact_raw_schemas",
    "proof_grade_carried_lc_entry_transition_canonical_round_trip",
    "proof_grade_carried_lc_entry_identifiers_match_and_are_unique",
    "proof_grade_carried_lc_entry_parent_clock_origin_is_exact_interval",
    "proof_grade_carried_lc_entry_source_ordinary_tube_freshly_certified",
    "proof_grade_carried_lc_entry_target_lc_tube_freshly_certified",
    "proof_grade_carried_lc_entry_common_planar_mass_problem",
    "proof_grade_carried_lc_entry_pair_is_canonical_ascending",
    "proof_grade_carried_lc_entry_outward_mass_arithmetic_certified",
    "proof_grade_carried_lc_entry_exact_source_right_to_lc_left_anchor",
    "proof_grade_carried_lc_entry_complete_source_endpoint_box_reconstructed",
    "proof_grade_carried_lc_entry_selected_pair_collision_free",
    "proof_grade_carried_lc_entry_canonical_square_root_atlas_reconstructed",
    "proof_grade_carried_lc_entry_derived_parity_graph_certified",
    "proof_grade_carried_lc_entry_physical_time_interval_exactly_derived",
    "proof_grade_carried_lc_entry_all_lift_patches_have_positive_rho",
    "proof_grade_carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
    "proof_grade_carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
    "proof_grade_carried_lc_entry_one_global_complement_contains_all_complete_patches",
)


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True


class _HostileString(str):
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _LCEntry:
    transition: object
    source_chart: object
    source_tube: object
    target_chart: object
    target_tube: object
    source_clock: tuple[Fraction, Fraction]

    @property
    def args(self) -> tuple[object, ...]:
        return (
            self.transition,
            self.source_chart,
            self.source_tube,
            self.target_chart,
            self.target_tube,
            self.source_clock,
        )


@lru_cache(maxsize=1)
def _lc_entries() -> tuple[_LCEntry, ...]:
    """Strictly load every checked-in LC leg with its carried source clock."""

    certificate = strict_load_raw_planar_chain(SUCCESS_RAW)
    replay = check_raw_planar_chain(certificate)
    assert replay.certified
    current_chart = certificate.initial_chart
    current_tube = certificate.initial_tube
    entries = []
    for index, segment in enumerate(certificate.segments):
        if type(segment) is OrdinaryBridgeV1Segment:
            current_chart = segment.target_chart
            current_tube = segment.target_tube
            continue
        assert type(segment) is PlanarLCPassageV1Segment
        cocycle = next(
            record
            for record in replay.cocycle_records
            if type(record) is PlanarLCPassageCocycleRecord
            and record.segment_index == index
        )
        entries.append(
            _LCEntry(
                transition=segment.entry_transition,
                source_chart=current_chart,
                source_tube=current_tube,
                target_chart=segment.lc_chart,
                target_tube=segment.lc_tube,
                source_clock=cocycle.source_clock_origin_interval,
            )
        )
        current_chart = segment.target_chart
        current_tube = segment.target_tube
    return tuple(entries)


def _first_lc_entry() -> _LCEntry:
    return _lc_entries()[0]


def _proof(entry: _LCEntry) -> ProofGradeCarriedPlanarLCEntryResult:
    result = check_proof_grade_carried_planar_lc_entry(*entry.args)
    assert type(result) is ProofGradeCarriedPlanarLCEntryResult
    return result


def test_proof_grade_first_lc_success_is_separate_and_exactly_versioned():
    entry = _first_lc_entry()
    proof = _proof(entry)
    compatibility = check_carried_planar_lc_entry(*entry.args)

    assert proof.certified
    assert proof.profile_id == (
        "binary64_outward_proof_grade_carried_planar_lc_entry_v04"
    )
    assert tuple(item.obligation for item in proof.obligations) == (
        _PROOF_GRADE_OBLIGATION_IDS
    )
    assert proof.missing_obligations == ()
    assert compatibility.certified


def test_proof_grade_certifies_every_checked_in_lc_pair_with_its_carried_clock():
    entries = _lc_entries()

    assert tuple(entry.target_chart.pair for entry in entries) == (
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1),
    )
    for entry in entries:
        proof = _proof(entry)
        assert proof.certified
        assert tuple(item.obligation for item in proof.obligations) == (
            _PROOF_GRADE_OBLIGATION_IDS
        )
        assert all(item.certified is True for item in proof.obligations)
        assert proof.missing_obligations == ()


def test_negative_claimed_tails_are_diagnostics_not_proof_prerequisites():
    entry = _first_lc_entry()
    baseline = _proof(entry)
    mutated = replace(
        entry,
        source_chart=replace(entry.source_chart, tail_bound=-1.0),
        target_chart=replace(entry.target_chart, tail_bound=-1.0),
    )
    proof = _proof(mutated)
    compatibility = check_carried_planar_lc_entry(*mutated.args)

    assert proof.certified
    assert proof.obligations == baseline.obligations
    assert proof.selected_assignment == baseline.selected_assignment
    assert proof.timed_lifted_patch_boxes == baseline.timed_lifted_patch_boxes
    assert proof.entry_time_interval == baseline.entry_time_interval
    assert proof.source_endpoint_state_box == baseline.source_endpoint_state_box
    assert proof.target_anchor == baseline.target_anchor
    assert proof.tested_containments == baseline.tested_containments
    assert proof.source_chart_result is not None
    assert proof.target_chart_result is not None
    assert not proof.source_chart_result.certified
    assert not proof.target_chart_result.certified
    assert not compatibility.certified


@pytest.mark.parametrize("diagnostic_name", ("source_chart_result", "target_chart_result"))
def test_proof_diagnostics_are_nondecisive_but_exactly_replayed(
    diagnostic_name: str,
):
    result = _proof(_first_lc_entry())
    diagnostic = getattr(result, diagnostic_name)
    assert diagnostic is not None
    forged_diagnostic = replace(
        diagnostic,
        max_coefficient_residual=diagnostic.max_coefficient_residual + 1.0,
    )
    forged = replace(result, **{diagnostic_name: forged_diagnostic})

    assert not forged.certified


def test_forged_direct_evidence_never_passes_exact_self_certification():
    result = _proof(_first_lc_entry())
    forged = replace(
        result,
        tested_lift_max_gaps=(
            result.tested_lift_max_gaps[0] + Fraction(1),
            result.tested_lift_max_gaps[1],
        ),
    )

    assert not forged.certified


def test_hostile_equality_payload_never_bypasses_exact_snapshot_types():
    result = _proof(_first_lc_entry())
    forged = replace(result, patch_vertex_ids=(_AlwaysEqual(),))

    assert not forged.certified


def test_hostile_string_subclass_identifier_never_bypasses_exact_snapshot_types():
    result = _proof(_first_lc_entry())
    forged = replace(
        result,
        transition_id=_HostileString(result.transition_id),
    )

    assert not forged.certified


@pytest.mark.parametrize(
    ("mutation", "expected_missing"),
    (
        (
            lambda entry: replace(
                entry,
                source_tube=replace(
                    entry.source_tube,
                    max_defect_bound=0.0,
                ),
            ),
            (
                "proof_grade_carried_lc_entry_source_ordinary_tube_freshly_certified",
                "proof_grade_carried_lc_entry_complete_source_endpoint_box_reconstructed",
                "proof_grade_carried_lc_entry_selected_pair_collision_free",
                "proof_grade_carried_lc_entry_canonical_square_root_atlas_reconstructed",
                "proof_grade_carried_lc_entry_derived_parity_graph_certified",
                "proof_grade_carried_lc_entry_physical_time_interval_exactly_derived",
                "proof_grade_carried_lc_entry_all_lift_patches_have_positive_rho",
                "proof_grade_carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
                "proof_grade_carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
                "proof_grade_carried_lc_entry_one_global_complement_contains_all_complete_patches",
            ),
        ),
        (
            lambda entry: replace(
                entry,
                transition=replace(
                    entry.transition,
                    source_right_parameter=math.nextafter(
                        entry.transition.source_right_parameter,
                        -math.inf,
                    ),
                ),
            ),
            (
                "proof_grade_carried_lc_entry_exact_source_right_to_lc_left_anchor",
                "proof_grade_carried_lc_entry_complete_source_endpoint_box_reconstructed",
                "proof_grade_carried_lc_entry_selected_pair_collision_free",
                "proof_grade_carried_lc_entry_canonical_square_root_atlas_reconstructed",
                "proof_grade_carried_lc_entry_derived_parity_graph_certified",
                "proof_grade_carried_lc_entry_physical_time_interval_exactly_derived",
                "proof_grade_carried_lc_entry_all_lift_patches_have_positive_rho",
                "proof_grade_carried_lc_entry_target_fourteen_dimensional_anchor_reconstructed",
                "proof_grade_carried_lc_entry_trusted_constrained_lift_deck_gauge_kernel",
                "proof_grade_carried_lc_entry_one_global_complement_contains_all_complete_patches",
            ),
        ),
    ),
)
def test_direct_source_or_endpoint_mutation_reports_exact_missing_profile_rows(
    mutation: object,
    expected_missing: tuple[str, ...],
):
    entry = _first_lc_entry()
    result = _proof(mutation(entry))

    assert not result.certified
    assert result.missing_obligations == expected_missing


@pytest.mark.parametrize(
    ("target_tube_change", "missing_obligation"),
    (
        (
            {"max_defect_bound": 0.0},
            "proof_grade_carried_lc_entry_target_lc_tube_freshly_certified",
        ),
        (
            {"initial_error_bound": 0.0},
            "proof_grade_carried_lc_entry_one_global_complement_contains_all_complete_patches",
        ),
    ),
)
def test_direct_tube_or_containment_mutation_rejects_proof_profile(
    target_tube_change: dict[str, float],
    missing_obligation: str,
):
    entry = _first_lc_entry()
    mutated = replace(
        entry,
        target_tube=replace(entry.target_tube, **target_tube_change),
    )
    result = _proof(mutated)

    assert not result.certified
    assert missing_obligation in result.missing_obligations
