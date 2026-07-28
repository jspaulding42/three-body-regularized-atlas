use three_body_planar_chain_verifier::{
    replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04,
    replay_raw_mixed_planar_chain_exact_rational_v04, CanonicalRawV1Admission,
    MixedChainRetainedRegion, OrdinaryChartReplayError, PlanarLcChartReplayError,
    PlanarLcSeriesError, ProofGradeMixedPlanarSegmentReplay,
    EXACT_RATIONAL_PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID,
    PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

fn proof(bytes: &[u8]) -> three_body_planar_chain_verifier::ProofGradeRawMixedPlanarChainReplay {
    let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
    replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap()
}

fn proof_and_compat(
    bytes: &[u8],
) -> (
    three_body_planar_chain_verifier::ProofGradeRawMixedPlanarChainReplay,
    three_body_planar_chain_verifier::RawMixedPlanarChainReplay,
) {
    let admission = CanonicalRawV1Admission::admit(bytes).unwrap();
    (
        replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap(),
        replay_raw_mixed_planar_chain_exact_rational_v04(&admission).unwrap(),
    )
}

fn all_tail_bounds_negative(raw: &str) -> String {
    let changed = raw.replace("\"tail_bound\":1e-11", "\"tail_bound\":-1.0");
    assert_ne!(changed, raw);
    changed
}

fn assert_same_direct_terminal_artifacts(
    baseline: &three_body_planar_chain_verifier::ProofGradeRawMixedPlanarChainReplay,
    changed: &three_body_planar_chain_verifier::ProofGradeRawMixedPlanarChainReplay,
) {
    assert_eq!(changed.root_replay(), baseline.root_replay());
    assert_eq!(changed.clock_ledger(), baseline.clock_ledger());
    assert_eq!(changed.cocycle_ledger(), baseline.cocycle_ledger());
    let baseline_chart = baseline.current_chart().expect("baseline current chart");
    let changed_chart = changed.current_chart().expect("changed current chart");
    assert_eq!(
        changed_chart.certificate_id(),
        baseline_chart.certificate_id()
    );
    assert_eq!(changed_chart.chart_id(), baseline_chart.chart_id());
    assert_eq!(changed_chart.source(), baseline_chart.source());
    assert_eq!(changed_chart.masses(), baseline_chart.masses());
    assert_eq!(
        changed_chart.parameter_interval(),
        baseline_chart.parameter_interval()
    );
    assert_eq!(
        changed_chart.physical_time_interval(),
        baseline_chart.physical_time_interval()
    );
    assert_eq!(
        changed.current_clock_origin(),
        baseline.current_clock_origin()
    );
    assert_eq!(changed.target_preimage(), baseline.target_preimage());
    assert_eq!(changed.final_tube_replay(), baseline.final_tube_replay());
    assert_eq!(changed.final_enclosure(), baseline.final_enclosure());
    assert_eq!(
        changed.maximum_component_width(),
        baseline.maximum_component_width()
    );
    assert_eq!(
        changed.covered_physical_interval(),
        baseline.covered_physical_interval()
    );
    assert_eq!(changed.retained_region(), baseline.retained_region());
    assert_eq!(
        changed.mathematical_to_target(),
        baseline.mathematical_to_target()
    );
}

#[test]
fn canonical_success_has_proof_grade_ledger_and_compatibility_direct_artifacts() {
    let (proof, compatibility) = proof_and_compat(SUCCESS_RAW);
    assert_eq!(
        proof.profile_id(),
        EXACT_RATIONAL_PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_V04_PROFILE_ID
    );
    assert_eq!(
        proof
            .obligations()
            .iter()
            .map(|row| row.id())
            .collect::<Vec<_>>(),
        PROOF_GRADE_RAW_MIXED_PLANAR_CHAIN_OBLIGATION_IDS
    );
    assert!(proof.profile_satisfied(), "{proof:#?}");
    assert!(proof.mathematical_to_target());
    assert_eq!(proof.certified_segment_count(), 5);
    assert_eq!(proof.segment_replays().len(), 5);
    assert_eq!(proof.clock_ledger(), compatibility.clock_ledger());
    assert_eq!(proof.cocycle_ledger(), compatibility.cocycle_ledger());
    assert_eq!(proof.target_preimage(), compatibility.target_preimage());
    assert_eq!(proof.final_enclosure(), compatibility.final_enclosure());
    assert_eq!(
        proof.maximum_component_width(),
        compatibility.maximum_component_width()
    );
    assert_eq!(
        proof.covered_physical_interval(),
        compatibility.covered_physical_interval()
    );
    assert!(matches!(
        proof.initial_claimed_tail_chart_diagnostic(),
        Some(Ok(replay)) if replay.conditional_profile_satisfied()
    ));
    for segment in proof.segment_replays() {
        if let ProofGradeMixedPlanarSegmentReplay::PlanarLc(local) = segment {
            assert!(local.replay().conditional_profile_satisfied());
            assert!(local
                .replay()
                .entry_replay()
                .conditional_profile_satisfied());
        }
    }
    let enclosure = proof.final_enclosure().expect("12D terminal enclosure");
    assert_eq!(enclosure.position_intervals().len(), 6);
    assert_eq!(enclosure.velocity_intervals().len(), 6);
}

#[test]
fn initial_and_all_tail_mutations_are_diagnostic_only() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let baseline = proof(SUCCESS_RAW);

    let initial_only = raw.replacen("\"tail_bound\":1e-11", "\"tail_bound\":-1.0", 1);
    let initial_admission = CanonicalRawV1Admission::admit(initial_only.as_bytes()).unwrap();
    let compatibility = replay_raw_mixed_planar_chain_exact_rational_v04(&initial_admission)
        .expect("compatibility tail rejection stays a Boolean unresolved replay");
    let initial_proof =
        replay_proof_grade_raw_mixed_planar_chain_exact_rational_v04(&initial_admission).unwrap();
    assert!(!compatibility.profile_satisfied());
    assert!(initial_proof.profile_satisfied());
    assert!(matches!(
        initial_proof.initial_claimed_tail_chart_diagnostic(),
        Some(Ok(replay)) if !replay.conditional_profile_satisfied()
    ));
    assert_same_direct_terminal_artifacts(&baseline, &initial_proof);
    assert_eq!(
        initial_proof.current_chart().unwrap().tail_bound(),
        baseline.current_chart().unwrap().tail_bound(),
        "the initial-only diagnostic mutation does not reach the retained final chart"
    );

    let all_negative = all_tail_bounds_negative(raw);
    let all_proof = proof(all_negative.as_bytes());
    assert!(all_proof.profile_satisfied());
    assert_same_direct_terminal_artifacts(&baseline, &all_proof);
    assert_ne!(
        all_proof.current_chart().unwrap().tail_bound(),
        baseline.current_chart().unwrap().tail_bound(),
        "the retained final chart records its own non-decisive tail mutation"
    );
    assert!(matches!(
        all_proof.initial_claimed_tail_chart_diagnostic(),
        Some(Ok(replay)) if !replay.conditional_profile_satisfied()
    ));
    for segment in all_proof.segment_replays() {
        if let ProofGradeMixedPlanarSegmentReplay::PlanarLc(local) = segment {
            assert!(matches!(
                local.replay().entry_replay().source_claimed_tail_chart_diagnostic(),
                Ok(replay) if !replay.conditional_profile_satisfied()
            ));
            assert!(matches!(
                local.replay().entry_replay().target_claimed_tail_chart_diagnostic(),
                Ok(replay) if !replay.conditional_profile_satisfied()
            ));
        }
    }
}

#[test]
fn initial_claimed_tail_resource_exhaustion_is_retained_without_gating_target() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let padded = extend_ordinary_coefficients_to(raw, "review-v03:n:certificate:0", 146);
    let replay = proof(padded.as_bytes());
    assert!(replay.profile_satisfied());
    assert!(matches!(
        replay.initial_claimed_tail_chart_diagnostic(),
        Some(Err(OrdinaryChartReplayError::WorkLimitExceeded {
            required: 2_018_400,
            limit: 2_000_000,
        }))
    ));
    assert_same_direct_terminal_artifacts(&proof(SUCCESS_RAW), &replay);
}

#[test]
fn target_lc_claimed_tail_resource_exhaustion_is_retained_without_gating_target() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let padded = extend_planar_lc_coefficients_to(raw, "review-v03:lc:certificate:0", 89);
    let replay = proof(padded.as_bytes());
    assert!(replay.profile_satisfied(), "{replay:#?}");
    assert!(replay.mathematical_to_target());
    let ProofGradeMixedPlanarSegmentReplay::PlanarLc(segment) = &replay.segment_replays()[1] else {
        panic!("segment one must be the first proof-grade LC passage")
    };
    assert!(matches!(
        segment
            .replay()
            .entry_replay()
            .target_claimed_tail_chart_diagnostic(),
        Err(PlanarLcChartReplayError::Series(
            PlanarLcSeriesError::WorkLimitExceeded {
                required: 4_055_552,
                limit: 4_000_000,
            }
        ))
    ));
    assert_same_direct_terminal_artifacts(&proof(SUCCESS_RAW), &replay);
}

#[test]
fn failed_revisit_retains_proof_grade_lc_right_and_tail_changes_do_not_change_it() {
    let baseline = proof(FAILED_REVISIT_RAW);
    assert_eq!(baseline.certified_segment_count(), 4);
    assert_eq!(baseline.failed_segment_index(), Some(4));
    assert_eq!(
        baseline.failed_local_obligation_ids(),
        &["proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection"]
    );
    assert!(matches!(
        baseline.retained_region(),
        Some(MixedChainRetainedRegion::LiftedPlanarLcRight(frontier))
            if frontier.failed_segment_index() == 4 && frontier.lifted_state().components().len() == 14
    ));
    let raw = std::str::from_utf8(FAILED_REVISIT_RAW).unwrap();
    let tail_only = proof(all_tail_bounds_negative(raw).as_bytes());
    assert_eq!(
        tail_only.certified_segment_count(),
        baseline.certified_segment_count()
    );
    assert_eq!(
        tail_only.failed_segment_index(),
        baseline.failed_segment_index()
    );
    assert_eq!(
        tail_only.failed_local_obligation_ids(),
        baseline.failed_local_obligation_ids()
    );
    assert_eq!(tail_only.clock_ledger(), baseline.clock_ledger());
    assert_eq!(tail_only.cocycle_ledger(), baseline.cocycle_ledger());
    assert_eq!(tail_only.retained_region(), baseline.retained_region());
}

#[test]
fn direct_lc_containment_failure_rolls_back_without_committing_later_segments() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let defective = raw.replacen(
        "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":1e-06",
        "\"chart_id\":\"review-v03:n:chart:2\",\"initial_error_bound\":0.0",
        1,
    );
    assert_ne!(defective, raw);
    let replay = proof(defective.as_bytes());
    assert_eq!(replay.certified_segment_count(), 1);
    assert_eq!(replay.failed_segment_index(), Some(1));
    assert_eq!(replay.segment_replays().len(), 2);
    assert_eq!(replay.clock_ledger().len(), 2);
    assert_eq!(replay.cocycle_ledger().len(), 1);
    assert!(replay
        .failed_local_obligation_ids()
        .contains(&"proof_grade_carried_lc_exit_target_initial_ball_contains_complete_projection"));
    assert!(matches!(
        replay.segment_replays()[1],
        ProofGradeMixedPlanarSegmentReplay::PlanarLc(_)
    ));
}

#[test]
fn malformed_initial_is_empty_and_malformed_lc_target_retains_proof_right() {
    let raw = std::str::from_utf8(SUCCESS_RAW).unwrap();
    let malformed_initial = raw.replacen(
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"ordinary_taylor\"",
        "\"chart_id\":\"review-v03:n:chart:0\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    let initial = proof(malformed_initial.as_bytes());
    assert!(initial.obligations().iter().all(|row| !row.satisfied()));
    assert!(initial.initial_claimed_tail_chart_diagnostic().is_none());
    assert!(initial.segment_replays().is_empty());

    let malformed_target = raw.replacen(
        "\"certificate_id\":\"review-v03:n:certificate:2\",\"chart_id\":\"review-v03:n:chart:2\",\"chart_type\":\"ordinary_taylor\"",
        "\"certificate_id\":\"review-v03:n:certificate:2\",\"chart_id\":\"review-v03:n:chart:2\",\"chart_type\":\"malformed_ordinary\"",
        1,
    );
    let target = proof(malformed_target.as_bytes());
    assert_eq!(target.certified_segment_count(), 1);
    assert_eq!(target.failed_segment_index(), Some(1));
    assert!(matches!(
        target.replay_failure(),
        Some(
            three_body_planar_chain_verifier::MixedChainReplayFailure::PlanarLcExitKernel {
                segment_index: 1,
                ..
            }
        )
    ));
    assert!(matches!(
        target.retained_region(),
        Some(MixedChainRetainedRegion::LiftedPlanarLcRight(frontier))
            if frontier.failed_segment_index() == 1 && frontier.lifted_state().components().len() == 14
    ));
}

fn matching_array_end(raw: &str, start: usize) -> usize {
    let bytes = raw.as_bytes();
    assert_eq!(bytes[start], b'[');
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'\"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'\"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    panic!("unterminated array")
}

fn containing_object_start(raw: &str, position: usize) -> usize {
    let mut object_stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in raw.as_bytes().iter().copied().enumerate().take(position) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'\"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'\"' => in_string = true,
            b'{' => object_stack.push(index),
            b'}' => {
                object_stack.pop().expect("unbalanced object");
            }
            _ => {}
        }
    }
    *object_stack.last().expect("anchor outside object")
}

fn extend_ordinary_coefficients_to(raw: &str, certificate_id: &str, count: usize) -> String {
    let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
    let zero = "[[0.0,0.0],[0.0,0.0],[0.0,0.0]]";
    let mut result = raw.to_owned();
    for field in ["position_coefficients", "velocity_coefficients"] {
        let anchor_start = result.find(&anchor).expect("missing record anchor");
        let object_start = containing_object_start(&result, anchor_start);
        let marker = format!("\"{field}\":[");
        let relative = result[object_start..]
            .find(&marker)
            .expect("missing anchored coefficient array");
        let array_start = object_start + relative + marker.len() - 1;
        let array_end = matching_array_end(&result, array_start);
        let inner = &result[array_start + 1..array_end];
        let mut depth = 0_usize;
        let mut existing = usize::from(!inner.is_empty());
        for byte in inner.as_bytes() {
            match byte {
                b'[' => depth += 1,
                b']' => depth -= 1,
                b',' if depth == 0 => existing += 1,
                _ => {}
            }
        }
        assert!(existing <= count);
        let padding = vec![zero; count - existing].join(",");
        let replacement = if padding.is_empty() {
            format!("[{inner}]")
        } else {
            format!("[{inner},{padding}]")
        };
        result = format!(
            "{}{}{}",
            &result[..array_start],
            replacement,
            &result[array_end + 1..]
        );
    }
    result
}

fn extend_planar_lc_coefficients_to(raw: &str, certificate_id: &str, count: usize) -> String {
    let anchor = format!("\"certificate_id\":\"{certificate_id}\"");
    let mut result = raw.to_owned();
    for (field, zero) in [
        ("z_coefficients", "[0.0,0.0]"),
        ("z_velocity_coefficients", "[0.0,0.0]"),
        ("pair_energy_coefficients", "0.0"),
        ("binary_center_coefficients", "[0.0,0.0]"),
        ("binary_center_velocity_coefficients", "[0.0,0.0]"),
        ("third_offset_coefficients", "[0.0,0.0]"),
        ("third_offset_velocity_coefficients", "[0.0,0.0]"),
        ("physical_time_coefficients", "0.0"),
    ] {
        let anchor_start = result.find(&anchor).expect("missing record anchor");
        let object_start = containing_object_start(&result, anchor_start);
        let marker = format!("\"{field}\":[");
        let relative = result[object_start..]
            .find(&marker)
            .expect("missing anchored coefficient array");
        let array_start = object_start + relative + marker.len() - 1;
        let array_end = matching_array_end(&result, array_start);
        let inner = &result[array_start + 1..array_end];
        let mut depth = 0_usize;
        let mut existing = usize::from(!inner.is_empty());
        for byte in inner.as_bytes() {
            match byte {
                b'[' => depth += 1,
                b']' => depth -= 1,
                b',' if depth == 0 => existing += 1,
                _ => {}
            }
        }
        assert!(existing <= count);
        let padding = vec![zero; count - existing].join(",");
        let replacement = if padding.is_empty() {
            format!("[{inner}]")
        } else {
            format!("[{inner},{padding}]")
        };
        result = format!(
            "{}{}{}",
            &result[..array_start],
            replacement,
            &result[array_end + 1..]
        );
    }
    result
}
